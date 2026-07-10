#!/usr/bin/env python3
"""Build and validate the frozen no-Top300 v4 project universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_full_project_universe_v3 import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/current_pipeline_v4.yaml"))
    parser.add_argument("--write-full-universe", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = build(
        config_path,
        output_subdir="full_untruncated_universe_v4",
        version_label="v4",
        use_legacy_recall_gate=False,
        write_full_universe=args.write_full_universe,
    )
    expected = int(config["contracts"]["project_cartesian_rows"])
    errors = []
    if summary["project_cartesian_rows"] != expected:
        errors.append(f"project rows {summary['project_cartesian_rows']} != {expected}")
    expected_physical = int(config["contracts"]["project_physical_pair_rows"])
    if summary.get("project_physical_pair_rows") != expected_physical:
        errors.append(
            f"physical pair rows {summary.get('project_physical_pair_rows')} != {expected_physical}"
        )
    if summary.get("project_unique_model_ligands") != 723:
        errors.append(
            f"unique model ligands {summary.get('project_unique_model_ligands')} != 723"
        )
    if summary["project_drugs"] != 750 or summary["project_targets"] != 463:
        errors.append(
            f"entity scope {summary['project_drugs']}x{summary['project_targets']} != 750x463"
        )
    if summary["project_drug_manifest_sha256"] != config["scope"]["project_drug_manifest_sha256"]:
        errors.append("project drug manifest hash mismatch")
    if summary["project_target_manifest_sha256"] != config["scope"]["project_target_manifest_sha256"]:
        errors.append("project target manifest hash mismatch")
    known = summary["known_calibration"]
    if known["known_union_rows"] != 491 or known["known_unique_active_moiety_target_rows"] != 473:
        errors.append(
            "known calibration contract mismatch: "
            f"id_pairs={known['known_union_rows']}, active_pairs={known['known_unique_active_moiety_target_rows']}"
        )
    if summary["selected_rows"] != int(config["contracts"]["refined_top3000_rows"]):
        errors.append(f"selected rows {summary['selected_rows']} != 3000")
    if summary["legacy_recall_gate_applied"]:
        errors.append("legacy recall gate is still enabled")
    if summary["active_moiety_target_duplicates_selected"]:
        errors.append("Top3000 contains active-moiety x target duplicates")
    if not summary["boltz_reuse_disabled"] or summary["delta_rows_requiring_boltz"] != 3000:
        errors.append("v4 must rerun all 3000 Boltz inputs without legacy reuse")
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
