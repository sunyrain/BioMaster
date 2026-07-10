#!/usr/bin/env python3
"""Finalize the no-Top300, active-moiety-deduplicated v4 package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_106k_reselection_v2 import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/current_pipeline_v4.yaml"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    stage_dir = ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v4"
    version_label = "v4_partial" if args.allow_partial else "v4_complete"
    output_subdir = "formal_full_universe_v4_partial" if args.allow_partial else "formal_full_universe_v4"
    summary = build(
        Path(args.config).resolve(),
        args.allow_partial,
        stage_dir=stage_dir,
        selected_filename="pre_boltz_top3000_v4_fully_audited.csv",
        input_package_name="boltz_full_input_package_v4_signed",
        run_name="boltz_full_run_v4_seeded",
        output_subdir=output_subdir,
        version_label=version_label,
        conplex_reference_path=(
            ROOT
            / "outputs/current_production_package_v2/full_untruncated_universe_v4_active_collapsed_sensitivity"
            / "conplex_reference_v4_active_collapsed_sensitivity.csv"
        ),
        result_source_label="v4_delta_refined",
        audit_pose_stability=not args.allow_partial,
        max_incomplete=0,
        reuse_old_results=False,
        require_input_signatures=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
