#!/usr/bin/env python3
"""Finalize sequence-matched known-positive Boltz96 with pose and hash audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.production import bool_series, file_sha256  # noqa: E402
from scripts.audit_boltz_pose_stability import (  # noqa: E402
    audit_pair,
    pocket_residue_ids_from_yaml,
)
from scripts.finalize_boltz_refined_3000_package import collect_result_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/current_pipeline_v4.yaml")
    parser.add_argument(
        "--source",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "known_control_boltz96_v4.csv"
        ),
    )
    parser.add_argument(
        "--input-package",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "known_control_boltz96_input_v4_sequence_matched_signed"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "known_control_boltz96_run_v4_sequence_matched_seeded"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "known_control_boltz96_refined_pose_audited_v4.csv"
        ),
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = pd.read_csv(args.source, low_memory=False).fillna("")
    package_dir = Path(args.input_package)
    manifest_path = package_dir / "boltz2_input_manifest.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    run_dir = Path(args.run_dir)
    run_plan = json.loads((run_dir / "run_plan.json").read_text(encoding="utf-8"))
    if len(source) != 96 or source["pair_id"].duplicated().any():
        raise RuntimeError("Known-control source row contract failed")
    if len(manifest) != 96 or manifest["pairId"].duplicated().any():
        raise RuntimeError("Known-control signed manifest row contract failed")
    if set(source["pair_id"].astype(str)) != set(manifest["pairId"].astype(str)):
        raise RuntimeError("Known-control source/manifest pair coverage differs")
    if run_plan.get("runParameters") != config["boltz_contract"]:
        raise RuntimeError("Known-control run parameters differ from v4 contract")
    if run_plan.get("inputManifestSha256") != file_sha256(manifest_path):
        raise RuntimeError("Known-control run plan manifest hash mismatch")

    extended_path = run_dir / "result_provenance_with_output_hashes.csv"
    extended_summary_path = run_dir / "result_provenance_with_output_hashes.summary.json"
    if not extended_path.exists() or not extended_summary_path.exists():
        raise FileNotFoundError("Known-control output-hash provenance is missing")
    extended = pd.read_csv(extended_path, low_memory=False).fillna("")
    extended_summary = json.loads(extended_summary_path.read_text(encoding="utf-8"))
    if len(extended) != 96 or extended["pairId"].duplicated().any():
        raise RuntimeError("Known-control output provenance row contract failed")
    if extended_summary.get("output_provenance_sha256") != file_sha256(extended_path):
        raise RuntimeError("Known-control output provenance hash mismatch")

    results = collect_result_files([("known_control_v4", run_dir / "batch_runs")])
    if len(results) != 96 or results["boltz_stem"].duplicated().any():
        raise RuntimeError(f"Known-control Boltz output row contract failed: {len(results)}")
    manifest["boltz_stem"] = manifest["yamlFile"].astype(str).map(lambda value: Path(value).stem)
    mapping = manifest[["pairId", "boltz_stem", "yamlFile", "inputSignatureSha256"]].rename(
        columns={"pairId": "pair_id"}
    )
    merged = source.merge(mapping, on="pair_id", how="left", validate="one_to_one").merge(
        results, on="boltz_stem", how="left", validate="one_to_one"
    )
    if not bool_series(merged, "boltz_completed_refined").all():
        raise RuntimeError("Known-control collector found incomplete or invalid outputs")
    hash_columns = [
        "boltz_confidence_sha256_refined",
        "boltz_affinity_sha256_refined",
        "boltz_cif_model0_sha256_refined",
        "boltz_cif_model1_sha256_refined",
    ]
    extended_hashes = extended.set_index("boltz_stem")[hash_columns].astype(str).sort_index()
    observed_hashes = merged.set_index("boltz_stem")[hash_columns].astype(str).sort_index()
    if not extended_hashes.equals(observed_hashes):
        raise RuntimeError("Known-control recomputed output hashes differ from provenance")

    pose_rows = []
    for _, row in merged.iterrows():
        yaml_path = package_dir / "inputs" / str(row["yamlFile"])
        pose_rows.append(
            audit_pair(
                Path(str(row["boltz_cif_path_refined"])),
                pocket_residue_ids_from_yaml(yaml_path),
                ligand_smiles=str(row.get("model_ligand_smiles", "")),
            )
        )
    pose = pd.DataFrame(pose_rows, index=merged.index)
    for column in pose.columns:
        merged[column] = pose[column]
    if not bool_series(merged, "pose_stability_completed").all():
        raise RuntimeError("Known-control pose audit is incomplete")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    summary = {
        "rows": len(merged),
        "completed_rows": int(bool_series(merged, "boltz_completed_refined").sum()),
        "pose_completed_rows": int(bool_series(merged, "pose_stability_completed").sum()),
        "pose_tiers": merged["pose_stability_tier"].value_counts().to_dict(),
        "sequence_match_status": merged["sequence_match_status"].value_counts().to_dict(),
        "source_sha256": file_sha256(Path(args.source)),
        "manifest_sha256": file_sha256(manifest_path),
        "run_plan_sha256": file_sha256(run_dir / "run_plan.json"),
        "output_provenance_sha256": file_sha256(extended_path),
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
