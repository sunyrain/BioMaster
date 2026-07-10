#!/usr/bin/env python3
"""Rebuild per-pair Boltz provenance with validated output-file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.finalize_boltz_refined_3000_package import collect_result_files


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--source-label", default="formal_seeded")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    run_dir = Path(args.run_dir)
    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    input_provenance_path = run_dir / "result_provenance.csv"
    if not input_provenance_path.exists():
        raise FileNotFoundError(input_provenance_path)
    input_provenance = pd.read_csv(input_provenance_path, low_memory=False).fillna("")
    results = collect_result_files([(args.source_label, run_dir / "batch_runs")])
    manifest["boltz_stem"] = manifest["yamlFile"].astype(str).map(lambda value: Path(value).stem)
    if len(manifest) != args.expected_rows or manifest["pairId"].duplicated().any():
        raise RuntimeError("Signed input manifest row contract failed")
    if len(input_provenance) != args.expected_rows or input_provenance["pairId"].duplicated().any():
        raise RuntimeError("Runner input provenance row contract failed")
    if results["boltz_stem"].duplicated().any():
        raise RuntimeError("Duplicate Boltz output stems")
    merged = manifest[
        ["pairId", "boltz_stem", "inputSignatureSha256", "yamlSha256"]
    ].merge(
        input_provenance[
            [
                "pairId",
                "batch",
                "batchInputSignature",
                "runParameterSignature",
                "seed",
                "resultCompletedVerified",
            ]
        ],
        on="pairId",
        how="left",
        validate="one_to_one",
    ).merge(results, on="boltz_stem", how="left", validate="one_to_one")
    completed = merged["boltz_completed_refined"].astype(str).str.lower().isin(
        {"true", "1", "1.0"}
    )
    hash_columns = [
        "boltz_confidence_sha256_refined",
        "boltz_affinity_sha256_refined",
        "boltz_cif_model0_sha256_refined",
        "boltz_cif_model1_sha256_refined",
    ]
    if len(merged) != args.expected_rows or not completed.all():
        raise RuntimeError(
            f"Boltz output completion contract failed: {int(completed.sum())}/{len(merged)}"
        )
    if merged[hash_columns].astype(str).eq("").any().any():
        raise RuntimeError("One or more completed Boltz outputs lack SHA-256")
    output = run_dir / "result_provenance_with_output_hashes.csv"
    merged.to_csv(output, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": len(merged),
        "completed_rows": int(completed.sum()),
        "manifest_sha256": file_sha(manifest_path),
        "runner_input_provenance_sha256": file_sha(input_provenance_path),
        "output_provenance_sha256": file_sha(output),
        "hash_columns": hash_columns,
    }
    summary_path = run_dir / "result_provenance_with_output_hashes.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
