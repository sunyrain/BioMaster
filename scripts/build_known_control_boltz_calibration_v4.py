#!/usr/bin/env python3
"""Select a diverse known-positive Boltz calibration panel from the v4 universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_calibration_v4.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_v4.csv",
    )
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument(
        "--sequence-audit",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "target_sequence_integrity_v4/project_targets_sequence_integrity_v4.csv"
        ),
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input, low_memory=False).fillna("")
    sequence_audit = pd.read_csv(args.sequence_audit, low_memory=False).fillna("")
    sequence_status = sequence_audit.drop_duplicates("sequence_key").set_index("sequence_key")[
        "sequence_match_status"
    ]
    data["sequence_match_status"] = data["sequence_key"].map(sequence_status).fillna("missing_audit")
    sequence_excluded = int(data["sequence_match_status"].ne("exact_match").sum())
    data = data[
        data["structure_bin"].isin(["A_strict_overlapping_pocket", "B_strict_supported_overlap"])
        & data["anchor_project_standard_direct_sm"].astype(str).str.lower().isin(["true", "1", "1.0"])
        & data["sequence_match_status"].eq("exact_match")
    ].copy()
    data["_score"] = pd.to_numeric(data["conplex_score"], errors="coerce").fillna(-1)
    data["_compound"] = data["active_moiety_smiles"].where(
        data["active_moiety_smiles"].astype(str).ne(""), data["drug_chembl_id"]
    )
    data = data.sort_values(["_score", "pair_id"], ascending=[False, True]).drop_duplicates(
        ["_compound", "sequence_key"]
    )
    selected = []
    compound_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    # Round-robin over family and score quantile prevents a high-score-only,
    # kinase-heavy control panel.
    data["control_score_band"] = pd.qcut(
        data["_score"].rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
    ).astype(str)
    groups = {
        key: group.sort_values(["_score", "pair_id"], ascending=[False, True])
        for key, group in data.groupby(["target_assay_family_v2", "control_score_band"])
    }
    positions = {key: 0 for key in groups}
    while len(selected) < args.size:
        added = False
        for key in sorted(groups):
            group = groups[key]
            while positions[key] < len(group):
                row = group.iloc[positions[key]]
                positions[key] += 1
                compound = str(row["_compound"])
                target = str(row["sequence_key"])
                if compound_counts.get(compound, 0) >= 2 or target_counts.get(target, 0) >= 3:
                    continue
                selected.append(row)
                compound_counts[compound] = compound_counts.get(compound, 0) + 1
                target_counts[target] = target_counts.get(target, 0) + 1
                added = True
                break
            if len(selected) >= args.size:
                break
        if not added:
            break
    result = pd.DataFrame(selected).drop(columns=["_score", "_compound"], errors="ignore")
    if len(result) != args.size:
        raise RuntimeError(f"Could select only {len(result)}/{args.size} known controls")
    result["externalQueueRank"] = range(1, len(result) + 1)
    result["pairId"] = result["pair_id"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "rows": len(result),
        "unique_active_moieties": int(result["active_moiety_smiles"].nunique()),
        "unique_targets": int(result["sequence_key"].nunique()),
        "assay_families": result["target_assay_family_v2"].value_counts().to_dict(),
        "score_bands": result["control_score_band"].value_counts().to_dict(),
        "sequence_mismatch_controls_excluded_before_selection": sequence_excluded,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
