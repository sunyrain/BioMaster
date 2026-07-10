#!/usr/bin/env python3
"""Compare Boltz signals for known positives with the unlabeled discovery queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def numeric(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def quantiles(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {key: None for key in ["min", "p10", "p25", "median", "p75", "p90", "max"]}
    values = clean.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist()
    return dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max"], map(float, values), strict=True))


def completed(df: pd.DataFrame) -> pd.Series:
    for column in ["boltz_completed_refined", "boltzCompleted"]:
        if column in df.columns:
            return df[column].astype(str).str.lower().isin({"true", "1", "1.0"})
    return pd.Series(False, index=df.index)


def stable_pose(df: pd.DataFrame) -> pd.Series:
    if "pose_stability_tier" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["pose_stability_tier"].astype(str).isin(
        {"A_stable_conditional_pose", "B_moderate_conditional_pose"}
    )


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known", required=True)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    known = pd.read_csv(args.known, low_memory=False).fillna("")
    discovery = pd.read_csv(args.discovery, low_memory=False).fillna("")
    known_completed = completed(known)
    discovery_completed = completed(discovery)
    if not known_completed.all() or not discovery_completed.all():
        raise ValueError(
            f"Calibration requires complete runs: known={int(known_completed.sum())}/{len(known)}, "
            f"discovery={int(discovery_completed.sum())}/{len(discovery)}"
        )
    known_affinity = numeric(known, ["boltz_affinity_probability_refined", "boltzAffinityProbabilityBinary"])
    discovery_affinity = numeric(
        discovery, ["boltz_affinity_probability_refined", "boltzAffinityProbabilityBinary"]
    )
    known_iptm = numeric(known, ["boltz_ligand_iptm_refined", "boltzLigandIptm"])
    discovery_iptm = numeric(discovery, ["boltz_ligand_iptm_refined", "boltzLigandIptm"])
    known_confidence = numeric(known, ["boltz_confidence_score_refined", "boltzConfidenceScore"])
    discovery_confidence = numeric(
        discovery, ["boltz_confidence_score_refined", "boltzConfidenceScore"]
    )
    reference = np.sort(discovery_affinity.dropna().to_numpy(dtype=float))
    known["affinity_percentile_within_unlabeled_discovery"] = [
        float(np.searchsorted(reference, value, side="right") / len(reference)) if pd.notna(value) and len(reference) else np.nan
        for value in known_affinity
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    known.to_csv(output, index=False)
    by_band = {}
    if "control_score_band" in known.columns:
        for band, group in known.groupby("control_score_band"):
            idx = group.index
            by_band[str(band)] = {
                "rows": len(group),
                "affinity_probability": quantiles(known_affinity.loc[idx]),
                "stable_pose_fraction": float(stable_pose(known).loc[idx].mean()),
            }
    summary = {
        "known_positive_rows": len(known),
        "unlabeled_discovery_rows": len(discovery),
        "known_affinity_probability": quantiles(known_affinity),
        "discovery_affinity_probability": quantiles(discovery_affinity),
        "known_ligand_iptm": quantiles(known_iptm),
        "discovery_ligand_iptm": quantiles(discovery_iptm),
        "known_confidence": quantiles(known_confidence),
        "discovery_confidence": quantiles(discovery_confidence),
        "known_stable_pose_fraction": float(stable_pose(known).mean()),
        "discovery_stable_pose_fraction": float(stable_pose(discovery).mean()),
        "known_affinity_percentile_within_unlabeled_discovery": quantiles(
            known["affinity_percentile_within_unlabeled_discovery"]
        ),
        "known_by_conplex_score_band": by_band,
        "interpretation_limit": (
            "Known positives test signal recovery only. The discovery queue is unlabeled, not a negative set; "
            "therefore this audit does not estimate specificity, AUROC, precision, or real-world affinity accuracy."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
