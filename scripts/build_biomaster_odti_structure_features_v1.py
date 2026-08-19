#!/usr/bin/env python3
"""Build a leakage-audited, pair-aligned structure-context feature store.

The safe default intentionally uses target-level structure/pocket descriptors
only.  It does *not* attach GNINA scores selected from positive/negative label
controls; those scores are useful for a separate diagnostic/experimental
branch, but they require split-local recalibration before entering a formal
test fold.  Missing target structure receives ``structure_mask=0`` and the
V2 model therefore falls back exactly to its sequence/chemistry score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DEFAULT_TARGET_STRUCTURE = ROOT / "outputs/affinity_first_remote_discovery_v1/DTA_STAGE1_REMOTE_STRICT_STRUCTURE_V1.csv.gz"
DEFAULT_TARGET_ATLAS = ROOT / "outputs/affinity_first_remote_discovery_v1/experimental_structure_atlas_v1/TARGET_EXPERIMENTAL_STRUCTURE_ATLAS_463_V1.csv"
DEFAULT_OUT = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
DEFAULT_MANIFEST = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_complete_by_target(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select one maximally populated target context row without label fields."""

    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame[["sequence_key"]].drop_duplicates().copy()
    work = frame[["sequence_key", *available]].copy()
    work["__coverage"] = work[available].notna().sum(axis=1)
    work = work.sort_values(["sequence_key", "__coverage"], ascending=[True, False])
    return work.drop_duplicates("sequence_key", keep="first").drop(columns="__coverage")


def build_structure_context(
    pairs_path: Path,
    target_structure_path: Path,
    atlas_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pairs = pd.read_csv(pairs_path, low_memory=False)
    required = {"calibration_pair_id", "sequence_key"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"pair table missing required columns: {missing}")
    if pairs["calibration_pair_id"].duplicated().any():
        raise ValueError("pair table has duplicate calibration_pair_id")

    dta = pd.read_csv(target_structure_path, low_memory=False)
    if "sequence_key" not in dta.columns:
        raise ValueError("target structure table requires sequence_key")
    # These columns describe receptor/pocket geometry or structure availability;
    # label-derived calibration columns are deliberately excluded.
    safe_dta = [
        "pdb_path",
        "top_pocket_score",
        "top_pocket_probability",
        "top_pocket_center_x",
        "top_pocket_center_y",
        "top_pocket_center_z",
        "p2rank26_top_pocket_volume",
        "p2rank26_top_pocket_num_residues",
        "p2rank_puresnet_overlap_fraction",
        "p2rank_puresnet_jaccard",
        "receptor_residue_count",
        "structure_bin",
        "strict_structure_tier",
        "sequence_match_status",
    ]
    target = first_complete_by_target(dta, safe_dta)

    atlas_summary: dict[str, object] = {"provided": False}
    if atlas_path is not None:
        atlas = pd.read_csv(atlas_path, low_memory=False)
        if "sequence_key" not in atlas.columns:
            raise ValueError("structure atlas requires sequence_key")
        safe_atlas = [
            "experimental_entry_count_top20",
            "candidate_holo_entry_count_top20",
            "has_candidate_experimental_holo",
            "best_holo_resolution",
            "best_holo_coverage",
        ]
        atlas_target = first_complete_by_target(atlas, safe_atlas)
        target = target.merge(atlas_target, on="sequence_key", how="outer", validate="one_to_one")
        atlas_summary = {
            "provided": True,
            "rows": int(len(atlas)),
            "targets": int(atlas["sequence_key"].nunique()),
        }

    output = pairs[["calibration_pair_id", "sequence_key"]].copy()
    output = output.merge(target, on="sequence_key", how="left", validate="many_to_one")

    def numeric(column: str) -> pd.Series:
        if column not in output.columns:
            return pd.Series(np.nan, index=output.index, dtype=np.float32)
        return pd.to_numeric(output[column], errors="coerce")

    def boolean(column: str) -> pd.Series:
        if column not in output.columns:
            return pd.Series(False, index=output.index, dtype=bool)
        value = output[column]
        if value.dtype == bool:
            return value.fillna(False)
        return value.astype(str).str.lower().isin({"true", "1", "yes", "y"})

    structure_available = output.get("pdb_path", pd.Series(index=output.index, dtype=object)).notna()
    pocket_available = numeric("top_pocket_probability").notna()
    mask = (structure_available & pocket_available).astype(np.float32)
    features = pd.DataFrame({
        "calibration_pair_id": output["calibration_pair_id"].astype(str),
        "structure_mask": mask,
        "structure_context_available": mask,
        "receptor_experimental_holo": boolean("has_candidate_experimental_holo").astype(np.float32),
        "pocket_top_score": numeric("top_pocket_score"),
        "pocket_top_probability": numeric("top_pocket_probability"),
        "pocket_center_x": numeric("top_pocket_center_x"),
        "pocket_center_y": numeric("top_pocket_center_y"),
        "pocket_center_z": numeric("top_pocket_center_z"),
        "pocket_volume": numeric("p2rank26_top_pocket_volume"),
        "pocket_num_residues": numeric("p2rank26_top_pocket_num_residues"),
        "pocket_puresnet_overlap": numeric("p2rank_puresnet_overlap_fraction"),
        "pocket_puresnet_jaccard": numeric("p2rank_puresnet_jaccard"),
        "receptor_residue_count": numeric("receptor_residue_count"),
        "experimental_entry_count_top20": numeric("experimental_entry_count_top20"),
        "candidate_holo_entry_count_top20": numeric("candidate_holo_entry_count_top20"),
        "best_holo_resolution": numeric("best_holo_resolution"),
        "best_holo_coverage": numeric("best_holo_coverage"),
        "structure_bin_is_strict": output.get("structure_bin", pd.Series("", index=output.index)).astype(str).str.startswith("A_").astype(np.float32),
        "structure_bin_is_manual_review": output.get("structure_bin", pd.Series("", index=output.index)).astype(str).str.startswith("C_").astype(np.float32),
        "sequence_exact_match": output.get("sequence_match_status", pd.Series("", index=output.index)).astype(str).eq("exact_match").astype(np.float32),
    })
    feature_columns = [column for column in features.columns if column not in {"calibration_pair_id", "structure_mask"}]
    features[feature_columns] = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    if features["calibration_pair_id"].duplicated().any():
        raise RuntimeError("generated structure feature table has duplicate pair IDs")
    if len(features) != len(pairs):
        raise RuntimeError("generated structure feature table does not cover the pair table")

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "SAFE_TARGET_STRUCTURE_CONTEXT_ONLY",
        "label_dependency": "NONE_IN_FEATURE_COLUMNS",
        "pair_rows": int(len(pairs)),
        "pair_targets": int(pairs["sequence_key"].nunique()),
        "targets_with_structure_context": int(
            output.loc[features["structure_mask"] > 0, "sequence_key"].nunique()
        ),
        "pairs_with_structure_context": int((features["structure_mask"] > 0).sum()),
        "present_pair_rows": int((features["structure_mask"] > 0).sum()),
        "feature_columns": feature_columns,
        "feature_dim": int(len(feature_columns)),
        "source_tables": {
            "pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "target_structure": {"path": str(target_structure_path), "sha256": sha256(target_structure_path)},
            "atlas": (
                {"path": str(atlas_path), "sha256": sha256(atlas_path)}
                if atlas_path is not None else None
            ),
        },
        "atlas_summary": atlas_summary,
        "excluded_label_derived_sources": [
            "GNINA_TARGET_CALIBRATION_LIGAND_SCORES_V2",
            "GNINA_TARGET_CHANNEL_CALIBRATION_V8",
            "any positive/negative control selection or target-local label calibration",
        ],
        "formal_use": "eligible as a target-context residual input; pair-level docking scores require split-local recalibration",
    }
    return features, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--target-structure", default=str(DEFAULT_TARGET_STRUCTURE))
    parser.add_argument("--atlas", default=str(DEFAULT_TARGET_ATLAS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    target_structure_path = Path(args.target_structure)
    atlas_path = Path(args.atlas) if args.atlas else None
    for path in [pairs_path, target_structure_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if atlas_path is not None and not atlas_path.is_file():
        raise FileNotFoundError(atlas_path)

    features, manifest = build_structure_context(pairs_path, target_structure_path, atlas_path)
    out = Path(args.out)
    manifest_path = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False, compression="gzip")
    manifest["output"] = {"path": str(out), "sha256": sha256(out)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
