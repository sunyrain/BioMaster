#!/usr/bin/env python3
"""Merge DrugCLIP six-fold embeddings into physical-pair retrieval scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "affinity_first_remote_discovery_v1"


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def rank_rows(values: np.ndarray, axis: int) -> np.ndarray:
    order = np.argsort(-values, axis=axis, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    if axis == 0:
        columns = np.arange(values.shape[1])[None, :]
        ranks[order, columns] = np.arange(1, values.shape[0] + 1)[:, None]
    elif axis == 1:
        rows = np.arange(values.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, values.shape[1] + 1)[None, :]
    else:
        raise ValueError(axis)
    return ranks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(BASE))
    args = parser.parse_args()
    base = Path(args.base_dir).resolve()
    input_dir = base / "drugclip_inputs_v1"
    inference_dir = base / "drugclip_inference_v1"
    mol_candidates = sorted((inference_dir / "mol_embeddings").glob("mol_reps*.npy"))
    if len(mol_candidates) != 1:
        raise ValueError(f"Expected one molecule embedding file, found {mol_candidates}")
    molecule_embeddings = np.load(mol_candidates[0])
    with (input_dir / "pocket_reps.pkl").open("rb") as handle:
        pocket_names, pocket_embeddings = pickle.load(handle)
    molecule_embeddings = np.asarray(molecule_embeddings, dtype=np.float32)
    pocket_embeddings = np.asarray(pocket_embeddings, dtype=np.float32)
    if molecule_embeddings.ndim != 3 or molecule_embeddings.shape[1:] != (6, 128):
        raise ValueError(f"Unexpected molecule embedding shape {molecule_embeddings.shape}")
    if pocket_embeddings.ndim != 3 or pocket_embeddings.shape[1:] != (6, 128):
        raise ValueError(f"Unexpected pocket embedding shape {pocket_embeddings.shape}")

    ligand_manifest = pd.read_csv(input_dir / "PROJECT723_LIGAND_LMDB_MANIFEST_V1.csv").fillna("")
    ligand_manifest = ligand_manifest[ligand_manifest["preparation_status"].eq("ok")].copy()
    ligand_manifest["lmdb_index"] = pd.to_numeric(ligand_manifest["lmdb_index"], errors="raise").astype(int)
    # Uni-Core enumerates LMDB byte keys lexicographically (0, 1, 10, 100,
    # ...), so reconstruct that exact order rather than numeric order.
    ligand_manifest["_lmdb_key"] = ligand_manifest["lmdb_index"].astype(str)
    ligand_manifest = ligand_manifest.sort_values("_lmdb_key", kind="mergesort").reset_index(drop=True)
    pocket_manifest = pd.read_csv(input_dir / "PROJECT308_POCKET_LMDB_MANIFEST_V1.csv").fillna("")
    pocket_manifest = pocket_manifest[pocket_manifest["preparation_status"].eq("ok")].copy()
    pocket_manifest["lmdb_index"] = pd.to_numeric(pocket_manifest["lmdb_index"], errors="raise").astype(int)
    pocket_manifest["_lmdb_key"] = pocket_manifest["lmdb_index"].astype(str)
    pocket_manifest = pocket_manifest.sort_values("_lmdb_key", kind="mergesort").reset_index(drop=True)
    if len(ligand_manifest) != molecule_embeddings.shape[0]:
        raise ValueError("Ligand embedding/manifest row mismatch")
    if len(pocket_manifest) != pocket_embeddings.shape[0]:
        raise ValueError("Pocket embedding/manifest row mismatch")
    if list(map(clean, pocket_names)) != pocket_manifest["sequence_key"].map(clean).tolist():
        raise ValueError("Pocket embedding order differs from signed manifest")

    fold_scores = np.einsum("lfd,tfd->ltf", molecule_embeddings, pocket_embeddings, optimize=True)
    mean_score = fold_scores.mean(axis=2)
    std_score = fold_scores.std(axis=2)
    target_mean = fold_scores.mean(axis=0, keepdims=True)
    target_std = fold_scores.std(axis=0, keepdims=True)
    target_std[target_std < 1e-6] = 1.0
    mean_target_zscore = ((fold_scores - target_mean) / target_std).mean(axis=2)
    rank_within_target = rank_rows(mean_target_zscore, axis=0)
    rank_within_ligand = rank_rows(mean_target_zscore, axis=1)

    ligand_index = np.repeat(np.arange(len(ligand_manifest)), len(pocket_manifest))
    pocket_index = np.tile(np.arange(len(pocket_manifest)), len(ligand_manifest))
    matrix = pd.DataFrame(
        {
            "model_ligand_smiles": ligand_manifest.loc[ligand_index, "model_ligand_smiles"].to_numpy(),
            "ligand_id": ligand_manifest.loc[ligand_index, "ligand_id"].to_numpy(),
            "sequence_key": pocket_manifest.loc[pocket_index, "sequence_key"].to_numpy(),
            "primary_gene": pocket_manifest.loc[pocket_index, "primary_gene"].to_numpy(),
            "drugclip_cosine_mean_v1": mean_score.reshape(-1),
            "drugclip_cosine_std_v1": std_score.reshape(-1),
            "drugclip_cosine_min_v1": fold_scores.min(axis=2).reshape(-1),
            "drugclip_cosine_max_v1": fold_scores.max(axis=2).reshape(-1),
            "drugclip_target_zscore_mean_v1": mean_target_zscore.reshape(-1),
            "drugclip_rank_within_target_v1": rank_within_target.reshape(-1),
            "drugclip_rank_within_ligand_v1": rank_within_ligand.reshape(-1),
        }
    )
    matrix["drugclip_top50_target_v1"] = matrix["drugclip_rank_within_target_v1"].le(50)
    matrix["drugclip_top50_ligand_v1"] = matrix["drugclip_rank_within_ligand_v1"].le(50)
    matrix["drugclip_bidirectional_top50_v1"] = (
        matrix["drugclip_top50_target_v1"] & matrix["drugclip_top50_ligand_v1"]
    )

    master = pd.read_csv(base / "PHYSICAL_PAIR_UNIVERSE_334749_HOMOLOGY_AUDITED_V1.csv.gz", low_memory=False)
    strict = master[as_bool(master["strict_structure_ready_v1"])].copy()
    scored = strict.merge(
        matrix,
        on=["model_ligand_smiles", "sequence_key", "primary_gene"],
        how="left",
        validate="one_to_one",
    )
    scored["drugclip_completed_v1"] = scored["drugclip_cosine_mean_v1"].notna()
    remote = scored[as_bool(scored["dta_stage1_strict_homology_audited_v1"])].copy()
    remote["conplex_top100_within_active_moiety_v1"] = pd.to_numeric(
        remote["rank_within_active_moiety"], errors="coerce"
    ).le(100)
    remote["drugclip_conplex_dual_support_v1"] = (
        as_bool(remote["drugclip_bidirectional_top50_v1"])
        & remote["conplex_top100_within_active_moiety_v1"]
    )

    compact_columns = [
        "physical_pair_id",
        "source_drug_ids",
        "source_drug_names",
        "model_ligand_smiles",
        "sequence_key",
        "primary_gene",
        "target_assay_family_v2",
        "structure_bin",
        "conplex_score",
        "rank_within_active_moiety",
        "rank_within_target_active_collapsed",
        "remote_novelty_status_v1",
        "max_known_active_similarity_v5",
        "expanded_known_target_count_v1",
        "max_known_target_mmseqs_identity_v1",
        "drugclip_completed_v1",
        "drugclip_cosine_mean_v1",
        "drugclip_cosine_std_v1",
        "drugclip_cosine_min_v1",
        "drugclip_cosine_max_v1",
        "drugclip_target_zscore_mean_v1",
        "drugclip_rank_within_target_v1",
        "drugclip_rank_within_ligand_v1",
        "drugclip_bidirectional_top50_v1",
        "conplex_top100_within_active_moiety_v1",
        "drugclip_conplex_dual_support_v1",
    ]
    inference_dir.mkdir(parents=True, exist_ok=True)
    matrix_output = inference_dir / "DRUGCLIP_PROJECT722_X_307_MATRIX_V1.csv.gz"
    remote_output = inference_dir / "REMOTE_STRICT_DTA_DRUGCLIP_SCORED_V1.csv.gz"
    matrix.to_csv(matrix_output, index=False, compression={"method": "gzip", "compresslevel": 5})
    remote[compact_columns].to_csv(
        remote_output, index=False, compression={"method": "gzip", "compresslevel": 5}
    )

    known = scored[
        as_bool(scored["any_known_fda_target_pair"]) | as_bool(scored["exact_known_active_smiles_v5"])
    ].copy()
    known_completed = known[known["drugclip_completed_v1"]]
    calibration = {
        f"top{k}": float((known_completed["drugclip_rank_within_ligand_v1"] <= k).mean())
        if len(known_completed)
        else None
        for k in [10, 50, 100, 300]
    }
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "molecule_embedding_shape": list(molecule_embeddings.shape),
        "pocket_embedding_shape": list(pocket_embeddings.shape),
        "matrix_rows": int(len(matrix)),
        "strict_physical_rows": int(len(strict)),
        "strict_scored_rows": int(scored["drugclip_completed_v1"].sum()),
        "remote_strict_rows": int(len(remote)),
        "remote_strict_scored_rows": int(remote["drugclip_completed_v1"].sum()),
        "bidirectional_top50_remote_rows": int(as_bool(remote["drugclip_bidirectional_top50_v1"]).sum()),
        "drugclip_conplex_dual_support_rows": int(as_bool(remote["drugclip_conplex_dual_support_v1"]).sum()),
        "known_control_rows_completed": int(len(known_completed)),
        "known_control_recall_by_ligand_rank": calibration,
        "interpretation": (
            "DrugCLIP values are six-fold pocket-ligand retrieval cosine scores and target-normalized z-scores, "
            "not Kd/IC50 estimates. Known-control recall is calibration with possible training overlap."
        ),
        "outputs": {
            "matrix": str(matrix_output.relative_to(ROOT)),
            "remote_scored": str(remote_output.relative_to(ROOT)),
        },
        "sha256": {"matrix": sha256(matrix_output), "remote_scored": sha256(remote_output)},
    }
    (inference_dir / "DRUGCLIP_PROJECT_INFERENCE_V1_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
