#!/usr/bin/env python3
"""Evaluate BioMaster-ODTI on pair-held-out BindingDB evidence.

This is deliberately a positive-only external retrieval protocol. BindingDB
rows provide observed positive evidence, not a trustworthy set of negatives;
the script therefore reports per-ligand Recall@K, best-positive rank and
average precision over the frozen target universe. Exact ChEMBL37 pair
overlaps are removed before scoring. Ligand/target entity overlap is reported
separately rather than silently called entity-cold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2  # noqa: E402
from train_biomaster_odti_v2 import load_drug_aux_availability, predict  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
STORE = BASE / "feature_store_v1"
PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = STORE / "MORGAN2048_UINT8_V1.npy"
PROTBERT = STORE / "PROTBERT1024_FLOAT32_V1.npy"
TARGET_AUX = BASE / "public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
DRUG_AUX = BASE / "public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_BERMOL768_FLOAT32_V1.npy"
DRUG_AUX_INDEX = BASE / "public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_BERMOL_DRUG_INDEX_V1.csv.gz"
BINDINGDB = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v9/BINDINGDB_EXACT_EVIDENCE_ROWS_V9.csv"
S4_ROOT = BASE / "biomaster_odti_v2_s4_esm2_formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: pd.Series) -> pd.Series:
    if value.dtype == bool:
        return value.fillna(False)
    return value.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def prepare_external_pairs() -> tuple[pd.DataFrame, dict[str, object]]:
    frozen = pd.read_csv(
        PAIRS,
        low_memory=False,
        usecols=[
            "calibration_pair_id",
            "parent_standard_inchi_key",
            "query_accession",
            "drug_feature_index",
            "target_feature_index",
            "target_assay_family",
            "target_chembl_id",
        ],
    )
    binding = pd.read_csv(BINDINGDB, low_memory=False)
    required = {
        "ligand_inchikey",
        "matched_uniprot_accession",
        "strict_single_chain_quantitative_support_le10um",
    }
    missing = required - set(binding.columns)
    if missing:
        raise ValueError(f"BindingDB evidence is missing columns: {sorted(missing)}")
    binding = binding[truthy(binding["strict_single_chain_quantitative_support_le10um"])].copy()
    binding["ligand_inchikey"] = binding["ligand_inchikey"].astype(str)
    binding["matched_uniprot_accession"] = binding["matched_uniprot_accession"].astype(str)
    evidence = (
        binding.groupby(["ligand_inchikey", "matched_uniprot_accession"], as_index=False)
        .agg(
            bindingdb_rows=("source_record_id", "nunique"),
            bindingdb_best_value_nm=("best_value_nm", "min"),
            bindingdb_sources=("source_snapshot", "nunique"),
        )
    )
    frozen_pairs = set(
        zip(frozen["parent_standard_inchi_key"].astype(str), frozen["query_accession"].astype(str))
    )
    evidence["exact_frozen_pair"] = [
        (ligand, target) in frozen_pairs
        for ligand, target in zip(evidence["ligand_inchikey"], evidence["matched_uniprot_accession"])
    ]
    exact_overlap_count = int(evidence["exact_frozen_pair"].sum())
    evidence = evidence.loc[~evidence["exact_frozen_pair"]].copy()
    heldout_before_alignment = evidence.copy()
    heldout_ligands_before_alignment = set(heldout_before_alignment["ligand_inchikey"])
    heldout_targets_before_alignment = set(heldout_before_alignment["matched_uniprot_accession"])
    drug_map = (
        frozen[["parent_standard_inchi_key", "drug_feature_index"]]
        .drop_duplicates("parent_standard_inchi_key")
        .rename(columns={"parent_standard_inchi_key": "ligand_inchikey"})
    )
    target_map = (
        frozen[["query_accession", "target_feature_index", "target_assay_family", "target_chembl_id"]]
        .drop_duplicates("query_accession")
    )
    evidence = evidence.merge(drug_map, on="ligand_inchikey", how="left", validate="many_to_one")
    evidence = evidence.merge(target_map, left_on="matched_uniprot_accession", right_on="query_accession", how="left", validate="many_to_one")
    evidence["ligand_entity_seen"] = evidence["drug_feature_index"].notna()
    evidence["target_entity_seen"] = evidence["target_feature_index"].notna()
    # Preserve the pre-alignment strata.  Dropping rows before recording this
    # table would hide the exact reason the current protocol is not
    # entity-cold: the feature store cannot score unseen ligands/targets.
    prealignment = evidence.copy()
    prealignment["entity_overlap_class"] = np.select(
        [
            prealignment["ligand_entity_seen"] & prealignment["target_entity_seen"],
            prealignment["ligand_entity_seen"] & ~prealignment["target_entity_seen"],
            ~prealignment["ligand_entity_seen"] & prealignment["target_entity_seen"],
        ],
        ["both_seen", "ligand_seen_target_unseen", "ligand_unseen_target_seen"],
        default="both_unseen",
    )
    evidence = evidence.dropna(subset=["drug_feature_index", "target_feature_index"]).copy()
    evidence["drug_feature_index"] = evidence["drug_feature_index"].astype(np.int64)
    evidence["target_feature_index"] = evidence["target_feature_index"].astype(np.int64)
    manifest = {
        "bindingdb_rows_after_strict_support": int(len(binding)),
        "bindingdb_unique_pairs_after_strict_support": int(
            binding[["ligand_inchikey", "matched_uniprot_accession"]].drop_duplicates().shape[0]
        ),
        "exact_frozen_pair_overlap_removed": exact_overlap_count,
        "pair_heldout_unique_pairs_before_feature_alignment": int(len(heldout_before_alignment)),
        "pair_heldout_unique_ligands_before_feature_alignment": int(len(heldout_ligands_before_alignment)),
        "pair_heldout_unique_targets_before_feature_alignment": int(len(heldout_targets_before_alignment)),
        "feature_alignment_dropped_pairs": int(len(heldout_before_alignment) - len(evidence)),
        "prealignment_entity_overlap_class_counts": {
            str(key): int(value)
            for key, value in prealignment["entity_overlap_class"].value_counts().to_dict().items()
        },
        "prealignment_entity_cold_pairs": int(
            (~prealignment["ligand_entity_seen"] | ~prealignment["target_entity_seen"]).sum()
        ),
        "prealignment_both_unseen_pairs": int(
            (~prealignment["ligand_entity_seen"] & ~prealignment["target_entity_seen"]).sum()
        ),
        "pair_heldout_unique_pairs": int(len(evidence)),
        "pair_heldout_unique_ligands": int(evidence["ligand_inchikey"].nunique()),
        "pair_heldout_unique_targets": int(evidence["matched_uniprot_accession"].nunique()),
        "aligned_unique_pairs": int(len(evidence)),
        "aligned_unique_ligands": int(evidence["ligand_inchikey"].nunique()),
        "aligned_unique_targets": int(evidence["matched_uniprot_accession"].nunique()),
        "ligand_entity_seen_pairs": int(evidence["ligand_entity_seen"].sum()),
        "target_entity_seen_pairs": int(evidence["target_entity_seen"].sum()),
        "ligand_entity_coverage_after_overlap": float(
            evidence["ligand_entity_seen"].mean() if len(evidence) else 0.0
        ),
        "target_entity_coverage_after_overlap": float(
            evidence["target_entity_seen"].mean() if len(evidence) else 0.0
        ),
        "protocol": "PAIR_HELDOUT_BINDINGDB_POSITIVE_ONLY_RETRIEVAL",
    }
    return evidence, manifest


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ODTIV2Config(**checkpoint["config"])
    model = RoutedInteractionRankerV2(
        family_count=len(checkpoint["families"]), config=config, use_conplex=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def score_candidates(
    frame: pd.DataFrame,
    checkpoint_paths: list[Path],
    device: torch.device,
    inference_batch_size: int,
) -> np.ndarray:
    drug_features = np.load(MORGAN, mmap_mode="r")
    available_drug_aux_features = np.load(DRUG_AUX, mmap_mode="r")
    drug_aux_availability, _ = load_drug_aux_availability(
        DRUG_AUX_INDEX, len(drug_features)
    )
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux_features = np.load(TARGET_AUX, mmap_mode="r")
    scores: list[np.ndarray] = []
    positions = np.arange(len(frame), dtype=np.int64)
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        # The S4 checkpoints were trained with the 19-dimensional audited
        # structure-context input.  External BindingDB pairs do not carry
        # structure evidence, so construct a width-compatible zero tensor and
        # set the explicit mask to zero.  The model's exact residual fallback
        # then guarantees final_logit == base_logit for every external row.
        structure_dim = int(checkpoint["config"].get("structure_input_dim", 0))
        structure_features = np.zeros((len(frame), structure_dim), dtype=np.float32)
        structure_mask = np.zeros(len(frame), dtype=np.float32)
        normalization = checkpoint["normalization"]
        drug_aux_dim = int(checkpoint["config"].get("drug_aux_input_dim", 0))
        drug_aux_features = available_drug_aux_features if drug_aux_dim > 0 else None
        if drug_aux_features is not None and drug_aux_features.shape[1] != drug_aux_dim:
            raise ValueError(
                f"BindingDB drug auxiliary width {drug_aux_features.shape[1]} != checkpoint {drug_aux_dim}"
            )
        families = checkpoint["families"]
        lookup = {name: index for index, name in enumerate(families)}
        family_index = frame["target_assay_family"].astype(str).map(lookup).fillna(lookup["__UNK__"]).to_numpy(dtype=np.int64)
        arrays = {
            "family_index": family_index,
            "conplex": np.zeros(len(frame), dtype=np.float32),
            "conplex_mean": float(normalization["conplex_mean"]),
            "conplex_std": float(normalization["conplex_std"]),
            "affinity": np.full(len(frame), np.nan, dtype=np.float32),
            "affinity_lower": np.full(len(frame), np.nan, dtype=np.float32),
            "affinity_upper": np.full(len(frame), np.nan, dtype=np.float32),
            "affinity_mean": float(normalization["affinity_mean"]),
            "affinity_std": float(normalization["affinity_std"]),
            "target_mean": np.asarray(normalization["target_mean"], dtype=np.float32),
            "target_std": np.asarray(normalization["target_std"], dtype=np.float32),
            "drug_aux_mean": np.asarray(
                normalization.get("drug_aux_mean", np.zeros(drug_aux_dim)), dtype=np.float32
            ),
            "drug_aux_std": np.asarray(
                normalization.get("drug_aux_std", np.ones(drug_aux_dim)), dtype=np.float32
            ),
            "target_aux_mean": np.asarray(normalization["target_aux_mean"], dtype=np.float32),
            "target_aux_std": np.asarray(normalization["target_aux_std"], dtype=np.float32),
            "structure_mean": np.asarray(
                normalization.get("structure_mean", np.zeros(structure_dim)),
                dtype=np.float32,
            ),
            "structure_std": np.asarray(
                normalization.get("structure_std", np.ones(structure_dim)),
                dtype=np.float32,
            ),
            "families": families,
        }
        result = predict(
            model,
            positions,
            frame,
            drug_features,
            drug_aux_features,
            target_features,
            target_aux_features,
            None,
            None,
            None,
            None,
            1022,
            None,
            None,
            structure_features,
            structure_mask,
            arrays,
            device,
            inference_batch_size,
            drug_aux_available=drug_aux_availability,
        )
        temperature = float(checkpoint.get("temperature", 1.0))
        logits = result["final_logit"] / max(temperature, 1e-6)
        scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))))
    return np.stack(scores, axis=0)


def ranking_metrics(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = []
    for ligand, part in scored.groupby("ligand_inchikey", sort=True):
        part = part.sort_values(["score_mean", "target_feature_index"], ascending=[False, True]).reset_index(drop=True)
        positive = part["external_positive"].to_numpy(dtype=bool)
        if not positive.any():
            continue
        ranks = np.flatnonzero(positive) + 1
        rows.append(
            {
                "ligand_inchikey": ligand,
                "candidate_targets": int(len(part)),
                "positive_targets": int(positive.sum()),
                "best_positive_rank": int(ranks.min()),
                "median_positive_rank": float(np.median(ranks)),
                "recall_at_5": float(positive[:5].sum() / positive.sum()),
                "recall_at_10": float(positive[:10].sum() / positive.sum()),
                "recall_at_20": float(positive[:20].sum() / positive.sum()),
                "average_precision": float(average_precision_score(positive.astype(np.int8), part["score_mean"])),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no evaluable ligand groups remain after external alignment")
    random_recall = {
        f"random_expected_recall_at_{k}": float(
            np.mean(np.minimum(k, table["candidate_targets"].to_numpy(dtype=np.float64))
                    / table["candidate_targets"].to_numpy(dtype=np.float64))
        )
        for k in (5, 10, 20)
    }
    random_best_rank = float(
        np.mean(
            (table["candidate_targets"].to_numpy(dtype=np.float64) + 1.0)
            / (table["positive_targets"].to_numpy(dtype=np.float64) + 1.0)
        )
    )
    summary = {
        "ligands_evaluable": int(len(table)),
        "positive_pairs": int(scored["external_positive"].sum()),
        "recall_at_5": float(table["recall_at_5"].mean()),
        "recall_at_10": float(table["recall_at_10"].mean()),
        "recall_at_20": float(table["recall_at_20"].mean()),
        "median_best_positive_rank": float(table["best_positive_rank"].median()),
        "mean_average_precision": float(table["average_precision"].mean()),
        **random_recall,
        "random_expected_best_positive_rank": random_best_rank,
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-glob", default=str(S4_ROOT / "S4_FIRST_SEEN_TEMPORAL_2023_2025__fold_-1__seed_*/BEST_MODEL_V2.pt"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    args = parser.parse_args()
    checkpoint_paths = sorted(Path(path) for path in __import__("glob").glob(args.checkpoint_glob))
    if not checkpoint_paths:
        raise FileNotFoundError("no checkpoint paths matched --checkpoint-glob")
    evidence, manifest = prepare_external_pairs()
    target_universe = (
        pd.read_csv(PAIRS, low_memory=False)[
            ["query_accession", "target_feature_index", "target_assay_family", "target_chembl_id"]
        ]
        .drop_duplicates("query_accession")
    )
    ligands = evidence[["ligand_inchikey", "drug_feature_index"]].drop_duplicates("ligand_inchikey")
    candidates = ligands.assign(__key=1).merge(target_universe.assign(__key=1), on="__key").drop(columns="__key")
    positives = evidence[["ligand_inchikey", "matched_uniprot_accession"]].drop_duplicates()
    positives["external_positive"] = True
    candidates = candidates.merge(
        positives,
        left_on=["ligand_inchikey", "query_accession"],
        right_on=["ligand_inchikey", "matched_uniprot_accession"],
        how="left",
    )
    candidates["external_positive"] = candidates["external_positive"].fillna(False).astype(bool)
    candidates["calibration_pair_id"] = (
        "BDB_PAIR_HELDOUT::" + candidates["ligand_inchikey"].astype(str) + "::" + candidates["query_accession"].astype(str)
    )
    candidates["binary_label"] = candidates["external_positive"].astype(np.float32)
    candidates["conplex_score"] = 0.0
    candidates["mean_pchembl"] = np.nan
    candidates["min_pchembl"] = np.nan
    candidates["max_pchembl"] = np.nan
    candidates["sequence_key"] = candidates["query_accession"].astype(str)
    candidates["parent_standard_inchi_key"] = candidates["ligand_inchikey"].astype(str)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    predictions = score_candidates(candidates, checkpoint_paths, device, args.inference_batch_size)
    candidates["score_mean"] = predictions.mean(axis=0)
    candidates["score_std"] = predictions.std(axis=0, ddof=1 if predictions.shape[0] > 1 else 0)
    candidates["ensemble_lower"] = np.clip(candidates["score_mean"] - 1.96 * candidates["score_std"], 0, 1)
    candidates["ensemble_upper"] = np.clip(candidates["score_mean"] + 1.96 * candidates["score_std"], 0, 1)
    per_ligand, ranking_summary = ranking_metrics(candidates)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "BINDINGDB_PAIR_HELDOUT_CANDIDATE_SCORES.csv.gz", index=False, compression="gzip")
    evidence.to_csv(out_dir / "BINDINGDB_PAIR_HELDOUT_EVIDENCE.csv.gz", index=False, compression="gzip")
    per_ligand.to_csv(out_dir / "BINDINGDB_PAIR_HELDOUT_PER_LIGAND_METRICS.csv", index=False)
    manifest.update(
        {
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pair_table_sha256": sha256(PAIRS),
            "bindingdb_sha256": sha256(BINDINGDB),
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "checkpoint_count": len(checkpoint_paths),
            "candidate_rows": int(len(candidates)),
            "candidate_ligands": int(candidates["ligand_inchikey"].nunique()),
            "target_universe": int(target_universe["target_feature_index"].nunique()),
            "ranking": ranking_summary,
            "claim_status": "PAIR_HELDOUT_BINDINGDB_POSITIVE_ONLY; NO_NEGATIVE_OR_SOTA_CLAIM",
        }
    )
    (out_dir / "BINDINGDB_PAIR_HELDOUT_SUMMARY.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
