#!/usr/bin/env python3
"""Train a drug-cold-start mutation-aware multi-task model on KiRHub variants.

The model never receives WT or variant experimental residual activity as an
input. It combines a Morgan drug encoder, frozen ProtBert WT target vectors,
explicit mutation tokens, and frozen WT retrieval priors. Outputs are variant
activity, variant-specific sensitization, and continuous residual activity.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
PANEL = RUN / "old_drug_innovation_v7/OLD_DRUG_MUTATION_APPLICATION_PANEL_V7.csv"
MODEL_VERSION = os.environ.get("BIOMASTER_MUTATION_VERSION", "V8").upper()
PRIOR = Path(os.environ.get(
    "BIOMASTER_MUTATION_PRIOR",
    str(RUN / "old_drug_advanced_ranker_v8/OLD_DRUG_ADVANCED_RANKER_ALL_276480_V8.csv.gz"),
))
if not PRIOR.is_absolute():
    PRIOR = ROOT / PRIOR
PRIOR_SCORE_COLUMN = os.environ.get(
    "BIOMASTER_MUTATION_SCORE_COLUMN", "old_drug_advanced_ranker_score_v8"
)
PRIOR_RANK_COLUMN = os.environ.get(
    "BIOMASTER_MUTATION_RANK_COLUMN", "old_drug_advanced_rank_within_drug_384_v8"
)
PAIR = RUN / "final_evidence_routing_v7/PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V7.csv.gz"
TARGET = RUN / "final_evidence_routing_v7/TARGET_EVIDENCE_LAYER_ROUTING_384_V7.csv"
SEQUENCE = ROOT / "outputs/target_universe_ch37_v2/TARGET_SET_SEQUENCE_DTA_ALL_V2.csv"
MORGAN = ROOT / "outputs/strict_dta_720x338_v1/conplex_cache/Morgan_features.h5"
PROTEIN_MAIN = ROOT / "outputs/strict_dta_720x338_v1/conplex_cache/ProtBert_features.h5"
PROTEIN_RECOVERED = ROOT / "outputs/recovered_dta_720x46_v1/conplex_cache/ProtBert_features.h5"
OUT = Path(os.environ.get(
    "BIOMASTER_MUTATION_OUT", str(RUN / f"mutation_aware_multitask_{MODEL_VERSION.lower()}")
))

SEED = 20260813
FOLDS = 5
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}
MAX_MUTATIONS = 4
BASELINE_FEATURES = [
    PRIOR_SCORE_COLUMN,
    PRIOR_RANK_COLUMN,
    "dta_cross_target_consensus_score",
    "conplex_percentile_within_ligand_384",
    "drugclip_percentile_within_ligand_382",
    "conplex_rank_within_ligand_384",
    "drugclip_rank_within_ligand_382",
    "dta_consensus_rank_within_ligand_384",
    "is_any_frozen_known_relationship",
    "local_chembl37_unreported_pair",
]


def versioned(stem: str) -> str:
    return f"{stem}_{MODEL_VERSION.lower()}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {str(key): np.asarray(handle[key], dtype=np.float32) for key in handle.keys()}


def mutation_tokens(construct: str, sequence_length: int) -> tuple[list[int], list[int], list[float], list[float]]:
    inside = construct[construct.find("(") + 1:construct.rfind(")")] if "(" in construct else construct
    substitutions = re.findall(
        r"(?:^|_)([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])(?=_|$)", inside
    )
    from_tokens = [0] * MAX_MUTATIONS
    to_tokens = [0] * MAX_MUTATIONS
    positions = [0.0] * MAX_MUTATIONS
    for index, (source, position, destination) in enumerate(substitutions[:MAX_MUTATIONS]):
        from_tokens[index] = AA_INDEX[source]
        to_tokens[index] = AA_INDEX[destination]
        positions[index] = min(float(position) / max(sequence_length, 1), 2.0)
    upper = construct.upper()
    flags = [
        min(len(substitutions), MAX_MUTATIONS) / MAX_MUTATIONS,
        float("DEL" in upper), float("INS" in upper), float("ITD" in upper),
        float("_" in construct and not substitutions),
        float("EX" in upper), float(len(substitutions) == 0),
        min(len(construct), 80) / 80.0,
    ]
    return from_tokens, to_tokens, positions, flags


def binary_metrics(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {
        "rows": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "micro_auroc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None,
        "micro_auprc": float(average_precision_score(y, score)) if np.unique(y).size == 2 else None,
    }
    aucs, aps = [], []
    recall = {10: [], 20: [], 50: []}
    for group in pd.unique(groups):
        mask = groups == group
        yy, ss = y[mask], score[mask]
        if np.unique(yy).size == 2:
            aucs.append(roc_auc_score(yy, ss))
            aps.append(average_precision_score(yy, ss))
        positives = int(yy.sum())
        if positives:
            order = np.argsort(-ss, kind="stable")
            for k in recall:
                recall[k].append(float(yy[order[:min(k, len(order))]].sum() / positives))
    output["macro_drug_auroc"] = float(np.mean(aucs)) if aucs else None
    output["macro_drug_auprc"] = float(np.mean(aps)) if aps else None
    output["drugs_with_two_classes"] = len(aucs)
    for k, values in recall.items():
        output[f"macro_recall_at_{k}"] = float(np.mean(values)) if values else None
    return output


class MutationAwareModel(torch.nn.Module):
    def __init__(self, drug_vectors: np.ndarray, target_vectors: np.ndarray, baseline_dim: int, mutation_flag_dim: int) -> None:
        super().__init__()
        self.register_buffer("drug_vectors", torch.as_tensor(drug_vectors, dtype=torch.float32))
        self.register_buffer("target_vectors", torch.as_tensor(target_vectors, dtype=torch.float32))
        self.drug_encoder = torch.nn.Sequential(
            torch.nn.Linear(drug_vectors.shape[1], 128), torch.nn.LayerNorm(128),
            torch.nn.GELU(), torch.nn.Dropout(0.12), torch.nn.Linear(128, 64), torch.nn.GELU(),
        )
        self.target_encoder = torch.nn.Sequential(
            torch.nn.Linear(target_vectors.shape[1], 128), torch.nn.LayerNorm(128),
            torch.nn.GELU(), torch.nn.Dropout(0.12), torch.nn.Linear(128, 64), torch.nn.GELU(),
        )
        self.from_aa = torch.nn.Embedding(len(AA_INDEX) + 1, 8, padding_idx=0)
        self.to_aa = torch.nn.Embedding(len(AA_INDEX) + 1, 8, padding_idx=0)
        mutation_input = MAX_MUTATIONS * 17 + mutation_flag_dim
        self.mutation_encoder = torch.nn.Sequential(
            torch.nn.Linear(mutation_input, 96), torch.nn.LayerNorm(96),
            torch.nn.GELU(), torch.nn.Dropout(0.10), torch.nn.Linear(96, 64), torch.nn.GELU(),
        )
        self.baseline_encoder = torch.nn.Sequential(
            torch.nn.Linear(baseline_dim, 32), torch.nn.LayerNorm(32), torch.nn.GELU(),
        )
        self.gate = torch.nn.Sequential(torch.nn.Linear(64 * 3 + 32, 64), torch.nn.Sigmoid())
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(64 * 6 + 32, 192), torch.nn.LayerNorm(192),
            torch.nn.GELU(), torch.nn.Dropout(0.15),
            torch.nn.Linear(192, 64), torch.nn.GELU(),
        )
        self.active_head = torch.nn.Linear(64, 1)
        self.sensitization_head = torch.nn.Linear(64, 1)
        self.residual_head = torch.nn.Linear(64, 1)

    def forward(
        self, drug_index: torch.Tensor, target_index: torch.Tensor,
        from_aa: torch.Tensor, to_aa: torch.Tensor, positions: torch.Tensor,
        flags: torch.Tensor, baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_drugs = self.drug_encoder(self.drug_vectors)
        all_targets = self.target_encoder(self.target_vectors)
        drug = all_drugs[drug_index]
        target = all_targets[target_index]
        mutation_tokens = torch.cat([
            self.from_aa(from_aa), self.to_aa(to_aa), positions.unsqueeze(-1)
        ], dim=-1).flatten(1)
        mutation = self.mutation_encoder(torch.cat([mutation_tokens, flags], dim=1))
        base = self.baseline_encoder(baseline)
        gate = self.gate(torch.cat([drug, target, mutation, base], dim=1))
        mutation_conditioned_target = gate * mutation + (1.0 - gate) * target
        hidden = self.fusion(torch.cat([
            drug, target, mutation, mutation_conditioned_target,
            drug * target, target * mutation, base,
        ], dim=1))
        return (
            self.active_head(hidden).squeeze(1),
            self.sensitization_head(hidden).squeeze(1),
            torch.sigmoid(self.residual_head(hidden).squeeze(1)),
        )


def make_pair_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    positive, negative = [], []
    for _, group in frame.groupby("kirhub_compound", sort=False):
        pos = group.index[group["active_label"].eq(1)].to_numpy(dtype=int)
        neg = group.index[group["active_label"].eq(0)].to_numpy(dtype=int)
        if not len(pos) or not len(neg):
            continue
        for item in pos:
            selected = rng.choice(neg, size=min(4, len(neg)), replace=False)
            positive.extend([item] * len(selected))
            negative.extend(selected.tolist())
    return np.asarray(positive), np.asarray(negative)


def tensor_bundle(frame: pd.DataFrame, baseline_mean: np.ndarray, baseline_scale: np.ndarray) -> dict[str, torch.Tensor]:
    baseline = frame[BASELINE_FEATURES].to_numpy(dtype=np.float32)
    baseline = (baseline - baseline_mean) / baseline_scale
    return {
        "drug_index": torch.as_tensor(frame["_drug_index"].to_numpy(copy=True), dtype=torch.long, device=DEVICE),
        "target_index": torch.as_tensor(frame["_target_index"].to_numpy(copy=True), dtype=torch.long, device=DEVICE),
        "from_aa": torch.as_tensor(np.stack(frame["_from_aa"]), dtype=torch.long, device=DEVICE),
        "to_aa": torch.as_tensor(np.stack(frame["_to_aa"]), dtype=torch.long, device=DEVICE),
        "positions": torch.as_tensor(np.stack(frame["_positions"]), dtype=torch.float32, device=DEVICE),
        "flags": torch.as_tensor(np.stack(frame["_mutation_flags"]), dtype=torch.float32, device=DEVICE),
        "baseline": torch.as_tensor(baseline, dtype=torch.float32, device=DEVICE),
    }


def forward_bundle(model: MutationAwareModel, bundle: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(**bundle)


def fit_model(
    train: pd.DataFrame, drug_vectors: np.ndarray, target_vectors: np.ndarray, seed: int,
) -> tuple[MutationAwareModel, np.ndarray, np.ndarray]:
    set_seeds(seed)
    baseline = train[BASELINE_FEATURES].to_numpy(dtype=np.float32)
    mean = baseline.mean(axis=0)
    scale = baseline.std(axis=0)
    scale[scale < 1e-8] = 1.0
    bundle = tensor_bundle(train, mean, scale)
    active = torch.as_tensor(train["active_label"].to_numpy(dtype=np.float32), device=DEVICE)
    sensitization = torch.as_tensor(train["sensitization_label"].to_numpy(dtype=np.float32), device=DEVICE)
    residual = torch.as_tensor(train["residual_scaled"].to_numpy(dtype=np.float32), device=DEVICE)
    pos_index, neg_index = make_pair_indices(train.reset_index(drop=True))
    pos_index = torch.as_tensor(pos_index, dtype=torch.long, device=DEVICE)
    neg_index = torch.as_tensor(neg_index, dtype=torch.long, device=DEVICE)
    model = MutationAwareModel(drug_vectors, target_vectors, len(BASELINE_FEATURES), 8).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    active_weight = torch.tensor([(len(active) - active.sum()).item() / max(active.sum().item(), 1)], device=DEVICE)
    sens_weight = torch.tensor(
        [(len(sensitization) - sensitization.sum()).item() / max(sensitization.sum().item(), 1)], device=DEVICE
    )
    active_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=active_weight)
    sens_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=sens_weight)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        active_logit, sens_logit, residual_prediction = forward_bundle(model, bundle)
        active_loss = active_loss_fn(active_logit, active)
        sensitization_loss = sens_loss_fn(sens_logit, sensitization)
        regression_loss = torch.nn.functional.smooth_l1_loss(residual_prediction, residual, beta=0.10)
        ranking_loss = torch.nn.functional.softplus(
            -(active_logit[pos_index] - active_logit[neg_index])
        ).mean()
        loss = active_loss + 0.30 * sensitization_loss + 0.25 * regression_loss + 0.30 * ranking_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model, mean, scale


def predict(
    model: MutationAwareModel, frame: pd.DataFrame, mean: np.ndarray, scale: np.ndarray,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active, sensitization, residual = [], [], []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            bundle = tensor_bundle(frame.iloc[start:start + batch_size], mean, scale)
            active_logit, sens_logit, residual_prediction = forward_bundle(model, bundle)
            active.append(torch.sigmoid(active_logit).cpu().numpy())
            sensitization.append(torch.sigmoid(sens_logit).cpu().numpy())
            residual.append((100.0 * residual_prediction).cpu().numpy())
    return np.concatenate(active), np.concatenate(sensitization), np.concatenate(residual)


def run_grouped_oof(
    panel: pd.DataFrame, groups: pd.Series, drug_vectors: np.ndarray,
    target_vectors: np.ndarray, seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    splitter = GroupKFold(n_splits=FOLDS)
    splits = list(splitter.split(panel, panel["active_label"], groups))
    active_oof = np.full(len(panel), np.nan, dtype=np.float32)
    sens_oof = np.full(len(panel), np.nan, dtype=np.float32)
    residual_oof = np.full(len(panel), np.nan, dtype=np.float32)
    assignment = np.zeros(len(panel), dtype=int)
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        model, mean, scale = fit_model(
            panel.iloc[train_index].copy().reset_index(drop=True),
            drug_vectors, target_vectors, SEED + seed_offset + fold,
        )
        active_prediction, sens_prediction, residual_prediction = predict(
            model, panel.iloc[test_index], mean, scale
        )
        active_oof[test_index] = active_prediction
        sens_oof[test_index] = sens_prediction
        residual_oof[test_index] = residual_prediction
        assignment[test_index] = fold
    if not np.isfinite(active_oof).all() or not np.isfinite(sens_oof).all() or not np.isfinite(residual_oof).all():
        raise RuntimeError("Mutation-aware grouped OOF predictions incomplete")
    return active_oof, sens_oof, residual_oof, assignment, splits


def main() -> None:
    inputs = [PANEL, PRIOR, PAIR, TARGET, SEQUENCE, MORGAN, PROTEIN_MAIN, PROTEIN_RECOVERED]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    set_seeds(SEED)

    panel = pd.read_csv(PANEL, low_memory=False)
    prior = pd.read_csv(PRIOR, low_memory=False)[[
        "pairId", PRIOR_SCORE_COLUMN, PRIOR_RANK_COLUMN
    ]]
    panel = panel.merge(prior, on="pairId", how="left", validate="many_to_one")
    if len(panel) != 19639 or panel[BASELINE_FEATURES].isna().any().any():
        raise RuntimeError(f"Mutation panel or {MODEL_VERSION} prior merge incomplete")
    panel["active_label"] = panel["variant_active_le30pct_residual"].astype(int)
    panel["sensitization_label"] = panel["mutation_response_class_v7"].eq(
        "M2_VARIANT_SPECIFIC_SENSITIZATION"
    ).astype(int)
    panel["residual_scaled"] = panel["variant_residual_activity_pct_1uM"].clip(0, 100) / 100.0

    pair_drugs = pd.read_csv(PAIR, usecols=["ligand_inchikey", "ligand_smiles"]).drop_duplicates(
        "ligand_inchikey"
    )
    panel["ligand_inchikey"] = panel["pairId"].str.split("__", n=1).str[0]
    drugs = panel[["ligand_inchikey", "kirhub_compound"]].drop_duplicates().sort_values(
        "ligand_inchikey"
    ).reset_index(drop=True).merge(pair_drugs, on="ligand_inchikey", how="left", validate="one_to_one")
    targets = panel[["target_chembl_id", "gene_symbol"]].drop_duplicates().sort_values(
        "target_chembl_id"
    ).reset_index(drop=True)
    drug_index = {key: index for index, key in enumerate(drugs["ligand_inchikey"])}
    target_index = {key: index for index, key in enumerate(targets["target_chembl_id"])}
    panel["_drug_index"] = panel["ligand_inchikey"].map(drug_index)
    panel["_target_index"] = panel["target_chembl_id"].map(target_index)

    morgan_map = load_h5(MORGAN)
    cache_smiles = drugs["ligand_smiles"].map(lambda value: value if value in morgan_map else value.replace("/", "|"))
    if not cache_smiles.isin(morgan_map).all():
        raise RuntimeError("Mutation drugs missing from Morgan cache")
    drug_vectors = np.stack([morgan_map[smiles] for smiles in cache_smiles]).astype(np.float32)

    sequences = pd.read_csv(SEQUENCE, usecols=["target_chembl_id", "sequence", "sequence_length"]).drop_duplicates(
        "target_chembl_id"
    ).set_index("target_chembl_id")
    protein_map = load_h5(PROTEIN_MAIN)
    protein_map.update(load_h5(PROTEIN_RECOVERED))
    target_vectors = np.stack([
        protein_map[str(sequences.loc[target, "sequence"])] for target in targets["target_chembl_id"]
    ]).astype(np.float32)
    target_vectors /= np.maximum(np.linalg.norm(target_vectors, axis=1, keepdims=True), 1e-12)
    lengths = sequences["sequence_length"].astype(int).to_dict()
    tokens = [mutation_tokens(row.variant_construct, lengths[row.target_chembl_id]) for row in panel.itertuples()]
    panel["_from_aa"] = [item[0] for item in tokens]
    panel["_to_aa"] = [item[1] for item in tokens]
    panel["_positions"] = [item[2] for item in tokens]
    panel["_mutation_flags"] = [item[3] for item in tokens]

    active_oof, sens_oof, residual_oof, fold_assignment, splits = run_grouped_oof(
        panel, panel["kirhub_compound"], drug_vectors, target_vectors, 0
    )
    variant_active_oof, variant_sens_oof, variant_residual_oof, variant_fold, variant_splits = run_grouped_oof(
        panel, panel["variant_construct"], drug_vectors, target_vectors, 100
    )
    gene_active_oof, gene_sens_oof, gene_residual_oof, gene_fold, gene_splits = run_grouped_oof(
        panel, panel["gene_symbol"], drug_vectors, target_vectors, 200
    )

    final_model, final_mean, final_scale = fit_model(panel.copy().reset_index(drop=True), drug_vectors, target_vectors, SEED)
    final_active, final_sens, final_residual = predict(final_model, panel, final_mean, final_scale)
    model_path = OUT / f"MUTATION_AWARE_MULTITASK_MODEL_{MODEL_VERSION}.pt"
    torch.save({
        "state_dict": final_model.state_dict(), "baseline_mean": final_mean, "baseline_scale": final_scale,
        "drug_index": drug_index, "target_index": target_index, "baseline_features": BASELINE_FEATURES,
        "amino_acid_index": AA_INDEX, "max_mutations": MAX_MUTATIONS,
    }, model_path)

    output_columns = [column for column in panel.columns if not column.startswith("_")]
    output = panel[output_columns].copy()
    active_oof_column = versioned("mutation_aware_active_probability_oof")
    sens_oof_column = versioned("mutation_aware_sensitization_probability_oof")
    output[active_oof_column] = active_oof
    output[sens_oof_column] = sens_oof
    output[versioned("mutation_aware_residual_pct_oof")] = residual_oof
    output[versioned("mutation_aware_drug_cold_start_fold")] = fold_assignment
    output[versioned("mutation_aware_active_probability_variant_cold_oof")] = variant_active_oof
    output[versioned("mutation_aware_sensitization_probability_variant_cold_oof")] = variant_sens_oof
    output[versioned("mutation_aware_residual_pct_variant_cold_oof")] = variant_residual_oof
    output[versioned("mutation_aware_variant_cold_start_fold")] = variant_fold
    output[versioned("mutation_aware_active_probability_gene_cold_oof")] = gene_active_oof
    output[versioned("mutation_aware_sensitization_probability_gene_cold_oof")] = gene_sens_oof
    output[versioned("mutation_aware_residual_pct_gene_cold_oof")] = gene_residual_oof
    output[versioned("mutation_aware_gene_cold_start_fold")] = gene_fold
    output[versioned("mutation_aware_active_probability_full")] = final_active
    output[versioned("mutation_aware_sensitization_probability_full")] = final_sens
    output[versioned("mutation_aware_residual_pct_full")] = final_residual
    output[versioned("mutation_aware_rank_within_drug_oof")] = output.groupby("kirhub_compound")[
        active_oof_column
    ].rank(method="min", ascending=False)
    output_path = OUT / f"MUTATION_AWARE_PANEL_19639_{MODEL_VERSION}.csv.gz"
    output.to_csv(output_path, index=False, compression="gzip")

    metric_rows = []
    active_scores = {
        "MUTATION_AWARE_DRUG_COLD_OOF": active_oof,
        "MUTATION_AWARE_VARIANT_COLD_OOF": variant_active_oof,
        "MUTATION_AWARE_GENE_COLD_OOF": gene_active_oof,
        f"{MODEL_VERSION}_WT_TARGET_PRIOR": panel[PRIOR_SCORE_COLUMN].to_numpy(),
        "FROZEN_DTA_WT_PRIOR": panel["dta_cross_target_consensus_score"].to_numpy(),
        "CONPLEX_WT_PRIOR": panel["conplex_percentile_within_ligand_384"].to_numpy(),
        "DRUGCLIP_WT_PRIOR": panel["drugclip_percentile_within_ligand_382"].to_numpy(),
    }
    slices = {
        "ALL_VARIANTS": np.ones(len(panel), dtype=bool),
        "COMMON_CLINICAL_HOTSPOTS": panel["common_clinical_hotspot_panel_v7"].astype(bool).to_numpy(),
        "STRICT_LOCAL_UNREPORTED_COMMON_HOTSPOTS": (
            panel["common_clinical_hotspot_panel_v7"].astype(bool)
            & panel["local_chembl37_unreported_pair"].astype(bool)
            & ~panel["is_any_frozen_known_relationship"].astype(bool)
        ).to_numpy(),
    }
    groups = panel["kirhub_compound"].to_numpy()
    y_active = panel["active_label"].to_numpy()
    for slice_name, mask in slices.items():
        for model_name, score in active_scores.items():
            metric_rows.append({
                "task": "VARIANT_ACTIVE_LE30PCT_RESIDUAL", "evaluation_slice": slice_name,
                "model_name": model_name, **binary_metrics(y_active[mask], score[mask], groups[mask]),
            })
    y_sens = panel["sensitization_label"].to_numpy()
    for slice_name, mask in slices.items():
        for model_name, score in {
            "MUTATION_AWARE_DRUG_COLD_OOF": sens_oof,
            "MUTATION_AWARE_VARIANT_COLD_OOF": variant_sens_oof,
            "MUTATION_AWARE_GENE_COLD_OOF": gene_sens_oof,
            f"{MODEL_VERSION}_WT_TARGET_PRIOR": panel[PRIOR_SCORE_COLUMN].to_numpy(),
        }.items():
            metric_rows.append({
                "task": "VARIANT_SPECIFIC_SENSITIZATION", "evaluation_slice": slice_name,
                "model_name": model_name, **binary_metrics(y_sens[mask], score[mask], groups[mask]),
            })
    metrics_table = pd.DataFrame(metric_rows)
    metrics_path = OUT / f"MUTATION_AWARE_DRUG_COLD_START_METRICS_{MODEL_VERSION}.csv"
    metrics_table.to_csv(metrics_path, index=False)

    all_active_model = metrics_table[
        metrics_table["task"].eq("VARIANT_ACTIVE_LE30PCT_RESIDUAL")
        & metrics_table["evaluation_slice"].eq("ALL_VARIANTS")
        & metrics_table["model_name"].eq("MUTATION_AWARE_DRUG_COLD_OOF")
    ].iloc[0]
    all_active_prior = metrics_table[
        metrics_table["task"].eq("VARIANT_ACTIVE_LE30PCT_RESIDUAL")
        & metrics_table["evaluation_slice"].eq("ALL_VARIANTS")
        & metrics_table["model_name"].eq(f"{MODEL_VERSION}_WT_TARGET_PRIOR")
    ].iloc[0]
    common_model = metrics_table[
        metrics_table["task"].eq("VARIANT_ACTIVE_LE30PCT_RESIDUAL")
        & metrics_table["evaluation_slice"].eq("COMMON_CLINICAL_HOTSPOTS")
        & metrics_table["model_name"].eq("MUTATION_AWARE_DRUG_COLD_OOF")
    ].iloc[0]
    sensitization_model = metrics_table[
        metrics_table["task"].eq("VARIANT_SPECIFIC_SENSITIZATION")
        & metrics_table["evaluation_slice"].eq("ALL_VARIANTS")
        & metrics_table["model_name"].eq("MUTATION_AWARE_DRUG_COLD_OOF")
    ].iloc[0]
    variant_cold_model = metrics_table[
        metrics_table["task"].eq("VARIANT_ACTIVE_LE30PCT_RESIDUAL")
        & metrics_table["evaluation_slice"].eq("ALL_VARIANTS")
        & metrics_table["model_name"].eq("MUTATION_AWARE_VARIANT_COLD_OOF")
    ].iloc[0]
    gene_cold_model = metrics_table[
        metrics_table["task"].eq("VARIANT_ACTIVE_LE30PCT_RESIDUAL")
        & metrics_table["evaluation_slice"].eq("ALL_VARIANTS")
        & metrics_table["model_name"].eq("MUTATION_AWARE_GENE_COLD_OOF")
    ].iloc[0]
    checks = {
        "exactly_19639_variant_measurements": len(panel) == 19639,
        "exactly_76_drugs_35_genes_299_constructs": (
            panel["kirhub_compound"].nunique() == 76 and panel["gene_symbol"].nunique() == 35
            and panel["variant_construct"].nunique() == 299
        ),
        "five_drug_cold_start_folds_complete": set(fold_assignment) == {1, 2, 3, 4, 5},
        "five_variant_and_gene_cold_start_folds_complete": set(variant_fold) == {1, 2, 3, 4, 5} and set(gene_fold) == {1, 2, 3, 4, 5},
        "no_drug_overlap_within_each_fold": all(
            not set(panel.iloc[train]["kirhub_compound"]) & set(panel.iloc[test]["kirhub_compound"])
            for train, test in splits
        ),
        "no_variant_or_gene_overlap_within_each_fold": (
            all(not set(panel.iloc[train]["variant_construct"]) & set(panel.iloc[test]["variant_construct"]) for train, test in variant_splits)
            and all(not set(panel.iloc[train]["gene_symbol"]) & set(panel.iloc[test]["gene_symbol"]) for train, test in gene_splits)
        ),
        "all_oof_predictions_finite": all(np.isfinite(values).all() for values in [
            active_oof, sens_oof, residual_oof, variant_active_oof, variant_sens_oof,
            variant_residual_oof, gene_active_oof, gene_sens_oof, gene_residual_oof,
        ]),
        "experimental_residuals_not_in_input_features": all("residual" not in column and "wt_min" not in column for column in BASELINE_FEATURES),
        "strict_variant_specific_sensitization_has_78_positives": int(panel["sensitization_label"].sum()) == 78,
        "model_outputs_complete": len(output) == 19639,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "architecture": {
            "name": "MORGAN_PROTBERT_MUTATION_TOKEN_GATED_MULTITASK",
            "drug_encoder": "Morgan-2048 -> 128 -> 64",
            "target_encoder": "frozen ProtBert-1024 -> 128 -> 64",
            "mutation_encoder": "up to four AA-from/AA-to/position tokens plus indel/fusion flags -> 64",
            "fusion": "mutation-conditioned target gate plus drug-target and target-mutation Hadamard interactions",
            "tasks": ["variant activity <=30% residual", "variant-specific sensitization", "continuous residual activity"],
            "loss": "class-balanced active BCE + 0.30 sensitization BCE + 0.25 Huber residual + 0.30 within-drug BPR",
        },
        "validation": "separate five-fold cold-drug, cold-variant-construct, and cold-gene OOF evaluations",
        "all_variant_active_metrics": {
            "mutation_model_auroc": float(all_active_model["micro_auroc"]),
            "mutation_model_auprc": float(all_active_model["micro_auprc"]),
            "wt_prior_name": f"{MODEL_VERSION}_WT_TARGET_PRIOR",
            "wt_prior_auroc": float(all_active_prior["micro_auroc"]),
            "wt_prior_auprc": float(all_active_prior["micro_auprc"]),
            "residual_mae_pct": float(mean_absolute_error(panel["variant_residual_activity_pct_1uM"], residual_oof)),
        },
        "common_hotspot_active_metrics": {
            "rows": int(common_model["rows"]), "positives": int(common_model["positives"]),
            "auroc": float(common_model["micro_auroc"]), "auprc": float(common_model["micro_auprc"]),
        },
        "harder_generalization_metrics": {
            "cold_variant_auroc": float(variant_cold_model["micro_auroc"]),
            "cold_variant_auprc": float(variant_cold_model["micro_auprc"]),
            "cold_gene_auroc": float(gene_cold_model["micro_auroc"]),
            "cold_gene_auprc": float(gene_cold_model["micro_auprc"]),
        },
        "variant_specific_sensitization_metrics": {
            "positives": int(sensitization_model["positives"]),
            "auroc": float(sensitization_model["micro_auroc"]),
            "auprc": float(sensitization_model["micro_auprc"]),
        },
        "claim_boundary": (
            "KiRHub supplies both training labels and the cold-drug/cold-variant/cold-gene OOF evaluations; results are internal cross-validation, not independent external validation or clinical evidence."
        ),
        "version": MODEL_VERSION,
        "wt_prior": {
            "path": str(PRIOR.relative_to(ROOT)), "score_column": PRIOR_SCORE_COLUMN,
            "rank_column": PRIOR_RANK_COLUMN,
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in (output_path, metrics_path, model_path)
        },
        "software": {"torch": torch.__version__, "device": str(DEVICE), "seed": SEED},
    }
    summary_path = OUT / f"MUTATION_AWARE_MULTITASK_SUMMARY_{MODEL_VERSION}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
