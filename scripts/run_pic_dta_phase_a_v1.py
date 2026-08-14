#!/usr/bin/env python3
"""Run frozen PIC-DTA paired interval-censored Phase A mechanism tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/biomaster_pic_dta_phase_a_freeze_20260814.json"
DEFAULT_OUTPUT = ROOT / "outputs/old_drug_target_sota_v1/pic_dta_phase_a_v1"
VARIANTS = [
    "ZERO_TAU_WT_ECHO",
    "RIDGE_POINT_EXACT",
    "DIRECT_MUTANT_TOBIT",
    "WT_LOCAL_ONLY_INTERVAL",
    "SHUFFLED_MODIFICATION_IDENTITY_WITHIN_GENE_INTERVAL",
    "NO_DRUG_CONDITIONING_INTERVAL",
    "FULL_POINT_EXACT_ONLY",
    "FULL_PAIRED_INTERVAL",
]
NEURAL_VARIANTS = VARIANTS[2:]
PROTOCOLS = ["VARIANT_CONSTRUCT_COLD", "BASE_GENE_COLD", "DRUG_SCAFFOLD_COLD"]
CATEGORIES = [
    "WT_UNCAPPED_MODIFIED_UNCAPPED",
    "WT_CAPPED_MODIFIED_UNCAPPED",
    "WT_UNCAPPED_MODIFIED_CAPPED",
    "WT_CAPPED_MODIFIED_CAPPED",
]
CATEGORY_CODE = {category: index for index, category in enumerate(CATEGORIES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def generic_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError(f"Invalid canonical SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return "ACYCLIC:" + Chem.MolToSmiles(molecule, canonical=True)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic, canonical=True)


def prepare_rows(labels: pd.DataFrame, constructs: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
    paired = labels[
        labels["is_modified_official"].astype(bool)
        & labels["has_paired_wildtype"].astype(bool)
    ].copy()
    if len(paired) != 3960:
        raise RuntimeError(f"Expected 3960 paired rows, observed {len(paired)}")
    paired = paired.merge(
        constructs[
            [
                "construct_feature_index",
                "protein",
                "base_wildtype_protein",
                "paired_wildtype_available",
            ]
        ],
        on=["protein", "base_wildtype_protein"],
        how="left",
        validate="many_to_one",
    ).merge(
        drugs[["drug_feature_index", "drug_name", "canonical_smiles_rdkit"]],
        on="drug_name",
        how="left",
        validate="many_to_one",
    )
    if paired[["construct_feature_index", "drug_feature_index"]].isna().any().any():
        raise RuntimeError("Feature joins are incomplete")
    paired["construct_feature_index"] = paired["construct_feature_index"].astype(int)
    paired["drug_feature_index"] = paired["drug_feature_index"].astype(int)
    paired["drug_scaffold_group"] = paired["canonical_smiles_rdkit"].map(generic_scaffold)
    wt_capped = paired["wildtype_is_capped_10uM"].astype(bool)
    modified_capped = paired["is_capped_10uM"].astype(bool)
    paired["censoring_category"] = np.select(
        [
            ~wt_capped & ~modified_capped,
            wt_capped & ~modified_capped,
            ~wt_capped & modified_capped,
            wt_capped & modified_capped,
        ],
        CATEGORIES,
        default="ERROR",
    )
    paired["category_code"] = paired["censoring_category"].map(CATEGORY_CODE).astype(int)
    paired["delta_exact"] = paired["y_pKd"] - paired["wildtype_y_pKd"]
    paired["delta_lower"] = paired["y_pKd"] - 5.0
    paired["delta_upper"] = 5.0 - paired["wildtype_y_pKd"]
    counts = paired["censoring_category"].value_counts().to_dict()
    expected = dict(zip(CATEGORIES, [1601, 134, 157, 2068]))
    if counts != expected:
        raise RuntimeError(f"Censoring count drift: {counts} != {expected}")
    paired = paired.sort_values(["protein", "drug_name"], kind="stable").reset_index(drop=True)
    paired["pair_row_index"] = np.arange(len(paired), dtype=int)
    return paired


def create_outer_folds(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    assignments: list[dict[str, Any]] = []
    proteins = sorted(rows["protein"].unique(), key=lambda value: stable_digest("PIC-DTA-VARIANT|" + value))
    variant_fold = {protein: index % 5 for index, protein in enumerate(proteins)}
    if sorted(pd.Series(variant_fold).value_counts().tolist()) != [11, 11, 11, 11, 11]:
        raise RuntimeError("Variant-cold fold sizes are not 11 constructs each")
    gene_groups = config["outer_protocols"]["BASE_GENE_COLD"]["fixed_test_groups"]
    gene_fold = {
        gene: int(fold)
        for fold, genes in gene_groups.items()
        for gene in genes
    }
    scaffold_sizes = rows[["drug_name", "drug_scaffold_group"]].drop_duplicates()[
        "drug_scaffold_group"
    ].value_counts().to_dict()
    scaffold_order = sorted(
        scaffold_sizes,
        key=lambda group: (-scaffold_sizes[group], stable_digest("PIC-DTA-DRUG|" + group)),
    )
    scaffold_load = [0] * 5
    scaffold_fold = {}
    for group in scaffold_order:
        fold = min(range(5), key=lambda candidate: (scaffold_load[candidate], candidate))
        scaffold_fold[group] = fold
        scaffold_load[fold] += int(scaffold_sizes[group])
    fold_maps = {
        "VARIANT_CONSTRUCT_COLD": rows["protein"].map(variant_fold),
        "BASE_GENE_COLD": rows["base_wildtype_protein"].map(gene_fold),
        "DRUG_SCAFFOLD_COLD": rows["drug_scaffold_group"].map(scaffold_fold),
    }
    for protocol, fold_series in fold_maps.items():
        if fold_series.isna().any() or set(fold_series.unique()) != set(range(5)):
            raise RuntimeError(f"Incomplete {protocol} fold assignment")
        for row_index, fold in zip(rows["pair_row_index"], fold_series.astype(int)):
            assignments.append(
                {
                    "protocol": protocol,
                    "pair_row_index": int(row_index),
                    "outer_fold": int(fold),
                }
            )
    return pd.DataFrame(assignments)


def protocol_group(rows: pd.DataFrame, protocol: str) -> pd.Series:
    if protocol == "VARIANT_CONSTRUCT_COLD":
        return rows["protein"]
    if protocol == "BASE_GENE_COLD":
        return rows["base_wildtype_protein"]
    if protocol == "DRUG_SCAFFOLD_COLD":
        return rows["drug_scaffold_group"]
    raise KeyError(protocol)


def inner_train_validation_indices(
    outer_train: pd.DataFrame, protocol: str, fold: int, fraction: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = sorted(
        protocol_group(outer_train, protocol).astype(str).unique(),
        key=lambda value: stable_digest(f"PIC-DTA-INNER|{protocol}|{fold}|{value}"),
    )
    validation_count = max(1, int(math.ceil(len(groups) * fraction)))
    validation_groups = set(groups[:validation_count])
    is_validation = protocol_group(outer_train, protocol).astype(str).isin(validation_groups)
    return (
        outer_train.index[~is_validation].to_numpy(dtype=int),
        outer_train.index[is_validation].to_numpy(dtype=int),
        sorted(validation_groups),
    )


class FixedPCA:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.rank = 0

    def fit(self, values: np.ndarray) -> "FixedPCA":
        scaled = self.scaler.fit_transform(values.astype(np.float64))
        self.rank = min(self.dimensions, scaled.shape[0] - 1, scaled.shape[1])
        if self.rank < 1:
            raise RuntimeError(f"Insufficient samples for PCA: {scaled.shape}")
        self.pca = PCA(n_components=self.rank, svd_solver="full")
        self.pca.fit(scaled)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA transformer is not fitted")
        transformed = self.pca.transform(self.scaler.transform(values.astype(np.float64)))
        output = np.zeros((len(values), self.dimensions), dtype=np.float32)
        output[:, : self.rank] = transformed.astype(np.float32)
        return output


class FoldFeatures:
    def __init__(
        self,
        rows: pd.DataFrame,
        outer_train_index: np.ndarray,
        morgan: np.ndarray,
        wt: np.ndarray,
        modified: np.ndarray,
        delta_mean: np.ndarray,
        delta_maxabs: np.ndarray,
        explicit: np.ndarray,
    ):
        train = rows.loc[outer_train_index]
        train_drugs = np.sort(train["drug_feature_index"].unique())
        train_constructs = np.sort(train["construct_feature_index"].unique())
        self.drug_pca = FixedPCA(32).fit(morgan[train_drugs])
        self.wt_pca = FixedPCA(16).fit(wt[train_constructs])
        self.modified_pca = FixedPCA(16).fit(modified[train_constructs])
        self.delta_mean_pca = FixedPCA(16).fit(delta_mean[train_constructs])
        self.delta_maxabs_pca = FixedPCA(16).fit(delta_maxabs[train_constructs])
        self.explicit_scaler = StandardScaler().fit(explicit[train_constructs].astype(np.float64))
        self.drug = self.drug_pca.transform(morgan)
        self.wt = self.wt_pca.transform(wt)
        self.modified = self.modified_pca.transform(modified)
        self.delta_mean = self.delta_mean_pca.transform(delta_mean)
        self.delta_maxabs = self.delta_maxabs_pca.transform(delta_maxabs)
        self.explicit = self.explicit_scaler.transform(explicit).astype(np.float32)

    def row_features(
        self,
        rows: pd.DataFrame,
        variant: str,
        permutation: dict[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        drug_index = rows["drug_feature_index"].to_numpy(dtype=int)
        construct_index = rows["construct_feature_index"].to_numpy(dtype=int)
        if permutation is not None:
            construct_index = np.array(
                [permutation.get(int(index), int(index)) for index in construct_index], dtype=int
            )
        drug = self.drug[drug_index].copy()
        wt = self.wt[construct_index]
        delta_mean = self.delta_mean[construct_index]
        delta_maxabs = self.delta_maxabs[construct_index]
        explicit = self.explicit[construct_index]
        if variant == "WT_LOCAL_ONLY_INTERVAL":
            delta_mean = np.zeros_like(delta_mean)
            delta_maxabs = np.zeros_like(delta_maxabs)
            explicit = np.zeros_like(explicit)
        if variant == "NO_DRUG_CONDITIONING_INTERVAL":
            drug = np.zeros_like(drug)
        if variant == "DIRECT_MUTANT_TOBIT":
            wt = self.modified[construct_index]
        modification = np.concatenate([wt, delta_mean, delta_maxabs, explicit], axis=1)
        if modification.shape[1] != 96 or drug.shape[1] != 32:
            raise RuntimeError(f"Feature shape drift: {drug.shape}/{modification.shape}")
        return drug.astype(np.float32), modification.astype(np.float32)


def permutation_within_gene(rows: pd.DataFrame, seed: int) -> dict[int, int]:
    unique = rows[
        ["construct_feature_index", "base_wildtype_protein"]
    ].drop_duplicates().sort_values(["base_wildtype_protein", "construct_feature_index"])
    rng = np.random.default_rng(seed)
    mapping: dict[int, int] = {}
    for _, group in unique.groupby("base_wildtype_protein", sort=True):
        original = group["construct_feature_index"].to_numpy(dtype=int)
        donor = original.copy()
        rng.shuffle(donor)
        mapping.update({int(source): int(target) for source, target in zip(original, donor)})
    return mapping


class PICDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.modification_encoder = nn.Sequential(
            nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 32)
        )
        self.main_effect = nn.Linear(32, 1)
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))
        self.sigma_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, drug: torch.Tensor, modification: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drug_latent = self.drug_encoder(drug)
        modification_latent = self.modification_encoder(modification)
        interaction = (drug_latent * modification_latent).sum(dim=1) / math.sqrt(32.0)
        mean = self.main_effect(modification_latent).squeeze(1) + self.interaction_scale * interaction
        sigma = torch.nn.functional.softplus(
            self.sigma_head(torch.cat([drug_latent, modification_latent], dim=1)).squeeze(1)
        ) + 0.02
        return mean, torch.clamp(sigma, min=0.02, max=3.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def interval_loss(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    category: torch.Tensor,
    exact: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lower_weight: float,
    upper_weight: float,
) -> torch.Tensor:
    losses = torch.zeros_like(mean)
    weights = torch.zeros_like(mean)
    exact_mask = category == 0
    lower_mask = category == 1
    upper_mask = category == 2
    if exact_mask.any():
        z = (exact[exact_mask] - mean[exact_mask]) / sigma[exact_mask]
        losses[exact_mask] = 0.5 * z.square() + torch.log(sigma[exact_mask]) + 0.5 * math.log(2 * math.pi)
        weights[exact_mask] = 1.0
    if lower_mask.any():
        z = (mean[lower_mask] - lower[lower_mask]) / sigma[lower_mask]
        losses[lower_mask] = -torch.special.log_ndtr(z)
        weights[lower_mask] = lower_weight
    if upper_mask.any():
        z = (upper[upper_mask] - mean[upper_mask]) / sigma[upper_mask]
        losses[upper_mask] = -torch.special.log_ndtr(z)
        weights[upper_mask] = upper_weight
    informative = weights > 0
    if not informative.any():
        raise RuntimeError("No informative interval observations in batch")
    return (losses[informative] * weights[informative]).sum() / weights[informative].sum()


def direct_tobit_loss(
    mean: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, capped: torch.Tensor
) -> torch.Tensor:
    losses = torch.zeros_like(mean)
    exact = ~capped
    if exact.any():
        z = (y[exact] - mean[exact]) / sigma[exact]
        losses[exact] = 0.5 * z.square() + torch.log(sigma[exact]) + 0.5 * math.log(2 * math.pi)
    if capped.any():
        losses[capped] = -torch.special.log_ndtr((5.0 - mean[capped]) / sigma[capped])
    return losses.mean()


def tensors_for_rows(
    rows: pd.DataFrame,
    features: FoldFeatures,
    variant: str,
    permutation: dict[int, int] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    drug, modification = features.row_features(rows, variant, permutation)
    return {
        "drug": torch.from_numpy(drug).to(device),
        "modification": torch.from_numpy(modification).to(device),
        "category": torch.from_numpy(rows["category_code"].to_numpy(dtype=np.int64)).to(device),
        "exact": torch.from_numpy(rows["delta_exact"].to_numpy(dtype=np.float32)).to(device),
        "lower": torch.from_numpy(rows["delta_lower"].to_numpy(dtype=np.float32)).to(device),
        "upper": torch.from_numpy(rows["delta_upper"].to_numpy(dtype=np.float32)).to(device),
        "mutant_y": torch.from_numpy(rows["y_pKd"].to_numpy(dtype=np.float32)).to(device),
        "mutant_capped": torch.from_numpy(rows["is_capped_10uM"].to_numpy(dtype=bool)).to(device),
    }


def train_neural(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    features: FoldFeatures,
    variant: str,
    train_permutation: dict[int, int] | None,
    validation_permutation: dict[int, int] | None,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[PICDTA, dict[str, Any]]:
    seed_everything(seed)
    model = PICDTA().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    train_tensors = tensors_for_rows(
        train_rows, features, variant, train_permutation, device
    )
    validation_tensors = tensors_for_rows(
        validation_rows, features, variant, validation_permutation, device
    )
    if variant == "FULL_POINT_EXACT_ONLY":
        train_eligible = np.flatnonzero(train_rows["category_code"].to_numpy() == 0)
        validation_eligible = np.flatnonzero(validation_rows["category_code"].to_numpy() == 0)
    elif variant == "DIRECT_MUTANT_TOBIT":
        train_eligible = np.arange(len(train_rows))
        validation_eligible = np.arange(len(validation_rows))
    else:
        train_eligible = np.flatnonzero(train_rows["category_code"].to_numpy() < 3)
        validation_eligible = np.flatnonzero(validation_rows["category_code"].to_numpy() < 3)
    if not len(train_eligible) or not len(validation_eligible):
        raise RuntimeError(f"Empty eligible train/validation rows for {variant}")
    rng = np.random.default_rng(seed)
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    patience = 0
    batch_size = int(config["model"]["batch_size"])
    lower_weight = float(config["paired_interval_targets"]["loss"]["lower_weight"])
    upper_weight = float(config["paired_interval_targets"]["loss"]["upper_weight"])
    for epoch in range(int(config["model"]["maximum_epochs"])):
        model.train()
        order = rng.permutation(train_eligible)
        for start in range(0, len(order), batch_size):
            index_np = order[start : start + batch_size]
            index = torch.from_numpy(index_np).to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            mean, sigma = model(
                train_tensors["drug"][index], train_tensors["modification"][index]
            )
            if variant == "DIRECT_MUTANT_TOBIT":
                loss = direct_tobit_loss(
                    mean,
                    sigma,
                    train_tensors["mutant_y"][index],
                    train_tensors["mutant_capped"][index],
                )
            else:
                loss = interval_loss(
                    mean,
                    sigma,
                    train_tensors["category"][index],
                    train_tensors["exact"][index],
                    train_tensors["lower"][index],
                    train_tensors["upper"][index],
                    lower_weight,
                    upper_weight,
                )
                identity_modification = torch.zeros_like(train_tensors["modification"][index])
                identity_modification[:, :16] = train_tensors["modification"][index, :16]
                identity_mean, _ = model(train_tensors["drug"][index], identity_modification)
                loss = loss + float(config["model"]["identity_consistency_weight"]) * identity_mean.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss for {variant}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["model"]["gradient_clip_norm"])
            )
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_index = torch.from_numpy(validation_eligible).to(device=device, dtype=torch.long)
            mean, sigma = model(
                validation_tensors["drug"][validation_index],
                validation_tensors["modification"][validation_index],
            )
            if variant == "DIRECT_MUTANT_TOBIT":
                validation_loss = direct_tobit_loss(
                    mean,
                    sigma,
                    validation_tensors["mutant_y"][validation_index],
                    validation_tensors["mutant_capped"][validation_index],
                )
            else:
                validation_loss = interval_loss(
                    mean,
                    sigma,
                    validation_tensors["category"][validation_index],
                    validation_tensors["exact"][validation_index],
                    validation_tensors["lower"][validation_index],
                    validation_tensors["upper"][validation_index],
                    lower_weight,
                    upper_weight,
                )
            value = float(validation_loss.cpu())
        if value < best_loss - 1e-7:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= int(config["model"]["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {variant}")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "best_validation_loss": best_loss,
        "parameter_count": parameter_count(model),
        "train_eligible_rows": int(len(train_eligible)),
        "validation_eligible_rows": int(len(validation_eligible)),
    }


def predict_neural(
    model: PICDTA,
    rows: pd.DataFrame,
    features: FoldFeatures,
    variant: str,
    permutation: dict[int, int] | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    drug, modification = features.row_features(rows, variant, permutation)
    with torch.inference_mode():
        mean, sigma = model(
            torch.from_numpy(drug).to(device),
            torch.from_numpy(modification).to(device),
        )
    mean_np = mean.float().cpu().numpy()
    sigma_np = sigma.float().cpu().numpy()
    if variant == "DIRECT_MUTANT_TOBIT":
        mean_np = mean_np - rows["wildtype_y_pKd"].to_numpy(dtype=float)
    return mean_np.astype(float), sigma_np.astype(float)


def pointwise_interval_nll(rows: pd.DataFrame, mean: np.ndarray, sigma: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    from scipy.special import log_ndtr

    category = rows["category_code"].to_numpy(dtype=int)
    sigma = np.clip(sigma, 0.02, 3.0)
    result = np.full(len(rows), np.nan, dtype=float)
    exact = category == 0
    lower = category == 1
    upper = category == 2
    z = (rows["delta_exact"].to_numpy(dtype=float)[exact] - mean[exact]) / sigma[exact]
    result[exact] = 0.5 * z**2 + np.log(sigma[exact]) + 0.5 * math.log(2 * math.pi)
    result[lower] = -log_ndtr(
        (mean[lower] - rows["delta_lower"].to_numpy(dtype=float)[lower]) / sigma[lower]
    )
    result[upper] = -log_ndtr(
        (rows["delta_upper"].to_numpy(dtype=float)[upper] - mean[upper]) / sigma[upper]
    )
    return result


def concordance_index(y: np.ndarray, prediction: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    for index in range(len(y)):
        difference = y[index + 1 :] - y[index]
        non_tied = difference != 0
        if not non_tied.any():
            continue
        prediction_difference = prediction[index + 1 :] - prediction[index]
        product = difference[non_tied] * prediction_difference[non_tied]
        concordant += float((product > 0).sum()) + 0.5 * float((product == 0).sum())
        comparable += int(non_tied.sum())
    return concordant / comparable if comparable else float("nan")


def safe_correlation(function: Any, y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(function(y, prediction).statistic)


def grouped_mean_pearson(frame: pd.DataFrame, groups: list[str]) -> float:
    values = []
    for _, group in frame.groupby(groups, observed=True):
        if len(group) < 3:
            continue
        value = safe_correlation(
            pearsonr,
            group["delta_exact"].to_numpy(dtype=float),
            group["predicted_delta"].to_numpy(dtype=float),
        )
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def metrics_for(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    exact = frame[frame["category_code"].eq(0)]
    y = exact["delta_exact"].to_numpy(dtype=float)
    prediction = exact["predicted_delta"].to_numpy(dtype=float)
    informative = frame[frame["category_code"].lt(3)]
    category_weights = np.array(
        [
            1.0,
            float(config["paired_interval_targets"]["loss"]["lower_weight"]),
            float(config["paired_interval_targets"]["loss"]["upper_weight"]),
        ]
    )
    weights = category_weights[informative["category_code"].to_numpy(dtype=int)]
    interval_nll = informative["interval_nll"].to_numpy(dtype=float)
    lower = informative["category_code"].eq(1).to_numpy()
    upper = informative["category_code"].eq(2).to_numpy()
    bound_violation = np.zeros(len(informative), dtype=float)
    bound_violation[lower] = np.maximum(
        0,
        informative.loc[lower, "delta_lower"].to_numpy(dtype=float)
        - informative.loc[lower, "predicted_delta"].to_numpy(dtype=float),
    )
    bound_violation[upper] = np.maximum(
        0,
        informative.loc[upper, "predicted_delta"].to_numpy(dtype=float)
        - informative.loc[upper, "delta_upper"].to_numpy(dtype=float),
    )
    strict = exact[exact["delta_exact"].abs().ge(0.3)]
    strict_y = strict["delta_exact"].gt(0).astype(int).to_numpy()
    strict_score = strict["predicted_delta"].to_numpy(dtype=float)
    strict_prediction = (strict_score > 0).astype(int)
    return {
        "rows": len(frame),
        "exact_rows": len(exact),
        "informative_interval_rows": len(informative),
        "strict_direction_rows": len(strict),
        "exact_delta_mse": float(mean_squared_error(y, prediction)),
        "exact_delta_rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "exact_delta_mae": float(mean_absolute_error(y, prediction)),
        "exact_delta_pearson": safe_correlation(pearsonr, y, prediction),
        "exact_delta_spearman": safe_correlation(spearmanr, y, prediction),
        "exact_delta_c_index": concordance_index(y, prediction),
        "weighted_interval_nll": float(np.average(interval_nll, weights=weights)),
        "one_sided_bound_violation": float(np.average(bound_violation, weights=weights)),
        "strict_direction_balanced_accuracy": (
            float(balanced_accuracy_score(strict_y, strict_prediction))
            if len(np.unique(strict_y)) == 2
            else float("nan")
        ),
        "strict_direction_auroc": (
            float(roc_auc_score(strict_y, strict_score))
            if len(np.unique(strict_y)) == 2
            else float("nan")
        ),
        "mean_within_modification_across_drug_pearson": grouped_mean_pearson(
            exact, ["protein"]
        ),
        "mean_within_drug_gene_across_modification_pearson": grouped_mean_pearson(
            exact, ["drug_name", "base_wildtype_protein"]
        ),
    }


def bootstrap_comparisons(oof: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    replicates = int(config["uncertainty"]["cluster_bootstrap_replicates"])
    rng = np.random.default_rng(int(config["model"]["phase_a_seed"]) + 991)
    rows = []
    for protocol in PROTOCOLS:
        selected = oof[oof["protocol"].eq(protocol)]
        cluster_column = {
            "VARIANT_CONSTRUCT_COLD": "protein",
            "BASE_GENE_COLD": "base_wildtype_protein",
            "DRUG_SCAFFOLD_COLD": "drug_scaffold_group",
        }[protocol]
        full = selected[selected["variant"].eq("FULL_PAIRED_INTERVAL")].set_index("pair_row_index")
        clusters = sorted(full[cluster_column].unique())
        for comparator_name in [variant for variant in VARIANTS if variant != "FULL_PAIRED_INTERVAL"]:
            comparator = selected[selected["variant"].eq(comparator_name)].set_index("pair_row_index")
            common = full.index.intersection(comparator.index)
            paired = full.loc[common].copy()
            paired["comparator_prediction"] = comparator.loc[common, "predicted_delta"]
            paired["comparator_interval_nll"] = comparator.loc[common, "interval_nll"]
            # A cluster bootstrap of a mean depends only on cluster-level
            # counts and sums.  Pre-aggregating these sufficient statistics is
            # exactly equivalent to concatenating duplicated cluster frames,
            # while avoiding millions of repeated DataFrame selections.
            cluster_statistics = []
            category_weight_map = {
                0: 1.0,
                1: float(config["paired_interval_targets"]["loss"]["lower_weight"]),
                2: float(config["paired_interval_targets"]["loss"]["upper_weight"]),
            }
            for cluster in clusters:
                cluster_frame = paired[paired[cluster_column].eq(cluster)]
                exact = cluster_frame[cluster_frame["category_code"].eq(0)]
                informative = cluster_frame[cluster_frame["category_code"].lt(3)]
                weights = informative["category_code"].map(category_weight_map).to_numpy(dtype=float)
                cluster_statistics.append(
                    {
                        "exact_count": len(exact),
                        "full_squared_error_sum": float(
                            ((exact["predicted_delta"] - exact["delta_exact"]) ** 2).sum()
                        ),
                        "comparator_squared_error_sum": float(
                            ((exact["comparator_prediction"] - exact["delta_exact"]) ** 2).sum()
                        ),
                        "interval_weight_sum": float(weights.sum()),
                        "full_weighted_nll_sum": float(
                            np.sum(informative["interval_nll"].to_numpy(dtype=float) * weights)
                        ),
                        "comparator_weighted_nll_sum": float(
                            np.sum(
                                informative["comparator_interval_nll"].to_numpy(dtype=float)
                                * weights
                            )
                        ),
                    }
                )
            statistic_frame = pd.DataFrame(cluster_statistics)
            sampled_indices = rng.integers(
                0, len(clusters), size=(replicates, len(clusters))
            )
            exact_count = statistic_frame["exact_count"].to_numpy(dtype=float)[
                sampled_indices
            ].sum(axis=1)
            full_squared_error = statistic_frame[
                "full_squared_error_sum"
            ].to_numpy(dtype=float)[sampled_indices].sum(axis=1)
            comparator_squared_error = statistic_frame[
                "comparator_squared_error_sum"
            ].to_numpy(dtype=float)[sampled_indices].sum(axis=1)
            mse_differences = (
                full_squared_error / exact_count
                - comparator_squared_error / exact_count
            )
            interval_weight = statistic_frame["interval_weight_sum"].to_numpy(dtype=float)[
                sampled_indices
            ].sum(axis=1)
            full_weighted_nll = statistic_frame["full_weighted_nll_sum"].to_numpy(dtype=float)[
                sampled_indices
            ].sum(axis=1)
            comparator_weighted_nll = statistic_frame[
                "comparator_weighted_nll_sum"
            ].to_numpy(dtype=float)[sampled_indices].sum(axis=1)
            nll_differences = (
                full_weighted_nll / interval_weight
                - comparator_weighted_nll / interval_weight
            )
            for metric, values in [
                ("exact_delta_mse_difference_full_minus_comparator", mse_differences),
                ("weighted_interval_nll_difference_full_minus_comparator", nll_differences),
            ]:
                array = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "protocol": protocol,
                        "comparator": comparator_name,
                        "metric": metric,
                        "replicates": len(array),
                        "mean": float(array.mean()),
                        "ci_lower": float(np.quantile(array, 0.025)),
                        "ci_upper": float(np.quantile(array, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_PHASE_A_IMPLEMENTATION_OR_MODEL_FITTING":
        raise RuntimeError("Unexpected Phase A freeze status")
    paths = {
        name: (Path(spec["path"]) if Path(spec["path"]).is_absolute() else ROOT / spec["path"])
        for name, spec in config["inputs"].items()
    }
    for name, path in paths.items():
        if sha256(path) != config["inputs"][name]["sha256"]:
            raise RuntimeError(f"Input hash mismatch for {name}")
    labels = pd.read_csv(paths["label_table"], low_memory=False)
    constructs = pd.read_csv(paths["construct_manifest"])
    drugs = pd.read_csv(paths["drug_manifest"])
    explicit = np.load(paths["explicit_modification_features"])
    morgan = np.load(paths["drug_morgan_features"])
    wt = np.load(paths["esm2_wt_local_features"])
    modified = np.load(paths["esm2_modified_local_features"])
    delta_mean = np.load(paths["esm2_delta_mean_features"])
    delta_maxabs = np.load(paths["esm2_delta_maxabs_features"])
    rows = prepare_rows(labels, constructs, drugs)
    fold_assignments = create_outer_folds(rows, config)
    split_checks = {
        "paired_rows_3960": len(rows) == 3960,
        "fold_assignment_rows_11880": len(fold_assignments) == 3 * len(rows),
        "five_folds_each_protocol": all(
            set(fold_assignments.loc[fold_assignments["protocol"].eq(protocol), "outer_fold"])
            == set(range(5))
            for protocol in PROTOCOLS
        ),
        "feature_shapes": (
            explicit.shape == (56, 48)
            and morgan.shape == (72, 2048)
            and wt.shape == modified.shape == delta_mean.shape == delta_maxabs.shape == (56, 1280)
        ),
        "all_raw_features_finite": all(
            np.isfinite(matrix).all()
            for matrix in [explicit, morgan, wt, modified, delta_mean, delta_maxabs]
        ),
        "unpaired_gcn2_excluded": "gcn2_kindom2s808g" not in set(rows["protein"]),
        "both_capped_not_exact": not rows.loc[
            rows["category_code"].eq(3), "category_code"
        ].eq(0).any(),
    }
    validation = {
        "config_sha256": sha256(config_path),
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "split_checks": split_checks,
        "censoring_counts": rows["censoring_category"].value_counts().to_dict(),
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID" if all(split_checks.values()) else "FAIL", **validation}, ensure_ascii=False, indent=2))
        if not all(split_checks.values()):
            raise SystemExit(1)
        return

    output = args.output_dir.resolve()
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    split_audit_rows = []
    oof_parts = []
    fit_audit_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = int(config["model"]["phase_a_seed"])
    for protocol_index, protocol in enumerate(PROTOCOLS):
        protocol_assignment = fold_assignments[fold_assignments["protocol"].eq(protocol)].set_index(
            "pair_row_index"
        )["outer_fold"]
        for fold in range(5):
            test_index = rows.loc[
                rows["pair_row_index"].map(protocol_assignment).eq(fold)
            ].index.to_numpy(dtype=int)
            outer_train_index = rows.index.difference(test_index).to_numpy(dtype=int)
            outer_train = rows.loc[outer_train_index]
            test_rows = rows.loc[test_index].copy()
            inner_train_index, validation_index, validation_groups = inner_train_validation_indices(
                outer_train,
                protocol,
                fold,
                float(config["inner_validation"]["fraction_of_outer_train_groups"]),
            )
            train_rows = rows.loc[inner_train_index].copy()
            validation_rows = rows.loc[validation_index].copy()
            group_column = {
                "VARIANT_CONSTRUCT_COLD": "protein",
                "BASE_GENE_COLD": "base_wildtype_protein",
                "DRUG_SCAFFOLD_COLD": "drug_scaffold_group",
            }[protocol]
            overlap_train_test = set(outer_train[group_column]) & set(test_rows[group_column])
            overlap_train_validation = set(train_rows[group_column]) & set(validation_rows[group_column])
            if overlap_train_test or overlap_train_validation:
                raise RuntimeError(f"Grouped split overlap in {protocol}/fold{fold}")
            split_audit_rows.append(
                {
                    "protocol": protocol,
                    "outer_fold": fold,
                    "outer_train_rows": len(outer_train),
                    "inner_train_rows": len(train_rows),
                    "validation_rows": len(validation_rows),
                    "test_rows": len(test_rows),
                    "outer_train_groups": outer_train[group_column].nunique(),
                    "inner_train_groups": train_rows[group_column].nunique(),
                    "validation_groups": validation_rows[group_column].nunique(),
                    "test_groups": test_rows[group_column].nunique(),
                    "validation_group_values_json": json.dumps(validation_groups),
                    "train_test_group_overlap": len(overlap_train_test),
                    "train_validation_group_overlap": len(overlap_train_validation),
                    "test_labels_used_before_scoring": False,
                }
            )
            features = FoldFeatures(
                rows,
                outer_train_index,
                morgan,
                wt,
                modified,
                delta_mean,
                delta_maxabs,
                explicit,
            )
            neural_parameter_counts = []
            train_exact = train_rows[train_rows["category_code"].eq(0)]
            zero_sigma = max(0.02, float(train_exact["delta_exact"].std(ddof=0)))
            for variant_index, variant in enumerate(VARIANTS):
                fit_seed = base_seed + protocol_index * 10000 + fold * 100 + variant_index
                train_permutation = validation_permutation = test_permutation = None
                if variant == "SHUFFLED_MODIFICATION_IDENTITY_WITHIN_GENE_INTERVAL":
                    train_permutation = permutation_within_gene(train_rows, fit_seed + 1)
                    validation_permutation = permutation_within_gene(validation_rows, fit_seed + 2)
                    test_permutation = permutation_within_gene(test_rows, fit_seed + 3)
                if variant == "ZERO_TAU_WT_ECHO":
                    prediction = np.zeros(len(test_rows), dtype=float)
                    sigma = np.full(len(test_rows), zero_sigma, dtype=float)
                    fit_audit = {
                        "best_epoch": -1,
                        "epochs_run": 0,
                        "best_validation_loss": float("nan"),
                        "parameter_count": 0,
                        "train_eligible_rows": len(train_exact),
                        "validation_eligible_rows": int(validation_rows["category_code"].eq(0).sum()),
                    }
                elif variant == "RIDGE_POINT_EXACT":
                    train_drug, train_modification = features.row_features(train_exact, variant)
                    test_drug, test_modification = features.row_features(test_rows, variant)
                    ridge = RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0, 100.0]))
                    ridge.fit(
                        np.concatenate([train_drug, train_modification], axis=1),
                        train_exact["delta_exact"].to_numpy(dtype=float),
                    )
                    prediction = ridge.predict(
                        np.concatenate([test_drug, test_modification], axis=1)
                    )
                    residual = train_exact["delta_exact"].to_numpy(dtype=float) - ridge.predict(
                        np.concatenate([train_drug, train_modification], axis=1)
                    )
                    sigma = np.full(len(test_rows), max(0.02, float(np.std(residual))), dtype=float)
                    np.savez_compressed(
                        checkpoints / f"{protocol.lower()}_fold{fold}_{variant.lower()}_v1.npz",
                        coef=ridge.coef_, intercept=ridge.intercept_, alpha=ridge.alpha_, sigma=sigma[0]
                    )
                    fit_audit = {
                        "best_epoch": -1,
                        "epochs_run": 0,
                        "best_validation_loss": float("nan"),
                        "parameter_count": int(ridge.coef_.size + 1),
                        "train_eligible_rows": len(train_exact),
                        "validation_eligible_rows": int(validation_rows["category_code"].eq(0).sum()),
                    }
                else:
                    model, fit_audit = train_neural(
                        train_rows,
                        validation_rows,
                        features,
                        variant,
                        train_permutation,
                        validation_permutation,
                        config,
                        fit_seed,
                        device,
                    )
                    neural_parameter_counts.append(fit_audit["parameter_count"])
                    prediction, sigma = predict_neural(
                        model,
                        test_rows,
                        features,
                        variant,
                        test_permutation,
                        device,
                    )
                    torch.save(
                        {
                            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "variant": variant,
                            "protocol": protocol,
                            "fold": fold,
                            "fit_audit": fit_audit,
                        },
                        checkpoints / f"{protocol.lower()}_fold{fold}_{variant.lower()}_v1.pt",
                    )
                if not np.isfinite(prediction).all() or not np.isfinite(sigma).all():
                    raise RuntimeError(f"Non-finite predictions for {protocol}/{fold}/{variant}")
                scored = test_rows.copy()
                scored["protocol"] = protocol
                scored["outer_fold"] = fold
                scored["variant"] = variant
                scored["predicted_delta"] = prediction
                scored["predicted_sigma"] = sigma
                scored["interval_nll"] = pointwise_interval_nll(scored, prediction, sigma, config)
                oof_parts.append(scored)
                fit_audit_rows.append(
                    {
                        "protocol": protocol,
                        "outer_fold": fold,
                        "variant": variant,
                        "seed": fit_seed,
                        **fit_audit,
                    }
                )
                print(
                    json.dumps(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "variant": variant,
                            "test_rows": len(test_rows),
                            "best_epoch": fit_audit["best_epoch"],
                        }
                    ),
                    flush=True,
                )
            fold_neural_counts = {
                row["parameter_count"]
                for row in fit_audit_rows
                if row["protocol"] == protocol
                and row["outer_fold"] == fold
                and row["variant"] in NEURAL_VARIANTS
            }
            if len(fold_neural_counts) != 1:
                raise RuntimeError(f"Neural parameter-count mismatch: {fold_neural_counts}")
    oof = pd.concat(oof_parts, ignore_index=True)
    split_audit = pd.DataFrame(split_audit_rows)
    fit_audit = pd.DataFrame(fit_audit_rows)
    expected_oof = len(rows) * len(PROTOCOLS) * len(VARIANTS)
    integrity_checks = {
        **split_checks,
        "all_15_outer_splits_group_disjoint": bool(
            split_audit["train_test_group_overlap"].eq(0).all()
            and split_audit["train_validation_group_overlap"].eq(0).all()
        ),
        "all_test_labels_isolated_before_scoring": bool(
            ~split_audit["test_labels_used_before_scoring"].astype(bool).any()
        ),
        "oof_rows_exact": len(oof) == expected_oof,
        "one_prediction_per_pair_protocol_variant": not oof.duplicated(
            ["pair_row_index", "protocol", "variant"]
        ).any(),
        "all_predictions_finite": bool(
            np.isfinite(oof["predicted_delta"]).all()
            and np.isfinite(oof["predicted_sigma"]).all()
        ),
        "all_neural_variants_capacity_matched": fit_audit[
            fit_audit["variant"].isin(NEURAL_VARIANTS)
        ]["parameter_count"].nunique()
        == 1,
        "both_capped_interval_nll_is_nan": bool(
            oof.loc[oof["category_code"].eq(3), "interval_nll"].isna().all()
        ),
    }
    metric_rows = []
    for (protocol, variant), frame in oof.groupby(["protocol", "variant"], sort=False):
        metric_rows.append({"protocol": protocol, "variant": variant, **metrics_for(frame, config)})
    metrics = pd.DataFrame(metric_rows)
    bootstrap = bootstrap_comparisons(oof, config)

    def metric(protocol: str, variant: str, name: str) -> float:
        return float(
            metrics.loc[
                metrics["protocol"].eq(protocol) & metrics["variant"].eq(variant), name
            ].iloc[0]
        )

    def bootstrap_upper(protocol: str, comparator: str, metric_name: str) -> float:
        return float(
            bootstrap.loc[
                bootstrap["protocol"].eq(protocol)
                & bootstrap["comparator"].eq(comparator)
                & bootstrap["metric"].eq(metric_name),
                "ci_upper",
            ].iloc[0]
        )

    full = "FULL_PAIRED_INTERVAL"
    primary = "VARIANT_CONSTRUCT_COLD"
    gates = {
        "gate_1_full_mse_below_wt_echo_variant_cold": (
            metric(primary, full, "exact_delta_mse")
            < metric(primary, "ZERO_TAU_WT_ECHO", "exact_delta_mse")
        ),
        "gate_2_full_beats_shuffled_mse_bootstrap": (
            bootstrap_upper(
                primary,
                "SHUFFLED_MODIFICATION_IDENTITY_WITHIN_GENE_INTERVAL",
                "exact_delta_mse_difference_full_minus_comparator",
            )
            < 0
        ),
        "gate_3_full_beats_no_drug_mse_bootstrap": (
            bootstrap_upper(
                primary,
                "NO_DRUG_CONDITIONING_INTERVAL",
                "exact_delta_mse_difference_full_minus_comparator",
            )
            < 0
        ),
        "gate_4_interval_beats_point_nll_without_mse_degradation": (
            bootstrap_upper(
                primary,
                "FULL_POINT_EXACT_ONLY",
                "weighted_interval_nll_difference_full_minus_comparator",
            )
            < 0
            and metric(primary, full, "exact_delta_mse")
            - metric(primary, "FULL_POINT_EXACT_ONLY", "exact_delta_mse")
            <= 0.01
        ),
        "gate_5_strict_direction_bacc_above_055_variant_and_gene": (
            metric(primary, full, "strict_direction_balanced_accuracy") > 0.55
            and metric(
                "BASE_GENE_COLD", full, "strict_direction_balanced_accuracy"
            )
            > 0.55
        ),
        "gate_6_all_integrity_checks": all(integrity_checks.values()),
    }
    decision = (
        "OPEN_FIVE_SEED_CONFIRMATION_AND_OFFICIAL_FEW_SHOT_COMPARISON"
        if all(gates.values())
        else "PHASE_A_MECHANISM_NOT_SUPPORTED"
    )
    split_path = output / "PIC_DTA_PHASE_A_SPLIT_AUDIT_V1.csv"
    fit_path = output / "PIC_DTA_PHASE_A_FIT_AUDIT_V1.csv"
    oof_path = output / "PIC_DTA_PHASE_A_OOF_V1.csv.gz"
    metrics_path = output / "PIC_DTA_PHASE_A_METRICS_V1.csv"
    bootstrap_path = output / "PIC_DTA_PHASE_A_BOOTSTRAP_V1.csv"
    split_audit.to_csv(split_path, index=False)
    fit_audit.to_csv(fit_path, index=False)
    oof.to_csv(oof_path, index=False, compression="gzip")
    metrics.to_csv(metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    output_paths = [split_path, fit_path, oof_path, metrics_path, bootstrap_path]
    summary = {
        "schema_version": "PIC_DTA_PHASE_A_SUMMARY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(integrity_checks.values()) else "FAIL",
        "decision": decision,
        "claim_boundary": (
            "PIC-DTA remains a preregistered candidate and is not a verified core algorithm innovation or SOTA result."
            if decision != "OPEN_FIVE_SEED_CONFIRMATION_AND_OFFICIAL_FEW_SHOT_COMPARISON"
            else "Phase A mechanism gates passed, authorizing confirmation only; no SOTA or validated-innovation claim before multiseed and external validation."
        ),
        "config": relative(config_path),
        "config_sha256": sha256(config_path),
        "device": str(device),
        "counts": {
            "paired_rows": len(rows),
            "protocols": len(PROTOCOLS),
            "variants": len(VARIANTS),
            "outer_fits_neural": int(fit_audit["variant"].isin(NEURAL_VARIANTS).sum()),
            "oof_rows": len(oof),
            "bootstrap_replicates_per_comparison": int(
                config["uncertainty"]["cluster_bootstrap_replicates"]
            ),
        },
        "integrity_checks": integrity_checks,
        "signal_gates": gates,
        "primary_metrics": metrics[metrics["variant"].eq(full)].to_dict("records"),
        "all_metrics": metrics.to_dict("records"),
        "bootstrap_comparisons": bootstrap.to_dict("records"),
        "inputs": {relative(path): sha256(path) for path in paths.values()},
        "script_sha256": sha256(Path(__file__)),
        "outputs": {relative(path): sha256(path) for path in output_paths},
        "interpretation_if_failed": (
            "A failed gate means the true modification identity, drug conditioning, or paired interval information did not demonstrate the frozen mechanism. "
            "Any apparent gain over WT echo alone is insufficient."
        ),
    }
    summary_path = output / "PIC_DTA_PHASE_A_SUMMARY_V1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "decision": decision,
        "signal_gates": gates,
        "primary_metrics": summary["primary_metrics"],
        "summary": relative(summary_path),
        "summary_sha256": sha256(summary_path),
    }, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
