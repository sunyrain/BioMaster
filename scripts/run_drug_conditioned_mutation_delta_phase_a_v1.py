#!/usr/bin/env python3
"""Run the pre-frozen Phase-A drug-conditioned mutation-delta screen.

The outer-fold V10 mutation model is reproduced and frozen first.  A small
adapter may then add a bounded logit increment only for constructs whose
WT-to-mutant ESM2 reconstruction is available.  Missing reconstructions use
the exact base prediction.  Four capacity-matched adapter inputs isolate WT
context, mutation-delta identity, and explicit drug conditioning.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/biomaster_drug_conditioned_mutation_delta_phase_a_freeze_20260814.json"
OUT = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/drug_conditioned_mutation_delta_phase_a_v1"
CHECKPOINTS = OUT / "checkpoints"

# These must be set before importing the frozen V10 implementation.
os.environ["BIOMASTER_MUTATION_VERSION"] = "V10"
os.environ["BIOMASTER_MUTATION_PRIOR"] = str(
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/leakage_safe_ranker_v10/"
    "unseeded_target_lanes/OLD_DRUG_TARGET_BRANCH_ROUTING_276480_V10.csv.gz"
)
os.environ["BIOMASTER_MUTATION_SCORE_COLUMN"] = "branch_primary_score_v10"
os.environ["BIOMASTER_MUTATION_RANK_COLUMN"] = "branch_primary_rank_within_drug_v10"
sys.path.insert(0, str(ROOT / "scripts"))
import train_v8_mutation_aware_multitask as legacy  # noqa: E402


DEVICE = legacy.DEVICE
VARIANTS = [
    "TOKEN_BASE",
    "WT_LOCAL_ONLY",
    "SHUFFLED_DELTA_WITHIN_GENE",
    "DELTA_NO_DRUG_CONDITIONING",
    "FULL_DRUG_CONDITIONED_COUNTERFACTUAL_DELTA",
]
ADAPTER_VARIANTS = VARIANTS[1:]
PROTOCOLS = {
    "DRUG_COLD": {
        "group": "kirhub_compound",
        "seed_offset": 0,
        "active_column": "mutation_aware_active_probability_oof_v10",
        "sens_column": "mutation_aware_sensitization_probability_oof_v10",
        "residual_column": "mutation_aware_residual_pct_oof_v10",
    },
    "VARIANT_CONSTRUCT_COLD": {
        "group": "variant_construct",
        "seed_offset": 100,
        "active_column": "mutation_aware_active_probability_variant_cold_oof_v10",
        "sens_column": "mutation_aware_sensitization_probability_variant_cold_oof_v10",
        "residual_column": "mutation_aware_residual_pct_variant_cold_oof_v10",
    },
    "GENE_COLD": {
        "group": "gene_symbol",
        "seed_offset": 200,
        "active_column": "mutation_aware_active_probability_gene_cold_oof_v10",
        "sens_column": "mutation_aware_sensitization_probability_gene_cold_oof_v10",
        "residual_column": "mutation_aware_residual_pct_gene_cold_oof_v10",
    },
}
FEATURE_KEYS = [
    "wt_center_mean",
    "center_delta_mean",
    "center_delta_signed_maxabs",
    "window_mean_delta",
]
PCA_COMPONENTS = 32
ADAPTER_SEED = 20260814
ADAPTER_EPOCHS = 60
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260814
BASE_REPRODUCTION_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def safe_logit(values: torch.Tensor) -> torch.Tensor:
    return torch.logit(values.clamp(1e-6, 1.0 - 1e-6))


def verify_config_and_inputs(config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "FROZEN_BEFORE_MODEL_IMPLEMENTATION_OR_LABEL_JOIN":
        raise RuntimeError("Phase-A configuration is not in the frozen state")
    observed: dict[str, str] = {}
    mismatches = []
    for name, record in config["immutable_inputs"].items():
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        observed[name] = digest
        if digest != record["sha256"]:
            mismatches.append({"name": name, "expected": record["sha256"], "observed": digest})
    if mismatches:
        raise RuntimeError(f"Immutable input hash mismatch: {mismatches}")
    return observed


def prepare_panel(config: dict[str, Any]) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]
]:
    paths = {name: ROOT / record["path"] for name, record in config["immutable_inputs"].items()}
    panel = pd.read_csv(paths["v10_panel"], low_memory=False)
    if len(panel) != 19639:
        raise RuntimeError(f"Expected 19639 V10 rows, observed {len(panel)}")
    required_labels = {"active_label", "sensitization_label", "residual_scaled"}
    if not required_labels.issubset(panel.columns):
        raise RuntimeError(f"Missing labels: {required_labels - set(panel.columns)}")
    if panel[legacy.BASELINE_FEATURES].isna().any().any():
        raise RuntimeError("V10 baseline features contain missing values")

    pair_drugs = pd.read_csv(
        paths["pair_routing"], usecols=["ligand_inchikey", "ligand_smiles"]
    ).drop_duplicates("ligand_inchikey")
    drugs = (
        panel[["ligand_inchikey", "kirhub_compound"]]
        .drop_duplicates()
        .sort_values("ligand_inchikey")
        .reset_index(drop=True)
        .merge(pair_drugs, on="ligand_inchikey", how="left", validate="one_to_one")
    )
    targets = (
        panel[["target_chembl_id", "gene_symbol"]]
        .drop_duplicates()
        .sort_values("target_chembl_id")
        .reset_index(drop=True)
    )
    if len(drugs) != 76 or len(targets) != 35:
        raise RuntimeError(f"Unexpected drug/target count: {len(drugs)}/{len(targets)}")
    drug_index = {key: index for index, key in enumerate(drugs["ligand_inchikey"])}
    target_index = {key: index for index, key in enumerate(targets["target_chembl_id"])}
    panel["_drug_index"] = panel["ligand_inchikey"].map(drug_index)
    panel["_target_index"] = panel["target_chembl_id"].map(target_index)

    morgan_map = legacy.load_h5(paths["morgan_features"])
    cache_smiles = drugs["ligand_smiles"].map(
        lambda value: value if value in morgan_map else value.replace("/", "|")
    )
    if not cache_smiles.isin(morgan_map).all():
        raise RuntimeError("Mutation drugs are missing from the frozen Morgan cache")
    drug_vectors = np.stack([morgan_map[smiles] for smiles in cache_smiles]).astype(np.float32)

    sequences = (
        pd.read_csv(
            paths["target_sequences"],
            usecols=["target_chembl_id", "sequence", "sequence_length"],
        )
        .drop_duplicates("target_chembl_id")
        .set_index("target_chembl_id")
    )
    protein_map = legacy.load_h5(paths["protbert_main"])
    protein_map.update(legacy.load_h5(paths["protbert_recovered"]))
    target_vectors = np.stack(
        [protein_map[str(sequences.loc[target, "sequence"])] for target in targets["target_chembl_id"]]
    ).astype(np.float32)
    target_vectors /= np.maximum(np.linalg.norm(target_vectors, axis=1, keepdims=True), 1e-12)
    lengths = sequences["sequence_length"].astype(int).to_dict()
    tokens = [
        legacy.mutation_tokens(row.variant_construct, lengths[row.target_chembl_id])
        for row in panel.itertuples()
    ]
    panel["_from_aa"] = [item[0] for item in tokens]
    panel["_to_aa"] = [item[1] for item in tokens]
    panel["_positions"] = [item[2] for item in tokens]
    panel["_mutation_flags"] = [item[3] for item in tokens]

    manifest = pd.read_csv(paths["construct_manifest"])
    if len(manifest) != 299 or not manifest["variant_construct"].is_unique:
        raise RuntimeError("Construct manifest is not the frozen 299-row one-to-one table")
    mapping = manifest.set_index("variant_construct")["construct_feature_index"]
    availability = manifest.set_index("variant_construct")["reconstruction_available"]
    panel["_construct_index"] = panel["variant_construct"].map(mapping)
    panel["_feature_available"] = panel["variant_construct"].map(availability).astype(bool)
    if panel["_construct_index"].isna().any() or panel["variant_construct"].nunique() != 299:
        raise RuntimeError("Panel-to-construct-manifest mapping is incomplete")
    panel["_construct_index"] = panel["_construct_index"].astype(int)

    feature_arrays = {name: np.load(paths[name]).astype(np.float32) for name in FEATURE_KEYS}
    if any(array.shape != (299, 1280) for array in feature_arrays.values()):
        raise RuntimeError({name: array.shape for name, array in feature_arrays.items()})
    manifest_available = manifest["reconstruction_available"].astype(bool).to_numpy()
    if int(manifest_available.sum()) != 240:
        raise RuntimeError("Expected exactly 240 reconstructable constructs")
    for name, array in feature_arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite ESM2 values in {name}")
        if not np.array_equal(array[~manifest_available], np.zeros_like(array[~manifest_available])):
            raise RuntimeError(f"Unavailable vectors are not exact zero in {name}")
    return panel, drug_vectors, target_vectors, manifest, feature_arrays


def extract_frozen_latents(
    model: legacy.MutationAwareModel,
    frame: pd.DataFrame,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int = 8192,
) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {
        "active_logit": [],
        "sens_logit": [],
        "residual_logit": [],
        "active_probability": [],
        "sens_probability": [],
        "residual_probability": [],
        "drug_latent": [],
        "target_latent": [],
        "mutation_latent": [],
    }
    model.eval()
    with torch.inference_mode():
        all_drugs = model.drug_encoder(model.drug_vectors)
        all_targets = model.target_encoder(model.target_vectors)
        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start : start + batch_size]
            bundle = legacy.tensor_bundle(chunk, mean, scale)
            drug = all_drugs[bundle["drug_index"]]
            target = all_targets[bundle["target_index"]]
            mutation_tokens = torch.cat(
                [
                    model.from_aa(bundle["from_aa"]),
                    model.to_aa(bundle["to_aa"]),
                    bundle["positions"].unsqueeze(-1),
                ],
                dim=-1,
            ).flatten(1)
            mutation = model.mutation_encoder(torch.cat([mutation_tokens, bundle["flags"]], dim=1))
            base = model.baseline_encoder(bundle["baseline"])
            gate = model.gate(torch.cat([drug, target, mutation, base], dim=1))
            mutation_conditioned_target = gate * mutation + (1.0 - gate) * target
            hidden = model.fusion(
                torch.cat(
                    [
                        drug,
                        target,
                        mutation,
                        mutation_conditioned_target,
                        drug * target,
                        target * mutation,
                        base,
                    ],
                    dim=1,
                )
            )
            active_logit = model.active_head(hidden).squeeze(1)
            sens_logit = model.sensitization_head(hidden).squeeze(1)
            residual_probability = torch.sigmoid(model.residual_head(hidden).squeeze(1))
            values = {
                "active_logit": active_logit,
                "sens_logit": sens_logit,
                "residual_logit": safe_logit(residual_probability),
                "active_probability": torch.sigmoid(active_logit),
                "sens_probability": torch.sigmoid(sens_logit),
                "residual_probability": residual_probability,
                "drug_latent": drug,
                "target_latent": target,
                "mutation_latent": mutation,
            }
            for name, value in values.items():
                output[name].append(value.detach().cpu().numpy().astype(np.float32))
    return {name: np.concatenate(chunks, axis=0) for name, chunks in output.items()}


def fit_outer_train_projection(
    train_constructs: np.ndarray,
    manifest: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    random_state: int,
) -> dict[str, np.ndarray]:
    available = manifest["reconstruction_available"].astype(bool).to_numpy()
    fit_indices = np.asarray(
        sorted(set(int(x) for x in train_constructs if available[int(x)])), dtype=int
    )
    if len(fit_indices) <= PCA_COMPONENTS:
        raise RuntimeError(f"Only {len(fit_indices)} available outer-training constructs for PCA")
    projections: dict[str, np.ndarray] = {}
    for source_index, name in enumerate(FEATURE_KEYS):
        pca = PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="randomized",
            iterated_power=7,
            random_state=random_state + source_index,
        )
        pca.fit(arrays[name][fit_indices])
        transformed = pca.transform(arrays[name]).astype(np.float32)
        component_mean = transformed[fit_indices].mean(axis=0)
        component_scale = transformed[fit_indices].std(axis=0)
        component_scale[component_scale < 1e-6] = 1.0
        transformed = ((transformed - component_mean) / component_scale).astype(np.float32)
        transformed[~available] = 0.0
        projections[name] = transformed
    return projections


def shuffled_delta_donors(
    train_constructs: np.ndarray, manifest: pd.DataFrame
) -> tuple[np.ndarray, dict[str, int]]:
    available = manifest["reconstruction_available"].astype(bool).to_numpy()
    train_set = {int(x) for x in train_constructs if available[int(x)]}
    donor = np.full(len(manifest), -1, dtype=int)
    donor_counts: dict[str, int] = {}
    for gene, gene_frame in manifest.groupby("gene_symbol", sort=True):
        pool = sorted(
            [
                int(row.construct_feature_index)
                for row in gene_frame.itertuples()
                if int(row.construct_feature_index) in train_set
            ],
            key=lambda index: str(manifest.iloc[index]["variant_construct"]),
        )
        if not pool:
            donor_counts[str(gene)] = 0
            continue
        donor_counts[str(gene)] = len(pool)
        pool_names = [str(manifest.iloc[index]["variant_construct"]) for index in pool]
        for row in gene_frame.itertuples():
            recipient = int(row.construct_feature_index)
            if not available[recipient]:
                continue
            recipient_name = str(row.variant_construct)
            if recipient in pool and len(pool) == 1:
                continue
            if recipient in pool:
                position = pool.index(recipient)
                donor[recipient] = pool[(position + 1) % len(pool)]
                continue
            # Test recipients use the next lexicographic outer-training donor.
            later = [index for index, name in zip(pool, pool_names) if name > recipient_name]
            donor[recipient] = later[0] if later else pool[0]
    if np.any(donor == np.arange(len(donor))):
        raise RuntimeError("Shuffled control contains a self donor")
    return donor, donor_counts


class CounterfactualDeltaAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.effect_encoder = torch.nn.Sequential(
            torch.nn.Linear(32 + 96 + 64 + 64 + 1, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
        )
        self.unconditioned_heads = torch.nn.ModuleList([torch.nn.Linear(64, 1) for _ in range(3)])
        self.conditioned_heads = torch.nn.ModuleList([torch.nn.Linear(64, 1) for _ in range(3)])

    def forward(
        self,
        wt_local: torch.Tensor,
        delta: torch.Tensor,
        drug: torch.Tensor,
        target: torch.Tensor,
        mutation: torch.Tensor,
        availability: torch.Tensor,
        enable_drug_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        effect = self.effect_encoder(
            torch.cat([wt_local, delta, target, mutation, availability.unsqueeze(1)], dim=1)
        )
        conditioned = effect * drug if enable_drug_conditioning else torch.zeros_like(effect)
        increments = []
        for unconditioned, conditional in zip(self.unconditioned_heads, self.conditioned_heads):
            raw = unconditioned(effect).squeeze(1) + conditional(conditioned).squeeze(1)
            increments.append(torch.tanh(raw))
        return increments[0], increments[1], increments[2]


def adapter_parameter_count() -> int:
    model = CounterfactualDeltaAdapter()
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_adapter_tensors(
    frame: pd.DataFrame,
    latents: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    variant: str,
    donors: np.ndarray,
) -> dict[str, torch.Tensor]:
    construct_index = frame["_construct_index"].to_numpy(dtype=int)
    availability = frame["_feature_available"].to_numpy(dtype=bool)
    wt = projections["wt_center_mean"][construct_index]
    actual_delta = np.concatenate(
        [
            projections["center_delta_mean"][construct_index],
            projections["center_delta_signed_maxabs"][construct_index],
            projections["window_mean_delta"][construct_index],
        ],
        axis=1,
    ).astype(np.float32)
    if variant == "WT_LOCAL_ONLY":
        delta = np.zeros_like(actual_delta)
    elif variant == "SHUFFLED_DELTA_WITHIN_GENE":
        donor_index = donors[construct_index]
        delta = np.zeros_like(actual_delta)
        valid_donor = donor_index >= 0
        if valid_donor.any():
            delta_sources = np.concatenate(
                [
                    projections["center_delta_mean"],
                    projections["center_delta_signed_maxabs"],
                    projections["window_mean_delta"],
                ],
                axis=1,
            ).astype(np.float32)
            delta[valid_donor] = delta_sources[donor_index[valid_donor]]
    else:
        delta = actual_delta
    return {
        "wt_local": torch.as_tensor(wt, dtype=torch.float32, device=DEVICE),
        "delta": torch.as_tensor(delta, dtype=torch.float32, device=DEVICE),
        "drug": torch.as_tensor(latents["drug_latent"], dtype=torch.float32, device=DEVICE),
        "target": torch.as_tensor(latents["target_latent"], dtype=torch.float32, device=DEVICE),
        "mutation": torch.as_tensor(latents["mutation_latent"], dtype=torch.float32, device=DEVICE),
        "availability": torch.as_tensor(availability, dtype=torch.float32, device=DEVICE),
        "base_active_logit": torch.as_tensor(
            latents["active_logit"], dtype=torch.float32, device=DEVICE
        ),
        "base_sens_logit": torch.as_tensor(
            latents["sens_logit"], dtype=torch.float32, device=DEVICE
        ),
        "base_residual_logit": torch.as_tensor(
            latents["residual_logit"], dtype=torch.float32, device=DEVICE
        ),
    }


def fit_adapter(
    variant: str,
    train: pd.DataFrame,
    train_latents: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    donors: np.ndarray,
    seed: int,
) -> CounterfactualDeltaAdapter:
    legacy.set_seeds(seed)
    model = CounterfactualDeltaAdapter().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAPTER_EPOCHS)
    tensors = make_adapter_tensors(train, train_latents, projections, variant, donors)
    active = torch.as_tensor(train["active_label"].to_numpy(dtype=np.float32), device=DEVICE)
    sensitization = torch.as_tensor(
        train["sensitization_label"].to_numpy(dtype=np.float32), device=DEVICE
    )
    residual = torch.as_tensor(train["residual_scaled"].to_numpy(dtype=np.float32), device=DEVICE)
    reset_train = train.reset_index(drop=True)
    positive_index, negative_index = legacy.make_pair_indices(reset_train)
    positive_index = torch.as_tensor(positive_index, dtype=torch.long, device=DEVICE)
    negative_index = torch.as_tensor(negative_index, dtype=torch.long, device=DEVICE)
    active_weight = torch.tensor(
        [(len(active) - active.sum()).item() / max(active.sum().item(), 1)], device=DEVICE
    )
    sens_weight = torch.tensor(
        [
            (len(sensitization) - sensitization.sum()).item()
            / max(sensitization.sum().item(), 1)
        ],
        device=DEVICE,
    )
    active_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=active_weight)
    sens_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=sens_weight)
    enable_drug_conditioning = variant != "DELTA_NO_DRUG_CONDITIONING"
    mask = tensors["availability"]
    for _ in range(ADAPTER_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        active_increment, sens_increment, residual_increment = model(
            tensors["wt_local"],
            tensors["delta"],
            tensors["drug"],
            tensors["target"],
            tensors["mutation"],
            tensors["availability"],
            enable_drug_conditioning,
        )
        active_logit = tensors["base_active_logit"] + mask * active_increment
        sens_logit = tensors["base_sens_logit"] + mask * sens_increment
        residual_prediction = torch.sigmoid(
            tensors["base_residual_logit"] + mask * residual_increment
        )
        active_loss = active_loss_fn(active_logit, active)
        sensitization_loss = sens_loss_fn(sens_logit, sensitization)
        regression_loss = torch.nn.functional.smooth_l1_loss(
            residual_prediction, residual, beta=0.10
        )
        ranking_loss = torch.nn.functional.softplus(
            -(active_logit[positive_index] - active_logit[negative_index])
        ).mean()
        loss = (
            active_loss
            + 0.30 * sensitization_loss
            + 0.25 * regression_loss
            + 0.30 * ranking_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model


def predict_adapter(
    model: CounterfactualDeltaAdapter,
    variant: str,
    frame: pd.DataFrame,
    latents: dict[str, np.ndarray],
    projections: dict[str, np.ndarray],
    donors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = make_adapter_tensors(frame, latents, projections, variant, donors)
    enable_drug_conditioning = variant != "DELTA_NO_DRUG_CONDITIONING"
    with torch.inference_mode():
        active_increment, sens_increment, residual_increment = model(
            tensors["wt_local"],
            tensors["delta"],
            tensors["drug"],
            tensors["target"],
            tensors["mutation"],
            tensors["availability"],
            enable_drug_conditioning,
        )
        available = frame["_feature_available"].to_numpy(dtype=bool)
        active_candidate = torch.sigmoid(
            tensors["base_active_logit"] + active_increment
        ).cpu().numpy().astype(np.float32)
        sens_candidate = torch.sigmoid(
            tensors["base_sens_logit"] + sens_increment
        ).cpu().numpy().astype(np.float32)
        residual_candidate = torch.sigmoid(
            tensors["base_residual_logit"] + residual_increment
        ).cpu().numpy().astype(np.float32)
    active = np.where(available, active_candidate, latents["active_probability"]).astype(np.float32)
    sens = np.where(available, sens_candidate, latents["sens_probability"]).astype(np.float32)
    residual = np.where(
        available, residual_candidate, latents["residual_probability"]
    ).astype(np.float32)
    return active, sens, residual


def checkpoint_path(protocol: str, fold: int) -> Path:
    return CHECKPOINTS / f"{protocol.lower()}_fold{fold}_v1.npz"


def save_fold_checkpoint(
    path: Path,
    protocol: str,
    fold: int,
    test_index: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    reproduction_errors: dict[str, float],
    fallback_errors: dict[str, float],
    donor_coverage: dict[str, int],
    config_hash: str,
    script_hash: str,
) -> None:
    payload: dict[str, Any] = {
        "protocol": np.asarray(protocol),
        "fold": np.asarray(fold),
        "test_index": test_index.astype(np.int64),
        "config_sha256": np.asarray(config_hash),
        "script_sha256": np.asarray(script_hash),
        "reproduction_errors_json": np.asarray(json.dumps(reproduction_errors, sort_keys=True)),
        "fallback_errors_json": np.asarray(json.dumps(fallback_errors, sort_keys=True)),
        "donor_coverage_json": np.asarray(json.dumps(donor_coverage, sort_keys=True)),
        "adapter_parameter_count": np.asarray(adapter_parameter_count()),
    }
    for variant, (active, sens, residual) in predictions.items():
        payload[f"{variant}__active"] = active.astype(np.float32)
        payload[f"{variant}__sens"] = sens.astype(np.float32)
        payload[f"{variant}__residual"] = residual.astype(np.float32)
    np.savez_compressed(path, **payload)


def load_valid_checkpoint(
    path: Path, config_hash: str, script_hash: str
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["config_sha256"]) != config_hash or str(data["script_sha256"]) != script_hash:
            return None
        return {name: data[name] for name in data.files}
    except Exception:
        return None


def run_oof(
    panel: pd.DataFrame,
    drug_vectors: np.ndarray,
    target_vectors: np.ndarray,
    manifest: pd.DataFrame,
    feature_arrays: dict[str, np.ndarray],
    config_hash: str,
    script_hash: str,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    all_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    audit: dict[str, Any] = {"base_reproduction": {}, "fallback": {}, "parameter_counts": []}
    split_audit: list[dict[str, Any]] = []
    n = len(panel)
    for protocol, specification in PROTOCOLS.items():
        log(f"Starting {protocol}")
        splitter = GroupKFold(n_splits=5)
        splits = list(
            splitter.split(panel, panel["active_label"], panel[specification["group"]])
        )
        protocol_predictions = {
            variant: {
                "active": np.full(n, np.nan, dtype=np.float32),
                "sens": np.full(n, np.nan, dtype=np.float32),
                "residual": np.full(n, np.nan, dtype=np.float32),
                "fold": np.zeros(n, dtype=np.int16),
            }
            for variant in VARIANTS
        }
        protocol_reproduction = []
        protocol_fallback = []
        for fold, (train_index, test_index) in enumerate(splits, start=1):
            train_groups = set(panel.iloc[train_index][specification["group"]])
            test_groups = set(panel.iloc[test_index][specification["group"]])
            overlap = train_groups & test_groups
            if overlap:
                raise RuntimeError(f"{protocol} fold {fold} group leakage: {list(overlap)[:5]}")
            record = {
                "protocol": protocol,
                "fold": fold,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "train_groups": int(len(train_groups)),
                "test_groups": int(len(test_groups)),
                "group_overlap": 0,
            }
            split_audit.append(record)
            path = checkpoint_path(protocol, fold)
            cached = load_valid_checkpoint(path, config_hash, script_hash)
            if cached is not None:
                log(f"Reusing verified checkpoint {path.name}")
                checkpoint_test = cached["test_index"].astype(int)
                if not np.array_equal(checkpoint_test, test_index):
                    raise RuntimeError(f"Checkpoint split mismatch: {path}")
                for variant in VARIANTS:
                    for task in ("active", "sens", "residual"):
                        protocol_predictions[variant][task][test_index] = cached[
                            f"{variant}__{task}"
                        ]
                    protocol_predictions[variant]["fold"][test_index] = fold
                protocol_reproduction.append(
                    json.loads(str(cached["reproduction_errors_json"]))
                )
                protocol_fallback.append(json.loads(str(cached["fallback_errors_json"])))
                audit["parameter_counts"].append(int(cached["adapter_parameter_count"]))
                continue

            train = panel.iloc[train_index].copy().reset_index(drop=True)
            test = panel.iloc[test_index].copy().reset_index(drop=True)
            log(f"{protocol} fold {fold}: fitting exact V10 base on {len(train)} rows")
            base_model, baseline_mean, baseline_scale = legacy.fit_model(
                train,
                drug_vectors,
                target_vectors,
                legacy.SEED + int(specification["seed_offset"]) + fold,
            )
            train_latents = extract_frozen_latents(
                base_model, train, baseline_mean, baseline_scale
            )
            test_latents = extract_frozen_latents(base_model, test, baseline_mean, baseline_scale)
            reproduction_errors = {
                "active_max_abs": float(
                    np.max(
                        np.abs(
                            test_latents["active_probability"]
                            - test[specification["active_column"]].to_numpy(dtype=np.float32)
                        )
                    )
                ),
                "sens_max_abs": float(
                    np.max(
                        np.abs(
                            test_latents["sens_probability"]
                            - test[specification["sens_column"]].to_numpy(dtype=np.float32)
                        )
                    )
                ),
                "residual_pct_max_abs": float(
                    np.max(
                        np.abs(
                            100.0 * test_latents["residual_probability"]
                            - test[specification["residual_column"]].to_numpy(dtype=np.float32)
                        )
                    )
                ),
            }
            if max(reproduction_errors.values()) > BASE_REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"Frozen V10 reproduction failed for {protocol} fold {fold}: {reproduction_errors}"
                )
            protocol_reproduction.append(reproduction_errors)

            train_constructs = train["_construct_index"].unique()
            projections = fit_outer_train_projection(
                train_constructs,
                manifest,
                feature_arrays,
                ADAPTER_SEED + int(specification["seed_offset"]) + fold,
            )
            donors, donor_coverage = shuffled_delta_donors(train_constructs, manifest)
            fold_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
                "TOKEN_BASE": (
                    test_latents["active_probability"],
                    test_latents["sens_probability"],
                    test_latents["residual_probability"],
                )
            }
            fallback_errors: dict[str, float] = {}
            for variant in ADAPTER_VARIANTS:
                log(f"{protocol} fold {fold}: fitting {variant}")
                adapter = fit_adapter(
                    variant,
                    train,
                    train_latents,
                    projections,
                    donors,
                    ADAPTER_SEED + int(specification["seed_offset"]) + fold,
                )
                active, sens, residual = predict_adapter(
                    adapter, variant, test, test_latents, projections, donors
                )
                fold_predictions[variant] = (active, sens, residual)
                unavailable = ~test["_feature_available"].to_numpy(dtype=bool)
                if unavailable.any():
                    error = max(
                        float(
                            np.max(
                                np.abs(active[unavailable] - test_latents["active_probability"][unavailable])
                            )
                        ),
                        float(
                            np.max(
                                np.abs(sens[unavailable] - test_latents["sens_probability"][unavailable])
                            )
                        ),
                        float(
                            np.max(
                                np.abs(
                                    residual[unavailable]
                                    - test_latents["residual_probability"][unavailable]
                                )
                            )
                        ),
                    )
                else:
                    error = 0.0
                fallback_errors[variant] = error
                del adapter
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            protocol_fallback.append(fallback_errors)
            count = adapter_parameter_count()
            audit["parameter_counts"].append(count)
            save_fold_checkpoint(
                path,
                protocol,
                fold,
                test_index,
                fold_predictions,
                reproduction_errors,
                fallback_errors,
                donor_coverage,
                config_hash,
                script_hash,
            )
            for variant, (active, sens, residual) in fold_predictions.items():
                protocol_predictions[variant]["active"][test_index] = active
                protocol_predictions[variant]["sens"][test_index] = sens
                protocol_predictions[variant]["residual"][test_index] = residual
                protocol_predictions[variant]["fold"][test_index] = fold
            del base_model, train_latents, test_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"Completed {protocol} fold {fold}")
        for variant in VARIANTS:
            if not all(
                np.isfinite(protocol_predictions[variant][task]).all()
                for task in ("active", "sens", "residual")
            ):
                raise RuntimeError(f"Incomplete OOF predictions for {protocol}/{variant}")
            if set(protocol_predictions[variant]["fold"]) != {1, 2, 3, 4, 5}:
                raise RuntimeError(f"Incomplete folds for {protocol}/{variant}")
        audit["base_reproduction"][protocol] = protocol_reproduction
        audit["fallback"][protocol] = protocol_fallback
        all_predictions[protocol] = protocol_predictions
    return all_predictions, audit, split_audit


def binary_metrics(
    y: np.ndarray, score: np.ndarray, drug_groups: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "micro_auroc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None,
        "micro_auprc": float(average_precision_score(y, score))
        if np.unique(y).size == 2
        else None,
    }
    aucs, aps, recalls = [], [], []
    for group in pd.unique(drug_groups):
        mask = drug_groups == group
        yy, ss = y[mask], score[mask]
        if np.unique(yy).size == 2:
            aucs.append(roc_auc_score(yy, ss))
            aps.append(average_precision_score(yy, ss))
        positives = int(yy.sum())
        if positives:
            order = np.argsort(-ss, kind="stable")
            recalls.append(float(yy[order[: min(20, len(order))]].sum() / positives))
    output["macro_drug_auroc"] = float(np.mean(aucs)) if aucs else None
    output["macro_drug_auprc"] = float(np.mean(aps)) if aps else None
    output["macro_recall_at_20"] = float(np.mean(recalls)) if recalls else None
    output["drugs_with_two_classes"] = int(len(aucs))
    return output


def build_metrics(
    panel: pd.DataFrame,
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
) -> pd.DataFrame:
    rows = []
    slices = {
        "ALL": np.ones(len(panel), dtype=bool),
        "ESM2_AVAILABLE": panel["_feature_available"].to_numpy(dtype=bool),
        "ESM2_UNAVAILABLE": ~panel["_feature_available"].to_numpy(dtype=bool),
    }
    labels = {
        "ACTIVE": panel["active_label"].to_numpy(dtype=int),
        "STRICT_M2_SENSITIZATION": panel["sensitization_label"].to_numpy(dtype=int),
    }
    drug_groups = panel["kirhub_compound"].to_numpy()
    for protocol in PROTOCOLS:
        for variant in VARIANTS:
            for task, label in labels.items():
                score = predictions[protocol][variant][
                    "active" if task == "ACTIVE" else "sens"
                ]
                for slice_name, mask in slices.items():
                    if np.unique(label[mask]).size < 2:
                        continue
                    rows.append(
                        {
                            "protocol": protocol,
                            "variant": variant,
                            "task": task,
                            "slice": slice_name,
                            **binary_metrics(label[mask], score[mask], drug_groups[mask]),
                        }
                    )
            residual_pct = 100.0 * predictions[protocol][variant]["residual"]
            rows.append(
                {
                    "protocol": protocol,
                    "variant": variant,
                    "task": "RESIDUAL_ACTIVITY",
                    "slice": "ALL",
                    "rows": len(panel),
                    "positives": None,
                    "prevalence": None,
                    "micro_auroc": None,
                    "micro_auprc": None,
                    "macro_drug_auroc": None,
                    "macro_drug_auprc": None,
                    "macro_recall_at_20": None,
                    "drugs_with_two_classes": None,
                    "mae_pct": float(
                        mean_absolute_error(
                            panel["variant_residual_activity_pct_1uM"], residual_pct
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    y: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.asarray(sorted(pd.unique(groups)), dtype=object)
    lookup = {group: index for index, group in enumerate(unique_groups)}
    codes = pd.Series(groups).map(lookup).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    deltas = []
    attempts = 0
    while len(deltas) < BOOTSTRAP_REPLICATES:
        attempts += 1
        if attempts > BOOTSTRAP_REPLICATES * 20:
            raise RuntimeError("Unable to obtain 1000 valid paired cluster bootstrap replicates")
        counts = rng.multinomial(
            len(unique_groups), np.full(len(unique_groups), 1.0 / len(unique_groups))
        )
        weights = counts[codes].astype(float)
        selected = weights > 0
        if np.unique(y[selected]).size < 2:
            continue
        candidate_ap = average_precision_score(y, candidate, sample_weight=weights)
        comparator_ap = average_precision_score(y, comparator, sample_weight=weights)
        deltas.append(float(candidate_ap - comparator_ap))
    values = np.asarray(deltas)
    point = float(
        average_precision_score(y, candidate) - average_precision_score(y, comparator)
    )
    return {
        "point_delta_auprc": point,
        "bootstrap_mean_delta_auprc": float(values.mean()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "attempts": attempts,
        "clusters": int(len(unique_groups)),
    }


def build_bootstrap(
    panel: pd.DataFrame,
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    nested: dict[str, Any] = {}
    task_labels = {
        "ACTIVE": panel["active_label"].to_numpy(dtype=int),
        "STRICT_M2_SENSITIZATION": panel["sensitization_label"].to_numpy(dtype=int),
    }
    full_name = "FULL_DRUG_CONDITIONED_COUNTERFACTUAL_DELTA"
    for protocol_index, (protocol, specification) in enumerate(PROTOCOLS.items()):
        groups = panel[specification["group"]].to_numpy()
        nested[protocol] = {}
        for task_index, (task, label) in enumerate(task_labels.items()):
            score_key = "active" if task == "ACTIVE" else "sens"
            candidate = predictions[protocol][full_name][score_key]
            nested[protocol][task] = {}
            for comparator_index, comparator_name in enumerate(VARIANTS[:-1]):
                result = paired_cluster_bootstrap(
                    label,
                    candidate,
                    predictions[protocol][comparator_name][score_key],
                    groups,
                    BOOTSTRAP_SEED
                    + 1000 * protocol_index
                    + 100 * task_index
                    + comparator_index,
                )
                nested[protocol][task][comparator_name] = result
                rows.append(
                    {
                        "protocol": protocol,
                        "task": task,
                        "candidate": full_name,
                        "comparator": comparator_name,
                        "cluster_unit": specification["group"],
                        **result,
                    }
                )
    return pd.DataFrame(rows), nested


def metric_value(
    metrics: pd.DataFrame, protocol: str, variant: str, task: str, slice_name: str
) -> float:
    row = metrics[
        metrics["protocol"].eq(protocol)
        & metrics["variant"].eq(variant)
        & metrics["task"].eq(task)
        & metrics["slice"].eq(slice_name)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Metric row is not unique: {protocol}/{variant}/{task}/{slice_name}")
    return float(row.iloc[0]["micro_auprc"])


def evaluate_gates(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: dict[str, Any],
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    audit: dict[str, Any],
    split_audit: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, bool], dict[str, Any]]:
    full = "FULL_DRUG_CONDITIONED_COUNTERFACTUAL_DELTA"
    controls = ["WT_LOCAL_ONLY", "SHUFFLED_DELTA_WITHIN_GENE", "DELTA_NO_DRUG_CONDITIONING"]
    active_full = metric_value(metrics, "DRUG_COLD", full, "ACTIVE", "ALL")
    active_base = metric_value(metrics, "DRUG_COLD", "TOKEN_BASE", "ACTIVE", "ALL")
    sens_full = metric_value(metrics, "DRUG_COLD", full, "STRICT_M2_SENSITIZATION", "ALL")
    sens_base = metric_value(metrics, "DRUG_COLD", "TOKEN_BASE", "STRICT_M2_SENSITIZATION", "ALL")
    variant_full = metric_value(metrics, "VARIANT_CONSTRUCT_COLD", full, "ACTIVE", "ALL")
    variant_base = metric_value(metrics, "VARIANT_CONSTRUCT_COLD", "TOKEN_BASE", "ACTIVE", "ALL")
    gene_full = metric_value(metrics, "GENE_COLD", full, "ACTIVE", "ALL")
    gene_base = metric_value(metrics, "GENE_COLD", "TOKEN_BASE", "ACTIVE", "ALL")
    available_full = metric_value(metrics, "DRUG_COLD", full, "ACTIVE", "ESM2_AVAILABLE")
    available_base = metric_value(metrics, "DRUG_COLD", "TOKEN_BASE", "ACTIVE", "ESM2_AVAILABLE")
    base_errors = [
        value
        for protocol in audit["base_reproduction"].values()
        for fold in protocol
        for value in fold.values()
    ]
    fallback_errors = [
        value
        for protocol in audit["fallback"].values()
        for fold in protocol
        for value in fold.values()
    ]
    integrity = {
        "all_input_hashes_match": True,
        "all_three_protocols_have_five_complete_nonoverlapping_outer_folds": (
            len(split_audit) == 15
            and all(record["group_overlap"] == 0 for record in split_audit)
            and {record["fold"] for record in split_audit} == {1, 2, 3, 4, 5}
        ),
        "base_oof_reproduces_frozen_v10_within_absolute_tolerance": (
            max(base_errors, default=float("inf")) <= BASE_REPRODUCTION_TOLERANCE
        ),
        "all_predictions_finite": all(
            np.isfinite(predictions[protocol][variant][task]).all()
            for protocol in PROTOCOLS
            for variant in VARIANTS
            for task in ("active", "sens", "residual")
        ),
        "all_unavailable_construct_predictions_exactly_equal_base": (
            max(fallback_errors, default=float("inf")) == 0.0
        ),
        "all_variants_use_identical_trainable_parameter_count_and_optimization": (
            len(set(audit["parameter_counts"])) == 1
            and set(audit["parameter_counts"]) == {adapter_parameter_count()}
        ),
        "no_outer_test_labels_used_before_final_scoring": True,
    }
    signal = {
        "drug_cold_active_full_minus_base_auprc_point_gt_0": active_full > active_base,
        "drug_cold_active_full_minus_base_drug_cluster_bootstrap_ci95_lower_gt_0": (
            bootstrap["DRUG_COLD"]["ACTIVE"]["TOKEN_BASE"]["ci95_lower"] > 0
        ),
        "drug_cold_sensitization_full_minus_base_auprc_point_gt_0": sens_full > sens_base,
        "drug_cold_sensitization_full_minus_base_drug_cluster_bootstrap_ci95_lower_gt_0": (
            bootstrap["DRUG_COLD"]["STRICT_M2_SENSITIZATION"]["TOKEN_BASE"][
                "ci95_lower"
            ]
            > 0
        ),
        "drug_cold_active_full_auprc_gt_each_wt_only_shuffled_and_no_drug_control": all(
            active_full > metric_value(metrics, "DRUG_COLD", control, "ACTIVE", "ALL")
            for control in controls
        ),
        "drug_cold_sensitization_full_auprc_gt_each_wt_only_shuffled_and_no_drug_control": all(
            sens_full
            > metric_value(
                metrics, "DRUG_COLD", control, "STRICT_M2_SENSITIZATION", "ALL"
            )
            for control in controls
        ),
        "drug_cold_active_full_minus_shuffled_bootstrap_ci95_lower_gt_0": (
            bootstrap["DRUG_COLD"]["ACTIVE"]["SHUFFLED_DELTA_WITHIN_GENE"][
                "ci95_lower"
            ]
            > 0
        ),
        "cold_variant_active_full_minus_base_point_ge_minus_0_002": (
            variant_full - variant_base >= -0.002
        ),
        "cold_gene_active_full_minus_base_point_ge_minus_0_002": (
            gene_full - gene_base >= -0.002
        ),
        "available_subset_active_full_minus_base_point_gt_0": available_full > available_base,
    }
    values = {
        "drug_cold_active_auprc": {"full": active_full, "base": active_base, "delta": active_full - active_base},
        "drug_cold_sensitization_auprc": {"full": sens_full, "base": sens_base, "delta": sens_full - sens_base},
        "cold_variant_active_auprc": {"full": variant_full, "base": variant_base, "delta": variant_full - variant_base},
        "cold_gene_active_auprc": {"full": gene_full, "base": gene_base, "delta": gene_full - gene_base},
        "drug_cold_available_active_auprc": {"full": available_full, "base": available_base, "delta": available_full - available_base},
        "maximum_base_reproduction_absolute_error": max(base_errors),
        "maximum_unavailable_fallback_absolute_error": max(fallback_errors),
        "adapter_parameter_count": adapter_parameter_count(),
    }
    return {key: bool(value) for key, value in integrity.items()}, {key: bool(value) for key, value in signal.items()}, values


def write_oof_predictions(
    panel: pd.DataFrame,
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    path: Path,
) -> None:
    identifiers = panel[
        [
            "kirhub_compound",
            "variant_construct",
            "gene_symbol",
            "pairId",
            "target_chembl_id",
            "active_label",
            "sensitization_label",
            "variant_residual_activity_pct_1uM",
        ]
    ].copy()
    identifiers["esm2_reconstruction_available"] = panel["_feature_available"].to_numpy()
    frames = []
    for protocol in PROTOCOLS:
        for variant in VARIANTS:
            frame = identifiers.copy()
            frame.insert(0, "row_index", np.arange(len(panel)))
            frame.insert(1, "protocol", protocol)
            frame.insert(2, "variant", variant)
            frame["outer_fold"] = predictions[protocol][variant]["fold"]
            frame["active_probability"] = predictions[protocol][variant]["active"]
            frame["sensitization_probability"] = predictions[protocol][variant]["sens"]
            frame["residual_activity_pct"] = 100.0 * predictions[protocol][variant]["residual"]
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, compression="gzip")


def main() -> None:
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    observed_hashes = verify_config_and_inputs(config)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(CONFIG)
    script_hash = sha256(Path(__file__))
    log(f"Configuration {config_hash}; script {script_hash}; device {DEVICE}")
    panel, drug_vectors, target_vectors, manifest, feature_arrays = prepare_panel(config)
    predictions, audit, split_audit = run_oof(
        panel,
        drug_vectors,
        target_vectors,
        manifest,
        feature_arrays,
        config_hash,
        script_hash,
    )
    log("Computing prespecified metrics and paired cluster bootstrap")
    metrics = build_metrics(panel, predictions)
    bootstrap_table, bootstrap_nested = build_bootstrap(panel, predictions)
    integrity_gates, signal_gates, gate_values = evaluate_gates(
        panel, metrics, bootstrap_nested, predictions, audit, split_audit
    )
    multiseed_authorized = all(integrity_gates.values()) and all(signal_gates.values())

    oof_path = OUT / "DRUG_CONDITIONED_MUTATION_DELTA_PHASE_A_OOF_V1.csv.gz"
    metrics_path = OUT / "DRUG_CONDITIONED_MUTATION_DELTA_PHASE_A_METRICS_V1.csv"
    bootstrap_path = OUT / "DRUG_CONDITIONED_MUTATION_DELTA_PHASE_A_BOOTSTRAP_V1.csv"
    split_path = OUT / "DRUG_CONDITIONED_MUTATION_DELTA_PHASE_A_SPLIT_AUDIT_V1.csv"
    write_oof_predictions(panel, predictions, oof_path)
    metrics.to_csv(metrics_path, index=False)
    bootstrap_table.to_csv(bootstrap_path, index=False)
    pd.DataFrame(split_audit).to_csv(split_path, index=False)

    summary = {
        "schema_version": "BIOMASTER_DRUG_CONDITIONED_MUTATION_DELTA_PHASE_A_SUMMARY_V1",
        "created_utc": utc_now(),
        "status": "PASS" if all(integrity_gates.values()) else "FAIL_INTEGRITY",
        "scope": config["scope"],
        "architecture": config["adapter_architecture"],
        "variants": VARIANTS,
        "config": {"path": str(CONFIG.relative_to(ROOT)), "sha256": config_hash},
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": script_hash},
        "observed_input_hashes": observed_hashes,
        "integrity_gates": integrity_gates,
        "signal_gates": signal_gates,
        "gate_values": gate_values,
        "paired_cluster_bootstrap": bootstrap_nested,
        "split_audit": split_audit,
        "multiseed_confirmation_authorized": multiseed_authorized,
        "decision": (
            "AUTHORIZE_FIVE_SEED_CONFIRMATION_WITH_NEW_PRE_FREEZE"
            if multiseed_authorized
            else "PHASE_A_MECHANISM_NOT_SUPPORTED; DO_NOT_OPEN_FIVE_SEED_CONFIRMATION"
        ),
        "full_algorithm_innovation_claim": False,
        "full_sota_claim": False,
        "claim_boundary": config["claim_boundary"],
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(DEVICE),
            "base_seed": legacy.SEED,
            "adapter_seed": ADAPTER_SEED,
        },
        "outputs": {},
    }
    for path in (oof_path, metrics_path, bootstrap_path, split_path):
        summary["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    summary_path = OUT / "DRUG_CONDITIONED_MUTATION_DELTA_PHASE_A_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(json.dumps({
        "status": summary["status"],
        "decision": summary["decision"],
        "gate_values": gate_values,
        "failed_signal_gates": [key for key, value in signal_gates.items() if not value],
        "summary": str(summary_path.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
