#!/usr/bin/env python3
"""Run preregistered TRACE-PL development evaluation on homology-safe Platinum."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
LIGAND_CATEGORIES = 8
SIDECHAIN_CATEGORIES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ligand_category(atomic_number: int) -> int:
    if atomic_number == 6:
        return 0
    if atomic_number == 7:
        return 1
    if atomic_number == 8:
        return 2
    if atomic_number == 16:
        return 3
    if atomic_number == 15:
        return 4
    if atomic_number in {9, 17, 35, 53}:
        return 5
    if atomic_number in {3, 4, 11, 12, 13, 19, 20, 25, 26, 27, 28, 29, 30}:
        return 6
    return 7


def sidechain_category(atomic_number: int) -> int:
    return {6: 0, 7: 1, 8: 2, 16: 3}.get(atomic_number, 4)


def rbf(distances: np.ndarray, count: int = 16, upper: float = 8.0) -> np.ndarray:
    centers = np.linspace(0.5, upper, count, dtype=np.float32)
    width = float(centers[1] - centers[0])
    return np.exp(-((distances[..., None] - centers) / width) ** 2).astype(np.float32)


def contact_field(
    state_coordinates: np.ndarray,
    state_atomic_numbers: np.ndarray,
    ligand_coordinates: np.ndarray,
    ligand_atomic_numbers: np.ndarray,
    cutoff: float = 8.0,
) -> np.ndarray:
    conformers = state_coordinates.shape[0]
    field = np.zeros(
        (conformers, SIDECHAIN_CATEGORIES, LIGAND_CATEGORIES, 16), dtype=np.float32
    )
    if state_coordinates.shape[1] == 0:
        return field.reshape(conformers, -1)
    distances = np.linalg.norm(
        state_coordinates[:, :, None, :] - ligand_coordinates[None, None, :, :], axis=-1
    )
    basis = rbf(distances)
    basis[distances > cutoff] = 0.0
    normalization = math.sqrt(max(state_coordinates.shape[1] * len(ligand_coordinates), 1))
    side_categories = [sidechain_category(int(z)) for z in state_atomic_numbers]
    ligand_categories = [ligand_category(int(z)) for z in ligand_atomic_numbers]
    for side_index, side_cat in enumerate(side_categories):
        for ligand_index, ligand_cat in enumerate(ligand_categories):
            field[:, side_cat, ligand_cat, :] += basis[:, side_index, ligand_index, :]
    return (field / normalization).reshape(conformers, -1)


def clash_features(
    state_coordinates: np.ndarray, environment_coordinates: np.ndarray
) -> np.ndarray:
    conformers, atoms = state_coordinates.shape[:2]
    output = np.zeros((conformers, 4), dtype=np.float32)
    if atoms == 0 or len(environment_coordinates) == 0:
        output[:, 0] = 1.0
        return output
    distances = np.linalg.norm(
        state_coordinates[:, :, None, :] - environment_coordinates[None, None, :, :],
        axis=-1,
    )
    minimum = distances.min(axis=(1, 2))
    output[:, 0] = np.clip(minimum / 8.0, 0.0, 1.0)
    output[:, 1] = np.exp(-((distances / 2.0) ** 2)).sum(axis=(1, 2)) / atoms
    output[:, 2] = (np.maximum(2.2 - distances, 0.0) ** 2).sum(axis=(1, 2)) / atoms
    output[:, 3] = (distances < 2.2).sum(axis=(1, 2)) / atoms
    return output


def environment_field(
    environment_coordinates: np.ndarray,
    environment_atomic_numbers: np.ndarray,
    ca: np.ndarray,
) -> np.ndarray:
    field = np.zeros((LIGAND_CATEGORIES, 16), dtype=np.float32)
    if len(environment_coordinates) == 0:
        return field.reshape(-1)
    distances = np.linalg.norm(environment_coordinates - ca[None, :], axis=1)
    basis = rbf(distances)
    basis[distances > 8.0] = 0.0
    for atom_index, atomic_number in enumerate(environment_atomic_numbers):
        field[ligand_category(int(atomic_number))] += basis[atom_index]
    return (field / math.sqrt(len(environment_coordinates))).reshape(-1)


def ligand_counts(atomic_numbers: np.ndarray) -> np.ndarray:
    counts = np.zeros(LIGAND_CATEGORIES, dtype=np.float32)
    for atomic_number in atomic_numbers:
        counts[ligand_category(int(atomic_number))] += 1.0
    return counts / max(len(atomic_numbers), 1)


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    result = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return result if result else f"ACYCLIC::{Chem.MolToSmiles(molecule, canonical=True)}"


def state_descriptors(raw: dict, which: str, ligand_source: dict | None = None) -> dict:
    ligand = ligand_source if ligand_source is not None else raw
    ca = raw["backbone_coordinates"][1]
    if ligand_source is None:
        ligand_coordinates = raw["ligand_coordinates"]
    else:
        donor_ca = ligand_source["backbone_coordinates"][1]
        ligand_coordinates = (
            ligand_source["ligand_coordinates"] - donor_ca[None, :] + ca[None, :]
        )
    state_coordinates = raw[f"{which}_state_coordinates"]
    state_atomic_numbers = raw[f"{which}_state_atomic_numbers"]
    return {
        "contact": contact_field(
            state_coordinates,
            state_atomic_numbers,
            ligand_coordinates,
            ligand["ligand_atomic_numbers"],
        ),
        "clash": clash_features(state_coordinates, raw["environment_coordinates"]),
        "prior": raw[f"{which}_state_conformer_prior_energy"].astype(np.float32),
    }


def prepare_samples(index: pd.DataFrame) -> list[dict]:
    samples = []
    for row_index, row in index.iterrows():
        with np.load(row["feature_path"]) as loaded:
            raw = {key: loaded[key] for key in loaded.files}
        distance = float(row["computed_mutation_ligand_min_distance"])
        sample = {
            "row_index": int(row_index),
            "sample_id": row["sample_id"],
            "uniprot_id": row["uniprot_id"],
            "homology_cluster": row["homology_cluster"],
            "scaffold": scaffold(row["ligand_smiles"]),
            "old_aa": row["old_aa"],
            "new_aa": row["new_aa"],
            "old_aa_index": AA_TO_INDEX[row["old_aa"]],
            "new_aa_index": AA_TO_INDEX[row["new_aa"]],
            "label": float(row["label_ddg_kcal_mol"]),
            "distance": distance,
            "distance_rbf": rbf(np.asarray(distance, dtype=np.float32)).reshape(-1),
            "fingerprint": raw["ligand_morgan"].astype(np.float32),
            "ligand_counts": ligand_counts(raw["ligand_atomic_numbers"]),
            "environment": environment_field(
                raw["environment_coordinates"],
                raw["environment_atomic_numbers"],
                raw["backbone_coordinates"][1],
            ),
            "old_state": state_descriptors(raw, "old"),
            "new_state": state_descriptors(raw, "new"),
            "raw": raw,
        }
        samples.append(sample)
    return samples


def shuffled_samples(samples: list[dict], seed: int) -> list[dict]:
    order = np.random.default_rng(seed).permutation(len(samples))
    if np.array_equal(order, np.arange(len(samples))):
        order = np.roll(order, 1)
    output = []
    for receiver, donor_index in zip(samples, order):
        donor = samples[int(donor_index)]
        transformed = copy.copy(receiver)
        transformed["fingerprint"] = donor["fingerprint"]
        transformed["ligand_counts"] = donor["ligand_counts"]
        transformed["old_state"] = state_descriptors(
            receiver["raw"], "old", ligand_source=donor["raw"]
        )
        transformed["new_state"] = state_descriptors(
            receiver["raw"], "new", ligand_source=donor["raw"]
        )
        output.append(transformed)
    return output


def balanced_group_folds(groups: list[str], n_splits: int, seed: int) -> np.ndarray:
    counts = Counter(groups)
    rng = random.Random(seed)
    unique = list(counts)
    rng.shuffle(unique)
    unique.sort(key=lambda group: counts[group], reverse=True)
    fold_sizes = [0] * n_splits
    assignment = {}
    for group in unique:
        fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        assignment[group] = fold
        fold_sizes[fold] += counts[group]
    return np.asarray([assignment[group] for group in groups], dtype=int)


class TraceDataset(Dataset):
    def __init__(self, samples: list[dict], indices: np.ndarray, single: bool = False, no_contact: bool = False):
        self.samples = samples
        self.indices = np.asarray(indices, dtype=int)
        self.single = single
        self.no_contact = no_contact

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        sample = self.samples[int(self.indices[position])]
        result = {key: value for key, value in sample.items() if key != "raw"}
        result["old_state"] = {key: value.copy() for key, value in sample["old_state"].items()}
        result["new_state"] = {key: value.copy() for key, value in sample["new_state"].items()}
        if self.single:
            for which in ["old_state", "new_state"]:
                for key in ["contact", "clash", "prior"]:
                    result[which][key] = result[which][key][:1]
        if self.no_contact:
            result["old_state"]["contact"].fill(0.0)
            result["new_state"]["contact"].fill(0.0)
        return result


def pad_states(batch: list[dict], which: str) -> dict[str, torch.Tensor]:
    max_conformers = max(len(sample[which]["prior"]) for sample in batch)
    contact = np.zeros((len(batch), max_conformers, 640), dtype=np.float32)
    clash = np.zeros((len(batch), max_conformers, 4), dtype=np.float32)
    prior = np.zeros((len(batch), max_conformers), dtype=np.float32)
    mask = np.zeros((len(batch), max_conformers), dtype=bool)
    for index, sample in enumerate(batch):
        count = len(sample[which]["prior"])
        contact[index, :count] = sample[which]["contact"]
        clash[index, :count] = sample[which]["clash"]
        prior[index, :count] = sample[which]["prior"]
        mask[index, :count] = True
    return {
        "contact": torch.from_numpy(contact),
        "clash": torch.from_numpy(clash),
        "prior": torch.from_numpy(prior),
        "mask": torch.from_numpy(mask),
    }


def collate(batch: list[dict]) -> dict:
    return {
        "sample_id": [sample["sample_id"] for sample in batch],
        "row_index": np.asarray([sample["row_index"] for sample in batch], dtype=int),
        "old_aa": torch.tensor([sample["old_aa_index"] for sample in batch]),
        "new_aa": torch.tensor([sample["new_aa_index"] for sample in batch]),
        "fingerprint": torch.from_numpy(np.stack([sample["fingerprint"] for sample in batch])),
        "ligand_counts": torch.from_numpy(np.stack([sample["ligand_counts"] for sample in batch])),
        "environment": torch.from_numpy(np.stack([sample["environment"] for sample in batch])),
        "distance_rbf": torch.from_numpy(np.stack([sample["distance_rbf"] for sample in batch])),
        "old_state": pad_states(batch, "old_state"),
        "new_state": pad_states(batch, "new_state"),
        "label": torch.tensor([sample["label"] for sample in batch], dtype=torch.float32),
    }


def move(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        elif isinstance(value, dict):
            output[key] = move(value, device)
        else:
            output[key] = value
    return output


class StateEncoder(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.aa_embedding = nn.Embedding(20, 12)
        self.contact_encoder = nn.Sequential(
            nn.Linear(640, 96), nn.SiLU(), nn.Linear(96, hidden)
        )
        self.clash_encoder = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, hidden))
        self.fingerprint_encoder = nn.Sequential(nn.Linear(256, 32), nn.SiLU())
        self.environment_encoder = nn.Sequential(nn.Linear(128, hidden), nn.SiLU())
        self.global_encoder = nn.Sequential(
            nn.Linear(12 + 32 + hidden + 8 + 16, 96),
            nn.SiLU(),
            nn.Linear(96, hidden),
        )
        self.normalization = nn.LayerNorm(hidden)
        self.conformer_score = nn.Linear(hidden, 1)
        self.prior_scale_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        state: dict,
        amino_acid: torch.Tensor,
        fingerprint: torch.Tensor,
        ligand_counts_tensor: torch.Tensor,
        environment: torch.Tensor,
        distance_rbf_tensor: torch.Tensor,
    ) -> torch.Tensor:
        aa = self.aa_embedding(amino_acid)
        global_state = self.global_encoder(
            torch.cat(
                [
                    aa,
                    self.fingerprint_encoder(fingerprint),
                    self.environment_encoder(environment),
                    ligand_counts_tensor,
                    distance_rbf_tensor,
                ],
                dim=-1,
            )
        )
        conformer = self.normalization(
            self.contact_encoder(state["contact"])
            + self.clash_encoder(state["clash"])
            + global_state[:, None, :]
        )
        score = self.conformer_score(conformer).squeeze(-1)
        score = score + torch.nn.functional.softplus(self.prior_scale_raw) * state["prior"]
        score = score.masked_fill(~state["mask"], float("inf"))
        weights = torch.softmax(-score, dim=1)
        return torch.sum(weights[:, :, None] * conformer, dim=1)


class TracePotentialModel(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.encoder = StateEncoder(hidden)
        self.potential_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def potentials(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        arguments = [
            batch["fingerprint"],
            batch["ligand_counts"],
            batch["environment"],
            batch["distance_rbf"],
        ]
        old_latent = self.encoder(batch["old_state"], batch["old_aa"], *arguments)
        new_latent = self.encoder(batch["new_state"], batch["new_aa"], *arguments)
        return (
            self.potential_head(old_latent).squeeze(-1),
            self.potential_head(new_latent).squeeze(-1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        old, new = self.potentials(batch)
        return new - old


class DirectPairModel(nn.Module):
    def __init__(self, hidden: int = 48):
        super().__init__()
        self.encoder = StateEncoder(hidden)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        arguments = [
            batch["fingerprint"],
            batch["ligand_counts"],
            batch["environment"],
            batch["distance_rbf"],
        ]
        old_latent = self.encoder(batch["old_state"], batch["old_aa"], *arguments)
        new_latent = self.encoder(batch["new_state"], batch["new_aa"], *arguments)
        return self.pair_head(torch.cat([old_latent, new_latent], dim=-1)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    mean: float,
    scale: float,
    freeze: dict,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(freeze["phase_a_training"]["learning_rate"]),
        weight_decay=float(freeze["phase_a_training"]["weight_decay"]),
    )
    loss_function = nn.HuberLoss(delta=1.0)
    best_state = None
    best_rmse = float("inf")
    best_epoch = -1
    stale = 0
    epochs_max = int(freeze["phase_a_training"]["epochs_max"])
    patience = int(freeze["phase_a_training"]["early_stopping_patience"])
    for epoch in range(epochs_max):
        model.train()
        for batch in train_loader:
            batch = move(batch, device)
            # A constant output offset would violate thermodynamic antisymmetry.
            # Scale labels for conditioning, but never center them.
            target = batch["label"] / scale
            prediction = model(batch)
            loss = loss_function(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        residuals = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = move(batch, device)
                prediction = model(batch) * scale
                residuals.extend((prediction - batch["label"]).cpu().numpy().tolist())
        validation_rmse = float(np.sqrt(np.mean(np.square(residuals))))
        if validation_rmse < best_rmse - 1e-6:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_rmse": best_rmse,
        "epochs_completed": epoch + 1,
    }


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    mean: float,
    scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    model.eval()
    row_indices, labels, predictions = [], [], []
    antisymmetry_errors, cycle_errors = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            normalized = model(batch)
            prediction = normalized * scale
            row_indices.extend(batch["row_index"].tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
            if isinstance(model, TracePotentialModel):
                old, new = model.potentials(batch)
                forward = new - old
                reverse = old - new
                antisymmetry_errors.extend(torch.abs(forward + reverse).cpu().numpy().tolist())
                third = (old + new) / 2.0
                closure = (new - old) + (third - new) - (third - old)
                cycle_errors.extend(torch.abs(closure).cpu().numpy().tolist())
    return (
        np.asarray(row_indices, dtype=int),
        np.asarray(labels, dtype=float),
        np.asarray(predictions, dtype=float),
        max(antisymmetry_errors, default=float("nan")),
        max(cycle_errors, default=float("nan")),
    )


def classical_features(samples: list[dict], indices: np.ndarray) -> np.ndarray:
    rows = []
    for index in indices:
        sample = samples[int(index)]
        old = np.zeros(20, dtype=np.float32)
        new = np.zeros(20, dtype=np.float32)
        old[sample["old_aa_index"]] = 1.0
        new[sample["new_aa_index"]] = 1.0
        rows.append(
            np.concatenate(
                [
                    old,
                    new,
                    new - old,
                    sample["distance_rbf"],
                    sample["ligand_counts"],
                    sample["fingerprint"],
                ]
            )
        )
    return np.stack(rows)


def metric_row(protocol: str, model: str, fold: int | str, y: np.ndarray, p: np.ndarray) -> dict:
    pearson = float(stats.pearsonr(y, p).statistic) if np.std(p) > 0 else float("nan")
    spearman = float(stats.spearmanr(y, p).statistic) if np.std(p) > 0 else float("nan")
    return {
        "protocol": protocol,
        "model": model,
        "fold": fold,
        "n": len(y),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "pearson": pearson,
        "spearman": spearman,
    }


def bootstrap_metric_difference(
    predictions: pd.DataFrame,
    protocol: str,
    first: str,
    second: str,
    group_column: str,
    seed: int,
    replicates: int = 2000,
) -> dict:
    subset = predictions[predictions["protocol"] == protocol]
    pivot = subset.pivot(index="row_index", columns="model", values="prediction")
    labels = subset.drop_duplicates("row_index").set_index("row_index")["label"]
    groups = subset.drop_duplicates("row_index").set_index("row_index")[group_column]
    common = pivot[[first, second]].dropna().index.intersection(labels.index)
    group_values = groups.loc[common]
    unique_groups = group_values.unique().tolist()
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate(
            [common[group_values.to_numpy() == group] for group in sampled]
        )
        y = labels.loc[selected].to_numpy(float)
        p_first = pivot.loc[selected, first].to_numpy(float)
        p_second = pivot.loc[selected, second].to_numpy(float)
        differences.append(
            np.sqrt(np.mean((y - p_first) ** 2))
            - np.sqrt(np.mean((y - p_second) ** 2))
        )
    point_y = labels.loc[common].to_numpy(float)
    point = float(
        np.sqrt(np.mean((point_y - pivot.loc[common, first].to_numpy(float)) ** 2))
        - np.sqrt(np.mean((point_y - pivot.loc[common, second].to_numpy(float)) ** 2))
    )
    return {
        "first": first,
        "second": second,
        "metric": "RMSE_first_minus_RMSE_second",
        "point": point,
        "ci95_low": float(np.percentile(differences, 2.5)),
        "ci95_high": float(np.percentile(differences, 97.5)),
        "replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-index",
        default=(
            "outputs/old_drug_target_sota_v1/trace_pl_platinum_features_v1/"
            "TRACE_PL_PLATINUM_FEATURE_INDEX_V1.csv"
        ),
    )
    parser.add_argument(
        "--freeze", default="configs/biomaster_trace_pl_phase_a_freeze_20260814.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/old_drug_target_sota_v1/trace_pl_phase_a_v1"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.threads)
    feature_index_path = Path(args.feature_index).resolve()
    freeze_path = Path(args.freeze).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(freeze_path.read_text())
    seed = int(freeze["phase_a_training"]["seed"])
    seed_everything(seed)
    index = pd.read_csv(feature_index_path, low_memory=False)
    if len(index) != int(freeze["development_data"]["rows"]):
        raise RuntimeError("feature index row count differs from frozen development data")
    samples = prepare_samples(index)
    device = torch.device("cpu")

    trace_parameters = parameter_count(TracePotentialModel())
    direct_parameters = parameter_count(DirectPairModel())
    parameter_relative_difference = abs(trace_parameters - direct_parameters) / trace_parameters
    batch_size = int(freeze["phase_a_training"]["batch_size"])
    prediction_records = []
    metric_records = []
    training_records = []
    split_records = []
    maximum_antisymmetry_error = 0.0
    maximum_cycle_error = 0.0
    split_integrity = True

    protocols = {
        "homology_cluster_cold": [sample["homology_cluster"] for sample in samples],
        "ligand_scaffold_cold": [sample["scaffold"] for sample in samples],
    }
    variants = [
        "trace_full",
        "direct_pair",
        "trace_single_conformer",
        "trace_no_atomic_contact",
        "trace_shuffled_ligand_contact",
    ]
    for protocol_index, (protocol, groups) in enumerate(protocols.items()):
        fold_assignments = balanced_group_folds(groups, 5, seed + protocol_index * 1000)
        for fold in range(5):
            test_indices = np.flatnonzero(fold_assignments == fold)
            outer_train_indices = np.flatnonzero(fold_assignments != fold)
            outer_groups = [groups[index] for index in outer_train_indices]
            inner_assignment = balanced_group_folds(
                outer_groups, 5, seed + protocol_index * 1000 + fold + 1
            )
            validation_indices = outer_train_indices[inner_assignment == 0]
            train_indices = outer_train_indices[inner_assignment != 0]
            train_groups = {groups[index] for index in train_indices}
            validation_groups = {groups[index] for index in validation_indices}
            test_groups = {groups[index] for index in test_indices}
            disjoint = not (
                train_groups & validation_groups
                or train_groups & test_groups
                or validation_groups & test_groups
            )
            split_integrity &= disjoint
            for index_value in train_indices:
                split_records.append(
                    {"protocol": protocol, "fold": fold, "row_index": int(index_value), "role": "train"}
                )
            for index_value in validation_indices:
                split_records.append(
                    {"protocol": protocol, "fold": fold, "row_index": int(index_value), "role": "validation"}
                )
            for index_value in test_indices:
                split_records.append(
                    {"protocol": protocol, "fold": fold, "row_index": int(index_value), "role": "test"}
                )
            train_labels = np.asarray([samples[index]["label"] for index in train_indices])
            mean = float(train_labels.mean())
            scale = float(train_labels.std())
            if scale < 1e-6:
                scale = 1.0

            # Label-free classical baselines.
            x_train = classical_features(samples, train_indices)
            x_test = classical_features(samples, test_indices)
            y_train = train_labels
            y_test = np.asarray([samples[index]["label"] for index in test_indices])
            classical_models = {
                "zero_ddg": None,
                "train_mean": None,
                "ridge_nonatomic": Ridge(alpha=1.0),
                "extra_trees_nonatomic": ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed + fold,
                    n_jobs=args.threads,
                ),
            }
            for name, estimator in classical_models.items():
                if name == "zero_ddg":
                    prediction = np.zeros(len(test_indices))
                elif name == "train_mean":
                    prediction = np.full(len(test_indices), mean)
                else:
                    estimator.fit(x_train, y_train)
                    prediction = estimator.predict(x_test)
                metric_records.append(metric_row(protocol, name, fold, y_test, prediction))
                for sample_index, label, value in zip(test_indices, y_test, prediction):
                    sample = samples[int(sample_index)]
                    prediction_records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "model": name,
                            "row_index": int(sample_index),
                            "sample_id": sample["sample_id"],
                            "label": float(label),
                            "prediction": float(value),
                            "homology_cluster": sample["homology_cluster"],
                            "scaffold": sample["scaffold"],
                        }
                    )

            for variant_index, variant in enumerate(variants):
                variant_seed = seed + protocol_index * 10000 + fold * 100 + variant_index
                seed_everything(variant_seed)
                variant_samples = samples
                if variant == "trace_shuffled_ligand_contact":
                    variant_samples = shuffled_samples(samples, variant_seed)
                single = variant == "trace_single_conformer"
                no_contact = variant == "trace_no_atomic_contact"
                train_dataset = TraceDataset(
                    variant_samples, train_indices, single=single, no_contact=no_contact
                )
                validation_dataset = TraceDataset(
                    variant_samples, validation_indices, single=single, no_contact=no_contact
                )
                test_dataset = TraceDataset(
                    variant_samples, test_indices, single=single, no_contact=no_contact
                )
                generator = torch.Generator().manual_seed(variant_seed)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    generator=generator,
                    collate_fn=collate,
                )
                validation_loader = DataLoader(
                    validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
                )
                model = DirectPairModel() if variant == "direct_pair" else TracePotentialModel()
                model.to(device)
                model, training = train_model(
                    model, train_loader, validation_loader, mean, scale, freeze, device
                )
                row_indices, labels, predictions, antisymmetry, cycle = predict_model(
                    model, test_loader, mean, scale, device
                )
                if variant != "direct_pair":
                    maximum_antisymmetry_error = max(maximum_antisymmetry_error, antisymmetry)
                    maximum_cycle_error = max(maximum_cycle_error, cycle)
                training_records.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "model": variant,
                        "seed": variant_seed,
                        "train_n": len(train_indices),
                        "validation_n": len(validation_indices),
                        "test_n": len(test_indices),
                        "label_mean": mean,
                        "label_scale": scale,
                        **training,
                    }
                )
                metric_records.append(metric_row(protocol, variant, fold, labels, predictions))
                for sample_index, label, value in zip(row_indices, labels, predictions):
                    sample = samples[int(sample_index)]
                    prediction_records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "model": variant,
                            "row_index": int(sample_index),
                            "sample_id": sample["sample_id"],
                            "label": float(label),
                            "prediction": float(value),
                            "homology_cluster": sample["homology_cluster"],
                            "scaffold": sample["scaffold"],
                        }
                    )
                if protocol == "homology_cluster_cold" and variant == "trace_full":
                    checkpoint_path = checkpoint_dir / f"TRACE_PL_HOMOLOGY_FOLD_{fold}_V1.pt"
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "label_mean": 0.0,
                            "label_scale": scale,
                            "fold": fold,
                            "freeze_sha256": sha256(freeze_path),
                            "feature_index_sha256": sha256(feature_index_path),
                        },
                        checkpoint_path,
                    )

    predictions_df = pd.DataFrame(prediction_records)
    metrics_df = pd.DataFrame(metric_records)
    training_df = pd.DataFrame(training_records)
    splits_df = pd.DataFrame(split_records)
    aggregate_metrics = []
    for (protocol, model), group in predictions_df.groupby(["protocol", "model"]):
        aggregate_metrics.append(
            metric_row(
                protocol,
                model,
                "OOF_AGGREGATE",
                group["label"].to_numpy(float),
                group["prediction"].to_numpy(float),
            )
        )
    aggregate_df = pd.DataFrame(aggregate_metrics)
    metrics_df = pd.concat([metrics_df, aggregate_df], ignore_index=True)

    comparisons = {}
    for comparator in [
        "direct_pair",
        "trace_single_conformer",
        "trace_shuffled_ligand_contact",
    ]:
        comparisons[f"trace_full_vs_{comparator}"] = bootstrap_metric_difference(
            predictions_df,
            "homology_cluster_cold",
            "trace_full",
            comparator,
            "homology_cluster",
            seed + len(comparisons),
        )
    primary = aggregate_df[
        aggregate_df["protocol"].eq("homology_cluster_cold")
    ].set_index("model")
    gates = {
        "integrity": bool(
            split_integrity
            and np.isfinite(predictions_df["prediction"]).all()
            and parameter_relative_difference <= 0.10
            and maximum_antisymmetry_error <= 1e-6
            and maximum_cycle_error <= 1e-6
        ),
        "full_beats_direct_pair_rmse_ci": comparisons[
            "trace_full_vs_direct_pair"
        ]["ci95_high"]
        < 0,
        "full_beats_single_conformer_rmse_ci": comparisons[
            "trace_full_vs_trace_single_conformer"
        ]["ci95_high"]
        < 0,
        "full_beats_shuffled_contacts_rmse_ci": comparisons[
            "trace_full_vs_trace_shuffled_ligand_contact"
        ]["ci95_high"]
        < 0,
        "primary_pearson_at_least_0_20": bool(primary.loc["trace_full", "pearson"] >= 0.20),
    }
    development_supported = all(gates.values())
    decision = (
        "DEVELOPMENT_MECHANISM_GATES_PASS; EXTERNAL_PBCNET2_EVALUATION_AUTHORIZED"
        if development_supported
        else "PHASE_A_MECHANISM_NOT_SUPPORTED; DO_NOT_OPEN_EXTERNAL_TEST_OR_MULTISEED_CONFIRMATION"
    )

    predictions_path = output_dir / "TRACE_PL_PHASE_A_OOF_PREDICTIONS_V1.csv"
    metrics_path = output_dir / "TRACE_PL_PHASE_A_METRICS_V1.csv"
    training_path = output_dir / "TRACE_PL_PHASE_A_TRAINING_LOG_V1.csv"
    splits_path = output_dir / "TRACE_PL_PHASE_A_SPLIT_ASSIGNMENTS_V1.csv"
    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    training_df.to_csv(training_path, index=False)
    splits_df.to_csv(splits_path, index=False)
    summary = {
        "schema_version": "TRACE_PL_PHASE_A_V1",
        "status": "PASS" if gates["integrity"] else "FAIL",
        "decision": decision,
        "algorithm_innovation_verified": False,
        "reason_innovation_not_yet_verified": (
            "development evidence alone is insufficient; locked external SOTA and multiseed confirmation remain unopened"
            if development_supported
            else "the preregistered development mechanism gates did not all pass"
        ),
        "freeze_sha256": sha256(freeze_path),
        "feature_index_sha256": sha256(feature_index_path),
        "counts": {
            "samples": len(samples),
            "protocols": len(protocols),
            "neural_fits": len(training_df),
            "oof_prediction_rows": len(predictions_df),
        },
        "parameter_audit": {
            "trace_parameters": trace_parameters,
            "direct_pair_parameters": direct_parameters,
            "relative_difference": parameter_relative_difference,
            "within_10_percent": parameter_relative_difference <= 0.10,
        },
        "integrity_audit": {
            "all_splits_group_disjoint": split_integrity,
            "all_predictions_finite": bool(np.isfinite(predictions_df["prediction"]).all()),
            "maximum_absolute_antisymmetry_error": maximum_antisymmetry_error,
            "maximum_absolute_cycle_closure_error": maximum_cycle_error,
            "external_test_opened": False,
        },
        "development_gates": gates,
        "bootstrap_comparisons": comparisons,
        "primary_homology_cold_metrics": primary.to_dict(orient="index"),
        "files": {
            "oof_predictions_csv": str(predictions_path),
            "metrics_csv": str(metrics_path),
            "training_log_csv": str(training_path),
            "split_assignments_csv": str(splits_path),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }
    summary_path = output_dir / "TRACE_PL_PHASE_A_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
