#!/usr/bin/env python3
"""Run official lightweight SCOPE-DTI checkpoints on the frozen old-drug test.

Only targets present in the official SCOPE target vocabulary are evaluated.
Missing targets and failed deterministic 3D conformers are reported and never
silently assigned a negative score.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / ".external/scope_dti_lightweight"
sys.path.insert(0, str(SCOPE))
sys.path.insert(0, str(ROOT / "scripts"))
from configs import get_cfg_defaults  # noqa: E402
from models import SCOPE as ScopeModel  # noqa: E402
from mol_graph import sdf_to_graphs  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
STACK = (
    BASE / "biomaster_odti_routed_ranker_v1/cold_regime_stack_v1"
    / "S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_20260813__CORE"
    / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
)
VOCABULARY = SCOPE / "protein_targets/Total_predict.parquet"
CHECKPOINT_DIR = SCOPE / "models_path/Filtered_Total_DrugBAN_3D"
OUT = BASE / "public_baselines_v1/scope_dti_old_drug_entity_cold_v1"
SEED = 20260813
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14,
    "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
    "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_sdf(smiles: str) -> tuple[str | None, str]:
    base_molecule = Chem.MolFromSmiles(smiles)
    if base_molecule is None:
        return None, "RDKIT_PARSE_FAILURE"
    molecule = Chem.AddHs(base_molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.pruneRmsThresh = 0.1
    status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        # Some constrained polycycles (for example quinidine) fail distance-
        # geometry only after explicit hydrogen expansion.  A deterministic
        # heavy-atom embedding followed by coordinate-preserving hydrogen
        # addition is the RDKit-supported fallback and retains a genuine 3-D
        # conformer rather than silently substituting a 2-D drawing.
        heavy = Chem.Mol(base_molecule)
        fallback_params = AllChem.ETKDGv3()
        fallback_params.randomSeed = SEED
        fallback_params.pruneRmsThresh = 0.1
        fallback_status = AllChem.EmbedMolecule(heavy, fallback_params)
        if fallback_status != 0:
            return None, f"ETKDG_EXPLICIT_H_{status}_HEAVY_ATOM_{fallback_status}"
        molecule = Chem.AddHs(heavy, addCoords=True)
        embed_route = f"ETKDGV3_HEAVY_ATOM_FALLBACK_FROM_EXPLICIT_H_{status}"
    else:
        embed_route = "ETKDGV3_EXPLICIT_H"
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            optimize_status = AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
            route = f"{embed_route}_MMFF_STATUS_{optimize_status}"
        else:
            optimize_status = AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
            route = f"{embed_route}_UFF_STATUS_{optimize_status}"
    except Exception as error:
        route = f"{embed_route}_OPTIMIZATION_EXCEPTION_{type(error).__name__}"
    return Chem.MolToMolBlock(molecule), route


def integer_label_protein(sequence: str, max_length: int = 2000) -> np.ndarray:
    encoding = np.zeros(max_length, dtype=np.float32)
    for index, letter in enumerate(sequence[:max_length]):
        encoding[index] = CHARPROTSET.get(letter.upper(), 0)
    return encoding


def custom_collate_fn(batch):
    graphs, proteins, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(proteins), torch.stack(labels)


class ScopePairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        graphs: dict[str, object],
        protein_encoding: dict[str, torch.Tensor],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.graphs = graphs
        self.protein_encoding = protein_encoding

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        return (
            self.graphs[row["model_ligand_smiles"]],
            self.protein_encoding[row["query_accession"]],
            torch.tensor(float(row["binary_label"]), dtype=torch.float32),
        )


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for drug, protein, _ in loader:
        drug = drug.to(device)
        protein = protein.to(device)
        _, _, score, _ = model(drug, protein, mode="eval")
        values.append(torch.sigmoid(score).squeeze(1).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    required = [PAIRS, STACK, VOCABULARY]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
    if len(checkpoints) != 5:
        raise RuntimeError(f"Expected five official Total checkpoints, found {len(checkpoints)}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    OUT.mkdir(parents=True, exist_ok=True)

    all_pairs = pd.read_csv(PAIRS, low_memory=False)
    test = all_pairs[all_pairs["is_deployment_old_drug"].astype(bool)].copy()
    if len(test) != 2556 or test["parent_standard_inchi_key"].nunique() != 302:
        raise RuntimeError("Frozen old-drug entity-cold test changed")
    vocabulary = pd.read_parquet(VOCABULARY)
    if vocabulary["target_uniprot_id"].duplicated().any():
        raise RuntimeError("SCOPE Total target vocabulary is not unique")
    sequence_lookup = vocabulary.set_index("target_uniprot_id")["sequence"].astype(str).to_dict()
    test["scope_target_available"] = test["query_accession"].isin(sequence_lookup)

    conformer_rows: list[dict[str, object]] = []
    graphs: dict[str, object] = {}
    for smiles in sorted(test.loc[test["scope_target_available"], "model_ligand_smiles"].unique()):
        sdf, route = deterministic_sdf(str(smiles))
        available = sdf is not None
        if available:
            try:
                graphs[str(smiles)] = sdf_to_graphs(sdf)
            except Exception as error:
                available = False
                route = f"GRAPH_BUILD_FAILURE_{type(error).__name__}: {str(error)[:160]}"
        conformer_rows.append({
            "model_ligand_smiles": smiles,
            "scope_drug_graph_available": available,
            "conformer_route": route,
        })
    conformers = pd.DataFrame(conformer_rows)
    test = test.merge(conformers, on="model_ligand_smiles", how="left", validate="many_to_one")
    evaluable = test[
        test["scope_target_available"] & test["scope_drug_graph_available"].fillna(False)
    ].reset_index(drop=True)
    if evaluable["binary_label"].nunique() != 2:
        raise RuntimeError("SCOPE evaluable subset lacks both classes")
    protein_encoding = {
        accession: torch.tensor(integer_label_protein(sequence), dtype=torch.float32)
        for accession, sequence in vocabulary[
            vocabulary["target_uniprot_id"].isin(evaluable["query_accession"])
        ][["target_uniprot_id", "sequence"]].itertuples(index=False)
    }
    dataset = ScopePairDataset(evaluable, graphs, protein_encoding)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_cfg_defaults()
    predictions = evaluable[[
        "calibration_pair_id", "query_accession", "target_chembl_id", "primary_gene",
        "parent_standard_inchi_key", "parent_molecule_chembl_id", "binary_label",
    ]].copy()
    for index, checkpoint in enumerate(checkpoints, start=1):
        model = ScopeModel(**config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        predictions[f"scope_checkpoint_{index}"] = predict(model, loader, device)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"checkpoint": index, "rows": len(predictions)}), flush=True)
    checkpoint_columns = [f"scope_checkpoint_{index}" for index in range(1, 6)]
    predictions["scope_mean"] = predictions[checkpoint_columns].mean(axis=1)

    stack = pd.read_csv(STACK)
    comparison = predictions.merge(
        stack[[
            "calibration_pair_id", "COLD_REGIME_ROUTED_STACK", "conplex_score",
            "train_positive_max_tanimoto",
        ]],
        on="calibration_pair_id",
        validate="one_to_one",
    )
    metric_rows = []
    for name in [
        "scope_mean", "COLD_REGIME_ROUTED_STACK", "conplex_score", "train_positive_max_tanimoto"
    ]:
        row = {"model": name, "evaluation_subset": "SCOPE_TARGET_AND_3D_GRAPH_COVERED"}
        row.update(metrics(comparison, comparison[name].to_numpy(dtype=np.float64)))
        metric_rows.append(row)
    metrics_frame = pd.DataFrame(metric_rows)

    prediction_path = OUT / "SCOPE_DTI_OLD_DRUG_ENTITY_COLD_PREDICTIONS_V1.csv.gz"
    metric_path = OUT / "SCOPE_DTI_SAME_COVERAGE_COMPARISON_METRICS_V1.csv"
    conformer_path = OUT / "SCOPE_DTI_CONFORMER_COVERAGE_V1.csv.gz"
    predictions.to_csv(prediction_path, index=False)
    metrics_frame.to_csv(metric_path, index=False)
    conformers.to_csv(conformer_path, index=False)
    checks = {
        "official_total_vocabulary_exact_4893": len(vocabulary) == 4893,
        "official_total_checkpoints_exact_five": len(checkpoints) == 5,
        "all_261_targets_in_old_drug_test_covered": test.loc[test["scope_target_available"], "query_accession"].nunique() == 261,
        "evaluable_predictions_complete": predictions[checkpoint_columns + ["scope_mean"]].notna().all().all(),
        "evaluable_predictions_bounded": ((predictions["scope_mean"] >= 0) & (predictions["scope_mean"] <= 1)).all(),
        "same_coverage_comparison_exact": len(comparison) == len(predictions),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": str(device),
        "official_repository_commit": "af8a553c1d038dd13be9532b1f599b154a8a6b1d",
        "test_population": {
            "frozen_pairs": int(len(test)),
            "frozen_old_drugs": int(test["parent_standard_inchi_key"].nunique()),
            "frozen_targets": int(test["query_accession"].nunique()),
            "scope_covered_targets_in_test": int(test.loc[test["scope_target_available"], "query_accession"].nunique()),
            "scope_missing_targets_in_test": sorted(test.loc[~test["scope_target_available"], "query_accession"].unique()),
            "evaluable_pairs": int(len(evaluable)),
            "evaluable_old_drugs": int(evaluable["parent_standard_inchi_key"].nunique()),
            "evaluable_targets": int(evaluable["query_accession"].nunique()),
            "failed_unique_drug_graphs": int((~conformers["scope_drug_graph_available"]).sum()),
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "same_coverage_metrics": metrics_frame.to_dict("records"),
        "limitations": [
            "SCOPE is externally pretrained on 13 public repositories; source overlap with these ChEMBL37 pairs has not been excluded.",
            "This is a frozen-checkpoint inference comparison, not retraining SCOPE on the BioMaster split.",
            "Deterministic ETKDGv3 conformers are generated locally; they may differ from conformers used by the authors.",
            "The official vocabulary covers all 261 targets in this old-drug test; across the wider 428-target calibration set three targets are absent and would be excluded rather than scored as negatives.",
        ],
        "artifacts": {
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metric_path),
            "conformer_coverage_sha256": sha256(conformer_path),
        },
        "claim_status": "PUBLIC_CHECKPOINT_COVERAGE_SUBSET_COMPARATOR_ONLY",
    }
    summary_path = OUT / "SCOPE_DTI_OLD_DRUG_ENTITY_COLD_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
