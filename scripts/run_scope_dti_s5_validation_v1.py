#!/usr/bin/env python3
"""Run official lightweight SCOPE-DTI on the frozen S5 validation role."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCOPE_ROOT = ROOT / ".external/scope_dti_lightweight"
sys.path.insert(0, str(SCOPE_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from configs import get_cfg_defaults  # noqa: E402
from models import SCOPE as ScopeModel  # noqa: E402
from mol_graph import sdf_to_graphs  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics, split_masks  # noqa: E402
from run_scope_dti_old_drug_entity_cold_v1 import (  # noqa: E402
    CHECKPOINT_DIR,
    PAIRS,
    ScopePairDataset,
    custom_collate_fn,
    deterministic_sdf,
    integer_label_protein,
    predict,
    sha256,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
VOCABULARY = SCOPE_ROOT / "protein_targets/Total_predict.parquet"
OUT = BASE / "public_baselines_v1/scope_dti_s5_validation_v1"
WORKERS = 16


def build_one(smiles: str) -> tuple[str, object | None, str]:
    sdf, route = deterministic_sdf(smiles)
    if sdf is None:
        return smiles, None, route
    try:
        return smiles, sdf_to_graphs(sdf), route
    except Exception as error:
        return smiles, None, f"GRAPH_BUILD_FAILURE_{type(error).__name__}: {str(error)[:160]}"


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PAIRS, low_memory=False)
    masks = split_masks(data, "S5_OLD_DRUG_ENTITY_COLD", -1)
    role = data.loc[masks["valid"] & data["drug_feature_available"].to_numpy(dtype=bool)].copy()
    if len(role) != 16668:
        raise RuntimeError(f"S5 validation role changed: {len(role)}")
    vocabulary = pd.read_parquet(VOCABULARY)
    sequence_lookup = vocabulary.set_index("target_uniprot_id")["sequence"].astype(str).to_dict()
    role["scope_target_available"] = role["query_accession"].isin(sequence_lookup)
    smiles_values = sorted(role.loc[role["scope_target_available"], "model_ligand_smiles"].unique())
    graphs: dict[str, object] = {}
    conformer_rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, (smiles, graph, route) in enumerate(pool.map(build_one, smiles_values), start=1):
            available = graph is not None
            if available:
                graphs[smiles] = graph
            conformer_rows.append({
                "model_ligand_smiles": smiles,
                "scope_drug_graph_available": available,
                "conformer_route": route,
            })
            if index % 500 == 0:
                print(json.dumps({"conformers_processed": index, "total": len(smiles_values)}), flush=True)
    conformers = pd.DataFrame(conformer_rows)
    role = role.merge(conformers, on="model_ligand_smiles", how="left", validate="many_to_one")
    evaluable = role[
        role["scope_target_available"] & role["scope_drug_graph_available"].fillna(False)
    ].reset_index(drop=True)
    protein_encoding = {
        accession: torch.tensor(integer_label_protein(sequence), dtype=torch.float32)
        for accession, sequence in vocabulary[
            vocabulary["target_uniprot_id"].isin(evaluable["query_accession"])
        ][["target_uniprot_id", "sequence"]].itertuples(index=False)
    }
    loader = DataLoader(
        ScopePairDataset(evaluable, graphs, protein_encoding),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
    if len(checkpoints) != 5:
        raise RuntimeError("Expected five official Total checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_cfg_defaults()
    predictions = evaluable[[
        "calibration_pair_id", "query_accession", "target_chembl_id", "primary_gene",
        "parent_standard_inchi_key", "parent_molecule_chembl_id", "binary_label",
    ]].copy()
    for index, checkpoint in enumerate(checkpoints, start=1):
        model = ScopeModel(**config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device)
        predictions[f"scope_checkpoint_{index}"] = predict(model, loader, device)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"checkpoint": index, "rows": len(predictions)}), flush=True)
    checkpoint_columns = [f"scope_checkpoint_{index}" for index in range(1, 6)]
    predictions["scope_mean"] = predictions[checkpoint_columns].mean(axis=1)
    prediction_path = OUT / "SCOPE_DTI_S5_VALIDATION_PREDICTIONS_V1.csv.gz"
    conformer_path = OUT / "SCOPE_DTI_S5_VALIDATION_CONFORMER_COVERAGE_V1.csv.gz"
    predictions.to_csv(prediction_path, index=False)
    conformers.to_csv(conformer_path, index=False)
    score_metrics = metrics(predictions, predictions["scope_mean"].to_numpy(dtype=np.float64))
    checks = {
        "frozen_validation_role_exact_16668": len(role) == 16668,
        "all_predictions_finite": predictions[checkpoint_columns + ["scope_mean"]].notna().all().all(),
        "prediction_subset_has_both_classes": predictions["binary_label"].nunique() == 2,
        "no_missing_target_or_graph_silently_scored": len(predictions) == len(evaluable),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": str(device),
        "workers": WORKERS,
        "coverage": {
            "validation_pairs": int(len(role)),
            "validation_targets": int(role["query_accession"].nunique()),
            "scope_covered_targets": int(role.loc[role["scope_target_available"], "query_accession"].nunique()),
            "missing_target_accessions": sorted(role.loc[~role["scope_target_available"], "query_accession"].unique()),
            "unique_smiles_attempted": int(len(conformers)),
            "failed_unique_graphs": int((~conformers["scope_drug_graph_available"]).sum()),
            "evaluable_pairs": int(len(evaluable)),
        },
        "validation_metrics": score_metrics,
        "checks": {key: bool(value) for key, value in checks.items()},
        "artifacts": {
            "predictions_sha256": sha256(prediction_path),
            "conformer_coverage_sha256": sha256(conformer_path),
        },
        "claim_status": "VALIDATION_FEATURE_FOR_PREDECLARED_STACK_ONLY",
    }
    (OUT / "SCOPE_DTI_S5_VALIDATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
