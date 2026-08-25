#!/usr/bin/env python3
"""Rank the frozen current new relations with the FULL_FIT 2026 ensemble.

The primary score is a label-blind 55% binary-rank / 45% affinity-rank
fusion frozen from the earlier S5 target-cold validation.  The script scores
both a zero-structure control and the safe V1 target-structure context.  It
never uses the reported new-relation labels for model, structure, or fusion
selection.
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2  # noqa: E402
from train_biomaster_odti_v2 import predict  # noqa: E402


FULL_FIT_ROOT = ROOT / "outputs/biomaster_bindingdb_full_fit_2026_v1"
SEEDS = (20260816, 20260817, 20260820)
SIRT3_ROOT = ROOT / "outputs/biomaster_recent_quantitative_case_v3"
NOVEL_ROOT = ROOT / "outputs/biomaster_recent_novel_target_external_v1"
STRUCTURE_ROOT = ROOT / "outputs/biomaster_full_fit_current_new_relations_v1/structure"
SIRT3_STRUCTURE_ROOT = ROOT / "outputs/biomaster_recent_quantitative_case_v3_structure_v1"
STORE = ROOT / "outputs/old_drug_target_sota_v1"
DRUG_INDEX = STORE / "deployment_720x384_feature_store_v1/OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz"
DRUG_FEATURES = STORE / "deployment_720x384_feature_store_v1/OLD_DRUG_MORGAN2048_UINT8_V1.npy"
STRUCTURE_CONTRACT = STORE / "feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1_MANIFEST.json"
TRAIN_STRUCTURE = STORE / "feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
FUSION_SOURCE = SIRT3_ROOT / "INTERNAL_TARGET_COLD_FUSION_SELECTION_V3.csv"

SIRT3_INDEX = SIRT3_ROOT / "RECENT_QUANTITATIVE_TARGET_INDEX_V3.csv"
SIRT3_PROTBERT = SIRT3_ROOT / "RECENT_QUANTITATIVE_TARGET_PROTBERT1024_FLOAT32_V3.npy"
SIRT3_ESM2 = SIRT3_ROOT / "RECENT_QUANTITATIVE_TARGET_ESM2_650M_1280_FLOAT32_V3.npy"
NOVEL_INDEX = NOVEL_ROOT / "RECENT_NOVEL_TARGET_INDEX_V1.csv"
NOVEL_PROTBERT = NOVEL_ROOT / "RECENT_NOVEL_TARGET_PROTBERT1024_FLOAT32_V1.npy"
NOVEL_ESM2 = NOVEL_ROOT / "RECENT_NOVEL_TARGET_ESM2_T33_650M_1280_FLOAT32_V1.npy"

SEPHS2_PDB = (
    STRUCTURE_ROOT
    / "boltz_output/boltz_results_SEPHS2_Q99611_CYS_PROXY/predictions/"
      "SEPHS2_Q99611_CYS_PROXY/SEPHS2_Q99611_CYS_PROXY_model_0.pdb"
)
SEPHS2_CONFIDENCE = SEPHS2_PDB.parent / "confidence_SEPHS2_Q99611_CYS_PROXY_model_0.json"
SEPHS2_P2RANK = STRUCTURE_ROOT / "p2rank/SEPHS2/SEPHS2_Q99611_CYS_PROXY_model_0.pdb_predictions.csv"
SEPHS2_DESCRIPTOR = STRUCTURE_ROOT / "p2rank/SEPHS2/SEPHS2_Q99611_CYS_PROXY_model_0.pdb_pocket_descriptors.csv.gz"
VPS37C_PDB = STRUCTURE_ROOT / "source/AF-A5D8V6-F1-model_v6.pdb"
VPS37C_P2RANK = STRUCTURE_ROOT / "p2rank/VPS37C/AF-A5D8V6-F1-model_v6.pdb_predictions.csv"
VPS37C_DESCRIPTOR = STRUCTURE_ROOT / "p2rank/VPS37C/AF-A5D8V6-F1-model_v6.pdb_pocket_descriptors.csv.gz"

AFFINITY_WEIGHT = 0.45

RELATIONS = (
    {
        "case_id": "SIRT3_SPR_2026",
        "gene_symbol": "SIRT3",
        "drug_name": "binimetinib",
        "drug_inchikey": "ACWZRVQXLIRSDF-UHFFFAOYSA-N",
        "reported_kd": 21.28,
        "reported_kd_unit": "nM",
        "reported_pkd": -np.log10(21.28e-9),
        "cold_start_class": "TARGET_COLD_SINGLE",
    },
    {
        "case_id": "SIRT3_SPR_2026",
        "gene_symbol": "SIRT3",
        "drug_name": "olaparib",
        "drug_inchikey": "FDLYAMZZIXQODN-UHFFFAOYSA-N",
        "reported_kd": 84.0,
        "reported_kd_unit": "nM",
        "reported_pkd": -np.log10(84.0e-9),
        "cold_start_class": "TARGET_COLD_SINGLE",
    },
    {
        "case_id": "SEPHS2_POSTIT_2024",
        "gene_symbol": "SEPHS2",
        "drug_name": "dasatinib",
        "drug_inchikey": "ZBNZXTGUTAYRHI-UHFFFAOYSA-N",
        "reported_kd": 9.1,
        "reported_kd_unit": "uM",
        "reported_pkd": -np.log10(9.1e-6),
        "cold_start_class": "TARGET_COLD_SINGLE",
    },
    {
        "case_id": "VPS37C_POSTIT_2024",
        "gene_symbol": "VPS37C",
        "drug_name": "hydroxychloroquine",
        "drug_inchikey": "XXSMGPRMXLTPCZ-UHFFFAOYSA-N",
        "reported_kd": 16.9,
        "reported_kd_unit": "uM",
        "reported_pkd": -np.log10(16.9e-6),
        "cold_start_class": "DOUBLE_COLD",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def rank_unit(frame: pd.DataFrame, score: str) -> np.ndarray:
    return frame.groupby("case_id", sort=False)[score].rank(method="average", pct=True).to_numpy(np.float64)


def load_targets() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sirt3 = pd.read_csv(SIRT3_INDEX, low_memory=False).iloc[[0]].copy()
    novel = pd.read_csv(NOVEL_INDEX, low_memory=False).copy()
    novel.insert(1, "case_id", ["SEPHS2_POSTIT_2024", "VPS37C_POSTIT_2024"])
    novel.insert(2, "case_tier", ["PRIMARY_SHOWCASE", "MANDATORY_CO_REPORT"])
    common = [
        "case_id", "case_tier", "uniprot_accession", "gene_symbol",
        "protein_name", "target_assay_family", "sequence_length", "sequence_sha256",
    ]
    targets = pd.concat([sirt3[common], novel[common]], ignore_index=True)
    targets.insert(0, "target_feature_index", np.arange(len(targets), dtype=np.int64))
    protbert = np.concatenate([np.load(SIRT3_PROTBERT), np.load(NOVEL_PROTBERT)], axis=0).astype(np.float32)
    esm2 = np.concatenate([np.load(SIRT3_ESM2), np.load(NOVEL_ESM2)], axis=0).astype(np.float32)
    if protbert.shape != (3, 1024) or esm2.shape != (3, 1280):
        raise RuntimeError(f"target feature shape failure: {protbert.shape}, {esm2.shape}")
    if not np.isfinite(protbert).all() or not np.isfinite(esm2).all():
        raise RuntimeError("target feature table contains non-finite values")
    return targets, protbert, esm2


def structure_features(targets: pd.DataFrame, out: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    contract = json.loads(STRUCTURE_CONTRACT.read_text())
    columns = [str(value) for value in contract["feature_columns"]]
    sirt3_manifest_path = SIRT3_STRUCTURE_ROOT / "RECENT_QUANTITATIVE_TARGET_STRUCTURE_MANIFEST_V1.json"
    sirt3_manifest = json.loads(sirt3_manifest_path.read_text())
    if sirt3_manifest.get("status") != "PASS":
        raise RuntimeError("frozen SIRT3 structure feature audit is not PASS")
    sirt3_table = pd.read_csv(
        SIRT3_STRUCTURE_ROOT / "RECENT_QUANTITATIVE_TARGET_STRUCTURE_FEATURES_V1.csv",
        low_memory=False,
    )
    if len(sirt3_table) != 1 or float(sirt3_table.iloc[0]["structure_mask"]) != 1.0:
        raise RuntimeError("SIRT3 structure feature row is invalid")

    training = pd.read_csv(TRAIN_STRUCTURE, usecols=["structure_mask", *columns], low_memory=False)
    active = training.loc[training["structure_mask"] > 0, columns]
    probability_min = float(active["pocket_top_probability"].min())
    probability_max = float(active["pocket_top_probability"].max())

    sephs2_prediction = pd.read_csv(SEPHS2_P2RANK, low_memory=False)
    sephs2_prediction.columns = [str(value).strip() for value in sephs2_prediction.columns]
    sephs2_descriptor = pd.read_csv(SEPHS2_DESCRIPTOR, low_memory=False)
    sephs2_descriptor.columns = [str(value).strip() for value in sephs2_descriptor.columns]
    sephs2_top = sephs2_prediction.sort_values("rank").iloc[0]
    sephs2_desc = sephs2_descriptor.sort_values("rank").iloc[0]
    sephs2_probability = float(sephs2_top["probability"])
    sephs2_confidence = json.loads(SEPHS2_CONFIDENCE.read_text())

    vps_prediction = pd.read_csv(VPS37C_P2RANK, low_memory=False)
    vps_descriptor = pd.read_csv(VPS37C_DESCRIPTOR, low_memory=False)
    if len(vps_prediction) or len(vps_descriptor):
        raise RuntimeError("VPS37C was expected to have no P2Rank pocket")

    rows = []
    for target in targets.itertuples(index=False):
        if target.case_id == "SIRT3_SPR_2026":
            row = {"case_id": target.case_id, "structure_mask": 1.0}
            row.update({column: float(sirt3_table.iloc[0][column]) for column in columns})
        else:
            row = {"case_id": target.case_id, "structure_mask": 0.0}
            row.update({column: 0.0 for column in columns})
        rows.append(row)
    table = pd.DataFrame(rows)
    feature_path = out / "CURRENT_NEW_TARGET_STRUCTURE_FEATURES_V1.csv"
    table.to_csv(feature_path, index=False)

    audits = [
        {
            "case_id": "SIRT3_SPR_2026",
            "gene_symbol": "SIRT3",
            "structure_source": "AlphaFold_DB_v6",
            "sequence_status": "exact_399_of_399",
            "p2rank_probability": float(sirt3_manifest["p2rank"]["probability"]),
            "p2rank_score": float(sirt3_manifest["p2rank"]["score"]),
            "p2rank_volume": float(sirt3_manifest["p2rank"]["volume"]),
            "p2rank_num_residues": int(sirt3_manifest["p2rank"]["num_residues"]),
            "pocket_tier": str(sirt3_manifest["p2rank"]["tier"]),
            "structure_mask": 1,
            "decision": "ENABLED_IN_PRIMARY_SCORE",
        },
        {
            "case_id": "SEPHS2_POSTIT_2024",
            "gene_symbol": "SEPHS2",
            "structure_source": "Boltz2_single_sequence_Cys_proxy_for_Sec60",
            "structure_sha256": sha256(SEPHS2_PDB),
            "boltz_confidence": sephs2_confidence,
            "sequence_status": "448_of_448_after_U60_to_C_proxy",
            "p2rank_probability": sephs2_probability,
            "p2rank_score": float(sephs2_top["score"]),
            "p2rank_volume": float(sephs2_desc["volume"]),
            "p2rank_num_residues": int(sephs2_desc["num_residues"]),
            "pocket_tier": "C_p2rank_weak_review_pocket",
            "structure_mask": 0,
            "decision": "AUDITED_BUT_MASKED_BELOW_TRAINING_ACTIVE_PROBABILITY_RANGE",
        },
        {
            "case_id": "VPS37C_POSTIT_2024",
            "gene_symbol": "VPS37C",
            "structure_source": "AlphaFold_DB_v6",
            "structure_sha256": sha256(VPS37C_PDB),
            "sequence_status": "exact_355_of_355",
            "p2rank_probability": None,
            "p2rank_score": None,
            "p2rank_volume": None,
            "p2rank_num_residues": 0,
            "pocket_tier": "D_no_p2rank_pocket",
            "structure_mask": 0,
            "decision": "AUDITED_BUT_MASKED_NO_POCKET",
        },
    ]
    audit = {
        "status": "PASS",
        "policy": "Only exact, model-contract-compatible target context within the active training range is enabled.",
        "training_active_pocket_probability_range": [probability_min, probability_max],
        "targets_pocket_audited": 3,
        "targets_enabled": 1,
        "targets_masked": 2,
        "targets": audits,
        "feature_table": str(feature_path.relative_to(ROOT)),
        "feature_table_sha256": sha256(feature_path),
        "contract": str(STRUCTURE_CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(STRUCTURE_CONTRACT),
        "sirt3_manifest": str(sirt3_manifest_path.relative_to(ROOT)),
        "sirt3_manifest_sha256": sha256(sirt3_manifest_path),
    }
    audit_path = out / "CURRENT_NEW_TARGET_POCKET_AUDIT_V1.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=json_safe) + "\n")
    return (
        table[columns].to_numpy(dtype=np.float32),
        table["structure_mask"].to_numpy(dtype=np.float32),
        {**audit, "audit_path": str(audit_path.relative_to(ROOT)), "audit_sha256": sha256(audit_path)},
    )


def inference_arrays(frame: pd.DataFrame, checkpoint: dict[str, object]) -> dict[str, object]:
    normalization = checkpoint["normalization"]
    family_lookup = {str(name): index for index, name in enumerate(checkpoint["families"])}
    family = frame["target_assay_family"].astype(str).map(family_lookup)
    family = family.fillna(family_lookup.get("__UNK__", -1))
    if family.lt(0).any():
        raise RuntimeError("target family is unavailable and checkpoint has no __UNK__")
    return {
        "families": checkpoint["families"],
        "family_index": family.to_numpy(dtype=np.int64),
        "drug_aux_mean": np.zeros(0, dtype=np.float32),
        "drug_aux_std": np.ones(0, dtype=np.float32),
        "target_mean": np.asarray(normalization["target_mean"], dtype=np.float32),
        "target_std": np.asarray(normalization["target_std"], dtype=np.float32),
        "target_aux_mean": np.asarray(normalization["target_aux_mean"], dtype=np.float32),
        "target_aux_std": np.asarray(normalization["target_aux_std"], dtype=np.float32),
        "target_token_mean": np.asarray(normalization["target_token_mean"], dtype=np.float32),
        "target_token_std": np.asarray(normalization["target_token_std"], dtype=np.float32),
        "conplex": frame["conplex_score"].to_numpy(dtype=np.float32),
        "conplex_mean": float(normalization["conplex_mean"]),
        "conplex_std": float(normalization["conplex_std"]),
        "affinity": frame["mean_pchembl"].to_numpy(dtype=np.float32),
        "affinity_lower": frame["min_pchembl"].to_numpy(dtype=np.float32),
        "affinity_upper": frame["max_pchembl"].to_numpy(dtype=np.float32),
        "affinity_mean": float(normalization["affinity_mean"]),
        "affinity_std": float(normalization["affinity_std"]),
        "structure_mean": np.asarray(normalization["structure_mean"], dtype=np.float32),
        "structure_std": np.asarray(normalization["structure_std"], dtype=np.float32),
    }


def load_checkpoints(
    full_fit_root: Path,
    checkpoint_dir_template: str,
    checkpoint_filename: str,
) -> list[tuple[int, Path, dict[str, object]]]:
    loaded = []
    for seed in SEEDS:
        run = full_fit_root / checkpoint_dir_template.format(seed=seed)
        summary_path = run / "FULL_FIT_RUN_SUMMARY_V1.json"
        checkpoint_path = run / checkpoint_filename
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "PASS" or summary.get("new_relation_labels_used_for_training_or_selection") is not False:
            raise RuntimeError(f"FULL_FIT audit failed: {summary_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded.append((seed, checkpoint_path, checkpoint))
    return loaded


def baseline_ranks() -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    sirt3 = pd.read_csv(SIRT3_STRUCTURE_ROOT / "RECENT_QUANTITATIVE_POSITIVE_RANKS_V3.csv")
    for row in sirt3.itertuples(index=False):
        result[("SIRT3_SPR_2026", str(row.old_drug_inchikey))] = int(row.rank_within_target_720)
    novel = pd.read_csv(NOVEL_ROOT / "RECENT_NOVEL_TARGET_POSITIVE_CONTROL_RANKS_V1.csv")
    case_by_gene = {"SEPHS2": "SEPHS2_POSTIT_2024", "VPS37C": "VPS37C_POSTIT_2024"}
    for row in novel.itertuples(index=False):
        result[(case_by_gene[str(row.gene_symbol)], str(row.ligand_inchikey))] = int(row.rank_within_target_720)
    return result


def training_entity_sets(
    full_fit_root: Path,
    known_relations_path: Path | None = None,
) -> tuple[set[str], set[str]]:
    """Resolve exact entities actually used by either base or augmented FULL_FIT."""

    if known_relations_path is not None:
        entities = pd.read_csv(
            known_relations_path,
            usecols=["parent_standard_inchi_key", "primary_gene"],
            low_memory=False,
        )
        drugs = set(entities["parent_standard_inchi_key"].dropna().astype(str))
        genes = set(entities["primary_gene"].dropna().astype(str).str.upper()) - {"", "NAN"}
        return drugs, genes
    base_pairs = pd.read_csv(
        STORE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz",
        usecols=["parent_standard_inchi_key", "primary_gene"],
        low_memory=False,
    )
    bdb_pairs = pd.read_csv(
        ROOT / "outputs/biomaster_bindingdb_affinity_feature_package_v1/BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz",
        usecols=["parent_standard_inchi_key", "primary_gene"],
        low_memory=False,
    )
    frames = [base_pairs, bdb_pairs]
    # Augmented checkpoints record the recovered relation source in every run
    # summary.  Load it only when that source is part of this FULL_FIT root.
    summary_path = full_fit_root / f"FULL_FIT_2026__seed_{SEEDS[0]}" / "FULL_FIT_RUN_SUMMARY_V1.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for source in summary.get("sources", {}):
            if str(source).endswith("RECOVERED_CHEMBL37_RELATIONS_V1.csv.gz"):
                source_path = ROOT / str(source)
                frames.append(pd.read_csv(
                    source_path,
                    usecols=["parent_standard_inchi_key", "primary_gene"],
                    low_memory=False,
                ))
    entities = pd.concat(frames, ignore_index=True)
    drugs = set(entities["parent_standard_inchi_key"].dropna().astype(str))
    genes = set(entities["primary_gene"].dropna().astype(str).str.upper()) - {"", "NAN"}
    return drugs, genes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/biomaster_full_fit_current_new_relations_v1")
    parser.add_argument("--full-fit-root", default=str(FULL_FIT_ROOT))
    parser.add_argument("--checkpoint-dir-template", default="FULL_FIT_2026__seed_{seed}")
    parser.add_argument("--checkpoint-filename", default="FULL_FIT_MODEL_BINDINGDB_AFFINITY_V1.pt")
    parser.add_argument("--known-relations", default="")
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    full_fit_root = Path(args.full_fit_root).resolve()
    known_relations_path = Path(args.known_relations).resolve() if args.known_relations else None
    out.mkdir(parents=True, exist_ok=True)
    required = [
        DRUG_INDEX, DRUG_FEATURES, STRUCTURE_CONTRACT, TRAIN_STRUCTURE, FUSION_SOURCE,
        SIRT3_INDEX, SIRT3_PROTBERT, SIRT3_ESM2, NOVEL_INDEX, NOVEL_PROTBERT, NOVEL_ESM2,
        SEPHS2_PDB, SEPHS2_CONFIDENCE, SEPHS2_P2RANK, SEPHS2_DESCRIPTOR,
        VPS37C_PDB, VPS37C_P2RANK, VPS37C_DESCRIPTOR,
    ]
    if known_relations_path is not None:
        required.append(known_relations_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    fusion = pd.read_csv(FUSION_SOURCE, low_memory=False)
    winner = fusion.sort_values(["selection_value", "affinity_weight"], ascending=[False, True]).iloc[0]
    if not np.isclose(float(winner["affinity_weight"]), AFFINITY_WEIGHT):
        raise RuntimeError("frozen target-cold fusion source is not aligned to 0.45")

    targets, target_protbert, target_esm2 = load_targets()
    drug_index = pd.read_csv(DRUG_INDEX, low_memory=False).sort_values("drug_feature_index")
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    if len(drug_index) != 720 or drug_features.shape != (720, 2048):
        raise RuntimeError("old-drug retrieval universe is not 720x2048")
    chunks = []
    for target in targets.itertuples(index=False):
        chunk = drug_index.copy()
        chunk["target_feature_index"] = int(target.target_feature_index)
        for column in [
            "case_id", "case_tier", "uniprot_accession", "gene_symbol",
            "protein_name", "target_assay_family", "sequence_sha256",
        ]:
            chunk[column] = getattr(target, column)
        chunk["calibration_pair_id"] = [f"{target.case_id}__{value}" for value in chunk["ligand_inchikey"]]
        chunk["conplex_score"] = 0.0
        chunk["mean_pchembl"] = np.nan
        chunk["min_pchembl"] = np.nan
        chunk["max_pchembl"] = np.nan
        chunk["binary_label"] = 0
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    if len(frame) != 2160 or frame["calibration_pair_id"].duplicated().any():
        raise RuntimeError("3x720 retrieval frame contract failed")

    target_structure, target_structure_mask, structure_audit = structure_features(targets, out)
    target_rows = frame["target_feature_index"].to_numpy(dtype=np.int64)
    safe_structure = target_structure[target_rows]
    safe_mask = target_structure_mask[target_rows]
    zero_structure = np.zeros_like(safe_structure, dtype=np.float32)
    zero_mask = np.zeros(len(frame), dtype=np.float32)
    positions = np.arange(len(frame), dtype=np.int64)
    checkpoints = load_checkpoints(
        full_fit_root, args.checkpoint_dir_template, args.checkpoint_filename
    )
    training_drugs, training_genes = training_entity_sets(full_fit_root, known_relations_path)
    selected_fit_summary = json.loads(
        (
            full_fit_root / args.checkpoint_dir_template.format(seed=SEEDS[0])
            / "FULL_FIT_RUN_SUMMARY_V1.json"
        ).read_text()
    )
    selected_fit_counts = selected_fit_summary.get(
        "fit_counts", selected_fit_summary.get("split_and_fit_counts", {})
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_audit = []

    for mode, structure, structure_mask in (
        ("zero", zero_structure, zero_mask),
        ("safe", safe_structure, safe_mask),
    ):
        for model_index, (seed, checkpoint_path, checkpoint) in enumerate(checkpoints):
            config = ODTIV2Config(**checkpoint["config"])
            if config.structure_input_dim != 19:
                raise RuntimeError("FULL_FIT checkpoint does not follow the V1 19D structure contract")
            model = RoutedInteractionRankerV2(len(checkpoint["families"]), config, use_conplex=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            arrays = inference_arrays(frame, checkpoint)
            output = predict(
                model, positions, frame, drug_features, None, target_protbert, target_esm2,
                None, None, None, None, config.target_token_max_len, None, None,
                structure, structure_mask, arrays, device, 4096,
            )
            binary = f"{mode}_binary_{model_index}"
            affinity = f"{mode}_affinity_{model_index}"
            binary_rank = f"{mode}_binary_rank_{model_index}"
            affinity_rank = f"{mode}_affinity_rank_{model_index}"
            frame[binary] = output["final_logit"]
            frame[affinity] = output["affinity"] * arrays["affinity_std"] + arrays["affinity_mean"]
            frame[binary_rank] = rank_unit(frame, binary)
            frame[affinity_rank] = rank_unit(frame, affinity)
            seed_score = f"{mode}_seed_fusion_score_{seed}"
            seed_rank = f"{mode}_seed_rank_{seed}"
            frame[seed_score] = (1.0 - AFFINITY_WEIGHT) * frame[binary_rank] + AFFINITY_WEIGHT * frame[affinity_rank]
            frame[seed_rank] = frame.groupby("case_id")[seed_score].rank(method="min", ascending=False).astype(int)
            if mode == "safe":
                checkpoint_audit.append(
                    {
                        "seed": seed,
                        "path": str(checkpoint_path.relative_to(ROOT)),
                        "sha256": sha256(checkpoint_path),
                        "parameter_count": int(sum(value.numel() for value in model.parameters())),
                        "conplex_enabled": False,
                        "structure_input_dim": config.structure_input_dim,
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        frame[f"{mode}_binary_rank_ensemble"] = frame[
            [f"{mode}_binary_rank_{index}" for index in range(len(checkpoints))]
        ].mean(axis=1)
        frame[f"{mode}_affinity_rank_ensemble"] = frame[
            [f"{mode}_affinity_rank_{index}" for index in range(len(checkpoints))]
        ].mean(axis=1)
        frame[f"{mode}_fusion_score"] = (
            (1.0 - AFFINITY_WEIGHT) * frame[f"{mode}_binary_rank_ensemble"]
            + AFFINITY_WEIGHT * frame[f"{mode}_affinity_rank_ensemble"]
        )
        frame[f"{mode}_rank_within_target_720"] = frame.groupby("case_id")[f"{mode}_fusion_score"].rank(
            method="min", ascending=False
        ).astype(int)

    frame["structure_mask"] = safe_mask
    score_path = out / "FULL_FIT_CURRENT_NEW_3X720_SCORES_V1.csv.gz"
    frame.to_csv(score_path, index=False)
    top20 = frame.loc[frame["safe_rank_within_target_720"] <= 20].sort_values(
        ["case_id", "safe_rank_within_target_720"]
    )
    top20_path = out / "FULL_FIT_CURRENT_NEW_TOP20_V1.csv"
    top20.to_csv(top20_path, index=False)

    prior = baseline_ranks()
    relation_rows = []
    seed_rank_columns = [f"safe_seed_rank_{seed}" for seed in SEEDS]
    for relation in RELATIONS:
        matched = frame.loc[
            frame["case_id"].eq(relation["case_id"])
            & frame["ligand_inchikey"].eq(relation["drug_inchikey"])
        ]
        if len(matched) != 1:
            raise RuntimeError(f"relation old drug is not exactly once in 720: {relation}")
        row = matched.iloc[0]
        seed_ranks = [int(row[column]) for column in seed_rank_columns]
        safe_rank = int(row["safe_rank_within_target_720"])
        zero_rank = int(row["zero_rank_within_target_720"])
        drug_seen = str(relation["drug_inchikey"]) in training_drugs
        target_seen = str(relation["gene_symbol"]).upper() in training_genes
        if drug_seen and target_seen:
            dynamic_cold_class = "BOTH_SEEN_RELATION_HELD_OUT"
        elif drug_seen:
            dynamic_cold_class = "TARGET_COLD_SINGLE"
        elif target_seen:
            dynamic_cold_class = "DRUG_COLD_SINGLE"
        else:
            dynamic_cold_class = "DOUBLE_COLD"
        relation_rows.append(
            {
                **relation,
                "prior_declared_cold_start_class": relation["cold_start_class"],
                "cold_start_class": dynamic_cold_class,
                "drug_seen_in_selected_full_fit": drug_seen,
                "target_seen_in_selected_full_fit": target_seen,
                "structure_mask": int(row["structure_mask"]),
                "full_fit_rank_within_target_720": safe_rank,
                "full_fit_percentile_top": 1.0 - (safe_rank - 1) / 720.0,
                "full_fit_top_5_percent": safe_rank <= 36,
                "full_fit_top_10_percent": safe_rank <= 72,
                "zero_structure_rank_within_target_720": zero_rank,
                "structure_rank_change_positive_is_better": zero_rank - safe_rank,
                "seed_rank_min": min(seed_ranks),
                "seed_rank_median": float(np.median(seed_ranks)),
                "seed_rank_max": max(seed_ranks),
                **{column: int(row[column]) for column in seed_rank_columns},
                "prior_s5_rank_within_target_720": prior.get((relation["case_id"], relation["drug_inchikey"])),
                "rank_change_from_prior_s5_positive_is_better": (
                    prior[(relation["case_id"], relation["drug_inchikey"])] - safe_rank
                    if (relation["case_id"], relation["drug_inchikey"]) in prior else None
                ),
                "fusion_score": float(row["safe_fusion_score"]),
                "binary_rank_ensemble": float(row["safe_binary_rank_ensemble"]),
                "affinity_rank_ensemble": float(row["safe_affinity_rank_ensemble"]),
            }
        )
    relations = pd.DataFrame(relation_rows).sort_values(
        ["full_fit_rank_within_target_720", "case_id", "drug_name"]
    ).reset_index(drop=True)
    relation_path = out / "FULL_FIT_CURRENT_NEW_RELATION_RANKS_V1.csv"
    relations.to_csv(relation_path, index=False)

    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "FULL_FIT_2026_CURRENT_NEW_RELATION_RETRIEVAL_V1",
        "ranking_universe": "720 frozen old drugs independently within each target",
        "targets": targets.to_dict("records"),
        "relations": relation_rows,
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoint_audit,
        "full_fit_root": str(full_fit_root.relative_to(ROOT)),
        "full_fit_training_rows": int(selected_fit_counts.get(
            "full_refit_rows", selected_fit_counts.get(
                "full_fit_rows", selected_fit_counts.get("feature_available_fit_rows", 0)
            )
        )),
        "full_fit_unique_drugs": int(selected_fit_counts.get(
            "unique_drugs_full_refit", selected_fit_counts.get(
                "full_fit_unique_drugs", selected_fit_counts.get("unique_drugs", 0)
            )
        )),
        "full_fit_unique_targets": int(selected_fit_counts.get(
            "unique_targets_full_refit", selected_fit_counts.get(
                "full_fit_unique_targets", selected_fit_counts.get("unique_targets", 0)
            )
        )),
        "external_new_relation_labels_used_for_training_model_structure_or_fusion_selection": False,
        "fusion": {
            "binary_weight": 1.0 - AFFINITY_WEIGHT,
            "affinity_weight": AFFINITY_WEIGHT,
            "selection_source": str(FUSION_SOURCE.relative_to(ROOT)),
            "selection_source_sha256": sha256(FUSION_SOURCE),
            "selection_rows": int(winner["rows"]),
            "selection_targets": int(winner["targets"]),
        },
        "conplex_mode": "DISABLED_BY_ORIGINAL_W08_AND_FULL_FIT_CHECKPOINT_CONTRACT",
        "structure": structure_audit,
        "claim_boundary": (
            "FULL_FIT is a deployment fit with no held-out checkpoint-selection metric. "
            "Ranks are external frozen-positive readouts, not an unbiased estimate of prospective success rate."
        ),
        "artifacts": {
            "scores": str(score_path.relative_to(ROOT)),
            "scores_sha256": sha256(score_path),
            "top20": str(top20_path.relative_to(ROOT)),
            "top20_sha256": sha256(top20_path),
            "relation_ranks": str(relation_path.relative_to(ROOT)),
            "relation_ranks_sha256": sha256(relation_path),
        },
    }
    summary_path = out / "FULL_FIT_CURRENT_NEW_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe) + "\n")
    print(relations.to_string(index=False))
    print(json.dumps({"status": "PASS", "summary": str(summary_path), "scores": str(score_path)}, indent=2))


if __name__ == "__main__":
    main()
