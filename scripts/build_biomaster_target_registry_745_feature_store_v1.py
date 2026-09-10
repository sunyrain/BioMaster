#!/usr/bin/env python3
"""Build checkpoint-compatible features for the routed 745-target registry.

The builder reuses exact features from the frozen 384-target core and the
comprehensive 843-target training vocabulary.  Remaining ProtBERT features are
read from the frozen sequence-keyed ConPLex cache; only missing ESM2 features
are inferred.  Pocket context is copied from audited historical targets when
available and otherwise projected from the exact-sequence AlphaFold/P2Rank
registry without treating missing pockets as target exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_biomaster_deployment_augmentation_v1 import target_structure_row  # noqa: E402
from build_biomaster_odti_davis_feature_store_v1 import build_esm2  # noqa: E402
from score_biomaster_deployment_augmented_720x384_v1 import augmented_structure  # noqa: E402


REGISTRY = ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_REGISTRY_NON_GPCR_745_V3.csv.gz"
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
CORE = ROOT / "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1"
CORE_INDEX = CORE / "PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz"
CORE_PAIRS = CORE / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
CORE_PROTBERT = CORE / "PROJECT384_PROTBERT1024_FLOAT32_V1.npy"
CORE_ESM2 = ROOT / (
    "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy"
)
COMPREHENSIVE = ROOT / "outputs/retrain_20260901/comprehensive_training_v1"
COMPREHENSIVE_RELATIONS = COMPREHENSIVE / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
COMPREHENSIVE_PROTBERT = ROOT / (
    "outputs/biomaster_deployment_augmentation_v1/"
    "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy"
)
COMPREHENSIVE_ESM2 = ROOT / (
    "outputs/biomaster_deployment_augmentation_v1/"
    "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy"
)
PROTBERT_CACHE = ROOT / (
    "outputs/current_production_package_v2/conplex_target_calibration_v5_official/"
    "conplex_cache/ProtBert_features.h5"
)
BASE_PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
BASE_STRUCTURE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
AUGMENTED_STRUCTURE = ROOT / (
    "outputs/biomaster_deployment_augmentation_v1/"
    "TARGET_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz"
)
DEFAULT_OUT = ROOT / "outputs/retrain_20260901/target_registry_745_feature_store_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_from_lane(lane: pd.Series) -> pd.Series:
    return lane.map(
        {
            "ENZYME_BIOCHEMICAL": "enzyme",
            "KINASE_BIOCHEMICAL": "kinase",
            "NUCLEAR_EPIGENETIC_DOMAIN": "nuclear_epigenetic",
            "ION_CHANNEL_FUNCTIONAL": "ion_channel",
            "TRANSPORTER_MEMBRANE_FUNCTIONAL": "transporter",
            "EXTRACELLULAR_SPECIAL": "other_assayable",
            "NONCANONICAL_REVIEW": "other_assayable",
            "NON_GPCR_MEMBRANE_SPECIAL": "other_assayable",
        }
    ).fillna("other_assayable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--esm2-batch-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-existing-embeddings",
        action="store_true",
        help="when forcing a metadata/structure rebuild, retain the existing 745 ESM2 array",
    )
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    index_path = out / "TARGET_REGISTRY_745_FEATURE_INDEX_V1.csv.gz"
    protbert_path = out / "TARGET_REGISTRY_745_PROTBERT1024_FLOAT32_V1.npy"
    esm2_path = out / "TARGET_REGISTRY_745_ESM2_650M_1280_FLOAT32_V1.npy"
    structure_path = out / "TARGET_REGISTRY_745_STRUCTURE_CONTEXT_19D_V1.csv.gz"
    summary_path = out / "TARGET_REGISTRY_745_FEATURE_STORE_SUMMARY_V1.json"
    if not args.force and summary_path.is_file():
        previous = json.loads(summary_path.read_text())
        if previous.get("status") == "PASS" and all(
            path.is_file() for path in [index_path, protbert_path, esm2_path, structure_path]
        ):
            print(json.dumps(previous, ensure_ascii=False, indent=2))
            return

    required = [
        REGISTRY, UNIVERSE, CORE_INDEX, CORE_PAIRS, CORE_PROTBERT, CORE_ESM2,
        COMPREHENSIVE_RELATIONS, COMPREHENSIVE_PROTBERT, COMPREHENSIVE_ESM2,
        PROTBERT_CACHE, BASE_PAIRS, BASE_STRUCTURE, AUGMENTED_STRUCTURE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to fill missing ESM2 registry features")
    out.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(REGISTRY, low_memory=False).sort_values(
        "target_registry_index"
    ).reset_index(drop=True)
    if len(registry) != 745 or registry["target_chembl_id"].duplicated().any():
        raise RuntimeError("745-target registry contract failed")
    registry["target_feature_index"] = np.arange(len(registry), dtype=np.int16)
    registry["target_assay_family"] = family_from_lane(registry["assay_lane"])

    core_index = pd.read_csv(CORE_INDEX, low_memory=False)
    core_by_id = core_index.set_index("target_chembl_id")["target_feature_index"].astype(int).to_dict()
    core_protbert = np.load(CORE_PROTBERT, mmap_mode="r")
    core_esm2 = np.load(CORE_ESM2, mmap_mode="r")
    relations = pd.read_csv(
        COMPREHENSIVE_RELATIONS,
        usecols=[
            "target_chembl_id", "sequence_sha256", "target_feature_index",
            "target_assay_family",
        ],
        low_memory=False,
    ).drop_duplicates("target_chembl_id")
    comprehensive_by_id = relations.set_index("target_chembl_id")["target_feature_index"].astype(int).to_dict()
    comprehensive_hash_by_id = relations.set_index("target_chembl_id")["sequence_sha256"].astype(str).to_dict()
    comprehensive_family_by_id = relations.set_index("target_chembl_id")["target_assay_family"].astype(str).to_dict()
    comprehensive_protbert = np.load(COMPREHENSIVE_PROTBERT, mmap_mode="r")
    comprehensive_esm2 = np.load(COMPREHENSIVE_ESM2, mmap_mode="r")

    protbert = np.empty((745, 1024), dtype=np.float32)
    esm2 = np.empty((745, 1280), dtype=np.float32)
    protbert_route: list[str] = []
    esm2_route: list[str] = []
    family_route: list[str] = []
    esm2_missing_positions: list[int] = []
    esm2_missing_sequences: list[str] = []
    with h5py.File(PROTBERT_CACHE, "r") as cache:
        for position, row in enumerate(registry.itertuples(index=False)):
            target_id = str(row.target_chembl_id)
            sequence = str(row.sequence)
            sequence_hash = str(row.sequence_sha256)
            if target_id in core_by_id:
                source = core_by_id[target_id]
                protbert[position] = core_protbert[source]
                esm2[position] = core_esm2[source]
                protbert_route.append("FROZEN_384_TARGET_ID_COPY")
                esm2_route.append("FROZEN_384_TARGET_ID_COPY")
                family_route.append(str(core_index.loc[
                    core_index["target_chembl_id"].astype(str).eq(target_id),
                    "target_assay_family",
                ].iloc[0]))
            else:
                comprehensive_index = comprehensive_by_id.get(target_id)
                if comprehensive_index is not None and comprehensive_hash_by_id[target_id] == sequence_hash:
                    protbert[position] = comprehensive_protbert[comprehensive_index]
                    esm2[position] = comprehensive_esm2[comprehensive_index]
                    protbert_route.append("COMPREHENSIVE_EXACT_SEQUENCE_COPY")
                    esm2_route.append("COMPREHENSIVE_EXACT_SEQUENCE_COPY")
                    family_route.append(comprehensive_family_by_id[target_id])
                else:
                    if sequence not in cache:
                        raise RuntimeError(f"ProtBERT cache miss outside frozen core: {target_id}")
                    protbert[position] = np.asarray(cache[sequence], dtype=np.float32)
                    protbert_route.append("FROZEN_PROTBERT_EXACT_SEQUENCE_CACHE")
                    esm2_route.append("NEW_ESM2_EXACT_SEQUENCE_INFERENCE")
                    esm2_missing_positions.append(position)
                    esm2_missing_sequences.append(sequence)
                    family_route.append(str(registry.iloc[position]["target_assay_family"]))

    if esm2_missing_positions:
        if args.reuse_existing_embeddings and esm2_path.is_file():
            previous_esm2 = np.load(esm2_path, mmap_mode="r")
            if previous_esm2.shape != (745, 1280):
                raise RuntimeError("existing registry ESM2 array has the wrong shape")
            missing = np.asarray(esm2_missing_positions, dtype=np.int64)
            esm2[missing] = previous_esm2[missing]
            used_lengths = []
        else:
            new_esm2, used_lengths = build_esm2(
                esm2_missing_sequences, batch_size=args.esm2_batch_size
            )
            esm2[np.asarray(esm2_missing_positions, dtype=np.int64)] = new_esm2
    else:
        used_lengths = []

    base_pairs = pd.read_csv(
        BASE_PAIRS, usecols=["calibration_pair_id", "target_chembl_id"], low_memory=False
    )
    base_structure = pd.read_csv(BASE_STRUCTURE, low_memory=False)
    feature_columns = [
        column for column in base_structure.columns
        if column not in {"calibration_pair_id", "structure_mask"}
    ]
    base_by_id = base_pairs.merge(
        base_structure, on="calibration_pair_id", how="inner", validate="one_to_one"
    ).drop_duplicates("target_chembl_id").set_index("target_chembl_id")
    # Reproduce the exact structure contract used when Stage-A was selected.
    # Nineteen core targets had no safe structure in the original pair-aligned
    # table but were filled by the audited deployment augmentation.  Taking the
    # first historical row by target would silently lose those replacements.
    core_pairs = pd.read_csv(CORE_PAIRS, low_memory=False)
    core_structure_values, core_structure_mask, _ = augmented_structure(core_pairs)
    core_first = core_pairs.drop_duplicates("target_chembl_id").index.to_numpy(dtype=np.int64)
    core_structure_by_id = {
        str(core_pairs.iloc[position]["target_chembl_id"]): (
            core_structure_values[position], float(core_structure_mask[position])
        )
        for position in core_first
    }
    augmented_by_id = pd.read_csv(AUGMENTED_STRUCTURE, low_memory=False).drop_duplicates(
        "target_chembl_id"
    ).set_index("target_chembl_id")
    universe = pd.read_csv(UNIVERSE, low_memory=False).set_index("target_chembl_id")
    structure_rows = []
    for row in registry.itertuples(index=False):
        target_id = str(row.target_chembl_id)
        if target_id in core_structure_by_id:
            raw_values, mask = core_structure_by_id[target_id]
            values = {
                column: float(raw_values[column_index])
                for column_index, column in enumerate(feature_columns)
            }
            route = "EXACT_FROZEN_384_AUGMENTED_STRUCTURE_COPY"
        elif target_id in base_by_id.index:
            source = base_by_id.loc[target_id]
            values = {column: float(source[column]) for column in feature_columns}
            mask = float(source["structure_mask"])
            route = "EXACT_FROZEN_TARGET_STRUCTURE_COPY"
        elif target_id in augmented_by_id.index:
            source = augmented_by_id.loc[target_id]
            values = {column: float(source[column]) for column in feature_columns}
            mask = float(source["structure_mask"])
            route = str(source.get("structure_route", "EXACT_AUGMENTED_TARGET_STRUCTURE_COPY"))
        else:
            source = universe.loc[target_id].copy()
            source["structure_ready"] = bool(source.get("structure_ready_permissive", False))
            projected = target_structure_row(source, feature_columns)
            mask = float(projected.pop("structure_mask"))
            values = projected
            route = (
                "EXACT_SEQUENCE_ALPHAFOLD_P2RANK_19D_PROJECTION"
                if mask > 0 else "NO_SAFE_POCKET_MASKED"
            )
        structure_rows.append(
            {
                "target_feature_index": int(row.target_feature_index),
                "target_chembl_id": target_id,
                "structure_mask": mask,
                "structure_route": route,
                **values,
            }
        )
    structure = pd.DataFrame(structure_rows)
    registry["protbert_feature_route"] = protbert_route
    registry["esm2_feature_route"] = esm2_route
    registry["target_assay_family"] = family_route
    registry["structure_mask"] = structure["structure_mask"].to_numpy(dtype=np.float32)
    registry["resolved_feature_contract"] = "PROTBERT1024_ESM2_1280_STRUCTURE19D"

    checks = {
        "exact_745_targets": len(registry) == 745,
        "dense_target_feature_index": registry["target_feature_index"].tolist() == list(range(745)),
        "protbert_shape": protbert.shape == (745, 1024),
        "esm2_shape": esm2.shape == (745, 1280),
        "structure_shape": structure[feature_columns].shape == (745, 19),
        "all_features_finite": bool(
            np.isfinite(protbert).all()
            and np.isfinite(esm2).all()
            and np.isfinite(structure[feature_columns].to_numpy(dtype=np.float32)).all()
        ),
        "all_targets_have_protbert": len(protbert_route) == 745,
        "all_targets_have_esm2": len(esm2_route) == 745,
        "pocket_context_not_used_as_exclusion": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    np.save(protbert_path, protbert, allow_pickle=False)
    np.save(esm2_path, esm2, allow_pickle=False)
    registry.to_csv(index_path, index=False, compression={"method": "gzip", "mtime": 0})
    structure.to_csv(structure_path, index=False, compression={"method": "gzip", "mtime": 0})
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_ROUTED_TARGET_REGISTRY_745_FEATURE_STORE_V1",
        "counts": {
            "targets": 745,
            "target_warm": int(registry["in_supervised_training_target_vocabulary"].sum()),
            "target_cold": int((~registry["in_supervised_training_target_vocabulary"].astype(bool)).sum()),
            "core_target_id_copies": int(sum(route == "FROZEN_384_TARGET_ID_COPY" for route in protbert_route)),
            "comprehensive_exact_copies": int(sum(route == "COMPREHENSIVE_EXACT_SEQUENCE_COPY" for route in protbert_route)),
            "protbert_cache_copies": int(sum(route == "FROZEN_PROTBERT_EXACT_SEQUENCE_CACHE" for route in protbert_route)),
            "new_esm2_inferences": len(esm2_missing_positions),
            "structure_mask_one": int(structure["structure_mask"].gt(0).sum()),
            "structure_mask_zero": int(structure["structure_mask"].le(0).sum()),
        },
        "routes": registry["production_route"].value_counts().sort_index().to_dict(),
        "esm2_new_used_lengths": used_lengths,
        "structure_columns": feature_columns,
        "checks": checks,
        "claim_boundary": (
            "This feature store enables routed scoring of all 745 registry targets. "
            "It does not make all routes biologically comparable and does not convert "
            "unobserved drug-target pairs into negatives."
        ),
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [index_path, protbert_path, esm2_path, structure_path]
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
