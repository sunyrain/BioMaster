#!/usr/bin/env python3
"""Build a leakage-audited deployment augmentation for the frozen BioMaster fit.

The frozen 86,674-row benchmark table is never modified.  This package adds
eligible ChEMBL37 positive/weak-inactive relations for exact frozen old-drug
entities that are absent from the current FULL_FIT data.  Relation-level
development and test roles are assigned only when both the exact ligand and
the exact target retain a different fitting relation (strict double-warm).

Drug Morgan features are copied from the frozen 720-drug deployment store.
Exact target ProtBERT/ESM2 features are reused when already present and are
extracted label-blind only for genuinely new target sequences.  Existing
target structure context is copied exactly; otherwise audited AlphaFold/P2Rank
context from the ChEMBL37 target universe is projected into the frozen 19-D
structure contract.
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
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STORE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1"
DEPLOY = ROOT / "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1"
BDB = ROOT / "outputs/biomaster_bindingdb_affinity_feature_package_v1"
FULL_BDB = ROOT / "outputs/biomaster_bindingdb_full_training_subset_v1"
TARGET_ROOT = ROOT / "outputs/target_universe_ch37_v2"

BASE_PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
BASE_TARGET_INDEX = STORE / "TARGET_FEATURE_INDEX_V1.csv.gz"
BASE_STRUCTURE = STORE / "ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
BASE_MORGAN = BDB / "MORGAN2048_UINT8_COMBINED_V1.npy"
BASE_PROTBERT = BDB / "PROTBERT1024_FLOAT32_COMBINED_V1.npy"
BASE_ESM2 = BDB / "ESM2_650M_1280_FLOAT32_COMBINED_V1.npy"
BDB_PAIRS = BDB / "BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz"
BDB_NEW_TARGETS = BDB / "BINDINGDB_NEW_TARGET_FEATURE_INDEX_V1.csv.gz"
DEPLOY_DRUG_INDEX = DEPLOY / "OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz"
DEPLOY_MORGAN = DEPLOY / "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
ALL888_PAIRS = TARGET_ROOT / "chembl37_calibration_all888_v2/TARGET_ALL888_STRICT_BINDING_PAIRS_V2.csv.gz"
TARGET_UNIVERSE = TARGET_ROOT / "TARGET_SET_SEQUENCE_DTA_ALL_V2.csv"
DEFAULT_OUT = ROOT / "outputs/biomaster_deployment_augmentation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(sequence: object) -> str:
    clean = "".join(character for character in str(sequence).upper() if character.isalpha())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def broad_family(lane: object) -> str:
    value = str(lane).upper()
    if "KINASE" in value:
        return "kinase"
    if "ENZYME" in value:
        return "enzyme"
    if "NUCLEAR" in value or "EPIGENETIC" in value:
        return "nuclear_epigenetic"
    if "ION_CHANNEL" in value:
        return "ion_channel"
    if "TRANSPORTER" in value:
        return "transporter"
    return "other_assayable"


def murcko(smiles: object) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"invalid augmentation SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


def assign_roles(frame: pd.DataFrame, base_target_hashes: set[str]) -> pd.Series:
    """Deterministically create relation-unseen, entity-warm roles."""

    roles: dict[int, str] = {}
    for _, part in frame.groupby("parent_standard_inchi_key", sort=True):
        indices = list(part.sort_values("__role_hash").index)
        if len(indices) >= 3:
            roles[indices[0]] = "test"
            roles[indices[1]] = "dev"
            roles.update({index: "train" for index in indices[2:]})
        elif len(indices) == 2:
            roles[indices[0]] = "test"
            roles[indices[1]] = "train"
        else:
            roles[indices[0]] = "train"
    role = pd.Series(roles).reindex(frame.index)
    # A target outside the original 428 is warm only if another recovered
    # relation for the exact sequence remains in training.
    for _ in range(10):
        train_targets = base_target_hashes | set(frame.loc[role.eq("train"), "sequence_sha256"])
        move = frame.index[role.isin(["dev", "test"]) & ~frame["sequence_sha256"].isin(train_targets)]
        if len(move) == 0:
            break
        role.loc[move] = "train"
    return role


def target_structure_row(target: pd.Series, columns: list[str]) -> dict[str, float]:
    completed = str(target.get("p2rank_status", "")) == "completed"
    ready = bool(target.get("structure_ready", False)) and completed
    numeric = lambda name: float(pd.to_numeric(pd.Series([target.get(name)]), errors="coerce").iloc[0])
    values = {
        "structure_context_available": 1.0 if ready else 0.0,
        "receptor_experimental_holo": 0.0,
        "pocket_top_score": numeric("p2rank_top_score") if ready else 0.0,
        "pocket_top_probability": numeric("p2rank_top_probability") if ready else 0.0,
        "pocket_center_x": numeric("p2rank_center_x") if ready else 0.0,
        "pocket_center_y": numeric("p2rank_center_y") if ready else 0.0,
        "pocket_center_z": numeric("p2rank_center_z") if ready else 0.0,
        # The all-888 P2Rank export does not contain a calibrated volume.
        # Keep the semantic column at zero instead of substituting a different
        # quantity such as SAS-point count.
        "pocket_volume": 0.0,
        "pocket_num_residues": numeric("pocket_residue_count") if ready else 0.0,
        "pocket_puresnet_overlap": 0.0,
        "pocket_puresnet_jaccard": 0.0,
        "receptor_residue_count": numeric("sequence_length") if ready else 0.0,
        "experimental_entry_count_top20": 0.0,
        "candidate_holo_entry_count_top20": 0.0,
        "best_holo_resolution": 0.0,
        "best_holo_coverage": 0.0,
        "structure_bin_is_strict": 1.0 if ready and bool(target.get("structure_ready_strict", False)) else 0.0,
        "structure_bin_is_manual_review": 1.0 if ready and not bool(target.get("structure_ready_strict", False)) else 0.0,
        "sequence_exact_match": 1.0 if ready else 0.0,
    }
    result = {column: float(values[column]) for column in columns}
    if not all(np.isfinite(list(result.values()))):
        raise RuntimeError(f"non-finite projected structure row for {target.get('target_chembl_id')}")
    result["structure_mask"] = 1.0 if ready else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--protbert-batch-size", type=int, default=2)
    parser.add_argument("--esm2-batch-size", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = [
        BASE_PAIRS, BASE_TARGET_INDEX, BASE_STRUCTURE, BASE_MORGAN, BASE_PROTBERT,
        BASE_ESM2, BDB_PAIRS, BDB_NEW_TARGETS, DEPLOY_DRUG_INDEX, DEPLOY_MORGAN,
        ALL888_PAIRS, TARGET_UNIVERSE,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.cpu:
        raise RuntimeError("new ProtBERT/ESM2 target extraction requires CUDA")

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    base = pd.read_csv(BASE_PAIRS, low_memory=False)
    bdb_pairs = pd.read_csv(BDB_PAIRS, low_memory=False)
    deploy_index = pd.read_csv(DEPLOY_DRUG_INDEX, low_memory=False).sort_values("drug_feature_index")
    if len(deploy_index) != 720 or deploy_index["ligand_inchikey"].duplicated().any():
        raise RuntimeError("frozen 720-drug index contract failed")
    warm_drugs = set(base["parent_standard_inchi_key"].dropna().astype(str)) | set(
        bdb_pairs["parent_standard_inchi_key"].dropna().astype(str)
    )
    cold_drugs = set(deploy_index["ligand_inchikey"].astype(str)) - warm_drugs

    all_pairs = pd.read_csv(ALL888_PAIRS, low_memory=False)
    recovered = all_pairs.loc[
        all_pairs["parent_standard_inchi_key"].astype(str).isin(cold_drugs)
        & all_pairs["calibration_label"].isin(["positive", "negative_or_inactive"])
    ].copy()
    if recovered.empty or recovered.duplicated(["target_chembl_id", "parent_standard_inchi_key"]).any():
        raise RuntimeError("recovered ChEMBL37 relation table is empty or non-unique")

    target_columns = [
        "target_chembl_id", "uniprot_accession", "sequence", "sequence_sha256",
        "sequence_key", "gene_symbol", "assay_lane", "sequence_length",
        "structure_ready", "structure_ready_strict", "p2rank_status",
        "p2rank_top_score", "p2rank_top_probability", "p2rank_center_x",
        "p2rank_center_y", "p2rank_center_z", "pocket_residue_count",
    ]
    universe = pd.read_csv(TARGET_UNIVERSE, usecols=target_columns, low_memory=False).rename(
        columns={
            "gene_symbol": "universe_gene_symbol",
            "assay_lane": "universe_assay_lane",
        }
    )
    if universe["target_chembl_id"].duplicated().any():
        raise RuntimeError("target universe target_chembl_id is not unique")
    recovered = recovered.merge(universe, on="target_chembl_id", how="left", validate="many_to_one")
    if recovered[["sequence", "sequence_sha256", "sequence_key"]].isna().any().any():
        raise RuntimeError("recovered relations are missing exact target sequences")
    computed_hash = recovered["sequence"].map(sequence_hash)
    if not computed_hash.eq(recovered["sequence_sha256"].astype(str)).all():
        raise RuntimeError("target universe sequence hash mismatch")

    # Existing target feature indices are append-only: 428 frozen targets,
    # followed by 385 BindingDB targets.
    base_target_index = pd.read_csv(BASE_TARGET_INDEX, low_memory=False)
    base_target_index["sequence_sha256"] = base_target_index["protein_sequence"].map(sequence_hash)
    target_lookup = dict(zip(
        base_target_index["sequence_sha256"].astype(str),
        base_target_index["target_feature_index"].astype(int),
        strict=True,
    ))
    bdb_target_index = pd.read_csv(BDB_NEW_TARGETS, low_memory=False)
    for row in bdb_target_index.itertuples(index=False):
        key = str(row.target_sequence_hash)
        index = int(row.target_feature_index)
        if key in target_lookup and target_lookup[key] != index:
            raise RuntimeError("target feature index conflict")
        target_lookup[key] = index
    base_protbert = np.load(BASE_PROTBERT, mmap_mode="r")
    base_esm2 = np.load(BASE_ESM2, mmap_mode="r")
    if base_protbert.shape != (813, 1024) or base_esm2.shape != (813, 1280):
        raise RuntimeError(f"unexpected combined target feature shapes: {base_protbert.shape}, {base_esm2.shape}")
    if set(target_lookup.values()) != set(range(813)):
        raise RuntimeError("existing target feature map is not dense 0..812")
    target_unique = recovered.drop_duplicates("sequence_sha256").sort_values("sequence_sha256")
    new_targets = target_unique.loc[~target_unique["sequence_sha256"].isin(target_lookup)].copy()
    for index, target_hash in enumerate(new_targets["sequence_sha256"].astype(str), start=813):
        target_lookup[target_hash] = index
    if len(new_targets):
        from build_biomaster_odti_davis_feature_store_v1 import build_esm2, build_protbert

        sequences = new_targets["sequence"].astype(str).tolist()
        new_protbert, protbert_lengths = build_protbert(sequences, batch_size=args.protbert_batch_size)
        new_esm2, esm2_lengths = build_esm2(sequences, batch_size=args.esm2_batch_size)
        new_targets["protbert_used_length"] = protbert_lengths
        new_targets["esm2_used_length"] = esm2_lengths
        protbert = np.concatenate([np.asarray(base_protbert), new_protbert.astype(np.float32)], axis=0)
        esm2 = np.concatenate([np.asarray(base_esm2), new_esm2.astype(np.float32)], axis=0)
    else:
        protbert = np.asarray(base_protbert)
        esm2 = np.asarray(base_esm2)
    new_targets["target_feature_index"] = new_targets["sequence_sha256"].map(target_lookup).astype(int)

    # Append exact Morgan fingerprints copied from the frozen 720 deployment
    # store; do not regenerate structures or change their identity contract.
    base_morgan = np.load(BASE_MORGAN, mmap_mode="r")
    deploy_morgan = np.load(DEPLOY_MORGAN, mmap_mode="r")
    if base_morgan.shape != (68301, 2048) or deploy_morgan.shape != (720, 2048):
        raise RuntimeError("Morgan feature matrix contract failed")
    drug_lookup: dict[str, int] = {}
    for row in base[["parent_standard_inchi_key", "drug_feature_index"]].drop_duplicates().itertuples(index=False):
        drug_lookup[str(row.parent_standard_inchi_key)] = int(row.drug_feature_index)
    for row in bdb_pairs[["parent_standard_inchi_key", "drug_feature_index"]].drop_duplicates().itertuples(index=False):
        key, index = str(row.parent_standard_inchi_key), int(row.drug_feature_index)
        if key in drug_lookup and drug_lookup[key] != index:
            raise RuntimeError("drug feature index conflict")
        drug_lookup[key] = index
    deploy_lookup = deploy_index.set_index("ligand_inchikey")
    new_drug_keys = sorted(set(recovered["parent_standard_inchi_key"].astype(str)) - set(drug_lookup))
    if len(new_drug_keys) != recovered["parent_standard_inchi_key"].nunique():
        raise RuntimeError("recovered drug was unexpectedly warm in the base feature pair set")
    appended_drug_features = []
    new_drug_rows = []
    for offset, key in enumerate(new_drug_keys):
        source = deploy_lookup.loc[key]
        source_index = int(source["drug_feature_index"])
        feature_index = len(base_morgan) + offset
        drug_lookup[key] = feature_index
        appended_drug_features.append(np.asarray(deploy_morgan[source_index], dtype=np.uint8))
        new_drug_rows.append({
            "drug_feature_index": feature_index,
            "ligand_inchikey": key,
            "ligand_smiles": str(source["ligand_smiles"]),
            "drug_names": str(source["drug_names"]),
            "source_deployment_feature_index": source_index,
            "on_bits": int(source["on_bits"]),
            "feature_route": "FROZEN_720_EXACT_MORGAN_COPY",
        })
    morgan = np.concatenate([np.asarray(base_morgan), np.stack(appended_drug_features)], axis=0)

    # Canonical target metadata preferentially reuses the frozen internal
    # representation for exact sequences already present in the 428 targets.
    frozen_target = (
        base.sort_values("calibration_pair_id")
        .drop_duplicates("target_feature_index")
        .set_index("target_feature_index")
    )
    recovered["target_feature_index"] = recovered["sequence_sha256"].map(target_lookup).astype(int)
    recovered["drug_feature_index"] = recovered["parent_standard_inchi_key"].map(drug_lookup).astype(int)
    for column in ["primary_gene", "query_accession", "target_assay_family", "target_homology_cluster"]:
        recovered[column] = ""
    for index in recovered.index:
        target_index = int(recovered.at[index, "target_feature_index"])
        if target_index in frozen_target.index:
            row = frozen_target.loc[target_index]
            recovered.at[index, "primary_gene"] = str(row["primary_gene"])
            recovered.at[index, "query_accession"] = str(row["query_accession"])
            recovered.at[index, "target_assay_family"] = str(row["target_assay_family"])
            recovered.at[index, "target_homology_cluster"] = str(row["target_homology_cluster"])
            recovered.at[index, "sequence_key"] = str(row["sequence_key"])
        else:
            recovered.at[index, "primary_gene"] = str(recovered.at[index, "universe_gene_symbol"])
            recovered.at[index, "query_accession"] = str(recovered.at[index, "uniprot_accession"])
            recovered.at[index, "target_assay_family"] = broad_family(recovered.at[index, "universe_assay_lane"])
            recovered.at[index, "target_homology_cluster"] = "AUG_H_" + str(recovered.at[index, "sequence_sha256"])[:12]

    smiles_lookup = deploy_lookup["ligand_smiles"].astype(str).to_dict()
    recovered["model_ligand_smiles"] = recovered["parent_standard_inchi_key"].map(smiles_lookup)
    recovered["murcko_scaffold"] = recovered["model_ligand_smiles"].map(murcko)
    recovered["scaffold_group"] = recovered["murcko_scaffold"]
    recovered["binary_label"] = recovered["calibration_label"].eq("positive").astype(np.int8)
    recovered["rdkit_parse_ok"] = True
    recovered["calibration_pair_id"] = (
        "AUG37_" + recovered["sequence_sha256"].astype(str).str[:16]
        + "_" + recovered["parent_standard_inchi_key"].astype(str)
    )
    recovered["__role_hash"] = recovered["calibration_pair_id"].map(stable_hash)
    base_target_hashes = set(base_target_index["sequence_sha256"].astype(str))
    recovered["augmentation_role"] = assign_roles(recovered, base_target_hashes)
    recovered["augmentation_source"] = "CHEMBL37_ALL888_STRICT_ELIGIBLE_EXACT_COLD_DRUG"
    recovered["random_pair_fold"] = recovered["__role_hash"].str[:8].map(lambda value: int(value, 16) % 5)
    recovered["scaffold_cold_fold"] = -1
    recovered["target_homology_cold_fold"] = -1
    recovered["is_deployment_old_drug"] = True
    recovered["has_deployment_old_drug_scaffold"] = True
    recovered["temporal_role"] = "DEPLOYMENT_AUGMENTATION"
    recovered["drug_feature_available"] = True
    recovered["conplex_score"] = 0.0
    recovered["parent_canonical_smiles"] = recovered["model_ligand_smiles"]
    recovered["relationship_types"] = ""

    required_pair_columns = list(base.columns) + [
        "sequence_sha256", "augmentation_role", "augmentation_source"
    ]
    for column in required_pair_columns:
        if column not in recovered:
            recovered[column] = np.nan
    recovered = recovered[required_pair_columns].sort_values("calibration_pair_id").reset_index(drop=True)
    if recovered["calibration_pair_id"].duplicated().any():
        raise RuntimeError("augmentation relation ID is not unique")

    # Pair-aligned 19-D structure table.  Existing exact targets copy their
    # audited V1 context; other targets use exact-sequence P2Rank projection.
    base_structure = pd.read_csv(BASE_STRUCTURE, low_memory=False)
    structure_columns = [column for column in base_structure.columns if column not in {"calibration_pair_id", "structure_mask"}]
    base_with_target = base[["calibration_pair_id", "target_feature_index"]].merge(
        base_structure, on="calibration_pair_id", how="inner", validate="one_to_one"
    )
    base_structure_by_target = base_with_target.drop_duplicates("target_feature_index").set_index("target_feature_index")
    universe_by_target = universe.set_index("target_chembl_id")
    recovered_structure_rows = []
    target_structure_rows = []
    for target_index, part in recovered.groupby("target_feature_index", sort=True):
        representative = part.iloc[0]
        if target_index in base_structure_by_target.index:
            source = base_structure_by_target.loc[target_index]
            structure_values = {column: float(source[column]) for column in structure_columns}
            structure_mask = float(source["structure_mask"])
            route = "EXACT_FROZEN_TARGET_STRUCTURE_COPY"
        else:
            target = universe_by_target.loc[str(representative["target_chembl_id"])]
            projected = target_structure_row(target, structure_columns)
            structure_mask = projected.pop("structure_mask")
            structure_values = projected
            route = "EXACT_SEQUENCE_ALPHAFOLD_P2RANK_19D_PROJECTION" if structure_mask else "NO_SAFE_POCKET_MASKED"
        target_structure_rows.append({
            "target_feature_index": int(target_index),
            "target_chembl_id": str(representative["target_chembl_id"]),
            "sequence_sha256": str(representative["sequence_sha256"]),
            "structure_mask": structure_mask,
            "structure_route": route,
            **structure_values,
        })
        for pair_id in part["calibration_pair_id"]:
            recovered_structure_rows.append({
                "calibration_pair_id": str(pair_id),
                "structure_mask": structure_mask,
                **structure_values,
            })
    recovered_structure = pd.DataFrame(recovered_structure_rows)
    augmented_structure = pd.concat([base_structure, recovered_structure], ignore_index=True)
    if augmented_structure["calibration_pair_id"].duplicated().any():
        raise RuntimeError("augmented structure table has duplicate pair IDs")

    # Final leakage/entity checks.
    train = recovered.loc[recovered["augmentation_role"].eq("train")]
    held = recovered.loc[recovered["augmentation_role"].isin(["dev", "test"])]
    if set(held["parent_standard_inchi_key"]) - set(train["parent_standard_inchi_key"]):
        raise RuntimeError("held-out relation contains a drug-cold entity")
    warm_target_hashes = base_target_hashes | set(train["sequence_sha256"])
    if set(held["sequence_sha256"]) - warm_target_hashes:
        raise RuntimeError("held-out relation contains a target-cold entity")
    base_relation_keys = set(zip(base["target_feature_index"].astype(int), base["parent_standard_inchi_key"].astype(str)))
    recovered_relation_keys = set(zip(recovered["target_feature_index"].astype(int), recovered["parent_standard_inchi_key"].astype(str)))
    if base_relation_keys & recovered_relation_keys:
        raise RuntimeError("recovered relation overlaps frozen base pair table")

    paths = {
        "recovered_pairs": out / "RECOVERED_CHEMBL37_RELATIONS_V1.csv.gz",
        "morgan": out / "MORGAN2048_UINT8_DEPLOYMENT_AUGMENTED_V1.npy",
        "protbert": out / "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy",
        "esm2": out / "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy",
        "new_drugs": out / "NEW_OLD_DRUG_FEATURE_INDEX_V1.csv.gz",
        "new_targets": out / "NEW_TARGET_FEATURE_INDEX_V1.csv.gz",
        "structure": out / "ODTI_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz",
        "target_structure": out / "TARGET_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz",
    }
    recovered.to_csv(paths["recovered_pairs"], index=False, compression="gzip")
    np.save(paths["morgan"], morgan.astype(np.uint8), allow_pickle=False)
    np.save(paths["protbert"], protbert.astype(np.float32), allow_pickle=False)
    np.save(paths["esm2"], esm2.astype(np.float32), allow_pickle=False)
    pd.DataFrame(new_drug_rows).to_csv(paths["new_drugs"], index=False, compression="gzip")
    new_targets.to_csv(paths["new_targets"], index=False, compression="gzip")
    augmented_structure.to_csv(paths["structure"], index=False, compression="gzip")
    pd.DataFrame(target_structure_rows).to_csv(paths["target_structure"], index=False, compression="gzip")

    role_counts = recovered.groupby("augmentation_role").size().to_dict()
    role_drugs = recovered.groupby("augmentation_role")["parent_standard_inchi_key"].nunique().to_dict()
    role_targets = recovered.groupby("augmentation_role")["sequence_sha256"].nunique().to_dict()
    structure_target = pd.DataFrame(target_structure_rows)
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_DEPLOYMENT_AUGMENTATION_V1",
        "contract": {
            "frozen_benchmark_modified": False,
            "exact_ligand_identity": "full InChIKey",
            "exact_target_identity": "SHA256 of exact amino-acid sequence",
            "eligible_labels": ["positive", "negative_or_inactive"],
            "unknown_pairs_as_negative": False,
            "dev_test_regime": "RELATION_UNSEEN_STRICT_DOUBLE_WARM",
            "conplex_enabled": False,
            "new_features_label_blind": True,
        },
        "counts": {
            "frozen_720_warm_before": 720 - len(cold_drugs),
            "frozen_720_cold_before": len(cold_drugs),
            "recovered_rows": len(recovered),
            "recovered_drugs": recovered["parent_standard_inchi_key"].nunique(),
            "recovered_targets": recovered["sequence_sha256"].nunique(),
            "role_rows": role_counts,
            "role_drugs": role_drugs,
            "role_targets": role_targets,
            "warm_after_all_recovered_relations": 720 - len(cold_drugs) + recovered["parent_standard_inchi_key"].nunique(),
            "new_drug_features": len(new_drug_rows),
            "new_target_features": len(new_targets),
            "target_features_total": len(protbert),
            "drug_features_total": len(morgan),
            "recovered_targets_with_structure": int(structure_target["structure_mask"].gt(0).sum()),
            "recovered_rows_with_structure": int(recovered_structure["structure_mask"].gt(0).sum()),
        },
        "checks": {
            "all_recovered_drugs_train_warm": not bool(set(held["parent_standard_inchi_key"]) - set(train["parent_standard_inchi_key"])),
            "all_recovered_targets_train_warm": not bool(set(held["sequence_sha256"]) - warm_target_hashes),
            "recovered_relations_disjoint_frozen_base": not bool(base_relation_keys & recovered_relation_keys),
            "all_drug_features_exact_frozen_copies": True,
            "all_target_sequences_exact": True,
            "structure_context_pair_aligned": len(recovered_structure) == len(recovered),
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
    }
    manifest_path = out / "DEPLOYMENT_AUGMENTATION_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
