#!/usr/bin/env python3
"""Index existing V4 inputs and preserve measurement semantics; no model fitting.

The output is a planning/data package, not a new split or training-ready claim.
Original assets are read-only. Large arrays and weights are referenced in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMP = Path("outputs/retrain_20260901/comprehensive_training_v1")
BASE = Path("outputs/old_drug_target_sota_v1")
BDB = Path("outputs/biomaster_bindingdb_full_training_subset_v1")
BFEAT = Path("outputs/biomaster_bindingdb_affinity_feature_package_v1")
AUG = Path("outputs/biomaster_deployment_augmentation_v1")
KIR = Path("outputs/kirhub_functional_adaptation_v1/expanded_pretrained_features_v1")
PRE = Path("outputs/biomaster_odti_pretrained_features_v1")
TOK = Path("outputs/biomaster_bindingdb_target_token_feature_package_v1")
DT = BASE / "public_retrained_v1/dtiam_official_feature_store_v1"
DEP = BASE / "public_retrained_v1/dtiam_deployment_feature_store_v1"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def bool_values(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["true", "1", "1.0"])


class Organizer:
    def __init__(self, out: Path):
        self.out = out
        self.assets: dict[str, dict] = {}
        self.outputs: dict[str, dict] = {}
        self.summary: dict = {"status": "DATA_ORGANIZATION_COMPLETE", "training_ready": False,
                              "scope": "Existing assets, conservative exact-key joins; no new model or split"}

    def asset(self, path: Path | str, role: str) -> Path:
        p = ROOT / path
        key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        if key not in self.assets:
            record = {"path": key, "exists": p.is_file(), "roles": [role]}
            if p.is_file():
                record["size_bytes"] = p.stat().st_size
                record["mtime_ns"] = p.stat().st_mtime_ns
                if p.stat().st_size <= 64 * 1024**2:
                    record["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
                    record["hash_status"] = "computed_current_bytes"
                else:
                    record["hash_status"] = "not_rehashed_large_asset_stat_only"
                if p.suffix == ".npy":
                    array = np.load(p, mmap_mode="r", allow_pickle=False)
                    record.update(shape=list(array.shape), dtype=str(array.dtype))
            self.assets[key] = record
        elif role not in self.assets[key]["roles"]:
            self.assets[key]["roles"].append(role)
        return p

    def table(self, path: Path | str, role: str) -> pd.DataFrame:
        p = self.asset(path, role)
        frame = pd.read_csv(p, low_memory=False)
        key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        self.assets[key].update(rows=len(frame), columns=frame.columns.tolist())
        return frame

    def save(self, frame: pd.DataFrame, name: str):
        p = self.out / name
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(p, index=False, compression=compression)
        self.outputs[name] = {"rows": len(frame), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def save_json(self, payload: dict, name: str):
        p = self.out / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        self.outputs[name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        print("Reading current entities and relation provenance", flush=True)
        drugs = self.table(COMP / "DRUG_FEATURE_INDEX_COMPREHENSIVE_V1.csv.gz", "current_molecule_identity")
        rel = self.table(COMP / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz", "legacy_aggregate_relations_not_raw_measurements")
        prepared = self.table("outputs/biomaster_v3_20260905/data/RELATIONS_V3.csv.gz", "historical_development_split")
        assert drugs.drug_feature_index.is_unique
        assert drugs.model_ligand_smiles.is_unique
        assert (drugs.drug_feature_index.to_numpy() == np.arange(len(drugs))).all()
        drugs["molecule_key"] = drugs.model_ligand_smiles.map(lambda s: "mol:" + sha_text(s))
        drugs["identity_policy"] = "exact_existing_model_isomeric_smiles_no_new_normalization"
        drugs["historical_v3_split"] = drugs.drug_feature_index.map(
            prepared.drop_duplicates("drug_feature_index").set_index("drug_feature_index").split
        ).fillna("not_in_v3_binary_split")
        split_counts = prepared.groupby("drug_feature_index").split.nunique()
        assert split_counts.max() == 1

        target_frames = []
        for path in [BASE / "feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz",
                     BFEAT / "BINDINGDB_NEW_TARGET_FEATURE_INDEX_V1.csv.gz",
                     AUG / "NEW_TARGET_FEATURE_INDEX_V1.csv.gz"]:
            frame = self.table(path, "current_target_identity")
            seq_col = "protein_sequence" if "protein_sequence" in frame else "sequence"
            part = frame[["target_feature_index", seq_col]].rename(columns={seq_col: "protein_sequence"})
            part["protein_sequence"] = part.protein_sequence.str.replace(r"\s+", "", regex=True).str.upper()
            part["sequence_sha256"] = part.protein_sequence.map(sha_text)
            target_frames.append(part)
        targets = pd.concat(target_frames, ignore_index=True)
        assert targets.groupby("target_feature_index").sequence_sha256.nunique().max() == 1
        targets = targets.drop_duplicates("target_feature_index").sort_values("target_feature_index")
        assert (targets.target_feature_index.to_numpy() == np.arange(len(targets))).all()
        targets["sequence_length"] = targets.protein_sequence.str.len()
        targets["target_key"] = "seq:" + targets.sequence_sha256
        targets["historical_index_is_construct_identity"] = False
        seq_map = targets.set_index("target_feature_index").sequence_sha256
        rel["resolved_sequence_sha256"] = rel.target_feature_index.map(seq_map)
        assert rel.resolved_sequence_sha256.notna().all()

        print("Matching pretrained feature banks by exact molecule identity", flush=True)
        bermol_banks = [
            (DT / "DTIAM_BERMOL_DRUG_INDEX_V1.csv.gz", DT / "DTIAM_BERMOL768_FLOAT32_V1.npy", "model_ligand_smiles", "dtiam_bermol_available"),
            (DEP / "DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz", DEP / "DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy", "ligand_smiles", "dtiam_bermol_available"),
            (KIR / "KIRHUB_EXPANDED_DRUG92_INDEX_V1.csv", KIR / "KIRHUB_EXPANDED_DRUG92_BERMOL768_FLOAT32_V1.npy", "ligand_smiles", None),
        ]
        lookup = {}
        for index_path, array_path, key_col, flag_col in bermol_banks:
            frame = self.table(index_path, "bermol_feature_identity")
            array = np.load(self.asset(array_path, "bermol_pooled_features"), mmap_mode="r")
            assert frame.drug_feature_index.between(0, len(array)-1).all()
            if flag_col:
                frame = frame.loc[bool_values(frame[flag_col])]
            for row in frame.to_dict("records"):
                i = int(row["drug_feature_index"])
                if np.isfinite(array[i]).all() and np.linalg.norm(array[i]) > 0:
                    lookup.setdefault(row[key_col], (str(array_path), i))
        matches = drugs.model_ligand_smiles.map(lookup)
        drugs["bermol_available"] = matches.notna()
        drugs["bermol_array"] = matches.map(lambda x: x[0] if isinstance(x, tuple) else "")
        drugs["bermol_row"] = matches.map(lambda x: x[1] if isinstance(x, tuple) else -1)
        molformer = self.table(PRE / "molformer_xl_both_10pct/MOLFORMER_XL_DRUG_INDEX_V1.csv.gz", "historical_molecular_feature_control")
        molformer = molformer.loc[bool_values(molformer.feature_available)]
        mf_map = molformer.set_index("model_ligand_smiles").drug_feature_index.to_dict()
        drugs["molformer_row"] = drugs.model_ligand_smiles.map(mf_map).fillna(-1).astype(int)
        self.asset(PRE / "molformer_xl_both_10pct/MOLFORMER_XL_768_FLOAT32_V1.npy", "historical_nonisomeric_feature_control")
        # Original feature indices from different banks are never interchangeable.
        self.save(drugs, "CURRENT_MOLECULE_ASSETS_V4.csv.gz")
        self.save(drugs.loc[~drugs.bermol_available, ["molecule_key", "drug_feature_index", "model_ligand_smiles", "historical_v3_split"]], "BERMOL_FEATURE_BUILD_QUEUE_V4.csv.gz")

        residue = self.table(TOK / "ESM2_650M_RESIDUE_INDEX_COMBINED_V1.csv.gz", "residue_offsets")
        token_array = np.load(self.asset(TOK / "ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy", "whole_sequence_frozen_ESM_tokens"), mmap_mode="r")
        assert (residue.token_offset + residue.token_length).le(len(token_array)).all()
        assert residue.target_feature_index.is_unique
        targets = targets.merge(residue[["target_feature_index", "token_offset", "token_length"]], how="left", on="target_feature_index", validate="one_to_one")
        targets["full_residue_available"] = targets.token_length.eq(targets.sequence_length)
        assert targets.loc[targets.token_length.notna(), "full_residue_available"].all()
        esmc = self.table(PRE / "esmc_600m/ESMC_600M_TARGET_INDEX_V1.csv.gz", "ESMC_legacy_target_index")
        targets["esmc_pooled_available"] = targets.target_feature_index.isin(esmc.loc[bool_values(esmc.feature_available), "target_feature_index"])
        self.asset(PRE / "esmc_600m/ESMC_600M_1152_FLOAT32_V1.npy", "ESMC_pooled_features")
        pocket = json.loads(self.asset("outputs/biomaster_v3_20260905/local_pockets/POCKET_REGIONS_V3.json", "mapped_pocket_regions_not_encoded_geometry").read_text())
        pocket_rows = []
        for digest, entry in pocket["targets"].items():
            for p in entry["pockets"]:
                receptor = entry["sources"]["receptor"]["path"]
                pocket_rows.append({"sequence_sha256": digest, "pocket_id": p["pocket_id"],
                                    "probability": p["probability"], "residue_count": p["residue_count"],
                                    "residue_indices_json": json.dumps(p["residue_indices"]),
                                    "receptor_path": receptor, "receptor_exists": Path(receptor).is_file(),
                                    "source_structure_sha256": entry["sources"]["receptor"]["sha256"],
                                    "geometry_encoder_input_ready": False})
        pocket_frame = pd.DataFrame(pocket_rows)
        self.save(pocket_frame, "POCKET_GEOMETRY_BUILD_QUEUE_V4.csv.gz")
        counts = pocket_frame.groupby("sequence_sha256").size()
        targets["selected_pocket_count"] = targets.sequence_sha256.map(counts).fillna(0).astype(int)
        self.save(targets, "CURRENT_TARGET_ASSETS_V4.csv.gz")
        self.save(targets.loc[~targets.full_residue_available, ["target_key", "target_feature_index", "protein_sequence", "sequence_length"]], "ESM_RESIDUE_BUILD_QUEUE_V4.csv.gz")

        print("Writing provenance indices and endpoint-preserving measurement tables", flush=True)
        fields = ["source_kind", "calibration_pair_id", "drug_feature_index", "target_feature_index", "resolved_sequence_sha256",
                  "parent_standard_inchi_key", "binary_label", "binary_observed", "affinity_observed",
                  "mean_pchembl", "min_pchembl", "max_pchembl", "standard_types", "assay_ids", "doc_ids",
                  "min_document_year", "max_document_year", "duplicate_of_comprehensive"]
        ledger = rel[[k for k in fields if k in rel]].copy()
        ledger.insert(0, "source_row", np.arange(len(ledger)))
        ledger["molecule_key"] = ledger.drug_feature_index.map(drugs.set_index("drug_feature_index").molecule_key)
        ledger["historical_v3_split"] = ledger.source_row.map(prepared.set_index("source_row").split).fillna("not_retained_in_v3")
        ledger["record_level"] = "legacy_pair_aggregate"
        ledger["v4_endpoint_training_ready"] = False
        ledger["range_is_censor_interval"] = False
        self.save(ledger, "LEGACY_RELATION_PROVENANCE_V4.csv.gz")
        self.save(prepared.groupby(["split", "target_feature_index", "binary_label"]).size().rename("rows").reset_index(), "HISTORICAL_SPLIT_TARGET_COUNTS_V4.csv")

        measurements = self.table(BDB / "BINDINGDB_HIGH_CONFIDENCE_MEASUREMENTS_V1.csv.gz", "endpoint_preserving_priority_affinity_measurements")
        assert measurements.relation.isin(["=", "<", ">", "<=", ">="]).all()
        assert np.isfinite(measurements.value_nM).all()
        assert measurements.value_nM.gt(0).all()
        source_digest = self.assets[str(BDB / "BINDINGDB_HIGH_CONFIDENCE_MEASUREMENTS_V1.csv.gz")]["sha256"][:12]
        measurements.insert(0, "source_record_id", [f"BDB_A0:{source_digest}:{i}" for i in range(len(measurements))])
        measurements["pactivity_threshold"] = 9 - np.log10(measurements.value_nM)
        measurements["pactivity_relation"] = measurements.relation.map({"=": "=", ">": "<", "<": ">", ">=": "<=", "<=": ">="})
        measurements["record_level"] = "measurement"
        measurements["binary_label"] = np.nan
        measurements["v4_split"] = "UNASSIGNED_REQUIRES_JOINT_SOURCE_SPLIT"
        measurements["assay_context_status"] = "document_provenance_preserved_full_assay_conditions_unresolved"
        measurements["quantitative_reference_delta_ready"] = False
        self.save(measurements, "BINDINGDB_PRIORITY_MEASUREMENTS_V4.csv.gz")

        kir = self.table(KIR / "KIRHUB_EXPANDED_FUNCTIONAL_PAIRS_V1.csv.gz", "functional_profile_development")
        kt = self.table(KIR / "KIRHUB_EXPANDED_TARGET345_INDEX_V1.csv.gz", "functional_construct_identity")
        kh = kt.set_index("target_feature_index").protein_sequence.map(lambda s: sha_text("".join(s.split()).upper()))
        keep = ["pairId", "ligand_inchikey", "ligand_smiles", "kirhub_wt_construct", "uniprot_accession",
                "target_feature_index", "residual_activity_pct_1uM", "inhibition_fraction_1uM", "strong_inhibition_label",
                "assay_type", "assay_concentration_uM", "label_semantics", "source_doi", "drug_scaffold_cold_fold"]
        functional = kir[keep].copy()
        functional["target_sequence_sha256"] = functional.target_feature_index.map(kh)
        functional["record_level"] = "functional_measurement"
        functional["evaluation_role"] = "nested_grouped_development_not_untouched_external"
        assert functional.target_sequence_sha256.notna().all()
        assert not any("score" in c or "probability" in c or "logit" in c for c in functional)
        self.save(functional, "KIRHUB_FUNCTIONAL_MEASUREMENTS_V4.csv.gz")

        print("Auditing BindingDB expansion against the current comprehensive package", flush=True)
        current_keys = set(zip(rel.parent_standard_inchi_key.fillna(""), rel.resolved_sequence_sha256))
        current_inchikeys = set(rel.parent_standard_inchi_key.dropna())
        target_keys = set(targets.sequence_sha256)
        expansion = []
        for name, path in [("priority_A0", BDB / "BINDINGDB_HIGH_CONFIDENCE_TRAINING_PAIRS_V1.csv.gz"),
                           ("all_accepted_legacy_filter", BDB / "BINDINGDB_TRAINING_SUBSET_ACCEPTED_PAIRS_V1.csv.gz")]:
            table = self.table(path, "expansion_candidates_not_automatically_trainable")
            table["current_exact_pair_overlap"] = [(d, t) in current_keys for d, t in zip(table.ligand_inchikey, table.target_sequence_hash)]
            table["current_target_sequence_indexed"] = table.target_sequence_hash.isin(target_keys)
            expansion.append({"pool": name, "pairs": len(table), "drugs": table.ligand_inchikey.nunique(),
                              "targets": table.target_sequence_hash.nunique(), "current_exact_pair_overlap": int(table.current_exact_pair_overlap.sum()),
                              "no_exact_pair_match": int((~table.current_exact_pair_overlap).sum()),
                              "rows_with_indexed_target": int(table.current_target_sequence_indexed.sum()),
                              "new_target_sequence_hashes": int(table.loc[~table.current_target_sequence_indexed, "target_sequence_hash"].nunique()),
                              "unmatched_source_InChIKeys": int(table.loc[~table.ligand_inchikey.isin(current_inchikeys), "ligand_inchikey"].nunique()),
                              "mapping_scope": "exact_source_InChIKey_plus_full_sequence_hash_only_not_full_canonical_or_document_dedup"})
            if name == "priority_A0":
                self.save(table, "BINDINGDB_PRIORITY_EXPANSION_AUDIT_V4.csv.gz")
        self.save(pd.DataFrame(expansion), "BINDINGDB_EXPANSION_COUNTS_V4.csv")

        # Supplementary assets are catalogued with explicit roles, not merged into training.
        supplemental = [
            (BDB / "BINDINGDB_TRAINING_SUBSET_AUDIT_V1.json", "historical_raw_source_and_filter_counts"),
            ("downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db", "raw_activities_reaggregate_before_time_split"),
            (".tmp/bindingdb_202608_full/BindingDB_All_202608_tsv.zip", "raw_affinity_measurements"),
            (BDB / "BINDINGDB_AUXILIARY_MEASUREMENTS_V1.csv.gz", "tiered_expansion_measurements_require_mapping_and_split"),
            (BDB / "BINDINGDB_HIGH_CONFIDENCE_DIRECT_KI_KD_MEASUREMENTS_V1.csv.gz", "priority_measurement_subset_do_not_add_twice"),
            (BFEAT / "BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz", "legacy_9778_pair_aggregate_already_in_comprehensive"),
            (BASE / "davis_complete_audit_v1/DAVIS_COMPLETE_LABEL_LEVEL_31824_V1.csv.gz", "reserved_DAVIS_complete_censored_construct_benchmark"),
            (BASE / "davis_complete_sequence_reconstruction_v1/DAVIS_COMPLETE_SEQUENCE_MANIFEST_V1.csv", "DAVIS_construct_sequence_mapping"),
            ("outputs/biomaster_odti_davis_feature_store_v1/DAVIS_ENTITY_COLD_FEATURE_INDEXED_PAIR_MANIFEST_V1.csv.gz", "historical_external_requires_new_overlap_audit"),
            ("outputs/biomaster_odti_w1_v3_semantic_adapter_v1/W1_V3_SEMANTIC_ADAPTER_AUDIT_V1.json", "pending_wetlab_not_training_data"),
            ("outputs/affinity_first_remote_discovery_v1/drugclip_inputs_v1/project723_ligands.lmdb", "existing_small_scope_3D_ligand_inputs"),
            ("outputs/affinity_first_remote_discovery_v1/drugclip_inputs_v1/project308_strict_pockets.lmdb", "existing_small_scope_3D_pocket_inputs"),
            ("outputs/affinity_first_remote_discovery_v1/drugclip_inputs_v1/PROJECT723_LIGAND_LMDB_MANIFEST_V1.csv", "3D_ligand_identity_requires_new_scope_join"),
            ("outputs/affinity_first_remote_discovery_v1/drugclip_inputs_v1/PROJECT308_POCKET_LMDB_MANIFEST_V1.csv", "3D_pocket_identity_requires_new_scope_join"),
            (COMP / "MORGAN2048_UINT8_COMPREHENSIVE_V1.npy", "retrieval_fingerprint_cache"),
            (AUG / "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy", "legacy_pooled_ESM_may_be_truncated"),
            (AUG / "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy", "legacy_pooled_control"),
        ]
        for path, role in supplemental:
            if str(path).endswith((".csv", ".csv.gz")) and (ROOT / path).is_file():
                self.table(path, role)
            else:
                self.asset(path, role)
        weight_paths = [("third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl", "pretrained_molecule_weight"),
                        ("/root/autodl-tmp/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt", "pretrained_sequence_weight")]
        weight_paths += [(f"third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds/fold_{i}.pt", "joint_3D_checkpoint_fold") for i in range(6)]
        for path in (ROOT / ".model_cache/huggingface/hub").glob("*/snapshots/*/model.safetensors"):
            weight_paths.append((path, "optional_public_pretrained_weight"))
        for path, role in weight_paths:
            self.asset(path, role)
        self.asset("third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds/DRUGCLIP_SIXFOLD_WEIGHTS_V1.json", "historical_weight_hashes")
        self.asset("configs/biomaster_odti_v4.yaml", "architecture_design")
        self.asset(Path(__file__).resolve(), "organizer_source_code")

        # Bounded, label-free storage estimate; oversized molecules retain the 2-D path.
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.warning")
        sample = drugs.sample(n=min(4096, len(drugs)), random_state=20260905)
        molecules = [Chem.MolFromSmiles(s) for s in sample.model_ligand_smiles]
        sizes = [m.GetNumHeavyAtoms() for m in molecules if m is not None]
        resource = {"scope": "Deterministic molecule sample; not a throughput benchmark", "sample_size": len(sample),
                    "sample_parse_failures": len(sample)-len(sizes), "molecules": len(drugs),
                    "sample_heavy_atoms_mean": float(np.mean(sizes)), "sample_heavy_atoms_p95": float(np.percentile(sizes, 95)),
                    "sample_heavy_atoms_max": max(sizes), "sample_above_128": sum(x > 128 for x in sizes),
                    "full_BerMol_float16_GiB": len(drugs)*768*2/1024**3,
                    "one_conformer_frozen_512_token_cache_estimated_GiB": len(drugs)*np.mean(sizes)*512*2/1024**3,
                    "one_conformer_128_atom_padded_upper_GiB": len(drugs)*128*512*2/1024**3,
                    "note": "Cache variable-length frozen tokens; no dense pair tensors. Metadata and graph storage are additional."}
        self.save_json(resource, "RESOURCE_ESTIMATE_V4.json")
        import torch
        checkpoint = "third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds/fold_0.pt"
        state = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False, mmap=True)["model"]
        prefixes = ["mol_model.", "pocket_model.", "mol_project.", "pocket_project.",
                    "cross_distance_project.", "holo_distance_project.", "classification_head.", "fuse_project."]
        self.save_json({"checkpoint": checkpoint, "inspection": "CPU mmap keys/shapes; no new inference",
                        "total_state_tensor_elements": sum(v.numel() for v in state.values()),
                        "prefix_tensor_elements": {p: sum(v.numel() for k, v in state.items() if k.startswith(p)) for p in prefixes},
                        "molecule_token_embedding_shape": list(state["mol_model.embed_tokens.weight"].shape),
                        "pocket_token_embedding_shape": list(state["pocket_model.embed_tokens.weight"].shape),
                        "joint_projection_output_dim": 128, "reuse_prefixes": prefixes[:4],
                        "do_not_assume_unused_heads_trained": prefixes[4:],
                        "provenance_note": "Local release; no claim of newest upstream checkpoint or supervision of all heads."},
                       "PRETRAINED_BACKBONE_INSPECTION_V4.json")

        self.summary.update(
            current_molecules=len(drugs), current_indexed_targets=len(targets),
            unique_sequence_hashes=int(targets.sequence_sha256.nunique()), legacy_relation_rows=len(rel),
            unique_v3_binary_relations=len(prepared),
            bermol_exact_available=int(drugs.bermol_available.sum()), bermol_build_queue=int((~drugs.bermol_available).sum()),
            molformer_exact_available=int(drugs.molformer_row.ge(0).sum()),
            full_ESM_residue_targets=int(targets.full_residue_available.sum()),
            ESM_residue_build_queue=int((~targets.full_residue_available).sum()),
            ESMC_pooled_targets=int(targets.esmc_pooled_available.sum()),
            pocket_region_targets=int(targets.selected_pocket_count.gt(0).sum()),
            pocket_and_full_ESM_targets=int((targets.selected_pocket_count.gt(0) & targets.full_residue_available).sum()),
            pocket_regions=len(pocket_frame), receptor_paths_all_exist=bool(pocket_frame.receptor_exists.all()),
            priority_bindingdb_measurements=len(measurements), kirhub_measurements=len(functional),
            bindingdb_expansion=expansion,
            disk_free_gib=round(shutil.disk_usage(ROOT).free / 1024**3, 2),
            training_blockers=["raw_ChEMBL_endpoint_and_asof_reaggregation", "joint_source_entity_scaffold_family_split",
                               "complete_pretrained_molecule_features_for_requested_training_scope", "3D_atom_pocket_feature_adapter",
                               "V4_network_and_trainer_not_implemented"],
            limitations=["No source labels used to construct feature cache joins",
                         "Exact smiles joins are conservative; unmatched equivalent structures may be cache misses",
                         "Sequence identity is not a complete assay construct or phosphorylation identity",
                         "No new untouched evaluation set has been claimed",
                         "Large referenced files are stat-checked; existing historical hashes are not fresh verification"])
        self.save_json(self.summary, "DATA_READINESS_V4.json")
        (self.out / "ASSET_CATALOG_V4.json").write_text(json.dumps(list(self.assets.values()), ensure_ascii=False, indent=2) + "\n")
        (self.out / "OUTPUT_MANIFEST_V4.json").write_text(json.dumps(self.outputs, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(self.summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/biomaster_odti_v4_plan_20260905")
    args = parser.parse_args()
    Organizer(args.out.resolve()).run()
