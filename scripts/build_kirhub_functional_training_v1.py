#!/usr/bin/env python3
"""Build a leakage-aware KiRHub functional-inhibition adaptation dataset.

This package deliberately does not replace BioMaster's direct-binding labels.
It represents a separate 1 uM biochemical residual-activity task and keeps the
pretrained BerMol/ESM2 embeddings frozen.  The primary training table excludes
gene-level mappings that collapse multiple experimental kinase constructs.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
KIRHUB_DIR = RUN / "old_drug_innovation_v7"
PAIR_BENCHMARK = KIRHUB_DIR / "OLD_DRUG_KIRHUB_PAIR_BENCHMARK_V7.csv"
TARGET_MAPPING = KIRHUB_DIR / "KIRHUB_WT_TARGET_MAPPING_384_V7.csv"
WORKBOOK = (
    RUN / "drug_centric_cross_target_v1/external_benchmark"
    / "41587_2026_3090_MOESM4_ESM.xlsx"
)
BASE = ROOT / "outputs/old_drug_target_sota_v1"
FEATURE_STORE = BASE / "public_retrained_v1/dtiam_deployment_feature_store_v1"
DRUG_INDEX = FEATURE_STORE / "DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz"
TARGET_INDEX = FEATURE_STORE / "DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz"
TARGET_CLUSTERS = BASE / "benchmark_splits_v1/TARGET_HOMOLOGY_CLUSTERS_428_V1.csv"
BASELINE_SCORES = (
    BASE / "drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
)
OUT = ROOT / "outputs/kirhub_functional_adaptation_v1"
SEED = 20260831
FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms():
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic, canonical=True)
    # Do not collapse every acyclic compound into one artificial group.
    return "ACYCLIC:" + Chem.MolToSmiles(mol, canonical=True)


def assign_drug_scaffold_folds(data: pd.DataFrame) -> dict[str, int]:
    """Assign each whole Murcko scaffold group to exactly one fold."""
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    y = data["strong_inhibition_label"].to_numpy(dtype=np.int8)
    groups = data["drug_scaffold_group"].to_numpy()
    fold_by_group: dict[str, int] = {}
    placeholder = np.zeros((len(data), 1), dtype=np.int8)
    for fold, (_, test_index) in enumerate(splitter.split(placeholder, y, groups)):
        for group in np.unique(groups[test_index]):
            if group in fold_by_group:
                raise RuntimeError(f"Scaffold assigned twice: {group}")
            fold_by_group[str(group)] = fold
    expected = set(data["drug_scaffold_group"].astype(str))
    if set(fold_by_group) != expected:
        raise RuntimeError("Incomplete scaffold fold assignment")
    return fold_by_group


def build_gap_manifests(
    raw_wt: pd.DataFrame,
    benchmark: pd.DataFrame,
    drug_index: pd.DataFrame,
    target_index: pd.DataFrame,
    target_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compound_match = benchmark[[
        "kirhub_compound", "ligand_inchikey", "drug_names"
    ]].drop_duplicates()
    if compound_match["kirhub_compound"].duplicated().any():
        raise RuntimeError("One KiRHub compound maps to multiple project drugs")
    drug_gap = pd.DataFrame({"kirhub_compound": raw_wt["Compound"].astype(str)})
    drug_gap = drug_gap.merge(compound_match, on="kirhub_compound", how="left")
    drug_gap = drug_gap.merge(
        drug_index[["drug_feature_index", "ligand_inchikey", "ligand_smiles"]],
        on="ligand_inchikey", how="left", validate="many_to_one",
    )
    drug_gap["bermol768_available"] = drug_gap["drug_feature_index"].notna()

    gene_target = benchmark[["gene_symbol", "target_chembl_id"]].drop_duplicates()
    gene_target = gene_target.merge(
        target_index[["target_feature_index", "target_chembl_id"]],
        on="target_chembl_id", how="left", validate="many_to_one",
    )
    associations: list[dict[str, object]] = []
    for row in target_mapping.itertuples(index=False):
        columns = [item for item in str(row.kirhub_wt_columns).split(";") if item and item != "nan"]
        for column in columns:
            associations.append({"kirhub_wt_construct": column, "gene_symbol": row.gene_symbol})
    association_frame = pd.DataFrame(associations).merge(
        gene_target, on="gene_symbol", how="left", validate="many_to_many"
    )
    grouped = association_frame.groupby("kirhub_wt_construct", sort=False).agg(
        project_gene_symbols=("gene_symbol", lambda x: ";".join(sorted(set(map(str, x))))),
        project_target_chembl_ids=(
            "target_chembl_id",
            lambda x: ";".join(sorted(set(map(str, x.dropna())))),
        ),
        project_target_count=("target_chembl_id", lambda x: int(x.dropna().nunique())),
        esm2_1280_available=("target_feature_index", lambda x: bool(x.notna().any())),
    ).reset_index()
    target_gap = pd.DataFrame({"kirhub_wt_construct": raw_wt.columns[1:].astype(str)})
    target_gap = target_gap.merge(grouped, on="kirhub_wt_construct", how="left")
    target_gap["project_target_count"] = target_gap["project_target_count"].fillna(0).astype(int)
    target_gap["esm2_1280_available"] = target_gap["esm2_1280_available"].fillna(False).astype(bool)
    target_gap["measured_drug_count"] = [
        int(raw_wt[column].notna().sum()) for column in target_gap["kirhub_wt_construct"]
    ]
    return drug_gap, target_gap


def main() -> None:
    required = [
        PAIR_BENCHMARK, TARGET_MAPPING, WORKBOOK, DRUG_INDEX, TARGET_INDEX,
        TARGET_CLUSTERS, BASELINE_SCORES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    benchmark = pd.read_csv(PAIR_BENCHMARK, low_memory=False)
    target_mapping = pd.read_csv(TARGET_MAPPING)
    drug_index = pd.read_csv(DRUG_INDEX, low_memory=False)
    target_index = pd.read_csv(TARGET_INDEX, low_memory=False)
    clusters = pd.read_csv(TARGET_CLUSTERS)
    raw_wt = pd.read_excel(WORKBOOK, sheet_name="Table S4", header=8)
    baselines = pd.read_csv(BASELINE_SCORES, low_memory=False)

    data = benchmark[[
        "pairId", "ligand_inchikey", "drug_names", "kirhub_compound",
        "target_chembl_id", "gene_symbol", "kirhub_wt_columns",
        "kirhub_wt_construct_count", "kirhub_wt_min_residual_activity_pct_1uM",
    ]].copy()
    data = data.rename(columns={
        "kirhub_wt_min_residual_activity_pct_1uM": "residual_activity_pct_1uM",
    })
    data["residual_activity_pct_1uM"] = pd.to_numeric(
        data["residual_activity_pct_1uM"], errors="raise"
    )
    data["inhibition_fraction_1uM"] = 1.0 - data["residual_activity_pct_1uM"] / 100.0
    data["strong_inhibition_label"] = data["inhibition_fraction_1uM"].ge(0.70).astype(np.int8)
    data["inhibition_ordinal_bin"] = pd.cut(
        data["inhibition_fraction_1uM"],
        bins=[-np.inf, 0.30, 0.50, 0.70, np.inf],
        labels=[0, 1, 2, 3], right=False,
    ).astype(np.int8)
    data["primary_single_construct_scope"] = data["kirhub_wt_construct_count"].eq(1)
    data["assay_type"] = "HotSpot biochemical residual kinase activity"
    data["assay_concentration_uM"] = 1.0
    data["label_semantics"] = "functional_inhibition_not_direct_binding_affinity"
    data["source_doi"] = "10.1038/s41587-026-03090-8"

    data = data.merge(
        drug_index[["drug_feature_index", "ligand_inchikey", "ligand_smiles"]],
        on="ligand_inchikey", how="left", validate="many_to_one",
    )
    data = data.merge(
        target_index[["target_feature_index", "target_chembl_id"]],
        on="target_chembl_id", how="left", validate="many_to_one",
    )
    data = data.merge(
        clusters[[
            "target_chembl_id", "target_homology_cluster", "target_homology_cold_fold"
        ]],
        on="target_chembl_id", how="left", validate="many_to_one",
    )
    baseline_columns = [
        "pairId", "old_drug_leakage_safe_score_v10",
        "biomaster_full_fit_v10_borda_score", "dtiam_probability",
        "ensemble_drug_to_target_logit",
    ]
    data = data.merge(
        baselines[baseline_columns], on="pairId", how="left", validate="one_to_one"
    )

    unique_drugs = data[["ligand_inchikey", "ligand_smiles"]].drop_duplicates().copy()
    unique_drugs["drug_scaffold_group"] = unique_drugs["ligand_smiles"].map(scaffold_smiles)
    data = data.merge(unique_drugs, on=["ligand_inchikey", "ligand_smiles"], validate="many_to_one")
    primary = data[data["primary_single_construct_scope"]].copy().reset_index(drop=True)
    scaffold_folds = assign_drug_scaffold_folds(primary)
    data["drug_scaffold_cold_fold"] = (
        data["drug_scaffold_group"].map(scaffold_folds).astype(np.int8)
    )
    primary = data[data["primary_single_construct_scope"]].copy().reset_index(drop=True)

    checks = {
        "raw_matrix_exact_92_by_409": raw_wt.shape == (92, 410),
        "overlap_exact_8058_79_103": (
            len(data) == 8058
            and data["ligand_inchikey"].nunique() == 79
            and data["target_chembl_id"].nunique() == 103
        ),
        "primary_exact_7505_79_96": (
            len(primary) == 7505
            and primary["ligand_inchikey"].nunique() == 79
            and primary["target_chembl_id"].nunique() == 96
        ),
        "primary_exact_987_strong": int(primary["strong_inhibition_label"].sum()) == 987,
        "no_duplicate_pairs": not data.duplicated(["ligand_inchikey", "target_chembl_id"]).any(),
        "all_continuous_labels_bounded": data["inhibition_fraction_1uM"].between(0, 1).all(),
        "all_pretrained_features_available": (
            data["drug_feature_index"].notna().all()
            and data["target_feature_index"].notna().all()
        ),
        "all_baseline_scores_available": data[baseline_columns[1:]].notna().all().all(),
        "all_target_clusters_available": data["target_homology_cluster"].notna().all(),
        "five_drug_scaffold_folds": set(primary["drug_scaffold_cold_fold"]) == set(range(FOLDS)),
        "five_target_homology_folds": set(primary["target_homology_cold_fold"]) == set(range(FOLDS)),
        "scaffolds_do_not_cross_folds": primary.groupby("drug_scaffold_group")[
            "drug_scaffold_cold_fold"
        ].nunique().eq(1).all(),
        "target_clusters_do_not_cross_folds": primary.groupby("target_homology_cluster")[
            "target_homology_cold_fold"
        ].nunique().eq(1).all(),
        "multi_construct_rows_excluded_from_primary": primary["kirhub_wt_construct_count"].eq(1).all(),
        "unknown_raw_measurements_not_materialized_as_negative": data["residual_activity_pct_1uM"].notna().all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    drug_gap, target_gap = build_gap_manifests(
        raw_wt, benchmark, drug_index, target_index, target_mapping
    )
    OUT.mkdir(parents=True, exist_ok=True)
    full_path = OUT / "KIRHUB_FUNCTIONAL_OVERLAP_8058_V1.csv.gz"
    primary_path = OUT / "KIRHUB_FUNCTIONAL_PRIMARY_SINGLE_CONSTRUCT_7505_V1.csv.gz"
    drug_gap_path = OUT / "KIRHUB_FULL92_DRUG_FEATURE_GAP_V1.csv"
    target_gap_path = OUT / "KIRHUB_FULL409_TARGET_FEATURE_GAP_V1.csv"
    data.to_csv(full_path, index=False, compression="gzip")
    primary.to_csv(primary_path, index=False, compression="gzip")
    drug_gap.to_csv(drug_gap_path, index=False)
    target_gap.to_csv(target_gap_path, index=False)

    measured_slots = int(raw_wt.iloc[:, 1:].notna().sum().sum())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "task": "separate_1uM_functional_kinase_inhibition_adapter",
        "representation_policy": "freeze pretrained BerMol768 and ESM2-650M-1280 embeddings",
        "raw_kirhub_wt": {
            "drugs": 92, "constructs": 409, "matrix_slots": 92 * 409,
            "measured_nonmissing_slots": measured_slots,
            "unknown_slots": 92 * 409 - measured_slots,
        },
        "immediate_overlap": {
            "rows": len(data), "drugs": data["ligand_inchikey"].nunique(),
            "targets": data["target_chembl_id"].nunique(),
            "strong_pairs_ge_70pct_inhibition": int(data["strong_inhibition_label"].sum()),
        },
        "primary_single_construct": {
            "rows": len(primary), "drugs": primary["ligand_inchikey"].nunique(),
            "targets": primary["target_chembl_id"].nunique(),
            "strong_pairs_ge_70pct_inhibition": int(primary["strong_inhibition_label"].sum()),
            "reason": "avoid assigning a minimum across multiple experimental constructs to one gene-level target embedding",
        },
        "full_feature_gap": {
            "drugs_with_existing_bermol": int(drug_gap["bermol768_available"].sum()),
            "drugs_missing_bermol": int((~drug_gap["bermol768_available"]).sum()),
            "constructs_with_project_esm2_route": int(target_gap["esm2_1280_available"].sum()),
            "constructs_missing_project_esm2_route": int((~target_gap["esm2_1280_available"]).sum()),
        },
        "split_contract": {
            "drug": "5-fold generic Bemis-Murcko scaffold cold",
            "target": "5-fold >=40% sequence-homology component cold",
            "seed": SEED,
        },
        "checks": checks,
        "claim_boundary": (
            "KiRHub is now adaptation/development data. It can no longer be called an untouched "
            "external benchmark for the adapted head; the prior KiRHub-free model remains frozen."
        ),
        "inputs_sha256": {path.name: sha256(path) for path in required},
        "artifacts": {
            full_path.name: sha256(full_path), primary_path.name: sha256(primary_path),
            drug_gap_path.name: sha256(drug_gap_path), target_gap_path.name: sha256(target_gap_path),
        },
    }
    summary_path = OUT / "KIRHUB_FUNCTIONAL_DATA_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
