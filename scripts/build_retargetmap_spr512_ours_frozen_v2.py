#!/usr/bin/env python3
"""Build the own-frozen-model 512-pair SPR package for old-drug target discovery.

The package deliberately separates the registry, routed discovery, historical
benchmark, and uniform soluble-protein SPR universes.  Candidate pairs are
selected before wet-lab outcomes are available and are balanced by target,
our frozen-model rank band, drug reuse, and chemical scaffold. External model
scores never enter candidate eligibility, quotas, or the optimization objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/retargetmap_spr512_ours_frozen_20260904"

RANK_PATH = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
SCORE_PATH = ROOT / "outputs/retrain_20260901/bidirectional_720x745_full_fit_rows4/BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_SCORES_V1.csv.gz"
REGISTRY_PATH = ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_REGISTRY_NON_GPCR_745_V3.csv.gz"
STRICT_PATH = ROOT / "outputs/target_universe_ch37_v2/chembl37_calibration_all888_v2/TARGET_ALL888_STRICT_BINDING_PAIRS_V2.csv.gz"
KIRHUB_PATH = ROOT / "outputs/kirhub_functional_adaptation_v1/KIRHUB_FUNCTIONAL_PRIMARY_SINGLE_CONSTRUCT_7505_V1.csv.gz"
FDA_PATH = ROOT / "outputs/final_design_space_2005_2026_v1/FINAL_FROZEN_FDA_DRUG_ENTITIES_2005_2026.csv"
TRAINING_PATH = ROOT / "outputs/retrain_20260901/comprehensive_training_v1/COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
BINDINGDB_ARTICLES = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/external_relation_audit_v7/raw/bindingdb_202607/BindingDB_BindingDB_Articles_202607_tsv.zip"
BINDINGDB_PUBCHEM = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/external_relation_audit_v7/raw/bindingdb_202607/BindingDB_PubChem_202607_tsv.zip"
GTOPDB_DIR = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/gtopdb_external_v11/raw_2026_2"

SEED = 20260903
N_SELECTED_DRUGS = 120
MIN_TARGETS_PER_DRUG = 2
MAX_TARGETS_PER_DRUG = 6
MAX_DRUGS_PER_SCAFFOLD = 5

BENCHMARK_TARGETS = [
    "CA7", "CA4", "MME", "FKBP1A", "NAMPT", "REN", "AKR1B1", "PNP",
    "PDE5A", "MMP12", "TPH1", "ELANE", "GSK3A", "CLK2", "LYN", "CDK5",
    "DYRK1A", "RIPK1", "NR3C2", "AR",
]
DISCOVERY_TARGETS = [
    "RORC", "CA9", "NT5E", "FOLH1", "AOC3", "FAP", "HDAC6", "HDAC8",
    "KDM1A", "PGR", "NR3C1", "PPARG",
]

OURS_ONLY_QUOTAS = {
    "OUR_FROZEN_MODEL_HIGH": 10,
    "OUR_FROZEN_MODEL_INTERMEDIATE": 3,
    "OUR_FROZEN_MODEL_LOW_BACKGROUND": 2,
}

CONTROL_NAMES = {
    "CA7": "ETHOXZOLAMIDE", "CA4": "ACETAZOLAMIDE", "MME": "OMAPATRILAT",
    "FKBP1A": "SIROLIMUS", "NAMPT": "CHS-828", "REN": "REMIKIREN",
    "AKR1B1": "ZOPOLRESTAT", "PNP": "CLADRIBINE", "PDE5A": "VARDENAFIL",
    "MMP12": "ILOMASTAT", "TPH1": "TELOTRISTAT", "ELANE": "ALVELESTAT",
    "GSK3A": "ABEMACICLIB", "CLK2": "SUNITINIB", "LYN": "DASATINIB ANHYDROUS",
    "CDK5": "DINACICLIB", "DYRK1A": "HARMINE", "RIPK1": "GSK2982772",
    "NR3C2": "ALDOSTERONE", "AR": "TESTOSTERONE", "RORC": "T091317",
    "CA9": "ZONISAMIDE", "NT5E": "QUEMLICLUSTAT", "FOLH1": "PIFLUFOLASTAT",
    "AOC3": "MOFEGILINE", "FAP": "LINAGLIPTIN", "HDAC6": "PANOBINOSTAT",
    "HDAC8": "QUISINOSTAT", "KDM1A": "SECLIDEMSTAT", "PGR": "PROGESTERONE",
    "NR3C1": "BUDESONIDE", "PPARG": "INT131",
}

CONSTRUCTS = {
    "CA7": "可溶性全长胞内酶；保留 Zn2+ 催化位点",
    "CA4": "可溶性胞外催化域；去信号肽和 GPI 锚定序列；保留 Zn2+",
    "MME": "可溶性胞外金属肽酶域；去胞内段、跨膜段和信号肽",
    "FKBP1A": "可溶性全长蛋白；单体构建",
    "NAMPT": "可溶性全长成熟蛋白；保持二聚状态",
    "REN": "成熟 renin 催化域；去信号肽和前肽",
    "AKR1B1": "可溶性全长酶；按需要加入 NADPH/NADP+ 状态对照",
    "PNP": "可溶性全长成熟酶；保持三聚状态",
    "PDE5A": "PDE5A 催化域；保留金属离子结合条件",
    "MMP12": "成熟催化域；去信号肽/前肽；确认 Zn2+ 与激活状态",
    "TPH1": "可溶性催化核心；明确 Fe2+/BH4 辅因子状态",
    "ELANE": "成熟中性粒细胞弹性蛋白酶；去信号肽和前肽",
    "GSK3A": "激酶催化域；记录磷酸化和 ATP/Mg2+ 条件",
    "CLK2": "激酶催化域；记录磷酸化和 ATP/Mg2+ 条件",
    "LYN": "激酶催化域；记录磷酸化和 ATP/Mg2+ 条件",
    "CDK5": "CDK5 激酶域；优先同时评估 CDK5–p25 复合构建",
    "DYRK1A": "激酶催化域；确认成熟自磷酸化状态",
    "RIPK1": "RIPK1 激酶域；使用可逆抑制剂兼容构建",
    "NR3C2": "矿皮质激素受体配体结合域（LBD）",
    "AR": "雄激素受体配体结合域（LBD）",
    "RORC": "ROR-gamma 配体结合域（LBD）",
    "CA9": "可溶性胞外催化域；去信号肽、跨膜段和胞内尾；保留 Zn2+",
    "NT5E": "可溶性胞外催化域；去信号肽/GPI 锚；保持二聚状态",
    "FOLH1": "可溶性胞外催化域；去胞内段、跨膜段和信号肽；保持二聚",
    "AOC3": "可溶性胞外域；去跨膜段；确认 Cu2+/TPQ 成熟和二聚状态",
    "FAP": "可溶性胞外催化域；去胞内段、跨膜段和信号肽；保持二聚",
    "HDAC6": "优先 HDAC6 catalytic domain 2；保留 Zn2+",
    "HDAC8": "可溶性全长/催化域；保留 Zn2+",
    "KDM1A": "含 SWIRM 与 amine-oxidase-like 的催化核心；FAD 结合态",
    "PGR": "孕激素受体配体结合域（LBD）",
    "NR3C1": "糖皮质激素受体配体结合域（LBD）",
    "PPARG": "PPAR-gamma 配体结合域（LBD）",
}

AFFINITY_COLUMNS = ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]
BINDINGDB_BASE_COLUMNS = [
    "BindingDB Reactant_set_id", "Ligand InChI Key", "BindingDB Ligand Name",
    "Target Name", "Target Source Organism According to Curator or DataSource",
    *AFFINITY_COLUMNS, "Curation/DataSource", "Article DOI", "BindingDB Entry DOI",
    "PMID", "Date of publication", "Date in BindingDB",
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "Link to Ligand-Target Pair in BindingDB",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def split_accessions(value: object) -> set[str]:
    return {part for part in re.split(r"[\s,;|/]+", clean(value)) if part}


def bindingdb_header(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError(f"Unexpected BindingDB archive members: {path}")
        with archive.open(names[0]) as handle:
            return handle.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")


def scan_bindingdb(path: Path, snapshot: str, pairs: pd.DataFrame) -> pd.DataFrame:
    columns = bindingdb_header(path)
    uniprot_columns = [
        column for column in columns
        if re.fullmatch(r"UniProt \(SwissProt\) Primary ID of Target Chain \d+", column)
    ]
    if set(BINDINGDB_BASE_COLUMNS) - set(columns):
        raise RuntimeError(f"BindingDB columns changed: {path}")
    pair_lookup = {
        (row.ligand_inchikey, row.uniprot_accession): row.pair_id
        for row in pairs[["ligand_inchikey", "uniprot_accession", "pair_id"]].drop_duplicates().itertuples(index=False)
    }
    ligand_keys = set(pairs["ligand_inchikey"])
    rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(
        path, sep="\t", usecols=BINDINGDB_BASE_COLUMNS + uniprot_columns, dtype=str,
        chunksize=25_000, on_bad_lines="skip", low_memory=False,
    ):
        subset = chunk[chunk["Ligand InChI Key"].isin(ligand_keys)]
        for _, row in subset.iterrows():
            ligand = clean(row["Ligand InChI Key"])
            accessions: set[str] = set()
            for column in uniprot_columns:
                accessions.update(split_accessions(row.get(column)))
            for accession in accessions:
                pair_id = pair_lookup.get((ligand, accession))
                if pair_id is None:
                    continue
                rows.append({
                    "pair_id": pair_id,
                    "source_snapshot": snapshot,
                    "source_record_id": clean(row["BindingDB Reactant_set_id"]),
                    "source_ligand_name": clean(row["BindingDB Ligand Name"]),
                    "source_target_name": clean(row["Target Name"]),
                    "affinity_fields": ";".join(
                        f"{column}={clean(row.get(column))}" for column in AFFINITY_COLUMNS
                        if clean(row.get(column))
                    ),
                    "article_doi": clean(row["Article DOI"]),
                    "pmid": clean(row["PMID"]),
                    "source_url": clean(row["Link to Ligand-Target Pair in BindingDB"]),
                })
    return pd.DataFrame(rows)


def scan_gtopdb(pairs: pd.DataFrame) -> pd.DataFrame:
    ligands = pd.read_csv(GTOPDB_DIR / "ligands.csv", skiprows=1, low_memory=False)
    interactions = pd.read_csv(GTOPDB_DIR / "interactions.csv", skiprows=1, low_memory=False)
    ligand_map = ligands[["Ligand ID", "InChIKey", "Name"]].rename(
        columns={"InChIKey": "ligand_inchikey", "Name": "gtopdb_ligand_name"}
    )
    rows = interactions.merge(ligand_map, on="Ligand ID", how="inner")
    rows = rows[rows["Target Species"].eq("Human")]
    rows["uniprot_accession"] = rows["Target UniProt ID"].map(clean)
    keys = pairs[["pair_id", "ligand_inchikey", "uniprot_accession"]].drop_duplicates()
    rows = rows.merge(keys, on=["ligand_inchikey", "uniprot_accession"], how="inner")
    keep = [
        "pair_id", "Ligand ID", "gtopdb_ligand_name", "Target", "Target ID", "Action",
        "Original Affinity Units", "Original Affinity Median nm", "PubMed ID", "Webpage URLs",
    ]
    return rows[keep].copy()


def molecule_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame[["ligand_inchikey", "ligand_smiles"]].drop_duplicates().itertuples(index=False):
        mol = Chem.MolFromSmiles(row.ligand_smiles)
        if mol is None:
            raise RuntimeError(f"Invalid SMILES for {row.ligand_inchikey}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        rows.append({
            "ligand_inchikey": row.ligand_inchikey,
            "ligand_smiles": row.ligand_smiles,
            "connectivity_key": row.ligand_inchikey[:14],
            "murcko_scaffold": scaffold,
            "rdkit_mw": Descriptors.MolWt(mol),
            "rdkit_clogp": Crippen.MolLogP(mol),
            "rdkit_tpsa": rdMolDescriptors.CalcTPSA(mol),
            "rdkit_hbd": Lipinski.NumHDonors(mol),
            "rdkit_hba": Lipinski.NumHAcceptors(mol),
            "rdkit_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        })
    return pd.DataFrame(rows)


def parse_id_set(value: object) -> set[str]:
    return {part.strip() for part in clean(value).split(";") if part.strip()}


def build_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(REGISTRY_PATH)
    selected_genes = BENCHMARK_TARGETS + DISCOVERY_TARGETS
    selected_targets = registry[registry["gene_symbol"].isin(selected_genes)].copy()
    if selected_targets["gene_symbol"].nunique() != len(selected_genes):
        missing = sorted(set(selected_genes) - set(selected_targets["gene_symbol"]))
        raise RuntimeError(f"Missing selected targets: {missing}")

    fda_columns = [
        "model_inchikey", "ingredient_name", "molecular_weight", "fragment_count", "formal_charge",
        "known_mechanism_of_action", "known_action_types", "known_target_chembl_ids",
        "known_target_names", "all_fda_brand_names", "entity_first_nda_window_approval_date",
        "final_drug_scope_status", "compute_readiness",
    ]
    fda = pd.read_csv(FDA_PATH, usecols=fda_columns).drop_duplicates("model_inchikey")
    fda = fda[fda["final_drug_scope_status"].eq("INCLUDE_CORE_DISCOVERY")].copy()

    strict = pd.read_csv(STRICT_PATH)
    local_pairs = set(zip(strict["parent_standard_inchi_key"].astype(str), strict["target_chembl_id"].astype(str)))
    training = pd.read_csv(
        TRAINING_PATH, usecols=["parent_standard_inchi_key", "target_chembl_id"]
    )
    training_pairs = set(zip(
        training["parent_standard_inchi_key"].astype(str),
        training["target_chembl_id"].astype(str),
    ))
    training_drugs = set(training["parent_standard_inchi_key"].dropna().astype(str))
    training_targets = set(training["target_chembl_id"].dropna().astype(str))
    kirhub = pd.read_csv(KIRHUB_PATH, usecols=["ligand_inchikey", "target_chembl_id"])
    kirhub_pairs = set(zip(kirhub["ligand_inchikey"].astype(str), kirhub["target_chembl_id"].astype(str)))

    score = pd.read_csv(SCORE_PATH)
    score_selected = score[score["gene_symbol"].isin(selected_genes)].copy()
    score_meta = score_selected[[
        "ligand_inchikey", "ligand_smiles", "drug_names", "target_chembl_id", "gene_symbol",
        "uniprot_accession", "assay_lane", "production_route", "target_model_warmth",
        "structure_evidence_route", "structure_mask", "ensemble_pair_logit",
        "ensemble_drug_to_target_logit", "diagnostic_rank_within_drug_745",
        "routed_rank_within_drug", "routed_candidate_target_count",
    ]].copy()

    rank = pd.read_csv(RANK_PATH)
    rank["retargetmap_rank_384"] = rank.groupby("ligand_inchikey", sort=False)[
        "independent_validation_rank_score"
    ].rank(method="first", ascending=False).astype(int)
    rank["dtiam_rank_384"] = rank.groupby("ligand_inchikey", sort=False)[
        "dtiam_probability"
    ].rank(method="first", ascending=False).astype(int)
    benchmark = rank[rank["gene_symbol"].isin(BENCHMARK_TARGETS)].merge(
        score_meta,
        on=["ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol"],
        how="inner", validate="one_to_one",
    )

    discovery = score_selected[score_selected["gene_symbol"].isin(DISCOVERY_TARGETS)].copy()
    discovery = discovery.rename(columns={"candidate_pair_id": "score_candidate_pair_id"})

    candidates = []
    for arm, frame in (("FROZEN_384_CORE_DISCOVERY", benchmark), ("EXTENDED_367_TARGET_DISCOVERY", discovery)):
        frame = frame.merge(fda, left_on="ligand_inchikey", right_on="model_inchikey", how="inner", validate="many_to_one")
        frame["pair_id"] = frame["ligand_inchikey"] + "__" + frame["target_chembl_id"]
        frame["experiment_arm"] = arm
        frame["local_chembl_pair_found"] = [
            (drug, target) in local_pairs for drug, target in zip(frame["ligand_inchikey"], frame["target_chembl_id"])
        ]
        frame["kirhub_pair_found"] = [
            (drug, target) in kirhub_pairs for drug, target in zip(frame["ligand_inchikey"], frame["target_chembl_id"])
        ]
        frame["known_moa_target_collision"] = [
            target in parse_id_set(ids) for target, ids in zip(frame["target_chembl_id"], frame["known_target_chembl_ids"])
        ]
        frame["exact_pair_in_full_fit_training"] = [
            (drug, target) in training_pairs
            for drug, target in zip(frame["ligand_inchikey"], frame["target_chembl_id"])
        ]
        frame["drug_seen_in_full_fit_training"] = frame["ligand_inchikey"].astype(str).isin(training_drugs)
        frame["target_seen_in_full_fit_training"] = frame["target_chembl_id"].astype(str).isin(training_targets)
        frame = frame[
            ~frame["local_chembl_pair_found"] & ~frame["kirhub_pair_found"]
            & ~frame["known_moa_target_collision"]
            & ~frame["exact_pair_in_full_fit_training"]
            & frame["molecular_weight"].between(120, 700)
            & frame["fragment_count"].eq(1)
            & frame["formal_charge"].abs().le(2)
        ].copy()

        if arm == "FROZEN_384_CORE_DISCOVERY":
            conditions = [
                frame["retargetmap_rank_384"].le(20),
                frame["retargetmap_rank_384"].between(80, 180),
                frame["retargetmap_rank_384"].gt(250),
            ]
            roles = [
                "OUR_FROZEN_MODEL_HIGH", "OUR_FROZEN_MODEL_INTERMEDIATE",
                "OUR_FROZEN_MODEL_LOW_BACKGROUND",
            ]
        else:
            conditions = [
                frame["routed_rank_within_drug"].le(30),
                frame["routed_rank_within_drug"].between(80, 180),
                frame["routed_rank_within_drug"].gt(250),
            ]
            roles = [
                "OUR_FROZEN_MODEL_HIGH", "OUR_FROZEN_MODEL_INTERMEDIATE",
                "OUR_FROZEN_MODEL_LOW_BACKGROUND",
            ]
        frame["selection_role"] = np.select(conditions, roles, default="")
        frame = frame[frame["selection_role"].ne("")].copy()
        candidates.append(frame)

    pool = pd.concat(candidates, ignore_index=True, sort=False)
    pool = pool.merge(
        selected_targets[[
            "target_chembl_id", "target_name", "target_class_l1", "target_class_all",
            "sequence", "sequence_length", "in_historical_scored_384", "positive_compounds",
            "negative_compounds", "p2rank_tier", "structure_ready_permissive",
        ]],
        on="target_chembl_id", how="left", validate="many_to_one",
    )
    pool = pool.merge(molecule_features(pool), on=["ligand_inchikey", "ligand_smiles"], how="left", validate="many_to_one")
    pool["chemical_risk_points"] = (
        pool["rdkit_mw"].gt(550).astype(int)
        + pool["rdkit_clogp"].gt(5).astype(int)
        + pool["rdkit_tpsa"].gt(140).astype(int)
        + pool["rdkit_rotatable_bonds"].gt(12).astype(int)
    )

    old_target_lanes = registry.set_index("target_chembl_id")["assay_lane"].to_dict()
    pool["known_target_lanes"] = pool["known_target_chembl_ids"].map(
        lambda value: ";".join(sorted({old_target_lanes[x] for x in parse_id_set(value) if x in old_target_lanes}))
    )
    pool["same_assay_family_as_known_target"] = [
        lane in clean(known_lanes).split(";") for lane, known_lanes in zip(pool["assay_lane"], pool["known_target_lanes"])
    ]
    pool["known_kinase_drug"] = pool["known_target_lanes"].str.contains("KINASE", na=False) | pool[
        "known_mechanism_of_action"
    ].fillna("").str.contains("kinase", case=False)
    return pool, selected_targets, strict, registry


def add_external_evidence(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_index = pool[["pair_id", "ligand_inchikey", "uniprot_accession"]].drop_duplicates()
    binding_rows = pd.concat([
        scan_bindingdb(BINDINGDB_ARTICLES, "BindingDB_Articles_2026-07", pair_index),
        scan_bindingdb(BINDINGDB_PUBCHEM, "BindingDB_PubChem_2026-07", pair_index),
    ], ignore_index=True)
    gtopdb_rows = scan_gtopdb(pair_index)
    binding_pairs = set(binding_rows["pair_id"]) if len(binding_rows) else set()
    gtopdb_pairs = set(gtopdb_rows["pair_id"]) if len(gtopdb_rows) else set()
    pool["bindingdb_exact_pair_found"] = pool["pair_id"].isin(binding_pairs)
    pool["gtopdb_exact_pair_found"] = pool["pair_id"].isin(gtopdb_pairs)
    pool = pool[~pool["bindingdb_exact_pair_found"] & ~pool["gtopdb_exact_pair_found"]].copy()
    return pool, binding_rows, gtopdb_rows


def edge_cost(row: pd.Series) -> float:
    role = row["selection_role"]
    if role == "OUR_FROZEN_MODEL_HIGH":
        if row["experiment_arm"] == "FROZEN_384_CORE_DISCOVERY":
            cost = row["retargetmap_rank_384"] / 20
        else:
            cost = row["routed_rank_within_drug"] / 30
        # The pair-backbone score is part of our frozen system and is used only
        # as a small tie-breaker after the directional production rank.
        cost += row["diagnostic_rank_within_drug_745"] / 74500
    elif role == "OUR_FROZEN_MODEL_INTERMEDIATE":
        if row["experiment_arm"] == "FROZEN_384_CORE_DISCOVERY":
            cost = abs(row["retargetmap_rank_384"] - 130) / 100
        else:
            cost = abs(row["routed_rank_within_drug"] - 130) / 100
    elif role == "OUR_FROZEN_MODEL_LOW_BACKGROUND":
        if row["experiment_arm"] == "FROZEN_384_CORE_DISCOVERY":
            cost = (384 - row["retargetmap_rank_384"]) / 384
        else:
            cost = (367 - row["routed_rank_within_drug"]) / 367
    else:
        raise ValueError(role)
    cost += 0.18 * row["chemical_risk_points"]
    cost += 0.08 * bool(row["same_assay_family_as_known_target"])
    cost += 0.03 * bool(row["known_kinase_drug"])
    # Deterministic microscopic jitter prevents arbitrary tie selection.
    cost += int(hashlib.sha256(row["pair_id"].encode()).hexdigest()[:8], 16) / 2**32 * 1e-6
    return float(cost)


def optimize_candidates(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.drop_duplicates(["pair_id", "selection_role"]).reset_index(drop=True)
    role_requirements = []
    for gene in BENCHMARK_TARGETS + DISCOVERY_TARGETS:
        role_requirements.extend((gene, role, count) for role, count in OURS_ONLY_QUOTAS.items())
    for gene, role, count in role_requirements:
        available = int(((pool["gene_symbol"] == gene) & (pool["selection_role"] == role)).sum())
        if available < count:
            raise RuntimeError(f"Insufficient candidates for {gene}/{role}: {available} < {count}")

    drugs = sorted(pool["ligand_inchikey"].unique())
    drug_to_y = {drug: len(pool) + i for i, drug in enumerate(drugs)}
    n_variables = len(pool) + len(drugs)
    objective = np.zeros(n_variables)
    objective[: len(pool)] = pool.apply(edge_cost, axis=1).to_numpy()
    objective[len(pool):] = 1e-7
    integrality = np.ones(n_variables, dtype=int)
    bounds = Bounds(np.zeros(n_variables), np.ones(n_variables))

    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: dict[int, float], lb: float, ub: float) -> None:
        row_index = len(lower)
        for column, value in coefficients.items():
            matrix_rows.append(row_index)
            matrix_cols.append(column)
            matrix_data.append(value)
        lower.append(lb)
        upper.append(ub)

    for gene, role, count in role_requirements:
        indices = pool.index[(pool["gene_symbol"] == gene) & (pool["selection_role"] == role)]
        add_constraint({int(index): 1.0 for index in indices}, count, count)

    for drug in drugs:
        indices = pool.index[pool["ligand_inchikey"] == drug]
        y = drug_to_y[drug]
        add_constraint({**{int(index): 1.0 for index in indices}, y: -MIN_TARGETS_PER_DRUG}, 0, np.inf)
        add_constraint({**{int(index): 1.0 for index in indices}, y: -MAX_TARGETS_PER_DRUG}, -np.inf, 0)

    add_constraint({drug_to_y[drug]: 1.0 for drug in drugs}, N_SELECTED_DRUGS, N_SELECTED_DRUGS)

    drug_meta = pool.drop_duplicates("ligand_inchikey").set_index("ligand_inchikey")
    scaffold_groups = defaultdict(list)
    for drug in drugs:
        scaffold = clean(drug_meta.loc[drug, "murcko_scaffold"])
        if scaffold:
            scaffold_groups[scaffold].append(drug)
    for scaffold_drugs in scaffold_groups.values():
        if len(scaffold_drugs) > MAX_DRUGS_PER_SCAFFOLD:
            add_constraint(
                {drug_to_y[drug]: 1.0 for drug in scaffold_drugs},
                -np.inf, MAX_DRUGS_PER_SCAFFOLD,
            )

    connectivity_groups = defaultdict(list)
    for drug in drugs:
        connectivity_groups[clean(drug_meta.loc[drug, "connectivity_key"])].append(drug)
    for connectivity_drugs in connectivity_groups.values():
        if len(connectivity_drugs) > 1:
            add_constraint(
                {drug_to_y[drug]: 1.0 for drug in connectivity_drugs}, -np.inf, 1
            )

    kinase_drugs = [drug for drug in drugs if bool(drug_meta.loc[drug, "known_kinase_drug"])]
    add_constraint({drug_to_y[drug]: 1.0 for drug in kinase_drugs}, -np.inf, 30)

    matrix = coo_matrix(
        (matrix_data, (matrix_rows, matrix_cols)), shape=(len(lower), n_variables)
    ).tocsr()
    result = milp(
        c=objective, integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 300, "mip_rel_gap": 0.001, "presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    selected = pool.loc[np.flatnonzero(result.x[: len(pool)] > 0.5)].copy()
    expected_candidates = sum(count for _, _, count in role_requirements)
    if len(selected) != expected_candidates:
        raise RuntimeError(f"Expected {expected_candidates} candidate pairs, got {len(selected)}")
    selected["optimization_objective_contribution"] = selected.apply(edge_cost, axis=1)
    selected["external_novelty_status"] = (
        "NO_EXACT_PAIR_IN_CHEMBL37_BINDINGDB_2026_07_GTOPDB_2026_2_KIRHUB;_PUBMED_PATENT_MANUAL_REVIEW_PENDING"
    )
    return selected


def build_controls(strict: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_meta = targets.drop_duplicates("gene_symbol").set_index("gene_symbol")
    for gene in BENCHMARK_TARGETS + DISCOVERY_TARGETS:
        name = CONTROL_NAMES[gene]
        matches = strict[
            strict["gene_symbol"].eq(gene)
            & strict["calibration_label"].eq("positive")
            & strict["parent_molecule_name"].fillna("").str.upper().eq(name)
        ].copy()
        if matches.empty:
            raise RuntimeError(f"No positive-control evidence row for {gene}/{name}")
        matches["has_kd_ki"] = matches["standard_types"].str.contains(
            r"(?:^|,)(?:Kd|Ki)(?:,|$)", regex=True
        )
        best = matches.sort_values(["has_kd_ki", "max_pchembl"], ascending=[False, False]).iloc[0]
        target = target_meta.loc[gene]
        rows.append({
            "pair_id": f"CONTROL::{best.parent_standard_inchi_key}__{target.target_chembl_id}",
            "ligand_inchikey": best.parent_standard_inchi_key,
            "ligand_smiles": best.parent_canonical_smiles,
            "drug_names": str(best.parent_molecule_name).title(),
            "target_chembl_id": target.target_chembl_id,
            "gene_symbol": gene,
            "uniprot_accession": target.uniprot_accession,
            "target_name": target.target_name,
            "target_class_l1": target.target_class_l1,
            "assay_lane": target.assay_lane,
            "experiment_arm": "TARGET_QC_POSITIVE_CONTROL",
            "selection_role": "POSITIVE_CONTROL",
            "control_chembl_id": best.parent_molecule_chembl_id,
            "control_standard_types": best.standard_types,
            "control_min_pchembl": best.min_pchembl,
            "control_max_pchembl": best.max_pchembl,
            "control_direct_kd_or_ki_evidence": bool(best.has_kd_ki),
            "control_evidence_years": f"{int(best.min_document_year) if pd.notna(best.min_document_year) else ''}-{int(best.max_document_year) if pd.notna(best.max_document_year) else ''}",
            "external_novelty_status": "KNOWN_POSITIVE_CONTROL_NOT_A_DISCOVERY_CLAIM",
        })
    return pd.DataFrame(rows)


def make_target_roster(targets: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    genes = BENCHMARK_TARGETS + DISCOVERY_TARGETS
    roster = targets[targets["gene_symbol"].isin(genes)].drop_duplicates("gene_symbol").copy()
    roster["target_order"] = roster["gene_symbol"].map({gene: i + 1 for i, gene in enumerate(genes)})
    roster["experiment_arm"] = roster["gene_symbol"].map(
        lambda gene: "FROZEN_384_CORE_DISCOVERY" if gene in BENCHMARK_TARGETS else "EXTENDED_367_TARGET_DISCOVERY"
    )
    roster["recommended_construct"] = roster["gene_symbol"].map(CONSTRUCTS)
    control_cols = [
        "gene_symbol", "drug_names", "control_chembl_id", "control_standard_types",
        "control_min_pchembl", "control_max_pchembl", "control_direct_kd_or_ki_evidence",
    ]
    roster = roster.merge(
        controls[control_cols].rename(columns={"drug_names": "positive_control_drug"}),
        on="gene_symbol", how="left", validate="one_to_one",
    )
    roster["positive_control_evidence_grade"] = np.where(
        roster["control_direct_kd_or_ki_evidence"],
        "A_KD_OR_KI_SUPPORTED",
        "B_BIOCHEMICAL_IC50_SUPPORTED_SPR_SCOUT_REQUIRED",
    )
    roster["positive_control_action"] = np.where(
        roster["control_direct_kd_or_ki_evidence"],
        "Proceed with the standard positive-control scout",
        "Run control first; if no specific SPR signal, add a literature Kd control before interpreting candidates",
    )
    roster["construct_boundary_status"] = "EXPERIMENT_TEAM_TO_CONFIRM_RESIDUE_BOUNDARIES_AGAINST_UNIPROT_ISOFORM"
    roster["candidate_pairs"] = 15
    roster["positive_control_pairs"] = 1
    return roster.sort_values("target_order")


def validate_design(all_rows: pd.DataFrame, selected: pd.DataFrame, controls: pd.DataFrame) -> dict[str, bool]:
    candidate_degree = selected.groupby("ligand_inchikey").size()
    selected_drugs = selected[["ligand_inchikey"]].drop_duplicates().copy()
    selected_drugs["connectivity_key"] = selected_drugs["ligand_inchikey"].str[:14]
    expected_roles = {
        "OUR_FROZEN_MODEL_HIGH": 320,
        "OUR_FROZEN_MODEL_INTERMEDIATE": 96,
        "OUR_FROZEN_MODEL_LOW_BACKGROUND": 64,
    }
    checks = {
        "total_rows_512": len(all_rows) == 512,
        "unique_pair_ids_512": all_rows["pair_id"].nunique() == 512,
        "targets_32": all_rows["gene_symbol"].nunique() == 32,
        "candidate_pairs_480": len(selected) == 480,
        "positive_controls_32": len(controls) == 32,
        "each_target_has_15_candidates": selected.groupby("gene_symbol").size().eq(15).all(),
        "each_target_has_one_control": controls.groupby("gene_symbol").size().eq(1).all(),
        "candidate_drugs_120": selected["ligand_inchikey"].nunique() == 120,
        "candidate_drug_degree_2_to_6": candidate_degree.between(2, 6).all(),
        "one_selected_drug_per_connectivity": selected_drugs["connectivity_key"].is_unique,
        "all_candidate_fda_identity_mapped": selected["ingredient_name"].notna().all(),
        "all_candidate_old_targets_documented": selected["known_target_names"].notna().all(),
        "all_candidates_in_spr_mw_window": selected["rdkit_mw"].between(120, 700).all(),
        "all_candidate_pairs_unique": ~selected.duplicated(["ligand_inchikey", "target_chembl_id"]).any(),
        "no_selected_pair_in_full_fit_training": ~selected["exact_pair_in_full_fit_training"].any(),
        "all_construct_recommendations_present": all_rows["construct_recommendation"].notna().all(),
        "all_target_controls_named": all_rows["known_positive_for_new_target"].notna().all(),
        "candidate_roles_exact": selected["selection_role"].value_counts().to_dict() == expected_roles,
        "no_selected_bindingdb_exact_pair": ~selected["bindingdb_exact_pair_found"].any(),
        "no_selected_gtopdb_exact_pair": ~selected["gtopdb_exact_pair_found"].any(),
    }
    return {key: bool(value) for key, value in checks.items()}


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="0B5A8F")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for column in sheet.columns:
            letter = column[0].column_letter
            max_len = min(45, max(10, max(len(str(cell.value or "")) for cell in column[:200]) + 2))
            sheet.column_dimensions[letter].width = max_len
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    workbook.save(path)


def write_outputs(
    selected: pd.DataFrame,
    controls: pd.DataFrame,
    targets: pd.DataFrame,
    binding_rows: pd.DataFrame,
    gtopdb_rows: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat([selected, controls], ignore_index=True, sort=False)
    all_rows["construct_recommendation"] = all_rows["gene_symbol"].map(CONSTRUCTS)
    all_rows["known_positive_for_new_target"] = all_rows["gene_symbol"].map(
        lambda gene: CONTROL_NAMES[gene].title()
    )
    all_rows["posthoc_only_dtiam_rank_384"] = all_rows.get("dtiam_rank_384")
    all_rows["posthoc_only_dtiam_probability"] = all_rows.get("dtiam_probability")
    all_rows["spr_concentration_plan"] = np.where(
        all_rows["selection_role"].eq("POSITIVE_CONTROL"),
        "0.001–3 µM, 8-point 3-fold; adjust after scout",
        "0.03–30 µM, 7-point 3-fold; solubility-limited rows stop at qualified maximum",
    )
    all_rows["primary_endpoint"] = "QC-qualified steady-state or kinetic KD"
    all_rows["result_bin_pre_registered"] = "KD<=0.1uM strong; 0.1<KD<=1uM positive; 1<KD<=10uM grey; KD>10uM/no specific binding negative"
    rng = np.random.default_rng(SEED)
    order = []
    for gene in BENCHMARK_TARGETS + DISCOVERY_TARGETS:
        block = all_rows[all_rows["gene_symbol"].eq(gene)].copy()
        control = block[block["selection_role"].eq("POSITIVE_CONTROL")]
        candidates = block[~block["selection_role"].eq("POSITIVE_CONTROL")].sample(frac=1, random_state=SEED + len(order))
        order.append(pd.concat([control, candidates], ignore_index=True))
    all_rows = pd.concat(order, ignore_index=True)
    all_rows["blind_experiment_id"] = [f"SPR512-{i:03d}" for i in range(1, len(all_rows) + 1)]
    all_rows["lab_row_type"] = np.where(
        all_rows["selection_role"].eq("POSITIVE_CONTROL"), "TARGET_POSITIVE_CONTROL", "BLINDED_CANDIDATE"
    )
    validation_checks = validate_design(all_rows, selected, controls)
    if not all(validation_checks.values()):
        raise RuntimeError({key: value for key, value in validation_checks.items() if not value})

    internal_columns = [
        "blind_experiment_id", "lab_row_type", "experiment_arm", "selection_role", "pair_id",
        "drug_names", "ingredient_name", "all_fda_brand_names", "ligand_inchikey", "ligand_smiles",
        "known_mechanism_of_action", "known_action_types", "known_target_names", "known_target_chembl_ids",
        "gene_symbol", "target_name", "target_chembl_id", "uniprot_accession", "target_class_l1",
        "assay_lane", "construct_recommendation", "known_positive_for_new_target",
        "retargetmap_rank_384", "routed_rank_within_drug",
        "routed_candidate_target_count", "diagnostic_rank_within_drug_745",
        "independent_validation_rank_score", "ensemble_drug_to_target_logit",
        "ensemble_pair_logit", "structure_mask", "structure_evidence_route",
        "connectivity_key", "murcko_scaffold", "rdkit_mw", "rdkit_clogp", "rdkit_tpsa", "rdkit_hbd", "rdkit_hba",
        "rdkit_rotatable_bonds", "chemical_risk_points", "same_assay_family_as_known_target",
        "known_kinase_drug", "external_novelty_status", "spr_concentration_plan", "primary_endpoint",
        "exact_pair_in_full_fit_training", "drug_seen_in_full_fit_training",
        "target_seen_in_full_fit_training",
        "result_bin_pre_registered", "control_chembl_id", "control_standard_types",
        "control_min_pchembl", "control_max_pchembl", "control_direct_kd_or_ki_evidence",
        "control_evidence_years", "optimization_objective_contribution",
        "posthoc_only_dtiam_rank_384", "posthoc_only_dtiam_probability",
    ]
    for column in internal_columns:
        if column not in all_rows:
            all_rows[column] = np.nan
    internal = all_rows[internal_columns]

    lab_columns = [
        "blind_experiment_id", "lab_row_type", "drug_names", "ingredient_name", "ligand_inchikey",
        "ligand_smiles", "known_target_names", "gene_symbol", "target_name", "target_chembl_id",
        "uniprot_accession", "construct_recommendation", "known_positive_for_new_target",
        "control_standard_types", "control_max_pchembl", "control_direct_kd_or_ki_evidence",
        "spr_concentration_plan", "primary_endpoint", "result_bin_pre_registered",
    ]
    lab = all_rows[lab_columns].rename(columns={
        "blind_experiment_id": "实验编号", "lab_row_type": "实验类型", "drug_names": "小分子药物",
        "ingredient_name": "FDA活性成分", "ligand_inchikey": "InChIKey", "ligand_smiles": "SMILES",
        "known_target_names": "药物旧靶点", "gene_symbol": "新靶点基因", "target_name": "新靶点蛋白",
        "target_chembl_id": "新靶点ChEMBL", "uniprot_accession": "新靶点UniProt",
        "construct_recommendation": "建议蛋白构建", "known_positive_for_new_target": "新靶点已知阳性药物",
        "control_standard_types": "本行阳性对照证据类型", "control_max_pchembl": "本行阳性对照最高pChEMBL",
        "control_direct_kd_or_ki_evidence": "本行阳性对照有Kd或Ki证据",
        "spr_concentration_plan": "SPR浓度建议", "primary_endpoint": "主终点",
        "result_bin_pre_registered": "预注册结果分层",
    })
    roster = make_target_roster(targets, controls)
    drug_procurement = selected.groupby(
        ["ligand_inchikey", "drug_names", "ingredient_name", "all_fda_brand_names", "ligand_smiles",
         "known_target_names", "known_mechanism_of_action", "connectivity_key", "murcko_scaffold", "rdkit_mw", "rdkit_clogp"],
        dropna=False, as_index=False,
    ).agg(
        target_count=("gene_symbol", "nunique"),
        target_genes=("gene_symbol", lambda values: ";".join(sorted(set(values)))),
        experiment_ids=("pair_id", "count"),
    ).sort_values(["target_count", "drug_names"], ascending=[False, True])

    internal.to_csv(OUT / "RETARGETMAP_SPR512_OURS_FROZEN_INTERNAL_MASTER_V2.csv", index=False)
    lab.to_csv(OUT / "RETARGETMAP_SPR512_OURS_FROZEN_LAB_EXECUTION_BLINDED_V2.csv", index=False)
    roster.to_csv(OUT / "RETARGETMAP_SPR512_OURS_FROZEN_TARGET_ROSTER_V2.csv", index=False)
    drug_procurement.to_csv(OUT / "RETARGETMAP_SPR512_OURS_FROZEN_DRUG_PROCUREMENT_V2.csv", index=False)
    binding_rows.to_csv(OUT / "RETARGETMAP_SPR512_OURS_FROZEN_BINDINGDB_EXCLUDED_ROWS_V2.csv", index=False)
    gtopdb_rows.to_csv(OUT / "RETARGETMAP_SPR512_OURS_FROZEN_GTOPDB_EXCLUDED_ROWS_V2.csv", index=False)

    summary_table = pd.DataFrame([
        ("完整登记空间", "720 drugs x 745 non-GPCR targets", 536400),
        ("主动路由空间", "720 x (450 primary + 42 special + 8 exploratory)", 360000),
        ("直接小分子主空间", "720 x 450 targets", 324000),
        ("统一可溶蛋白SPR主空间", "720 x 367 biochemical targets", 264240),
        ("历史公平比较空间", "720 x 384 frozen targets", 276480),
        ("本轮候选", "32 targets x 15 blinded candidates", 480),
        ("本轮阳性对照", "32 targets x 1 control", 32),
        ("本轮总计", "32 targets x 16 unique pairs", 512),
    ], columns=["层级", "定义", "pair数"])

    workbook_path = OUT / "RETARGETMAP_SPR512_OURS_FROZEN_LAB_PACKAGE_V2.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        lab.to_excel(writer, sheet_name="实验执行表", index=False)
        roster.to_excel(writer, sheet_name="蛋白构建清单", index=False)
        drug_procurement.to_excel(writer, sheet_name="药物采购去重", index=False)
        internal.to_excel(writer, sheet_name="内部冻结表", index=False)
        summary_table.to_excel(writer, sheet_name="空间与分母", index=False)
    format_workbook(workbook_path)

    candidate_counts = selected["selection_role"].value_counts().sort_index().to_dict()
    drug_counts = selected.groupby("ligand_inchikey").size()
    drug_warmth = selected.drop_duplicates("ligand_inchikey")["drug_seen_in_full_fit_training"]
    double_warm_pairs = selected[
        "drug_seen_in_full_fit_training"
    ].astype(bool) & selected["target_seen_in_full_fit_training"].astype(bool)
    scaffold_counts = selected.drop_duplicates("ligand_inchikey")["murcko_scaffold"].value_counts()
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "design_spaces": {
            "registry_720x745": 536400,
            "active_routed_720x500": 360000,
            "primary_direct_sm_720x450": 324000,
            "uniform_biochemical_spr_720x367": 264240,
            "uniform_biochemical_spr_after_local_chembl_and_kirhub_exclusion": 253474,
            "historical_benchmark_720x384": 276480,
        },
        "final_design": {
            "targets": int(all_rows["gene_symbol"].nunique()),
            "frozen_384_core_targets": len(BENCHMARK_TARGETS),
            "extended_367_targets": len(DISCOVERY_TARGETS),
            "candidate_selection_policy": "OUR_FROZEN_MODEL_ONLY",
            "dtiam_selection_role": "NONE_POSTHOC_ANNOTATION_ONLY",
            "candidate_pairs": int(len(selected)),
            "positive_control_pairs": int(len(controls)),
            "total_pairs": int(len(all_rows)),
            "candidate_drugs": int(selected["ligand_inchikey"].nunique()),
            "candidate_drugs_seen_in_full_fit_training": int(drug_warmth.sum()),
            "candidate_drugs_unseen_in_full_fit_training": int((~drug_warmth.astype(bool)).sum()),
            "candidate_double_warm_pairs": int(double_warm_pairs.sum()),
            "candidate_drug_cold_target_warm_pairs": int((~selected["drug_seen_in_full_fit_training"].astype(bool) & selected["target_seen_in_full_fit_training"].astype(bool)).sum()),
            "candidate_drug_degree_min": int(drug_counts.min()),
            "candidate_drug_degree_median": float(drug_counts.median()),
            "candidate_drug_degree_max": int(drug_counts.max()),
            "unique_murcko_scaffolds_nonempty": int(selected.loc[selected["murcko_scaffold"].ne(""), "murcko_scaffold"].nunique()),
            "max_drugs_per_nonempty_scaffold": int(scaffold_counts.max()),
            "candidate_role_counts": {str(key): int(value) for key, value in candidate_counts.items()},
            "direct_kd_or_ki_supported_controls": int(controls["control_direct_kd_or_ki_evidence"].sum()),
            "exact_candidate_pairs_in_full_fit_training": int(selected["exact_pair_in_full_fit_training"].sum()),
        },
        "external_exclusions": {
            "bindingdb_evidence_rows": int(len(binding_rows)),
            "bindingdb_exact_pairs_excluded": int(binding_rows["pair_id"].nunique()) if len(binding_rows) else 0,
            "gtopdb_evidence_rows": int(len(gtopdb_rows)),
            "gtopdb_exact_pairs_excluded": int(gtopdb_rows["pair_id"].nunique()) if len(gtopdb_rows) else 0,
        },
        "validation_checks": validation_checks,
        "claim_boundary": (
            "Candidate pairs have no exact relationship in the FULL_FIT training set, local ChEMBL 37 "
            "strict pairs, KiRHub, BindingDB 2026-07, or GtoPdb 2026.2. PubMed full text and patent manual review remains "
            "required before claiming international novelty."
        ),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [RANK_PATH, SCORE_PATH, REGISTRY_PATH, STRICT_PATH, KIRHUB_PATH, FDA_PATH, TRAINING_PATH,
                         BINDINGDB_ARTICLES, BINDINGDB_PUBCHEM]
        },
    }
    (OUT / "RETARGETMAP_SPR512_OURS_FROZEN_SUMMARY_V2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# ReTargetMap SPR512 冻结模型独立候选包

本包锁定 **32 个蛋白 × 16 个不同 pair = 512 个实验**：32 个蛋白阳性对照和 480 个盲化候选。

## 直接使用

- `RETARGETMAP_SPR512_OURS_FROZEN_LAB_PACKAGE_V2.xlsx`：主交付文件；实验组优先查看“实验执行表”“蛋白构建清单”“药物采购去重”。
- `RETARGETMAP_SPR512_OURS_FROZEN_LAB_EXECUTION_BLINDED_V2.csv`：不含模型分组与排名的执行表。
- `RETARGETMAP_SPR512_OURS_FROZEN_INTERNAL_MASTER_V2.csv`：内部揭盲与分析表，实验结果冻结前不发给执行人员。

## 已锁定设计

- 冻结 384 核心内 20 个蛋白、扩展 367 空间 12 个蛋白；每蛋白 1 个阳性对照 + 15 个候选。
- 所有候选的准入、分层、优化目标和配额只使用本项目冻结模型；DTIAM 不参与选择，只在内部表保留为实验后的同集事后比较字段。
- 480 个候选来自 {selected['ligand_inchikey'].nunique()} 种 FDA 活性成分；每药 {int(drug_counts.min())}–{int(drug_counts.max())} 个新靶点，中位数 {drug_counts.median():.0f}。
- 候选已排除 FULL_FIT 训练精确 pair、本地 ChEMBL 37 精确关系、KiRHub pair、BindingDB 2026-07 精确 InChIKey–UniProt 关系和 GtoPdb 2026.2 精确关系。
- 仍需在对外宣称“国际首报”前完成人工 PubMed 全文与专利复核；这不阻止按盲化清单启动采购和蛋白预实验。

## 结果判定

主终点为 QC 合格的 SPR KD：`<=100 nM` 强阳性，`100 nM–1 µM` 阳性，`1–10 µM` 灰区，`>10 µM` 或无特异结合为阴性；技术失败不计作阴性。
"""
    (OUT / "README_ZH.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    required = [
        RANK_PATH, SCORE_PATH, REGISTRY_PATH, STRICT_PATH, KIRHUB_PATH, FDA_PATH, TRAINING_PATH,
        BINDINGDB_ARTICLES, BINDINGDB_PUBCHEM, GTOPDB_DIR / "ligands.csv",
        GTOPDB_DIR / "interactions.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    pool, targets, strict, _ = build_candidate_pool()
    pool, binding_rows, gtopdb_rows = add_external_evidence(pool)
    selected = optimize_candidates(pool)
    controls = build_controls(strict, targets)
    write_outputs(selected, controls, targets, binding_rows, gtopdb_rows)


if __name__ == "__main__":
    main()
