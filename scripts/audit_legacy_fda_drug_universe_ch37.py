#!/usr/bin/env python3
"""Audit the legacy 915-row FDA-labelled drug library before rebuilding it."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path(__file__).resolve().parents[1]
DRUGS = ROOT / "data/processed/drug_library_active_moiety_v4.csv"
PROJECT_DRUGS = ROOT / "configs/project_drugs_v4.csv"
CHEMBL = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
OUTDIR = ROOT / "outputs/drug_universe_rebuild_v1"

SALT_WORDS = {
    "acetate", "aluminum", "ammonium", "benzoate", "besylate", "bitartrate", "bromide",
    "calcium", "chloride", "choline", "citrate", "dihydrochloride", "diphosphate", "disodium",
    "ditromethamine", "fumarate", "hemifumarate", "hemisulfate", "hydrobromide", "hydrochloride",
    "hyclate", "lactate", "lysine", "magnesium", "maleate", "malate", "mesylate", "nitrate",
    "olamine", "oxalate", "pamoate", "phosphate", "potassium", "sodium", "strontium", "succinate",
    "sulfate", "sulphate", "tartrate", "tosylate", "tromethamine",
}

NONTHERAPEUTIC_PATTERN = re.compile(
    r"\b(air|medical air|oxygen|nitrogen|xenon|contrast|diagnostic|imaging|fluorescein|"
    r"gallium|technetium|radium|iobenguane|ioflupane|florbetapir|flortaucipir|flutemetamol|"
    r"fluoroestradiol|fluorodopa|flurpiridaz|fluciclovine|piflufolastat|gozetotide|edotreotide)\b",
    re.IGNORECASE,
)


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def core_name(value: object) -> str:
    tokens = normalize_name(value).split()
    while tokens and tokens[-1] in SALT_WORDS:
        tokens.pop()
    return " ".join(tokens)


def standardize_smiles(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return ""
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    molecule = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def load_chembl() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    connection = sqlite3.connect(CHEMBL)
    metadata = pd.read_sql_query(
        """
        SELECT md.chembl_id,
               md.pref_name AS chembl_pref_name_local,
               md.max_phase AS chembl_max_phase_local,
               md.therapeutic_flag,
               md.dosed_ingredient,
               md.molecule_type,
               md.structure_type,
               md.first_approval AS chembl_first_approval_local,
               md.prodrug AS chembl_prodrug,
               md.inorganic_flag,
               md.polymer_flag,
               md.withdrawn_flag,
               parent.chembl_id AS parent_chembl_id,
               active.chembl_id AS active_chembl_id,
               parent_structure.canonical_smiles AS parent_chembl_smiles,
               active_structure.canonical_smiles AS active_chembl_smiles
        FROM molecule_dictionary md
        LEFT JOIN molecule_hierarchy hierarchy ON hierarchy.molregno = md.molregno
        LEFT JOIN molecule_dictionary parent ON parent.molregno = hierarchy.parent_molregno
        LEFT JOIN molecule_dictionary active ON active.molregno = hierarchy.active_molregno
        LEFT JOIN compound_structures parent_structure ON parent_structure.molregno = hierarchy.parent_molregno
        LEFT JOIN compound_structures active_structure ON active_structure.molregno = hierarchy.active_molregno
        """,
        connection,
    )
    synonyms = pd.read_sql_query(
        """
        SELECT md.chembl_id, synonym.synonyms
        FROM molecule_dictionary md
        JOIN molecule_synonyms synonym ON synonym.molregno = md.molregno
        WHERE synonym.synonyms IS NOT NULL
        """,
        connection,
    )
    connection.close()
    name_map: dict[str, set[str]] = {}
    for chembl_id, group in synonyms.groupby("chembl_id"):
        name_map[str(chembl_id)] = {normalize_name(value) for value in group["synonyms"] if normalize_name(value)}
    for row in metadata.itertuples(index=False):
        if pd.notna(row.chembl_pref_name_local):
            name_map.setdefault(str(row.chembl_id), set()).add(normalize_name(row.chembl_pref_name_local))
    return metadata, name_map


def best_name_match(query: str, candidates: set[str]) -> tuple[str, float]:
    if not query or not candidates:
        return "", 0.0
    best = max(candidates, key=lambda value: SequenceMatcher(None, query, value).ratio())
    return best, SequenceMatcher(None, query, best).ratio()


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    drugs = pd.read_csv(DRUGS, low_memory=False)
    project = pd.read_csv(PROJECT_DRUGS, low_memory=False)
    metadata, name_map = load_chembl()
    if len(drugs) != 915 or drugs["drug_id"].nunique() != 915:
        raise ValueError("Legacy drug table must contain 915 unique record IDs")

    data = drugs.merge(metadata, on="chembl_id", how="left", validate="many_to_one")
    data["drug_name_normalized"] = data["drug_name"].map(normalize_name)
    data["drug_core_name"] = data["drug_name"].map(core_name)
    data["name_candidate_count"] = data["chembl_id"].map(lambda value: len(name_map.get(str(value), set())))
    best = [best_name_match(name, name_map.get(str(chembl_id), set()))
            for name, chembl_id in zip(data["drug_name_normalized"], data["chembl_id"])]
    data["best_chembl_name_match"] = [value[0] for value in best]
    data["best_chembl_name_similarity"] = [round(value[1], 4) for value in best]
    data["exact_name_match"] = [
        name in name_map.get(str(chembl_id), set())
        for name, chembl_id in zip(data["drug_name_normalized"], data["chembl_id"])
    ]
    data["core_name_match"] = [
        bool(core) and any(core_name(candidate) == core for candidate in name_map.get(str(chembl_id), set()))
        for core, chembl_id in zip(data["drug_core_name"], data["chembl_id"])
    ]
    data["combination_product_name"] = data["drug_name"].fillna("").astype(str).str.contains(
        r"\s+and\s+|,", case=False, regex=True
    )
    data["chembl37_id_present"] = data["chembl_pref_name_local"].notna() | data["molecule_type"].notna()
    data["identity_status"] = "NAME_ID_MISMATCH_REVIEW"
    data.loc[data["core_name_match"], "identity_status"] = "CORE_NAME_SUPPORTED"
    data.loc[data["exact_name_match"], "identity_status"] = "EXACT_NAME_SUPPORTED"
    data.loc[
        data["chembl37_id_present"] & data["name_candidate_count"].eq(0),
        "identity_status",
    ] = "NO_CHEMBL_NAME_EVIDENCE"
    data.loc[data["combination_product_name"], "identity_status"] = "COMBINATION_PRODUCT_SINGLE_STRUCTURE"
    data.loc[~data["chembl37_id_present"], "identity_status"] = "CHEMBL37_ID_MISSING"

    data["chembl_id_record_count"] = data.groupby("chembl_id")["drug_id"].transform("size")
    data["chembl_id_core_name_count"] = data.groupby("chembl_id")["drug_core_name"].transform("nunique")
    data["chembl_id_conflicting_names"] = (
        data["chembl_id_record_count"].gt(1) & data["chembl_id_core_name_count"].gt(1)
    )
    data["model_structure_record_count"] = data.groupby("model_ligand_smiles")["drug_id"].transform("size")
    data["duplicate_model_structure"] = data["model_structure_record_count"].gt(1)

    data["model_ligand_smiles_standardized_audit"] = data["model_ligand_smiles"].map(standardize_smiles)
    data["active_chembl_smiles_standardized"] = data["active_chembl_smiles"].map(standardize_smiles)
    data["active_chembl_id_differs"] = (
        data["active_chembl_id"].notna() & data["active_chembl_id"].ne(data["chembl_id"])
    )
    data["model_differs_from_active_chembl_structure"] = (
        data["active_chembl_smiles_standardized"].ne("")
        & data["model_ligand_smiles_standardized_audit"].ne(data["active_chembl_smiles_standardized"])
    )
    data["legacy_active_moiety_is_only_structure_standardization"] = True

    text = (
        data["drug_name"].fillna("").astype(str) + " "
        + data["therapeutic_area"].fillna("").astype(str) + " "
        + data["mechanism_of_action"].fillna("").astype(str)
    )
    data["likely_diagnostic_gas_or_nontherapeutic"] = (
        text.str.contains(NONTHERAPEUTIC_PATTERN, na=False)
        | data["therapeutic_area"].fillna("").astype(str).str.contains("Diagnostic/Imaging", case=False)
    )
    project_ids = set(project["drug_chembl_id"].astype(str))
    data["in_legacy_project750"] = data["drug_id"].astype(str).isin(project_ids)

    data["rebuild_status"] = "PROVISIONAL_ENTITY_PASS"
    data.loc[data["duplicate_model_structure"], "rebuild_status"] = "HOLD_DUPLICATE_STRUCTURE_COLLAPSE"
    data.loc[
        data["active_chembl_id_differs"] | data["model_differs_from_active_chembl_structure"],
        "rebuild_status",
    ] = "HOLD_ACTIVE_SPECIES_REVIEW"
    data.loc[
        data["identity_status"].isin({
            "NAME_ID_MISMATCH_REVIEW", "NO_CHEMBL_NAME_EVIDENCE",
            "COMBINATION_PRODUCT_SINGLE_STRUCTURE", "CHEMBL37_ID_MISSING"
        })
        | data["chembl_id_conflicting_names"],
        "rebuild_status",
    ] = "HOLD_IDENTITY_REVIEW"
    data.loc[data["likely_diagnostic_gas_or_nontherapeutic"], "rebuild_status"] = "EXCLUDE_NONTHERAPEUTIC_SCOPE"

    identity_counts = data["identity_status"].value_counts().to_dict()
    summary = {
        "legacy_rows": 915,
        "legacy_unique_chembl_ids": int(data["chembl_id"].nunique()),
        "legacy_unique_model_structures": int(data["model_ligand_smiles"].nunique()),
        "chembl37_ids_present": int(data["chembl37_id_present"].sum()),
        "chembl37_ids_missing_rows": int((~data["chembl37_id_present"]).sum()),
        "identity_status_counts": identity_counts,
        "chembl_id_conflicting_name_rows": int(data["chembl_id_conflicting_names"].sum()),
        "chembl_id_conflicting_name_groups": int(
            data.loc[data["chembl_id_conflicting_names"], "chembl_id"].nunique()
        ),
        "duplicate_model_structure_rows": int(data["duplicate_model_structure"].sum()),
        "duplicate_model_structure_groups": int(
            data.loc[data["duplicate_model_structure"], "model_ligand_smiles"].nunique()
        ),
        "combination_product_rows": int(data["combination_product_name"].sum()),
        "likely_diagnostic_gas_or_nontherapeutic_rows": int(
            data["likely_diagnostic_gas_or_nontherapeutic"].sum()
        ),
        "chembl_prodrug_rows": int(pd.to_numeric(data["chembl_prodrug"], errors="coerce").eq(1).sum()),
        "active_chembl_id_differs_rows": int(data["active_chembl_id_differs"].sum()),
        "model_differs_from_active_chembl_structure_rows": int(
            data["model_differs_from_active_chembl_structure"].sum()
        ),
        "legacy_project_rows": int(data["in_legacy_project750"].sum()),
        "legacy_project_unique_model_structures": int(project["model_ligand_smiles"].nunique()),
        "rebuild_status_counts": data["rebuild_status"].value_counts().to_dict(),
        "primary_fda_application_or_label_identifier_available": 0,
        "conclusion": "The legacy 915/750/723 spaces are audit inputs, not a reusable frozen FDA active-entity universe.",
    }
    actions = pd.DataFrame([
        {"顺序": 1, "模块": "FDA批准身份", "必须补充": "application/NDA/BLA、批准日期、活性成分、官方标签来源", "当前问题": "915行没有FDA主键或官方来源"},
        {"顺序": 2, "模块": "名称与ChEMBL映射", "必须补充": "逐条确认generic ingredient与ChEMBL/parent ID", "当前问题": "存在名称-ID冲突、复方被映成单一成分和ChEMBL37缺失ID"},
        {"顺序": 3, "模块": "实体层级", "必须补充": "product→ingredient→parent→administered species→active species", "当前问题": "旧active_moiety仅做最大片段和去电荷"},
        {"顺序": 4, "模块": "结构标准", "必须补充": "立体化学、同位素、前药/代谢物、盐型和多组分结构", "当前问题": "不能用一个SMILES同时代表产品和活性物"},
        {"顺序": 5, "模块": "范围排除", "必须补充": "气体、诊断显像、聚合物/树脂、非治疗性和非直接target-engagement分子", "当前问题": "旧915仍含明显范围外记录"},
        {"顺序": 6, "模块": "唯一计算实体", "必须补充": "冻结compound_key和唯一模型结构，同时保留产品到实体映射", "当前问题": "915记录仅有876结构；旧750仅有723结构"},
        {"顺序": 7, "模块": "pair空间", "必须补充": "338×最终唯一实体，并单独标记exact-known和控制", "当前问题": "药物实体未冻结，不能生成正式pair"},
    ])

    data.to_csv(OUTDIR / "LEGACY_FDA_DRUG_LIBRARY_AUDIT_915.csv", index=False)
    data.loc[data["rebuild_status"].ne("PROVISIONAL_ENTITY_PASS")].to_csv(
        OUTDIR / "LEGACY_FDA_DRUG_HIGH_RISK_ROWS.csv", index=False
    )
    actions.to_csv(OUTDIR / "DRUG_UNIVERSE_REBUILD_ACTIONS_ZH.csv", index=False)
    (OUTDIR / "LEGACY_FDA_DRUG_AUDIT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
