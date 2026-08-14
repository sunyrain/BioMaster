#!/usr/bin/env python3
"""Build a residue-level known-pocket atlas and compare it with P2Rank predictions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
PDBE = ROOT / "outputs/chembl37_known_pocket_atlas/pdbe/PDBE_EXPERIMENTAL_LIGAND_POCKET_INSTANCES.csv.gz"
SPECIALIZED = ROOT / "outputs/chembl37_known_pocket_atlas/specialized_sources"
SCPDB = ROOT / "outputs/chembl37_known_pocket_atlas/scpdb/SCPDB_PROJECT_POCKET_MAPPINGS.csv.gz"
P2RANK = ROOT / "outputs/target_universe_ch37_v2/p2rank_all_exact_v2"
OUTDIR = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas"

MINIMUM_RESIDUES = 3
JACCARD_THRESHOLD = 0.10
KNOWN_RECALL_THRESHOLD = 0.25
CENTER_THRESHOLDS = (4.0, 8.0)


def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalized_keys(frame: pd.DataFrame, chain_column: str) -> pd.DataFrame:
    output = frame.copy()
    output["uniprot_accession"] = clean_series(output["uniprot_accession"]).str.upper()
    output["pdb_id"] = clean_series(output["pdb_id"]).str.lower()
    output[chain_column] = clean_series(output[chain_column])
    output["ligand_id"] = clean_series(output["ligand_id"]).str.upper()
    return output


def make_key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    return set(map(tuple, frame[columns].drop_duplicates().itertuples(index=False, name=None)))


def read_source_annotations() -> tuple[set[tuple[str, ...]], set[tuple[str, ...]], set[tuple[str, ...]], set[tuple[str, ...]], set[str]]:
    biolip = pd.read_csv(
        SPECIALIZED / "BIOLIP2_PROJECT_INTERACTION_SITES.csv.gz", low_memory=False, dtype=str
    ).rename(columns={"receptor_chain": "chain_id"})
    biolip = normalized_keys(biolip, "chain_id")
    biolip_keys = make_key_set(biolip, ["uniprot_accession", "pdb_id", "chain_id", "ligand_id"])
    affinity_tokens = {"", "-", "none", "nan", "na", "n/a"}
    affinity_columns = ["affinity_manual", "affinity_moad", "affinity_pdbbind_cn", "affinity_bindingdb"]
    affinity_mask = biolip[affinity_columns].fillna("").apply(
        lambda row: any(str(value).strip().lower() not in affinity_tokens for value in row), axis=1
    )
    biolip_affinity_keys = make_key_set(
        biolip.loc[affinity_mask], ["uniprot_accession", "pdb_id", "chain_id", "ligand_id"]
    )

    klifs = pd.read_csv(SPECIALIZED / "KLIFS_PROJECT_STRUCTURES.csv.gz", low_memory=False, dtype=str)
    klifs = klifs.rename(columns={"pdb": "pdb_id", "chain": "chain_id"})
    klifs["uniprot_accession"] = clean_series(klifs["uniprot_accession"]).str.upper()
    klifs["pdb_id"] = clean_series(klifs["pdb_id"]).str.lower()
    klifs["chain_id"] = clean_series(klifs["chain_id"])
    klifs_structure_keys = make_key_set(klifs, ["uniprot_accession", "pdb_id", "chain_id"])
    klifs_ligand = klifs.rename(columns={"ligand": "ligand_id"})
    klifs_ligand["ligand_id"] = clean_series(klifs_ligand["ligand_id"]).str.upper()
    klifs_ligand_keys = make_key_set(
        klifs_ligand.loc[klifs_ligand["ligand_id"].ne("")],
        ["uniprot_accession", "pdb_id", "chain_id", "ligand_id"],
    )

    scpdb = pd.read_csv(SCPDB, low_memory=False, dtype=str)
    scpdb_ids = set(
        clean_series(scpdb.loc[scpdb["scpdb_mapping_level"].eq("EXACT_CHAIN_LIGAND"), "pdbe_pocket_instance_id"])
    ) - {""}
    return biolip_keys, biolip_affinity_keys, klifs_structure_keys, klifs_ligand_keys, scpdb_ids


def preferred_resolution(method: str, resolution: Any) -> bool:
    value = pd.to_numeric(resolution, errors="coerce")
    method_lower = str(method).lower()
    if "nmr" in method_lower:
        return True
    if pd.isna(value):
        return False
    if "electron" in method_lower:
        return float(value) <= 3.5
    return float(value) <= 3.0


def usable_resolution(method: str, resolution: Any) -> bool:
    value = pd.to_numeric(resolution, errors="coerce")
    method_lower = str(method).lower()
    if "nmr" in method_lower:
        return True
    if pd.isna(value):
        return False
    return float(value) <= 4.0


def annotate_pdbe() -> pd.DataFrame:
    pdbe = normalized_keys(pd.read_csv(PDBE, low_memory=False), "chain_id")
    biolip, biolip_affinity, klifs_structures, klifs_ligands, scpdb_ids = read_source_annotations()
    key4 = list(zip(pdbe["uniprot_accession"], pdbe["pdb_id"], pdbe["chain_id"], pdbe["ligand_id"]))
    key3 = list(zip(pdbe["uniprot_accession"], pdbe["pdb_id"], pdbe["chain_id"]))
    pdbe["biolip2_support"] = [value in biolip for value in key4]
    pdbe["biolip2_affinity_support"] = [value in biolip_affinity for value in key4]
    pdbe["klifs_structure_support"] = [value in klifs_structures for value in key3]
    pdbe["klifs_ligand_support"] = [value in klifs_ligands for value in key4]
    pdbe["scpdb_support"] = pdbe["pocket_instance_id"].isin(scpdb_ids)
    pdbe["preferred_structure_quality"] = [
        preferred_resolution(method, resolution)
        for method, resolution in zip(pdbe["experimental_method"], pdbe["resolution"])
    ]
    pdbe["usable_structure_quality"] = [
        usable_resolution(method, resolution)
        for method, resolution in zip(pdbe["experimental_method"], pdbe["resolution"])
    ]
    pdbe["canonical_residue_mapping_valid"] = pdbe["pdbe_sequence_exact"].fillna(False).astype(bool)
    pdbe["benchmark_eligible"] = (
        pdbe["canonical_residue_mapping_valid"]
        & pdbe["binding_residue_count"].ge(MINIMUM_RESIDUES)
        & ~pdbe["ligand_class"].isin({"SOLVENT_OR_ADDITIVE", "ION_OR_TINY_FRAGMENT"})
    )
    specialized = pdbe[["biolip2_support", "scpdb_support", "klifs_structure_support"]].any(axis=1)
    pdbe["known_pocket_grade"] = np.select(
        [
            pdbe["ligand_class"].eq("DRUG_MAPPED") & (pdbe["preferred_structure_quality"] | specialized),
            specialized,
            pdbe["ligand_class"].isin({"DRUG_MAPPED", "DRUGLIKE_UNMAPPED"}),
            pdbe["ligand_class"].isin({"COFACTOR", "SMALL_FUNCTIONAL_OR_FRAGMENT"}),
        ],
        [
            "K1_DRUG_MAPPED_EXPERIMENTAL",
            "K2_SPECIALIZED_CURATED_SITE",
            "K3_EXPERIMENTAL_DRUGLIKE_SITE",
            "K4_FUNCTIONAL_OR_FRAGMENT_SITE",
        ],
        default="EXCLUDED",
    )
    pdbe.loc[~pdbe["benchmark_eligible"], "known_pocket_grade"] = "EXCLUDED"
    return pdbe


def gpcrdb_rows(master: pd.DataFrame) -> pd.DataFrame:
    path = SPECIALIZED / "GPCRDB_PROJECT_INTERACTION_POCKETS.csv.gz"
    gpcr = pd.read_csv(path, low_memory=False)
    if gpcr.empty:
        return gpcr
    meta = master.set_index("uniprot_accession")[["target_chembl_id", "gene_symbol"]]
    gpcr = gpcr.join(meta, on="uniprot_accession", how="inner")
    gpcr["residue_set_key"] = clean_series(gpcr["uniprot_residue_positions"]).map(
        lambda value: ";".join(map(str, sorted({int(token) for token in value.split(";") if token})))
    )
    gpcr = gpcr[gpcr["binding_residue_count"].ge(MINIMUM_RESIDUES)].copy()
    gpcr["known_pocket_grade"] = "K2_SPECIALIZED_CURATED_SITE"
    gpcr["pdbe_instance_count"] = 0
    gpcr["unique_pdb_count"] = 1
    gpcr["unique_ligand_count"] = 1
    gpcr["representative_pdb_id"] = gpcr["pdb_id"]
    gpcr["representative_ligand_id"] = gpcr["ligand_name"]
    gpcr["representative_ligand_name"] = gpcr["ligand_name"]
    gpcr["evidence_sources"] = "GPCRDB"
    for column in [
        "drug_mapped_support", "biolip2_support", "biolip2_affinity_support",
        "scpdb_support", "klifs_structure_support", "klifs_ligand_support",
    ]:
        gpcr[column] = False
    gpcr["gpcrdb_support"] = True
    gpcr["preferred_structure_quality"] = [
        preferred_resolution(method, resolution)
        for method, resolution in zip(gpcr["experimental_method"], gpcr["resolution"])
    ]
    gpcr["usable_structure_quality"] = [
        usable_resolution(method, resolution)
        for method, resolution in zip(gpcr["experimental_method"], gpcr["resolution"])
    ]
    return gpcr


GRADE_ORDER = {
    "K1_DRUG_MAPPED_EXPERIMENTAL": 1,
    "K2_SPECIALIZED_CURATED_SITE": 2,
    "K3_EXPERIMENTAL_DRUGLIKE_SITE": 3,
    "K4_FUNCTIONAL_OR_FRAGMENT_SITE": 4,
}


def collapse_pockets(pdbe: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    eligible = pdbe[pdbe["benchmark_eligible"]].copy()
    eligible["residue_set_key"] = clean_series(eligible["uniprot_residue_positions"]).map(
        lambda value: ";".join(map(str, sorted({int(token) for token in value.split(";") if token})))
    )
    eligible["grade_order"] = eligible["known_pocket_grade"].map(GRADE_ORDER)
    eligible = eligible.sort_values(
        ["uniprot_accession", "residue_set_key", "grade_order", "preferred_structure_quality"],
        ascending=[True, True, True, False],
    )
    rows: list[dict[str, Any]] = []
    boolean_columns = [
        "biolip2_support", "biolip2_affinity_support", "scpdb_support",
        "klifs_structure_support", "klifs_ligand_support",
        "preferred_structure_quality", "usable_structure_quality",
    ]
    for (accession, residue_key), group in eligible.groupby(
        ["uniprot_accession", "residue_set_key"], sort=False
    ):
        first = group.iloc[0]
        flags = {column: bool(group[column].any()) for column in boolean_columns}
        flags["gpcrdb_support"] = False
        flags["drug_mapped_support"] = bool(group["ligand_class"].eq("DRUG_MAPPED").any())
        sources = ["PDBE"]
        for flag, source in [
            (flags["biolip2_support"], "BIOLIP2"),
            (flags["scpdb_support"], "SCPDB"),
            (flags["klifs_structure_support"], "KLIFS"),
        ]:
            if flag:
                sources.append(source)
        rows.append({
            "target_chembl_id": first["target_chembl_id"],
            "gene_symbol": first["gene_symbol"],
            "uniprot_accession": accession,
            "residue_set_key": residue_key,
            "binding_residue_count": len(residue_key.split(";")),
            "known_pocket_grade": first["known_pocket_grade"],
            "pdbe_instance_count": len(group),
            "unique_pdb_count": group["pdb_id"].nunique(),
            "unique_ligand_count": group["ligand_id"].nunique(),
            "representative_pdb_id": first["pdb_id"],
            "representative_chain_id": first["chain_id"],
            "representative_ligand_id": first["ligand_id"],
            "representative_ligand_name": first["ligand_name"],
            "representative_method": first["experimental_method"],
            "representative_resolution": first["resolution"],
            "evidence_sources": ";".join(sources),
            **flags,
        })
    collapsed = pd.DataFrame(rows)
    gpcr = gpcrdb_rows(master)
    if not gpcr.empty:
        gpcr_columns = list(collapsed.columns)
        for column in gpcr_columns:
            if column not in gpcr:
                gpcr[column] = pd.NA
        collapsed = pd.concat([collapsed, gpcr[gpcr_columns]], ignore_index=True)

    # Merge records with the same canonical residue set. This removes repeated
    # structures without conflating overlapping but genuinely distinct sites.
    final_rows: list[dict[str, Any]] = []
    for (accession, residue_key), group in collapsed.groupby(
        ["uniprot_accession", "residue_set_key"], sort=False
    ):
        group = group.copy()
        group["grade_order"] = group["known_pocket_grade"].map(GRADE_ORDER)
        group = group.sort_values("grade_order")
        first = group.iloc[0]
        sources = sorted({source for value in group["evidence_sources"].dropna() for source in str(value).split(";")})
        row = first.drop(labels=["grade_order"]).to_dict()
        for column in [
            "drug_mapped_support", "biolip2_support", "biolip2_affinity_support",
            "scpdb_support", "klifs_structure_support", "klifs_ligand_support",
            "gpcrdb_support", "preferred_structure_quality", "usable_structure_quality",
        ]:
            row[column] = bool(group[column].fillna(False).any())
        row["pdbe_instance_count"] = int(pd.to_numeric(group["pdbe_instance_count"], errors="coerce").fillna(0).sum())
        row["unique_pdb_count"] = int(pd.to_numeric(group["unique_pdb_count"], errors="coerce").fillna(0).sum())
        row["unique_ligand_count"] = int(pd.to_numeric(group["unique_ligand_count"], errors="coerce").fillna(0).sum())
        row["evidence_sources"] = ";".join(sources)
        row["known_pocket_id"] = "KP:" + accession + ":" + hashlib.sha1(residue_key.encode()).hexdigest()[:12]
        final_rows.append(row)
    return pd.DataFrame(final_rows)


def parse_p2rank_predictions(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in master.itertuples(index=False):
        path_value = str(target.p2rank_file) if pd.notna(target.p2rank_file) else ""
        path = Path(path_value)
        if not path.is_file():
            fallback = P2RANK / f"AF-{target.uniprot_accession}-F1-model_v6.pdb_predictions.csv"
            path = fallback
        if not path.is_file() or path.stat().st_size <= 120:
            continue
        frame = pd.read_csv(path)
        frame.columns = [value.strip() for value in frame.columns]
        for record in frame.to_dict(orient="records"):
            residue_ids = str(record.get("residue_ids", ""))
            positions = sorted({int(value) for value in re.findall(r"_(\d+)", residue_ids)})
            rows.append({
                "uniprot_accession": target.uniprot_accession,
                "p2rank_name": str(record.get("name", "")).strip(),
                "p2rank_rank": int(record["rank"]),
                "p2rank_score": float(record["score"]),
                "p2rank_probability": float(record["probability"]),
                "p2rank_center_x": float(record["center_x"]),
                "p2rank_center_y": float(record["center_y"]),
                "p2rank_center_z": float(record["center_z"]),
                "p2rank_residue_positions": ";".join(map(str, positions)),
                "p2rank_residue_count": len(positions),
            })
    return pd.DataFrame(rows)


def ca_coordinates(path: str) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    file_path = Path(path)
    if not file_path.is_file():
        return output
    with file_path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            try:
                position = int(line[22:26])
                output[position] = np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ])
            except ValueError:
                continue
    return output


def scope_metrics(known: set[int], centroid: np.ndarray | None, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "max_known_recall": math.nan,
            "max_jaccard": math.nan,
            "min_center_distance": math.nan,
            "residue_match": False,
            "center_match_4a": False,
            "center_match_8a": False,
            "combined_match_8a": False,
        }
    recalls: list[float] = []
    jaccards: list[float] = []
    distances: list[float] = []
    for prediction in predictions:
        predicted = {int(value) for value in prediction["p2rank_residue_positions"].split(";") if value}
        intersection = len(known & predicted)
        recalls.append(intersection / len(known) if known else 0.0)
        union = len(known | predicted)
        jaccards.append(intersection / union if union else 0.0)
        if centroid is not None:
            center = np.array([
                prediction["p2rank_center_x"], prediction["p2rank_center_y"], prediction["p2rank_center_z"]
            ])
            distances.append(float(np.linalg.norm(centroid - center)))
    max_recall = max(recalls)
    max_jaccard = max(jaccards)
    min_distance = min(distances) if distances else math.nan
    residue_match = max_recall >= KNOWN_RECALL_THRESHOLD or max_jaccard >= JACCARD_THRESHOLD
    center4 = bool(pd.notna(min_distance) and min_distance <= CENTER_THRESHOLDS[0])
    center8 = bool(pd.notna(min_distance) and min_distance <= CENTER_THRESHOLDS[1])
    return {
        "max_known_recall": max_recall,
        "max_jaccard": max_jaccard,
        "min_center_distance": min_distance,
        "residue_match": residue_match,
        "center_match_4a": center4,
        "center_match_8a": center8,
        "combined_match_8a": residue_match or center8,
    }


def compare_p2rank(pockets: pd.DataFrame, predictions: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    prediction_groups = {
        key: value.sort_values("p2rank_rank").to_dict(orient="records")
        for key, value in predictions.groupby("uniprot_accession")
    }
    master_index = master.set_index("uniprot_accession")
    coordinate_cache: dict[str, dict[int, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for record in pockets.to_dict(orient="records"):
        accession = record["uniprot_accession"]
        target = master_index.loc[accession]
        coordinates = coordinate_cache.setdefault(accession, ca_coordinates(str(target["af_pdb_path"])))
        known = {int(value) for value in str(record["residue_set_key"]).split(";") if value}
        known_coords = [coordinates[value] for value in known if value in coordinates]
        centroid = np.mean(known_coords, axis=0) if len(known_coords) >= max(1, math.ceil(0.5 * len(known))) else None
        preds = prediction_groups.get(accession, [])
        output = dict(record)
        output.update({
            "af_exact_sequence_model": bool(target["af_exact_sequence_model"]),
            "p2rank_status": target["p2rank_status"],
            "p2rank_tier": target["p2rank_tier"],
            "known_centroid_residue_coverage": len(known_coords) / len(known) if known else math.nan,
            "known_centroid_x": centroid[0] if centroid is not None else math.nan,
            "known_centroid_y": centroid[1] if centroid is not None else math.nan,
            "known_centroid_z": centroid[2] if centroid is not None else math.nan,
        })
        for label, subset in [("top1", preds[:1]), ("top3", preds[:3]), ("all", preds)]:
            for metric, value in scope_metrics(known, centroid, subset).items():
                output[f"p2rank_{label}_{metric}"] = value
        rows.append(output)
    return pd.DataFrame(rows)


def target_summary(master: pd.DataFrame, compared: pd.DataFrame, pdbe: pd.DataFrame) -> pd.DataFrame:
    output = master[[
        "target_chembl_id", "gene_symbol", "uniprot_accession", "target_class_l1",
        "ot_project_assay_family", "is_gpcr", "chembl_kinase",
        "ot_sm_structure_with_ligand", "ot_sm_high_quality_pocket", "ot_sm_med_quality_pocket",
        "af_exact_sequence_model", "p2rank_status", "p2rank_tier", "p2rank_top_score",
        "p2rank_top_probability", "pocket_residue_count", "pocket_mean_plddt",
    ]].copy()
    rows: list[dict[str, Any]] = []
    for accession, group in compared.groupby("uniprot_accession", sort=False):
        row: dict[str, Any] = {
            "uniprot_accession": accession,
            "known_unique_pocket_count": len(group),
            "known_k1_pocket_count": group["known_pocket_grade"].eq("K1_DRUG_MAPPED_EXPERIMENTAL").sum(),
            "known_k1_k2_pocket_count": group["known_pocket_grade"].isin({"K1_DRUG_MAPPED_EXPERIMENTAL", "K2_SPECIALIZED_CURATED_SITE"}).sum(),
            "known_drug_mapped": bool(group["drug_mapped_support"].any()),
            "known_biolip2": bool(group["biolip2_support"].any()),
            "known_biolip2_affinity": bool(group["biolip2_affinity_support"].any()),
            "known_scpdb": bool(group["scpdb_support"].any()),
            "known_klifs": bool(group["klifs_structure_support"].any()),
            "known_gpcrdb": bool(group["gpcrdb_support"].any()),
        }
        for scope in ["top1", "top3", "all"]:
            for metric in ["residue_match", "center_match_4a", "center_match_8a", "combined_match_8a"]:
                row[f"p2rank_{scope}_any_known_{metric}"] = bool(group[f"p2rank_{scope}_{metric}"].any())
            row[f"p2rank_{scope}_best_known_recall"] = group[f"p2rank_{scope}_max_known_recall"].max()
            row[f"p2rank_{scope}_best_jaccard"] = group[f"p2rank_{scope}_max_jaccard"].max()
            row[f"p2rank_{scope}_minimum_center_distance"] = group[f"p2rank_{scope}_min_center_distance"].min()
        for grade_label, mask in [
            ("k1", group["known_pocket_grade"].eq("K1_DRUG_MAPPED_EXPERIMENTAL")),
            ("k1_k2", group["known_pocket_grade"].isin({"K1_DRUG_MAPPED_EXPERIMENTAL", "K2_SPECIALIZED_CURATED_SITE"})),
        ]:
            subset = group[mask]
            for scope in ["top1", "top3", "all"]:
                row[f"p2rank_{scope}_any_{grade_label}_combined_match_8a"] = bool(
                    len(subset) and subset[f"p2rank_{scope}_combined_match_8a"].any()
                )
        rows.append(row)
    known = pd.DataFrame(rows)
    output = output.merge(known, on="uniprot_accession", how="left", validate="one_to_one")
    count_columns = [value for value in output.columns if value.endswith("_count")]
    output[count_columns] = output[count_columns].fillna(0).astype(int)
    boolean_columns = [value for value in output.columns if value.startswith("known_") and value not in count_columns]
    boolean_columns += [value for value in output.columns if "_any_" in value and value.startswith("p2rank_")]
    output[boolean_columns] = output[boolean_columns].fillna(False).astype(bool)

    raw = pdbe.groupby("uniprot_accession").agg(
        pdbe_raw_ligand_instance_count=("pocket_instance_id", "size"),
        pdbe_raw_unique_pdb_count=("pdb_id", "nunique"),
    ).reset_index()
    output = output.merge(raw, on="uniprot_accession", how="left", validate="one_to_one")
    output[["pdbe_raw_ligand_instance_count", "pdbe_raw_unique_pdb_count"]] = output[[
        "pdbe_raw_ligand_instance_count", "pdbe_raw_unique_pdb_count"
    ]].fillna(0).astype(int)
    return output


def comparison_tables(targets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    known = targets[targets["known_unique_pocket_count"].gt(0)].copy()
    evaluable = known[known["af_exact_sequence_model"] & known["p2rank_status"].isin({"completed", "completed_no_pocket"})].copy()
    rows: list[dict[str, Any]] = []
    for label, subset in [
        ("全部实验口袋靶点", evaluable),
        ("K1药物映射实验口袋", evaluable[evaluable["known_k1_pocket_count"].gt(0)]),
        ("K1+K2高置信实验口袋", evaluable[evaluable["known_k1_k2_pocket_count"].gt(0)]),
        ("非GPCR实验口袋", evaluable[~evaluable["is_gpcr"]]),
        ("GPCR实验口袋", evaluable[evaluable["is_gpcr"]]),
        ("激酶实验口袋", evaluable[evaluable["chembl_kinase"]]),
    ]:
        denominator = len(subset)
        for scope in ["top1", "top3", "all"]:
            for metric, metric_zh in [
                ("residue_match", "残基重叠达标"),
                ("center_match_4a", "中心距离<=4A"),
                ("center_match_8a", "中心距离<=8A"),
                ("combined_match_8a", "残基或中心<=8A"),
            ]:
                column = f"p2rank_{scope}_any_known_{metric}"
                numerator = int(subset[column].sum()) if denominator else 0
                rows.append({
                    "评估集合": label,
                    "P2Rank范围": scope.upper(),
                    "匹配口径": metric_zh,
                    "可评估靶点数": denominator,
                    "匹配靶点数": numerator,
                    "匹配率": numerator / denominator if denominator else math.nan,
                })
    specialized_summary = json.loads(
        (SPECIALIZED / "SPECIALIZED_SOURCE_COLLECTION_SUMMARY.json").read_text(encoding="utf-8")
    )
    scpdb_summary = json.loads(
        (SCPDB.parent / "SCPDB_COLLECTION_SUMMARY.json").read_text(encoding="utf-8")
    )
    pdbe_summary = json.loads(
        (PDBE.parent / "PDBE_COLLECTION_SUMMARY.json").read_text(encoding="utf-8")
    )
    pdbe_raw_targets = pdbe_summary["targets_with_any_ligand_pocket"]
    source_rows = [
        {"来源/口径": "PDBe原始任意配体接触位点", "覆盖靶点数": pdbe_raw_targets, "占888比例": pdbe_raw_targets / 888},
        {"来源/口径": "BioLiP2原始项目匹配", "覆盖靶点数": specialized_summary["biolip2_targets"], "占888比例": specialized_summary["biolip2_targets"] / 888},
        {"来源/口径": "KLIFS项目结构覆盖", "覆盖靶点数": specialized_summary["klifs_targets_with_structures"], "占888比例": specialized_summary["klifs_targets_with_structures"] / 888},
        {"来源/口径": "GPCRdb项目结构覆盖", "覆盖靶点数": specialized_summary["gpcrdb_targets_with_structures"], "占888比例": specialized_summary["gpcrdb_targets_with_structures"] / 888},
        {"来源/口径": "scPDB链+配体精确映射", "覆盖靶点数": scpdb_summary["exact_mapped_targets"], "占888比例": scpdb_summary["exact_mapped_targets"] / 888},
    ]
    for column, label in [
        ("known_unique_pocket_count", "PDBe/统一实验口袋"),
        ("known_drug_mapped", "PDBe药物映射口袋"),
        ("known_biolip2", "统一基准内BioLiP2支持"),
        ("known_biolip2_affinity", "统一基准内BioLiP2带亲和记录"),
        ("known_scpdb", "统一基准内scPDB支持"),
        ("known_klifs", "统一基准内KLIFS支持"),
        ("known_gpcrdb", "统一基准内GPCRdb支持"),
        ("ot_sm_structure_with_ligand", "Open Targets结构配体标记"),
        ("ot_sm_high_quality_pocket", "Open Targets高质量口袋标记"),
        ("ot_sm_med_quality_pocket", "Open Targets中质量口袋标记"),
    ]:
        values = targets[column].gt(0) if column.endswith("_count") else targets[column].fillna(False).astype(bool)
        source_rows.append({"来源/口径": label, "覆盖靶点数": int(values.sum()), "占888比例": float(values.mean())})
    return {
        "p2rank_comparison": pd.DataFrame(rows),
        "source_coverage": pd.DataFrame(source_rows),
    }


def representative_pockets(compared: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = compared.copy()
    frame["grade_order"] = frame["known_pocket_grade"].map(GRADE_ORDER)
    frame["source_count"] = frame["evidence_sources"].fillna("").map(
        lambda value: len(set(str(value).split(";")) - {""})
    )
    frame = frame.sort_values(
        [
            "uniprot_accession", "grade_order", "source_count", "pdbe_instance_count",
            "unique_pdb_count", "binding_residue_count",
        ],
        ascending=[True, True, False, False, False, False],
    ).drop_duplicates("uniprot_accession")
    target_meta = master[["uniprot_accession", "is_gpcr", "chembl_kinase"]]
    frame = frame.merge(target_meta, on="uniprot_accession", how="left", validate="one_to_one")
    evaluable = frame[
        frame["af_exact_sequence_model"]
        & frame["p2rank_status"].isin({"completed", "completed_no_pocket"})
    ]
    rows: list[dict[str, Any]] = []
    for label, subset in [
        ("全部靶点代表口袋", evaluable),
        ("K1药物映射代表口袋", evaluable[evaluable["known_pocket_grade"].eq("K1_DRUG_MAPPED_EXPERIMENTAL")]),
        ("K1+K2高置信代表口袋", evaluable[evaluable["grade_order"].le(2)]),
        ("非GPCR代表口袋", evaluable[~evaluable["is_gpcr"]]),
        ("GPCR代表口袋", evaluable[evaluable["is_gpcr"]]),
        ("激酶代表口袋", evaluable[evaluable["chembl_kinase"]]),
    ]:
        denominator = len(subset)
        for scope in ["top1", "top3", "all"]:
            for metric, metric_zh in [
                ("residue_match", "残基重叠达标"),
                ("center_match_4a", "中心距离<=4A"),
                ("center_match_8a", "中心距离<=8A"),
                ("combined_match_8a", "残基或中心<=8A"),
            ]:
                column = f"p2rank_{scope}_{metric}"
                numerator = int(subset[column].sum()) if denominator else 0
                rows.append({
                    "评估集合": label,
                    "P2Rank范围": scope.upper(),
                    "匹配口径": metric_zh,
                    "可评估靶点数": denominator,
                    "匹配靶点数": numerator,
                    "匹配率": numerator / denominator if denominator else math.nan,
                })
    return frame, pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER, low_memory=False)
    if len(master) != 888:
        raise RuntimeError("Expected official 888 target universe")
    pdbe = annotate_pdbe()
    pockets = collapse_pockets(pdbe, master)
    predictions = parse_p2rank_predictions(master)
    compared = compare_p2rank(pockets, predictions, master)
    targets = target_summary(master, compared, pdbe)
    tables = comparison_tables(targets)
    representatives, representative_counts = representative_pockets(compared, master)

    pdbe.to_csv(OUTDIR / "KNOWN_POCKET_INSTANCES_ANNOTATED_FULL.csv.gz", index=False, compression="gzip")
    pockets.to_csv(OUTDIR / "KNOWN_POCKET_CANONICAL_RESIDUE_SETS.csv.gz", index=False, compression="gzip")
    predictions.to_csv(OUTDIR / "P2RANK_ALL_PREDICTED_POCKETS_875.csv.gz", index=False, compression="gzip")
    compared.to_csv(OUTDIR / "KNOWN_POCKET_VS_P2RANK_INSTANCE_COMPARISON.csv.gz", index=False, compression="gzip")
    targets.to_csv(OUTDIR / "TARGET_KNOWN_POCKET_AND_P2RANK_SUMMARY_888.csv", index=False)
    representatives.to_csv(OUTDIR / "TARGET_REPRESENTATIVE_KNOWN_POCKET_737.csv", index=False)
    tables["p2rank_comparison"].to_csv(OUTDIR / "P2RANK_COMPARISON_COUNTS_ZH.csv", index=False)
    representative_counts.to_csv(OUTDIR / "P2RANK_REPRESENTATIVE_POCKET_COMPARISON_ZH.csv", index=False)
    tables["source_coverage"].to_csv(OUTDIR / "KNOWN_POCKET_SOURCE_COVERAGE_ZH.csv", index=False)

    summary = {
        "target_universe": 888,
        "pdbe_raw_instances": int(len(pdbe)),
        "pdbe_benchmark_eligible_instances": int(pdbe["benchmark_eligible"].sum()),
        "canonical_unique_known_pockets": int(len(pockets)),
        "targets_with_known_pocket": int(targets["known_unique_pocket_count"].gt(0).sum()),
        "targets_with_k1_drug_mapped_pocket": int(targets["known_k1_pocket_count"].gt(0).sum()),
        "targets_with_k1_k2_high_conf_pocket": int(targets["known_k1_k2_pocket_count"].gt(0).sum()),
        "p2rank_prediction_targets": int(predictions["uniprot_accession"].nunique()),
        "p2rank_predicted_pockets": int(len(predictions)),
        "known_pocket_targets_with_exact_af": int((targets["known_unique_pocket_count"].gt(0) & targets["af_exact_sequence_model"]).sum()),
        "thresholds": {
            "minimum_binding_residues": MINIMUM_RESIDUES,
            "jaccard": JACCARD_THRESHOLD,
            "known_residue_recall": KNOWN_RECALL_THRESHOLD,
            "center_distance_angstrom": list(CENTER_THRESHOLDS),
        },
    }
    (OUTDIR / "KNOWN_POCKET_ATLAS_AND_P2RANK_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
