#!/usr/bin/env python3
"""Audit and score the frozen 2024 recent-target external panel.

This is a target-to-drug retrieval evaluation: for each frozen new protein,
the unchanged five-seed BioMaster-ODTI V2 ensemble ranks all 720 old drugs.
The two literature positives are used only after scoring to read out rank.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import sys
import zipfile
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


FREEZE = ROOT / "configs/biomaster_recent_novel_target_external_freeze_20260819.json"
OUT = ROOT / "outputs/biomaster_recent_novel_target_external_v1"
FEATURE_AUDIT = OUT / "RECENT_NOVEL_TARGET_FEATURE_AUDIT_V1.json"
TARGET_INDEX = OUT / "RECENT_NOVEL_TARGET_INDEX_V1.csv"
TARGET_PROTBERT = OUT / "RECENT_NOVEL_TARGET_PROTBERT1024_FLOAT32_V1.npy"
TARGET_ESM2 = OUT / "RECENT_NOVEL_TARGET_ESM2_T33_650M_1280_FLOAT32_V1.npy"

BASE = ROOT / "outputs/old_drug_target_sota_v1"
STORE = BASE / "feature_store_v1"
DRUG_INDEX = BASE / "deployment_720x384_feature_store_v1/OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz"
DRUG_FEATURES = BASE / "deployment_720x384_feature_store_v1/OLD_DRUG_MORGAN2048_UINT8_V1.npy"
TRAIN_TARGET_INDEX = STORE / "TARGET_FEATURE_INDEX_V1.csv.gz"
TRAIN_TARGET_PROTBERT = STORE / "PROTBERT1024_FLOAT32_V1.npy"
TRAIN_TARGET_ESM2 = BASE / "public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
TRAIN_PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DEPLOY_TARGET_INDEX = BASE / "deployment_720x384_feature_store_v1/PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz"
PROJECT_888 = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
RUNS = BASE / "biomaster_odti_v2_s5_esm2_formal"
CHEMBL_DB = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
BINDINGDB = [
    ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/external_relation_audit_v7/raw/bindingdb_202607/BindingDB_BindingDB_Articles_202607_tsv.zip",
    ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/external_relation_audit_v7/raw/bindingdb_202607/BindingDB_PubChem_202607_tsv.zip",
]

DATABASE_AUDIT_OUT = OUT / "RECENT_NOVEL_TARGET_DATABASE_AUDIT_V1.json"
SCORES_OUT = OUT / "RECENT_NOVEL_TARGET_720_DRUG_SCORES_V1.csv.gz"
TOP20_OUT = OUT / "RECENT_NOVEL_TARGET_TOP20_V1.csv"
POSITIVES_OUT = OUT / "RECENT_NOVEL_TARGET_POSITIVE_CONTROL_RANKS_V1.csv"
SUMMARY_OUT = OUT / "RECENT_NOVEL_TARGET_EXTERNAL_SUMMARY_V1.json"
REPORT_OUT = OUT / "RECENT_NOVEL_TARGET_EXTERNAL_REPORT_ZH_V1.md"

SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def json_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def scan_zip_tokens(path: Path, tokens: list[str]) -> dict[str, int]:
    """Detect case-insensitive ASCII tokens without unpacking large exports."""

    encoded = {token: token.upper().encode("ascii") for token in tokens}
    found: set[str] = set()
    overlap_width = max(map(len, encoded.values())) - 1
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            previous = b""
            with archive.open(name) as handle:
                while True:
                    chunk = handle.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    blob = (previous + chunk).upper()
                    for token, raw in encoded.items():
                        if raw in blob:
                            found.add(token)
                    previous = blob[-overlap_width:] if overlap_width else b""
    return {token: int(token in found) for token in tokens}


def chembl_target_audit(
    connection: sqlite3.Connection,
    target: dict[str, object],
) -> dict[str, object]:
    accession = str(target["uniprot_accession"])
    gene = str(target["gene_symbol"])
    inchikey = str(target["experimental_positive"]["old_drug_inchikey"])
    components = connection.execute(
        "SELECT component_id, accession, description, sequence FROM component_sequences WHERE accession = ?",
        (accession,),
    ).fetchall()
    description_matches = connection.execute(
        "SELECT COUNT(*) FROM component_sequences WHERE UPPER(description) LIKE ?",
        (f"%{gene.upper()}%",),
    ).fetchone()[0]
    synonym_matches = connection.execute(
        "SELECT COUNT(*) FROM component_synonyms WHERE UPPER(component_synonym) = ?",
        (gene.upper(),),
    ).fetchone()[0]
    target_name_matches = connection.execute(
        "SELECT COUNT(*) FROM target_dictionary WHERE UPPER(pref_name) LIKE ?",
        (f"%{gene.upper()}%",),
    ).fetchone()[0]
    target_rows = connection.execute(
        """
        SELECT COUNT(DISTINCT td.tid)
        FROM target_dictionary td
        JOIN target_components tc ON tc.tid = td.tid
        JOIN component_sequences cs ON cs.component_id = tc.component_id
        WHERE cs.accession = ?
        """,
        (accession,),
    ).fetchone()[0]
    pair_activity_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM activities act
        JOIN assays ass ON ass.assay_id = act.assay_id
        JOIN target_components tc ON tc.tid = ass.tid
        JOIN component_sequences cs ON cs.component_id = tc.component_id
        JOIN compound_structures cmp ON cmp.molregno = act.molregno
        WHERE cs.accession = ? AND cmp.standard_inchi_key = ?
        """,
        (accession, inchikey),
    ).fetchone()[0]
    drugs = connection.execute(
        """
        SELECT md.molregno, md.chembl_id, md.pref_name, md.max_phase
        FROM molecule_dictionary md
        JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE cs.standard_inchi_key = ?
        ORDER BY md.chembl_id
        """,
        (inchikey,),
    ).fetchall()
    return {
        "uniprot_accession": accession,
        "gene_symbol": gene,
        "component_accession_rows": len(components),
        "component_description_gene_matches": int(description_matches),
        "component_synonym_exact_gene_matches": int(synonym_matches),
        "target_dictionary_gene_name_matches": int(target_name_matches),
        "target_dictionary_accession_linked_rows": int(target_rows),
        "experimental_pair_activity_rows": int(pair_activity_rows),
        "old_drug_rows": [
            {"molregno": row[0], "chembl_id": row[1], "pref_name": row[2], "max_phase": row[3]}
            for row in drugs
        ],
        "target_absent_from_chembl37": len(components) == 0 and int(target_rows) == 0,
        "experimental_pair_absent_from_chembl37": int(pair_activity_rows) == 0,
        "old_drug_present_in_chembl37": len(drugs) > 0,
    }


def database_audit(freeze: dict[str, object], drug_index: pd.DataFrame) -> dict[str, object]:
    training_targets = pd.read_csv(TRAIN_TARGET_INDEX, low_memory=False)
    deployment_targets = pd.read_csv(DEPLOY_TARGET_INDEX, low_memory=False)
    universe = pd.read_csv(PROJECT_888, low_memory=False)
    pairs = pd.read_csv(
        TRAIN_PAIRS,
        usecols=["query_accession", "primary_gene", "target_feature_index"],
        low_memory=False,
    )
    tokens = []
    for target in freeze["targets"]:
        tokens.extend([str(target["uniprot_accession"]), str(target["gene_symbol"])])
    bindingdb_results = {rel(path): scan_zip_tokens(path, tokens) for path in BINDINGDB}
    connection = sqlite3.connect(f"file:{CHEMBL_DB}?mode=ro", uri=True)
    version_rows = connection.execute("SELECT name, creation_date, comments FROM version").fetchall()
    target_results = []
    for target in freeze["targets"]:
        accession = str(target["uniprot_accession"])
        gene = str(target["gene_symbol"])
        sequence = str(target["sequence"])
        inchikey = str(target["experimental_positive"]["old_drug_inchikey"])
        old_drug_rows = drug_index.loc[drug_index["ligand_inchikey"].astype(str).eq(inchikey)]
        chembl = chembl_target_audit(connection, target)
        binding_hits = {
            path: {accession: counts[accession], gene: counts[gene]}
            for path, counts in bindingdb_results.items()
        }
        target_results.append(
            {
                "uniprot_accession": accession,
                "gene_symbol": gene,
                "training_target_exact_sequence_rows": int(training_targets["protein_sequence"].astype(str).eq(sequence).sum()),
                "training_pair_accession_rows": int(pairs["query_accession"].astype(str).eq(accession).sum()),
                "training_pair_gene_rows": int(pairs["primary_gene"].astype(str).str.upper().eq(gene.upper()).sum()),
                "deployment_384_accession_rows": int(deployment_targets.get("query_accession", pd.Series(dtype=str)).astype(str).eq(accession).sum()),
                "deployment_384_exact_sequence_rows": int(deployment_targets.get("protein_sequence", pd.Series(dtype=str)).astype(str).eq(sequence).sum()),
                "project_888_accession_rows": int(universe["uniprot_accession"].astype(str).eq(accession).sum()),
                "project_888_exact_sequence_rows": int(universe["sequence"].astype(str).eq(sequence).sum()),
                "old_drug_720_rows": int(len(old_drug_rows)),
                "old_drug_720_record": old_drug_rows.to_dict("records"),
                "bindingdb_literal_token_hits": binding_hits,
                "bindingdb_target_tokens_absent": all(
                    counts[accession] == 0 and counts[gene] == 0
                    for counts in bindingdb_results.values()
                ),
                "chembl37": chembl,
            }
        )
    connection.close()
    checks = {
        "training_target_count_is_428": len(training_targets) == 428,
        "deployment_target_count_is_384": len(deployment_targets) == 384,
        "project_target_universe_is_888": len(universe) == 888,
        "old_drug_universe_is_720": len(drug_index) == 720,
        "both_targets_absent_from_training_sequences": all(row["training_target_exact_sequence_rows"] == 0 for row in target_results),
        "both_targets_absent_from_training_pair_metadata": all(
            row["training_pair_accession_rows"] == 0 and row["training_pair_gene_rows"] == 0
            for row in target_results
        ),
        "both_targets_absent_from_deployment_384": all(
            row["deployment_384_accession_rows"] == 0 and row["deployment_384_exact_sequence_rows"] == 0
            for row in target_results
        ),
        "both_targets_absent_from_project_888": all(
            row["project_888_accession_rows"] == 0 and row["project_888_exact_sequence_rows"] == 0
            for row in target_results
        ),
        "both_targets_absent_from_chembl37": all(row["chembl37"]["target_absent_from_chembl37"] for row in target_results),
        "both_pairs_absent_from_chembl37": all(row["chembl37"]["experimental_pair_absent_from_chembl37"] for row in target_results),
        "both_target_tokens_absent_from_bindingdb_202607": all(row["bindingdb_target_tokens_absent"] for row in target_results),
        "both_experimental_old_drugs_exactly_once_in_720": all(row["old_drug_720_rows"] == 1 for row in target_results),
        "both_experimental_old_drugs_present_in_chembl37": all(row["chembl37"]["old_drug_present_in_chembl37"] for row in target_results),
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminology_boundary": "DATABASE-DISJOINT means absent from the frozen DTI graph, ChEMBL 37 target/pair tables and audited BindingDB 2026-07 exports. Both proteins have UniProt records.",
        "chembl_version": [
            {"name": row[0], "creation_date": row[1], "comments": row[2]} for row in version_rows
        ],
        "bindingdb_exports": {rel(path): sha256(path) for path in BINDINGDB},
        "checks": checks,
        "targets": target_results,
    }
    DATABASE_AUDIT_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_value) + "\n")
    if result["status"] != "PASS":
        raise RuntimeError(f"database audit failed: {json.dumps(checks, indent=2)}")
    return result


def load_checkpoint(seed: int, device: torch.device) -> tuple[RoutedInteractionRankerV2, dict[str, object], dict[str, object]]:
    run = RUNS / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}"
    summary = json.loads((run / "RUN_SUMMARY_V2.json").read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"checkpoint seed {seed} is not PASS")
    checkpoint = torch.load(run / "BEST_MODEL_V2.pt", map_location="cpu", weights_only=False)
    config = ODTIV2Config(**checkpoint["config"])
    model = RoutedInteractionRankerV2(
        family_count=len(checkpoint["families"]),
        config=config,
        use_conplex=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint, summary


def inference_arrays(frame: pd.DataFrame, checkpoint: dict[str, object]) -> dict[str, object]:
    normalization = checkpoint["normalization"]
    family_lookup = {str(name): index for index, name in enumerate(checkpoint["families"])}
    family_index = frame["target_assay_family"].astype(str).map(family_lookup)
    if family_index.isna().any():
        missing = sorted(frame.loc[family_index.isna(), "target_assay_family"].unique())
        raise RuntimeError(f"frozen target family is unavailable in checkpoint: {missing}")
    return {
        "families": checkpoint["families"],
        "family_index": family_index.to_numpy(dtype=np.int64),
        "drug_aux_mean": np.zeros(0, dtype=np.float32),
        "drug_aux_std": np.ones(0, dtype=np.float32),
        "target_mean": np.asarray(normalization["target_mean"], dtype=np.float32),
        "target_std": np.asarray(normalization["target_std"], dtype=np.float32),
        "target_aux_mean": np.asarray(normalization["target_aux_mean"], dtype=np.float32),
        "target_aux_std": np.asarray(normalization["target_aux_std"], dtype=np.float32),
        "target_token_mean": np.zeros(0, dtype=np.float32),
        "target_token_std": np.ones(0, dtype=np.float32),
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


def build_pair_frame(
    freeze: dict[str, object],
    target_index: pd.DataFrame,
    drug_index: pd.DataFrame,
) -> pd.DataFrame:
    chunks = []
    targets_by_accession = {str(row["uniprot_accession"]): row for row in freeze["targets"]}
    for target_row in target_index.sort_values("target_feature_index").to_dict("records"):
        target = targets_by_accession[str(target_row["uniprot_accession"])]
        chunk = drug_index.copy()
        chunk["target_feature_index"] = int(target_row["target_feature_index"])
        chunk["panel_role"] = target["panel_role"]
        chunk["uniprot_accession"] = target["uniprot_accession"]
        chunk["gene_symbol"] = target["gene_symbol"]
        chunk["protein_name"] = target["protein_name"]
        chunk["target_assay_family"] = target["fixed_target_assay_family"]
        chunk["experimental_positive_old_drug_name"] = target["experimental_positive"]["old_drug_name"]
        chunk["experimental_positive_old_drug_inchikey"] = target["experimental_positive"]["old_drug_inchikey"]
        chunk["is_literature_experimental_positive"] = chunk["ligand_inchikey"].astype(str).eq(
            str(target["experimental_positive"]["old_drug_inchikey"])
        )
        chunk["binary_label"] = 0
        chunk["conplex_score"] = 0.0
        chunk["mean_pchembl"] = np.nan
        chunk["min_pchembl"] = np.nan
        chunk["max_pchembl"] = np.nan
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    if len(frame) != 1440 or frame["ligand_inchikey"].nunique() != 720:
        raise RuntimeError("frozen retrieval frame is not exactly 2 x 720")
    if int(frame["is_literature_experimental_positive"].sum()) != 2:
        raise RuntimeError("the two frozen experimental positive drugs are not uniquely represented")
    return frame


def nearest_training_targets(
    target_index: pd.DataFrame,
    target_protbert: np.ndarray,
    target_esm2: np.ndarray,
) -> dict[str, object]:
    train_index = pd.read_csv(TRAIN_TARGET_INDEX).sort_values("target_feature_index").reset_index(drop=True)
    train_protbert = np.asarray(np.load(TRAIN_TARGET_PROTBERT, mmap_mode="r"), dtype=np.float32)
    train_esm2 = np.asarray(np.load(TRAIN_TARGET_ESM2, mmap_mode="r"), dtype=np.float32)
    pair_meta = pd.read_csv(
        TRAIN_PAIRS,
        usecols=["target_feature_index", "primary_gene", "query_accession", "target_chembl_id", "target_assay_family"],
        low_memory=False,
    )

    def collapsed(values: pd.Series) -> str:
        return ";".join(sorted({str(value) for value in values.dropna()}))

    metadata = pair_meta.groupby("target_feature_index", as_index=False).agg(
        primary_gene=("primary_gene", collapsed),
        query_accession=("query_accession", collapsed),
        target_chembl_id=("target_chembl_id", collapsed),
        target_assay_family=("target_assay_family", collapsed),
    )
    metadata = train_index[["target_feature_index", "sequence_key", "sequence_length"]].merge(
        metadata, on="target_feature_index", how="left"
    )

    def similarities(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        reference_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
        return query_norm @ reference_norm.T

    protbert_similarity = similarities(target_protbert, train_protbert)
    esm2_similarity = similarities(target_esm2, train_esm2)
    combined = (protbert_similarity + esm2_similarity) / 2.0
    output = {}
    for row in target_index.itertuples(index=False):
        target_position = int(row.target_feature_index)
        order = np.argsort(-combined[target_position])[:5]
        neighbors = []
        for rank, training_position in enumerate(order, start=1):
            meta = metadata.iloc[int(training_position)].to_dict()
            neighbors.append(
                {
                    "rank": rank,
                    **{key: json_value(value) for key, value in meta.items()},
                    "protbert_cosine": float(protbert_similarity[target_position, training_position]),
                    "esm2_cosine": float(esm2_similarity[target_position, training_position]),
                    "mean_cosine": float(combined[target_position, training_position]),
                }
            )
        output[str(row.uniprot_accession)] = {
            "maximum_protbert_cosine": float(protbert_similarity[target_position].max()),
            "maximum_esm2_cosine": float(esm2_similarity[target_position].max()),
            "maximum_mean_cosine": float(combined[target_position].max()),
            "top5_by_mean_cosine": neighbors,
        }
    return output


def report_markdown(summary: dict[str, object], positives: pd.DataFrame, top20: pd.DataFrame) -> str:
    lines = [
        "# BioMaster 近两年新靶点—老药外部验证 V1",
        "",
        f"生成时间：{summary['created_utc']}",
        "",
        "## 结论",
        "",
        str(summary["conclusion_zh"]),
        "",
        "这是已知实验结果上的**回顾性、靶点冷启动、数据库关系隔离验证**。候选关系在评分前冻结、模型未重训，但论文先于本次模型执行，因此不能称为真正前瞻预测，也不能替代湿实验。",
        "",
        "## 两个冻结实验阳性关系",
        "",
        "| 靶点 | 老药 | 实验 Kd | 模型排名/720 | 百分位 | 判定 |",
        "|---|---|---:|---:|---:|---|",
    ]
    kd_lookup = {
        str(target["uniprot_accession"]): target["experimental_positive"]
        for target in summary["frozen_targets"]
    }
    for row in positives.sort_values("target_feature_index").itertuples(index=False):
        evidence = kd_lookup[str(row.uniprot_accession)]
        verdict = "Top 1%" if row.rank_within_target_720 <= 8 else (
            "Top 5%" if row.rank_within_target_720 <= 36 else "未达预设 Top 5%"
        )
        lines.append(
            f"| {row.gene_symbol} ({row.uniprot_accession}) | {row.drug_names} | "
            f"{evidence['reported_kd_micromolar']} ± {evidence['reported_kd_sd_micromolar']} μM | "
            f"{int(row.rank_within_target_720)}/720 | {row.percentile_within_target_720:.2%} | {verdict} |"
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "",
            "- 两个蛋白均不在冻结的 428 靶点训练特征表、384 靶点部署表、888 靶点项目全集和 ChEMBL 37 靶点字典中。",
            "- 两个实验关系在 ChEMBL 37 中均无活性记录；相关蛋白名/UniProt 号也未命中本地 BindingDB 2026-07 两个导出。",
            "- 两种老药均在固定 720 老药全集中且各唯一出现一次。",
            "- “不在数据库”严格指不在上述 DTI 关系/训练数据库；SEPHS2 与 VPS37C 本身都有 UniProt 条目。",
            "- 两个靶点没有提供结构分支输入，结构掩码恒为 0；逐种子验证最终 logit 与序列/化学基础 logit 完全相同。",
            "",
            "## 每个靶点模型 Top 5 老药（仅计算假设）",
            "",
        ]
    )
    for accession, group in top20.groupby("uniprot_accession", sort=False):
        gene = str(group.iloc[0]["gene_symbol"])
        lines.extend([f"### {gene} ({accession})", "", "| 排名 | 老药 | 集成分数 | 种子标准差 |", "|---:|---|---:|---:|"])
        for row in group.nsmallest(5, "rank_within_target_720").itertuples(index=False):
            lines.append(
                f"| {int(row.rank_within_target_720)} | {row.drug_names} | "
                f"{row.biomaster_ensemble_score:.6f} | {row.model_seed_std:.6f} |"
            )
        lines.extend(["", "本表为模型计算候选，均尚无本次实验验证。", ""])
    lines.extend(
        [
            "## 复现命令",
            "",
            "```bash",
            "PYTHONPATH=third_party/ConPLex .venvs/conplex/bin/python scripts/build_biomaster_recent_novel_target_features_v1.py --force",
            "python scripts/score_biomaster_recent_novel_targets_v1.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    required = [
        FREEZE, FEATURE_AUDIT, TARGET_INDEX, TARGET_PROTBERT, TARGET_ESM2,
        DRUG_INDEX, DRUG_FEATURES, TRAIN_TARGET_INDEX, TRAIN_TARGET_PROTBERT,
        TRAIN_TARGET_ESM2, TRAIN_PAIRS, DEPLOY_TARGET_INDEX, PROJECT_888,
        CHEMBL_DB, *BINDINGDB,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE.read_text())
    feature_audit = json.loads(FEATURE_AUDIT.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_MODEL_SCORING" or feature_audit.get("status") != "PASS":
        raise RuntimeError("frozen panel and feature audit must both pass before scoring")
    if list(freeze["model_policy"]["seeds"]) != SEEDS:
        raise RuntimeError("frozen seed list changed")
    target_index = pd.read_csv(TARGET_INDEX)
    drug_index = pd.read_csv(DRUG_INDEX)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    target_protbert = np.load(TARGET_PROTBERT, mmap_mode="r")
    target_esm2 = np.load(TARGET_ESM2, mmap_mode="r")
    if drug_features.shape != (720, 2048):
        raise RuntimeError(f"unexpected old-drug feature shape: {drug_features.shape}")
    if target_protbert.shape != (2, 1024) or target_esm2.shape != (2, 1280):
        raise RuntimeError("recent target feature shapes changed")
    database = database_audit(freeze, drug_index)
    frame = build_pair_frame(freeze, target_index, drug_index)
    positions = np.arange(len(frame), dtype=np.int64)
    structure = np.zeros((len(frame), 19), dtype=np.float32)
    structure_mask = np.zeros(len(frame), dtype=np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    probability_rows = []
    logit_rows = []
    checkpoint_metadata = []
    expected_checkpoint_config = None
    expected_families = None
    parameter_count = None
    fallback_max_abs = {}
    for seed in SEEDS:
        model, checkpoint, run_summary = load_checkpoint(seed, device)
        if expected_checkpoint_config is None:
            expected_checkpoint_config = checkpoint["config"]
            expected_families = checkpoint["families"]
            parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        elif checkpoint["config"] != expected_checkpoint_config or checkpoint["families"] != expected_families:
            raise RuntimeError("five-seed checkpoint architecture or family vocabulary mismatch")
        arrays = inference_arrays(frame, checkpoint)
        drug_cache = torch.from_numpy(np.asarray(drug_features, dtype=np.float32)).to(device)
        target_cache = torch.from_numpy(
            (np.asarray(target_protbert, dtype=np.float32) - arrays["target_mean"]) / arrays["target_std"]
        ).to(device)
        target_aux_cache = torch.from_numpy(
            (np.asarray(target_esm2, dtype=np.float32) - arrays["target_aux_mean"]) / arrays["target_aux_std"]
        ).to(device)
        output = predict(
            model,
            positions,
            frame,
            drug_features,
            None,
            target_protbert,
            target_esm2,
            None,
            None,
            None,
            None,
            int(checkpoint.get("target_token_max_len", 1022)),
            None,
            None,
            structure,
            structure_mask,
            arrays,
            device,
            4096,
            drug_feature_cache=drug_cache,
            target_feature_cache=target_cache,
            target_aux_feature_cache=target_aux_cache,
        )
        final_logit = np.asarray(output["final_logit"], dtype=np.float64)
        base_logit = np.asarray(output["base_logit"], dtype=np.float64)
        difference = float(np.max(np.abs(final_logit - base_logit)))
        fallback_max_abs[str(seed)] = difference
        if difference > 1e-7:
            raise RuntimeError(f"structure fallback is not exact for seed {seed}: {difference}")
        temperature = float(checkpoint["temperature"])
        probability = 1.0 / (1.0 + np.exp(-np.clip(final_logit / temperature, -60.0, 60.0)))
        probability_rows.append(probability)
        logit_rows.append(final_logit)
        frame[f"score_seed_{seed}"] = probability
        checkpoint_metadata.append(
            {
                "seed": seed,
                "temperature": temperature,
                "checkpoint": rel(RUNS / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}/BEST_MODEL_V2.pt"),
                "checkpoint_sha256": sha256(RUNS / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}/BEST_MODEL_V2.pt"),
                "test_micro_auprc": run_summary["test_metrics"]["micro_auprc"],
                "fallback_max_abs_final_minus_base_logit": difference,
            }
        )
        print(json.dumps({"scored_seed": seed, "pairs": len(frame), "device": str(device)}), flush=True)
        del model, drug_cache, target_cache, target_aux_cache, output
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probability_matrix = np.stack(probability_rows)
    logit_matrix = np.stack(logit_rows)
    frame["biomaster_ensemble_score"] = probability_matrix.mean(axis=0)
    frame["biomaster_ensemble_logit"] = logit_matrix.mean(axis=0)
    frame["model_seed_std"] = probability_matrix.std(axis=0)
    frame["rank_within_target_720"] = frame.groupby("uniprot_accession")["biomaster_ensemble_score"].rank(
        method="min", ascending=False
    ).astype(np.int16)
    frame["percentile_within_target_720"] = 1.0 - (frame["rank_within_target_720"] - 1) / 719.0
    frame["top_1_percent"] = frame["rank_within_target_720"].le(8)
    frame["top_5_percent"] = frame["rank_within_target_720"].le(36)
    frame["top_10"] = frame["rank_within_target_720"].le(10)
    score_columns = [
        column for column in frame.columns
        if column not in {"binary_label", "conplex_score", "mean_pchembl", "min_pchembl", "max_pchembl"}
    ]
    frame[score_columns].to_csv(SCORES_OUT, index=False, compression="gzip")
    top20 = frame.loc[frame["rank_within_target_720"].le(20), score_columns].sort_values(
        ["target_feature_index", "rank_within_target_720"]
    )
    positives = frame.loc[frame["is_literature_experimental_positive"], score_columns].sort_values(
        "target_feature_index"
    )
    top20.to_csv(TOP20_OUT, index=False)
    positives.to_csv(POSITIVES_OUT, index=False)
    nearest = nearest_training_targets(
        target_index,
        np.asarray(target_protbert, dtype=np.float32),
        np.asarray(target_esm2, dtype=np.float32),
    )
    positive_records = [
        {key: json_value(value) for key, value in row.items()}
        for row in positives.to_dict("records")
    ]
    primary = positives.loc[positives["panel_role"].eq("PRIMARY_SHOWCASE")].iloc[0]
    successful = int(positives["top_5_percent"].sum())
    stretch = int(positives["top_1_percent"].sum())
    if successful == 2:
        conclusion = "两个冻结实验阳性关系均进入各自靶点的老药 Top 5%，构成双案例回顾性 target-cold 命中。"
        claim_level = "TWO_OF_TWO_TOP5_PERCENT_RETROSPECTIVE_TARGET_COLD_HIT"
    elif bool(primary["top_5_percent"]):
        conclusion = "主案例 SEPHS2–达沙替尼进入 Top 5%，但共同报告案例未达阈值；可展示主案例，不能宣称双案例稳定命中。"
        claim_level = "PRIMARY_ONLY_TOP5_PERCENT_RETROSPECTIVE_TARGET_COLD_HIT"
    else:
        conclusion = "主案例未进入预设 Top 5%；本次验证应作为诚实失败/边界结果报告，不能用于宣称新靶点老药发现能力。"
        claim_level = "PRIMARY_MISSED_PREDECLARED_TOP5_PERCENT"
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "evaluation_type": "RETROSPECTIVE_TARGET_COLD_DATABASE_RELATION_DISJOINT_OLD_DRUG_RETRIEVAL",
        "prospective_claim_allowed": False,
        "claim_level": claim_level,
        "conclusion_zh": conclusion,
        "freeze_id": freeze["freeze_id"],
        "frozen_targets": freeze["targets"],
        "ranking_space": {"targets": 2, "old_drugs_per_target": 720, "pairs": 1440, "direction": "old drugs ranked within each target"},
        "predeclared_threshold_results": {
            "top5_percent_cutoff_rank": 36,
            "top1_percent_cutoff_rank": 8,
            "positive_controls_top5_percent": successful,
            "positive_controls_top1_percent": stretch,
            "positive_controls_total": 2,
        },
        "positive_control_results": positive_records,
        "model": {
            "name": "BioMaster-ODTI V2 ESM2 S5 five-seed ensemble",
            "parameter_count_per_seed": parameter_count,
            "checkpoint_config": expected_checkpoint_config,
            "families": expected_families,
            "seeds": SEEDS,
            "checkpoint_metadata": checkpoint_metadata,
            "score_semantics": "temperature-scaled ensemble ranking score; not a calibrated probability of physical binding",
        },
        "structure_fallback": {
            "structure_mask_zero_rows": int((structure_mask == 0).sum()),
            "exact_fallback_required": True,
            "max_abs_final_minus_base_logit_by_seed": fallback_max_abs,
        },
        "applicability": {"nearest_training_targets": nearest},
        "database_audit_status": database["status"],
        "limitations": [
            "The experimental outcomes were published before this evaluation; this is retrospective rather than prospective discovery.",
            "Panel selection used known 2024 literature outcomes, although target/family choices and both-pair reporting were frozen before model scoring.",
            "Database-disjoint means absent from the audited DTI relationship stores, not absent from UniProt.",
            "The frozen model was optimized primarily on observed ChEMBL DTI labels and is not guaranteed to rank every target class equally.",
        ],
        "artifacts": {
            "database_audit": rel(DATABASE_AUDIT_OUT),
            "scores": rel(SCORES_OUT),
            "top20": rel(TOP20_OUT),
            "positive_control_ranks": rel(POSITIVES_OUT),
            "report_zh": rel(REPORT_OUT),
        },
    }
    REPORT_OUT.write_text(report_markdown(summary, positives, top20), encoding="utf-8")
    summary["artifact_sha256"] = {
        rel(path): sha256(path)
        for path in [DATABASE_AUDIT_OUT, SCORES_OUT, TOP20_OUT, POSITIVES_OUT, REPORT_OUT]
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_value) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_value))


if __name__ == "__main__":
    main()
