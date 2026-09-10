"""Read-only, validated disease evidence for the complete project catalog.

TxGNN arrays are memory mapped in their physical CSV order. Large Open Targets,
EC-KG and pair coverage tables are streamed into a fingerprinted SQLite cache;
only one page is materialized at query time. No legacy cancer cache is consulted.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)
SNAPSHOT = "outputs/biomaster_disease_evidence_720x888_20260909"
CACHE_VERSION = 1
LOGITS = "TXGNN_INDICATION_LOGITS.npy"
MASK = "TXGNN_EXCLUSION_MASK.npy"
OT = "OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet"
EC = "ECKG_RELEVANT_RELATION_AUDIT.parquet"
PAIR = "PAIR_COVERAGE_720X888.parquet"
NODES = "ECKG_PROJECT_AND_DISEASE_NODES.parquet"
REQUIRED = (
    "VALIDATION.json", "SOURCE_AND_ARTIFACT_MANIFEST.json", "FINAL_SUMMARY.json",
    "TXGNN_RUN_COMPLETE.json", "TXGNN_SCORE_DRUG_ORDER.csv",
    "TXGNN_SCORE_DISEASE_ORDER.csv", "DRUG_COVERAGE_FINAL_720.csv",
    "TARGET_COVERAGE_888.csv", "OT_TARGET_COVERAGE_888.csv",
    "TXGNN_ALL_DISEASE_DIRECTIONS.csv", "DISEASE_ID_CROSSWALK.csv",
    LOGITS, MASK, OT, EC, PAIR, NODES,
)
DIRECTION_LABELS = {
    "breast": "乳腺", "cardiovascular": "心血管", "chromosomal": "染色体",
    "connective_tissue": "结缔组织", "dermatologic": "皮肤",
    "ear_nose_throat": "耳鼻喉", "endocrine": "内分泌", "eye_and_adnexa": "眼及附属器",
    "gastrointestinal": "胃肠", "hematologic": "血液", "immune": "免疫",
    "infection": "感染", "metabolic": "代谢", "musculoskeletal": "肌肉骨骼",
    "neoplasm": "肿瘤", "neurological": "神经", "obstetric": "产科",
    "poisoning_and_toxicity": "中毒与毒性", "psychiatric": "精神",
    "reproductive": "生殖", "respiratory": "呼吸", "renal_and_urinary": "肾及泌尿",
    "syndromic": "综合征", "unclassified": "未分类",
}
EXCLUSION_REASONS = {
    1: "旧图已知 indication / off-label / contraindication 关系",
    2: "EC-KG 治疗、试验或禁忌来源的保守排除标注（非已核实适应证清单）",
    4: "药物结构与图标识冲突，暂扣疾病线索解释资格",
}
SCORE_SEMANTICS = "raw_logit_not_calibrated_probability"
BOUNDARIES = [
    "TxGNN 原始 logit 未经临床或 SPR 概率校准；疾病假设不证明治疗有效或直接结合。",
    "默认仅展示 exclusion_mask == 0 的疾病线索；原始追溯包含已知关系及身份暂扣行。",
    "合并疾病节点保持完整 ID；合并组分数不能解释为每个子病种的特异性预测。",
    "23 个来源本体方向是多标签归类，计数不可相加；未分类疾病仍保留。",
    "EC-KG 的关系、谓词、来源与歧义分别保留，不合并为已知结合阳性。",
    "疾病一致性不证明抑制或激动方向，不改变原结合排名或 SPR 设计，未进行新训练。",
    "图外药物的来源关系补齐不意味着旧 TxGNN 已获得该药物的推理能力。",
]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(rows: list[dict], field: str, label: str) -> dict[str, dict]:
    result = {row.get(field, ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError(f"Missing or duplicate {label} identifiers")
    return result


def _sql_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid source column name: {name!r}")
    return '"' + name + '"'


class DiseaseEvidenceStore:
    """Optional all-direction snapshot with fail-closed validation.

    ``root`` is the repository root; drug and target mappings use the explorer's
    exact model InChIKeys and ChEMBL target IDs. A missing snapshot disables this
    store. A present but incomplete, corrupt or inconsistent snapshot raises
    ValueError and must not be silently replaced by older evidence.
    """

    def __init__(self, root: Path, drugs: dict, targets: dict):
        self.root = Path(root).resolve()
        self.path = self.root / SNAPSHOT
        self.drugs, self.targets = drugs, targets
        self.enabled = False
        self.counts: dict[str, int] = {}
        self.sources: list[dict] = []
        self.summary: dict[str, Any] = {"status": "snapshot_missing", "enabled": False}
        if not self.path.exists():
            return
        try:
            self._initialize()
        except (ValueError, OSError, KeyError, TypeError, sqlite3.Error, pa.ArrowException) as error:
            raise ValueError(f"Invalid all-direction disease snapshot {self.path}: {error}") from error
        self.enabled = True

    def _initialize(self) -> None:
        missing = [name for name in REQUIRED if not (self.path / name).is_file()]
        if missing:
            raise ValueError("Missing required artifacts: " + ", ".join(missing))
        self.validation = _json(self.path / "VALIDATION.json")
        self.manifest = _json(self.path / "SOURCE_AND_ARTIFACT_MANIFEST.json")
        self.final = _json(self.path / "FINAL_SUMMARY.json")
        run = _json(self.path / "TXGNN_RUN_COMPLETE.json")
        if self.validation.get("all_pass") is not True or not all(self.validation.get("checks", {}).values()):
            raise ValueError("Snapshot validation did not pass")
        if self.manifest.get("no_training") is not True or run.get("relation") != "indication":
            raise ValueError("Snapshot does not declare the reviewed indication inference contract")
        hashes = {}
        expected = self.manifest.get("artifact_sha256", {})
        for name in REQUIRED:
            digest = _sha256(self.path / name)
            hashes[name] = digest
            if name != "SOURCE_AND_ARTIFACT_MANIFEST.json":
                declared = expected.get(f"{SNAPSHOT}/{name}") or expected.get(name)
                if not declared or declared != digest:
                    raise ValueError(f"Artifact SHA256 mismatch or missing manifest entry: {name}")
        self._drug_coverage = _unique(_rows(self.path / "DRUG_COVERAGE_FINAL_720.csv"), "ligand_inchikey", "drug coverage")
        self._target_coverage = _unique(_rows(self.path / "TARGET_COVERAGE_888.csv"), "target_chembl_id", "target coverage")
        self._ot_coverage = _unique(_rows(self.path / "OT_TARGET_COVERAGE_888.csv"), "target_chembl_id", "OT coverage")
        if set(self._drug_coverage) != set(self.drugs) or set(self._target_coverage) != set(self.targets):
            raise ValueError("Snapshot identity universe differs from the explorer catalog")
        if set(self._ot_coverage) != set(self.targets):
            raise ValueError("OT coverage does not preserve the complete target universe")
        drug_order = _rows(self.path / "TXGNN_SCORE_DRUG_ORDER.csv")
        _unique(drug_order, "ligand_inchikey", "matrix drug order")
        # drug_idx is the OLD GRAPH index, not a matrix row. Never use it here.
        self._drug_rows = {row["ligand_inchikey"]: i for i, row in enumerate(drug_order)}
        if not set(self._drug_rows).issubset(self.drugs):
            raise ValueError("Matrix contains drugs outside the reviewed catalog")
        self._diseases = _rows(self.path / "TXGNN_SCORE_DISEASE_ORDER.csv")
        _unique(self._diseases, "id", "matrix disease order")
        self.logits = np.load(self.path / LOGITS, mmap_mode="r", allow_pickle=False)
        self.mask = np.load(self.path / MASK, mmap_mode="r", allow_pickle=False)
        shape = (len(drug_order), len(self._diseases))
        if self.logits.shape != shape or self.mask.shape != shape or len(shape) != 2:
            raise ValueError("Matrix dimensions disagree with physical CSV orders")
        if not np.issubdtype(self.logits.dtype, np.floating) or not np.issubdtype(self.mask.dtype, np.integer):
            raise ValueError("Invalid score or exclusion mask dtype")
        if not np.isfinite(self.logits).all() or np.any(self.mask < 0) or np.any(self.mask > 7):
            raise ValueError("Nonfinite logits or unsupported exclusion mask bits")
        self._eligible_counts = np.count_nonzero(self.mask == 0, axis=1)
        self._holds = set()
        for identifier, index in self._drug_rows.items():
            coverage = self._drug_coverage[identifier]
            holds = bool(np.any(self.mask[index] & 4)) or "CONFLICT_HOLD" in coverage.get("identity_concordance", "")
            if holds:
                self._holds.add(identifier)
                if not np.all(self.mask[index] & 4) or self._eligible_counts[index]:
                    raise ValueError(f"Identity-held row is not completely excluded: {identifier}")
            if _bool(coverage.get("interpretation_eligible")) != (not holds):
                raise ValueError(f"Interpretation eligibility and final mask disagree: {identifier}")
        self._load_diseases()
        self._nodes = {}
        for batch in pq.ParquetFile(self.path / NODES).iter_batches(batch_size=8192, columns=["id", "name"]):
            self._nodes.update({row["id"]: row["name"] for row in batch.to_pylist()})
        identity_names = {f"{kind}:{key}": _text(value.get("name")) for kind, entities in (("drug", self.drugs), ("target", self.targets)) for key, value in entities.items()}
        fingerprint = {"cache_version": CACHE_VERSION, "source_hashes": hashes, "entity_names": identity_names}
        self.fingerprint = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        self.cache_path = self.root / "outputs/biomaster_explorer/cache" / f"disease_v{CACHE_VERSION}_{self.fingerprint}.sqlite"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self._cache_valid():
                self._build_cache()
        self._load_counts()
        self._validate_counts(run)
        self._publish_summary()

    def _load_diseases(self) -> None:
        classified = _unique(_rows(self.path / "TXGNN_ALL_DISEASE_DIRECTIONS.csv"), "id", "disease directions")
        if set(classified) != {row["id"] for row in self._diseases}:
            raise ValueError("Direction labels do not cover the score disease universe")
        self._members: dict[int, list[str]] = defaultdict(list)
        self._member_column: dict[str, int] = {}
        scopes: dict[int, str] = {}
        for row in _rows(self.path / "DISEASE_ID_CROSSWALK.csv"):
            column = int(row["score_column"])
            if not 0 <= column < len(self._diseases) or row["txgnn_disease_id"] != self._diseases[column]["id"]:
                raise ValueError("Disease crosswalk column disagrees with the physical score order")
            member = row["disease_id"]
            if member in self._member_column and self._member_column[member] != column:
                raise ValueError("Disease member maps to multiple score columns")
            if member not in self._members[column]:
                self._members[column].append(member)
            self._member_column[member] = column
            scopes[column] = row["mapping_scope"]
        self._direction_masks = {key: np.zeros(len(self._diseases), dtype=bool) for key in DIRECTION_LABELS}
        self._disease_search = []
        for index, row in enumerate(self._diseases):
            directions = [part.removeprefix("speciality_") for part in classified[row["id"]].get("direction_labels", "").split(";") if part and part != "UNCLASSIFIED_BY_EC_ONTOLOGY"]
            if any(key not in DIRECTION_LABELS or key == "unclassified" for key in directions):
                raise ValueError(f"Unknown source ontology direction: {directions}")
            row["directions"] = list(dict.fromkeys(directions))
            row["mapping_scope"] = scopes.get(index, "UNMAPPED_TO_DISEASE_ONTOLOGY")
            for key in directions or ["unclassified"]:
                self._direction_masks[key][index] = True
            self._disease_search.append(" ".join([row.get("node_name", ""), row["id"], *self._members[index]]).casefold())

    def _cache_valid(self) -> bool:
        if not self.cache_path.is_file():
            return False
        try:
            with self._connection() as connection:
                row = connection.execute("SELECT value FROM cache_meta WHERE key='fingerprint'").fetchone()
                return bool(row and row[0] == self.fingerprint)
        except sqlite3.Error:
            return False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.cache_path.as_uri() + "?mode=ro", uri=True, timeout=60)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _entity_name(self, key: str) -> str:
        if key.startswith("drug:"):
            return _text(self.drugs.get(key[5:], {}).get("name")) or key[5:]
        if key.startswith("target:"):
            return _text(self.targets.get(key[7:], {}).get("name")) or key[7:]
        return self._nodes.get(key) or key

    def _direction_column(self, column: Any) -> list[str]:
        column = _int(column, -1)
        return self._diseases[column]["directions"] if 0 <= column < len(self._diseases) else []

    def _build_cache(self) -> None:
        temporary = self.cache_path.with_name(self.cache_path.name + "." + uuid.uuid4().hex + ".tmp")
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("CREATE TABLE cache_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            connection.execute("CREATE TABLE ec_entities(entity_key TEXT NOT NULL,row_id INTEGER NOT NULL,PRIMARY KEY(entity_key,row_id)) WITHOUT ROWID")
            for table, source in (("ot", OT), ("ec", EC), ("pairs", PAIR)):
                parquet = pq.ParquetFile(self.path / source)
                schema = parquet.schema_arrow
                columns = schema.names
                fields = []
                for field in schema:
                    kind = "INTEGER" if pa.types.is_integer(field.type) or pa.types.is_boolean(field.type) else "REAL" if pa.types.is_floating(field.type) else "TEXT"
                    fields.append(f"{_sql_name(field.name)} {kind}")
                connection.execute(f"CREATE TABLE {table}(_ordinal INTEGER PRIMARY KEY,{','.join(fields)},_search TEXT NOT NULL,_directions TEXT NOT NULL)")
                placeholders = ",".join("?" for _ in range(len(columns) + 3))
                ordinal = 0
                for batch in parquet.iter_batches(batch_size=8192):
                    values, entities = [], []
                    for row in batch.to_pylist():
                        ordinal += 1
                        if table == "ot":
                            if row["target_chembl_id"] not in self.targets:
                                raise ValueError("OT table contains an unregistered target")
                            search = " ".join(_text(row.get(key)) for key in ("disease_name", "disease_id", "therapeutic_areas", "source", "txgnn_mapping_scope"))
                            directions = self._direction_column(row.get("txgnn_score_column"))
                        elif table == "ec":
                            project, other = row["project_key"], row["other_id"]
                            if not (project.startswith("drug:") and project[5:] in self.drugs or project.startswith("target:") and project[7:] in self.targets):
                                raise ValueError("EC table contains an unregistered project endpoint")
                            search = " ".join([self._entity_name(project), self._entity_name(other), *[_text(value) for value in row.values()]])
                            directions = self._direction_column(self._member_column.get(other))
                            entities.append((project, ordinal))
                            if (other.startswith("drug:") and other[5:] in self.drugs or other.startswith("target:") and other[7:] in self.targets) and other != project:
                                entities.append((other, ordinal))
                        else:
                            if row["ligand_inchikey"] not in self.drugs or row["target_chembl_id"] not in self.targets:
                                raise ValueError("Pair coverage contains an unregistered endpoint")
                            search, directions = "", []
                        clean = [json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else None if isinstance(row.get(key), float) and not math.isfinite(row[key]) else row.get(key) for key in columns]
                        values.append((ordinal, *clean, search.casefold(), "|" + "|".join(directions or ["unclassified"]) + "|"))
                    connection.executemany(f"INSERT INTO {table} VALUES({placeholders})", values)
                    if entities:
                        connection.executemany("INSERT INTO ec_entities VALUES(?,?)", entities)
                    connection.commit()
                LOG.info("Disease explorer indexed %s: %s rows", table, ordinal)
                connection.execute("INSERT INTO cache_meta VALUES(?,?)", (table + "_rows", str(ordinal)))
            connection.execute("CREATE UNIQUE INDEX ot_identity ON ot(target_chembl_id,disease_id)")
            connection.execute("CREATE INDEX ot_ranking ON ot(target_chembl_id,overall_score DESC,_ordinal)")
            connection.execute("CREATE INDEX ec_pair ON ec(project_key,other_id,_ordinal)")
            connection.execute("CREATE UNIQUE INDEX pair_identity ON pairs(ligand_inchikey,target_chembl_id)")
            connection.execute("INSERT INTO cache_meta VALUES('fingerprint',?)", (self.fingerprint,))
            connection.commit()
            connection.close()
            os.replace(temporary, self.cache_path)
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise

    def _load_counts(self) -> None:
        with self._connection() as connection:
            self._ot_counts = {row[0]: row[1] for row in connection.execute("SELECT target_chembl_id,count(*) FROM ot GROUP BY target_chembl_id")}
            self._ec_counts = {row[0]: row[1] for row in connection.execute("SELECT entity_key,count(*) FROM ec_entities GROUP BY entity_key")}
            ec_classes = dict(connection.execute("SELECT relation_class,count(*) FROM ec GROUP BY relation_class"))
            raw_counts = dict(connection.execute("SELECT key,value FROM cache_meta"))
        eligible = int(self._eligible_counts.sum())
        self.counts = {
            "txgnn_raw_scored_drugs": len(self._drug_rows),
            "txgnn_drugs": len(self._drug_rows) - len(self._holds),
            "txgnn_diseases": len(self._diseases), "txgnn_associations": eligible,
            "txgnn_raw_scores": int(self.logits.size), "txgnn_masked_scores": int(self.logits.size) - eligible,
            "txgnn_directions": len(DIRECTION_LABELS) - 1, "txgnn_identity_holds": len(self._holds),
            "target_diseases": int(raw_counts["ot_rows"]), "target_disease_associations": int(raw_counts["ot_rows"]),
            "disease_targets": len(self._ot_counts), "eckg_relations": int(raw_counts["ec_rows"]),
            "disease_pair_coverage": int(raw_counts["pairs_rows"]),
            "eckg_drugs": sum(_int(row.get("ec_node_count")) > 0 for row in self._drug_coverage.values()),
            "eckg_targets": sum(_int(row.get("ec_node_count")) > 0 for row in self._target_coverage.values()),
            "eckg_drug_disease_relations": ec_classes.get("DRUG_DISEASE", 0),
            "eckg_drug_target_relations": ec_classes.get("DRUG_TARGET", 0),
            "eckg_target_disease_relations": ec_classes.get("TARGET_DISEASE", 0),
            "txgnn_source_evidence_only_drugs": sum(row.get("disease_evidence_status") == "SOURCE_EVIDENCE_ONLY_NO_USABLE_TXGNN_SCORE" for row in self._drug_coverage.values()),
            "disease_unavailable_drugs": sum(row.get("disease_evidence_status") == "IDENTIFIED_ENTITY_NO_DISEASE_EVIDENCE_IN_THIS_RUN" for row in self._drug_coverage.values()),
        }

    def _validate_counts(self, run: dict) -> None:
        expected = {
            "project_drugs": len(self.drugs), "project_targets": len(self.targets),
            "raw_scored_drugs": self.counts["txgnn_raw_scored_drugs"],
            "scored_diseases": self.counts["txgnn_diseases"], "score_count": self.counts["txgnn_raw_scores"],
            "interpretation_eligible_drugs": self.counts["txgnn_drugs"],
            "total_masked_query_cells": self.counts["txgnn_masked_scores"],
            "identity_conflict_hold_drugs": len(self._holds), "ot_evidence_rows": self.counts["target_diseases"],
            "all_pair_bookkeeping_rows": len(self.drugs) * len(self.targets),
        }
        for key, value in expected.items():
            if self.final.get(key) != value:
                raise ValueError(f"Final summary {key} disagrees with artifacts: {self.final.get(key)} != {value}")
        if self.counts["disease_pair_coverage"] != len(self.drugs) * len(self.targets):
            raise ValueError("Pair coverage does not preserve the full Cartesian catalog")
        if run.get("drugs") != len(self._drug_rows) or run.get("diseases") != len(self._diseases):
            raise ValueError("TxGNN run metadata disagrees with matrix orders")
        for identifier, row in self._ot_coverage.items():
            if _int(row.get("count")) != self._ot_counts.get(identifier, 0):
                raise ValueError(f"OT per-target count mismatch: {identifier}")
        for key, count in self.final.get("direction_disease_node_counts", {}).items():
            direction = key.removeprefix("speciality_")
            if direction not in self._direction_masks or int(self._direction_masks[direction].sum()) != count:
                raise ValueError(f"Direction count mismatch: {key}")

    def _publish_summary(self) -> None:
        specs = [
            ("TxGNN 全疾病原始 logit", LOGITS, self.counts["txgnn_raw_scores"]),
            ("TxGNN 已知关系与身份排除遮罩", MASK, self.counts["txgnn_masked_scores"]),
            ("TxGNN 23 方向来源本体分类", "TXGNN_ALL_DISEASE_DIRECTIONS.csv", len(self._diseases)),
            ("Open Targets 26.06 全靶点疾病证据", OT, self.counts["target_diseases"]),
            ("EC-KG 来源关系与增量审计", EC, self.counts["eckg_relations"]),
            ("720×888 配对疾病覆盖审查", PAIR, self.counts["disease_pair_coverage"]),
            ("药物身份与疾病解释资格", "DRUG_COVERAGE_FINAL_720.csv", len(self.drugs)),
            ("靶点疾病覆盖与身份缺口", "TARGET_COVERAGE_888.csv", len(self.targets)),
        ]
        self.sources = [{"name": name, "path": f"{SNAPSHOT}/{file}", "source_path": f"{SNAPSHOT}/{file}", "available": True, "rows": count, "version": "2026-09-09", "read_only": True} for name, file, count in specs]
        directions = [{"id": key, "label": label, "count": int(self._direction_masks[key].sum())} for key, label in DIRECTION_LABELS.items() if key != "unclassified"]
        self.summary = {
            "status": "validated", "enabled": True, "version": "2026-09-09", "snapshot": SNAPSHOT,
            "counts": dict(self.counts), **self.counts, "project_drugs": len(self.drugs), "project_targets": len(self.targets),
            "directions": directions, "unclassified_count": int(self._direction_masks["unclassified"].sum()),
            "unclassified_direction": {"id": "unclassified", "label": "未分类", "count": int(self._direction_masks["unclassified"].sum())},
            "score_semantics": SCORE_SEMANTICS, "relation": "indication", "default_scope": "eligible",
            "exclusion_bits": [{"bit": key, "reason": value} for key, value in EXCLUSION_REASONS.items()],
            "boundaries": BOUNDARIES, "validation": self.validation, "source_path": f"{SNAPSHOT}/SOURCE_AND_ARTIFACT_MANIFEST.json",
            "identity_hold_drugs": [{"id": key, "name": self.drugs[key].get("name", key)} for key in sorted(self._holds)],
            "sources": self.sources, "fingerprint": self.fingerprint,
        }

    def metadata(self, kind: str, identifier: str) -> dict:
        if not self.enabled:
            return {}
        self._check(kind, identifier)
        coverage = dict((self._drug_coverage if kind == "drug" else self._target_coverage)[identifier])
        ec = {"node_count": _int(coverage.get("ec_node_count")), "relation_count": self._ec_counts.get(f"{kind}:{identifier}", 0), "source_path": f"{SNAPSHOT}/{EC}", "interpretation": "来源关系审计；不等同于直接结合或已证实适应证"}
        if kind == "target":
            ot = self._ot_coverage[identifier]
            return {"disease_coverage": {**coverage, "status": ot.get("status"), "ot_status": ot.get("status"), "opentargets_count": self._ot_counts.get(identifier, 0), "ot_count": self._ot_counts.get(identifier, 0), "ot_ids": ot.get("ot_id", ""), "source_path": f"{SNAPSHOT}/{OT}", "version": "OpenTargets_26.06"}, "eckg_coverage": ec}
        row = self._drug_rows.get(identifier)
        raw_count = len(self._diseases) if row is not None else 0
        eligible_count = int(self._eligible_counts[row]) if row is not None else 0
        hold = identifier in self._holds
        status = "identity_hold" if hold else "scored" if row is not None else "source_evidence_only" if _int(coverage.get("ec_disease_endpoint_count")) else "unavailable"
        direction_count = sum(bool(np.any((self.mask[row] == 0) & mask)) for key, mask in self._direction_masks.items() if key != "unclassified") if row is not None and not hold else 0
        return {"txgnn": {
            **coverage, "status": status, "source_status": coverage.get("disease_evidence_status"),
            "raw_available": row is not None, "interpretation_eligible": row is not None and not hold,
            "raw_count": raw_count, "eligible_count": eligible_count, "excluded_count": raw_count - eligible_count,
            "identity_hold": hold, "legacy_identity_hold_field": _bool(coverage.get("identity_hold")),
            "mapping_rule": coverage.get("mapping_rule", ""), "mapping_status": coverage.get("mapping_status", ""),
            "graph_drug_id": coverage.get("graph_drug_id", ""), "graph_drug_name": coverage.get("graph_drug_name", ""),
            "direction_count": direction_count, "score_semantics": SCORE_SEMANTICS,
            "source_path": f"{SNAPSHOT}/{LOGITS}", "mask_source_path": f"{SNAPSHOT}/{MASK}",
            "coverage_source_path": f"{SNAPSHOT}/DRUG_COVERAGE_FINAL_720.csv", "version": "2026-09-09",
        }, "eckg_coverage": ec}

    def _check(self, kind: str, identifier: str, section: str | None = None) -> None:
        if kind not in {"drug", "target"}:
            raise ValueError("kind must be drug or target")
        if identifier not in (self.drugs if kind == "drug" else self.targets):
            raise KeyError(identifier)
        if section and section not in ({"txgnn_diseases", "eckg"} if kind == "drug" else {"target_diseases", "eckg"}):
            raise ValueError(f"Unsupported disease section {section!r} for {kind}")

    def total(self, kind: str, identifier: str, section: str) -> int:
        if not self.enabled:
            return 0
        self._check(kind, identifier, section)
        if section == "txgnn_diseases":
            row = self._drug_rows.get(identifier)
            return int(self._eligible_counts[row]) if row is not None else 0
        if section == "target_diseases":
            return self._ot_counts.get(identifier, 0)
        return self._ec_counts.get(f"{kind}:{identifier}", 0)

    @lru_cache(maxsize=32)
    def _rank_arrays(self, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(-self.logits[row], kind="stable")
        raw_ranks = np.empty(len(order), dtype=np.int32)
        raw_ranks[order] = np.arange(1, len(order) + 1)
        eligible_order = order[self.mask[row, order] == 0]
        eligible_ranks = np.zeros(len(order), dtype=np.int32)
        eligible_ranks[eligible_order] = np.arange(1, len(eligible_order) + 1)
        return order, raw_ranks, eligible_ranks

    def query(self, kind: str, identifier: str, section: str, page: int = 1,
              page_size: int = 50, search: str = "", direction: str = "",
              scope: str = "eligible") -> dict:
        page, page_size = max(1, int(page)), max(1, min(1000, int(page_size)))
        if scope not in {"eligible", "all"}:
            raise ValueError("scope must be eligible or all")
        direction = direction.removeprefix("speciality_")
        if direction and direction not in DIRECTION_LABELS:
            raise ValueError("Unknown disease direction")
        if not self.enabled:
            return {"items": [], "total": 0, "all_total": 0, "page": page, "page_size": page_size, "status": "snapshot_missing"}
        self._check(kind, identifier, section)
        if section == "txgnn_diseases":
            return self._query_txgnn(identifier, page, page_size, search.strip().casefold(), direction, scope)
        return self._query_sql(kind, identifier, section, page, page_size, search.strip().casefold(), direction, scope)

    def _query_txgnn(self, identifier: str, page: int, page_size: int, search: str, direction: str, scope: str) -> dict:
        row = self._drug_rows.get(identifier)
        metadata = self.metadata("drug", identifier)["txgnn"]
        raw_total, eligible_total = metadata["raw_count"], metadata["eligible_count"]
        all_total = raw_total if scope == "all" else eligible_total
        result = {"items": [], "total": 0, "all_total": all_total, "page": page, "page_size": page_size, "scope": scope, "direction": direction,
                  "raw_total": raw_total, "eligible_total": eligible_total, "excluded_total": raw_total - eligible_total,
                  "denominator": all_total, "score_semantics": SCORE_SEMANTICS, "metadata": metadata, "source_path": f"{SNAPSHOT}/{LOGITS}"}
        if row is None:
            return result
        order, raw_ranks, eligible_ranks = self._rank_arrays(row)
        keep = np.ones(len(self._diseases), dtype=bool) if scope == "all" else self.mask[row] == 0
        if direction:
            keep &= self._direction_masks[direction]
        if search:
            keep &= np.fromiter((search in value for value in self._disease_search), dtype=bool, count=len(self._diseases))
        selected = order[keep[order]]
        result["total"] = len(selected)
        for index in selected[(page - 1) * page_size:page * page_size]:
            column = int(index)
            disease, exclusion = self._diseases[column], int(self.mask[row, column])
            score = float(self.logits[row, column])
            directions = disease["directions"]
            result["items"].append({
                "id": disease["id"], "name": disease.get("node_name") or disease["id"],
                "score": score, "logit": score, "rank": int(raw_ranks[column] if scope == "all" else eligible_ranks[column]),
                "denominator": all_total, "raw_rank": int(raw_ranks[column]), "raw_denominator": raw_total,
                "eligible_rank": int(eligible_ranks[column]) or None, "eligible_denominator": eligible_total,
                "ranking_scope": "raw_all_diseases" if scope == "all" else "eligible_unmasked_diseases",
                "directions": directions, "direction_labels": [DIRECTION_LABELS[key] for key in directions] or ["未分类"],
                "exclusion_mask": exclusion, "exclusion_reasons": [reason for bit, reason in EXCLUSION_REASONS.items() if exclusion & bit],
                "interpretation_eligible": exclusion == 0, "identity_hold": bool(exclusion & 4),
                "mapping_scope": disease["mapping_scope"], "member_disease_ids": self._members[column],
                "source": "TxGNN 原权重 · 2026-09-09 全疾病推理", "source_path": f"{SNAPSHOT}/{LOGITS}",
                "mask_source_path": f"{SNAPSHOT}/{MASK}", "score_column": column, "physical_drug_row": row,
                "relation": "indication", "score_semantics": SCORE_SEMANTICS,
            })
        return result

    def _query_sql(self, kind: str, identifier: str, section: str, page: int, page_size: int, search: str, direction: str, scope: str) -> dict:
        where, filters = [], []
        if search:
            where.append("instr(_search,?)>0")
            filters.append(search)
        if direction:
            where.append("instr(_directions,?)>0")
            filters.append("|" + direction + "|")
        condition = " WHERE " + " AND ".join(where) if where else ""
        if section == "target_diseases":
            base = "SELECT *,row_number() OVER (ORDER BY overall_score DESC,_ordinal) AS _rank FROM ot WHERE target_chembl_id=?"
            key, source = identifier, OT
        else:
            base = "SELECT ec.*,row_number() OVER (ORDER BY ec._ordinal) AS _rank FROM ec JOIN ec_entities ON ec._ordinal=ec_entities.row_id WHERE ec_entities.entity_key=?"
            key, source = f"{kind}:{identifier}", EC
        with self._connection() as connection:
            total = connection.execute(f"WITH ranked AS ({base}) SELECT count(*) FROM ranked{condition}", [key, *filters]).fetchone()[0]
            rows = connection.execute(f"WITH ranked AS ({base}) SELECT * FROM ranked{condition} ORDER BY _rank LIMIT ? OFFSET ?", [key, *filters, page_size, (page - 1) * page_size]).fetchall()
        all_total = self.total(kind, identifier, section)
        items = [self._format_ot(dict(row), all_total) if section == "target_diseases" else self._format_ec(dict(row), kind, identifier, all_total) for row in rows]
        return {"items": items, "total": total, "all_total": all_total, "page": page, "page_size": page_size,
                "direction": direction, "scope": scope, "denominator": all_total, "source_path": f"{SNAPSHOT}/{source}"}

    def _format_ot(self, row: dict, denominator: int) -> dict:
        rank = row.pop("_rank", None)
        row.pop("_ordinal", None); row.pop("_search", None); row.pop("_directions", None)
        disease_id = row["disease_id"]
        data = row.get("datatype_scores_json")
        try:
            data = json.loads(data) if data else {}
        except (ValueError, TypeError):
            raise ValueError("Invalid datatype_scores_json in Open Targets source")
        row.update(id=disease_id, name=row["disease_name"], score=row.get("overall_score"),
                   datatype_scores=data, rank=rank, denominator=denominator,
                   source_path=f"{SNAPSHOT}/{OT}", url=f"https://platform.opentargets.org/evidence/{quote(row.get('source_ensembl_id') or row.get('ot_id') or '', safe='')}/{quote(disease_id, safe='')}",
                   relation="target_disease_association", treatment_direction_established=_bool(row.get("treatment_direction_established")))
        directions = self._direction_column(row.get("txgnn_score_column"))
        row.update(directions=directions, direction_labels=[DIRECTION_LABELS[key] for key in directions])
        return row

    def _format_ec(self, row: dict, kind: str, identifier: str, denominator: int) -> dict:
        rank, ordinal = row.pop("_rank", None), row.pop("_ordinal", None)
        row.pop("_search", None); row.pop("_directions", None)
        caller = f"{kind}:{identifier}"
        counterpart = row["other_id"] if row["project_key"] == caller else row["project_key"]
        row.update(id=f"eckg:{ordinal}", name=self._entity_name(counterpart), counterpart_id=counterpart,
                   relation=row.get("predicate"), source=row.get("primary_sources") or "EC-KG 来源关系审计",
                   source_path=f"{SNAPSHOT}/{EC}", rank=rank, denominator=denominator,
                   mapping_ambiguous=_bool(row.get("mapping_ambiguous")),
                   subject_name=self._entity_name(row.get("ec_subject", "")), object_name=self._entity_name(row.get("ec_object", "")),
                   interpretation="来源关系审计；谓词、证据性质及身份歧义均须单独核对，不等于已知结合")
        return row

    def preview(self, kind: str, identifier: str, section: str, limit: int = 100) -> list:
        return self.query(kind, identifier, section, page_size=limit)["items"]

    def pair(self, drug_id: str, target_id: str) -> dict:
        if not self.enabled:
            return {}
        self._check("drug", drug_id); self._check("target", target_id)
        drug_key, target_key = "drug:" + drug_id, "target:" + target_id
        with self._connection() as connection:
            coverage = connection.execute("SELECT * FROM pairs WHERE ligand_inchikey=? AND target_chembl_id=?", (drug_id, target_id)).fetchone()
            total = connection.execute("SELECT count(*) FROM ec WHERE (project_key=? AND other_id=?) OR (project_key=? AND other_id=?)", (drug_key, target_key, target_key, drug_key)).fetchone()[0]
            rows = connection.execute("SELECT * FROM ec WHERE (project_key=? AND other_id=?) OR (project_key=? AND other_id=?) ORDER BY _ordinal LIMIT 25", (drug_key, target_key, target_key, drug_key)).fetchall()
        if coverage is None:
            raise ValueError("Validated pair coverage is missing an in-catalog pair")
        record = {key: value for key, value in dict(coverage).items() if not key.startswith("_")}
        record["eckg_drug_target_relation_present"] = _bool(record.get("eckg_drug_target_relation_present"))
        record["source_path"] = f"{SNAPSHOT}/{PAIR}"
        return {"coverage": record, "pair_coverage": record, "eckg_records": [self._format_ec(dict(row), "drug", drug_id, total) for row in rows],
                "eckg_total": total, "eckg_preview_limit": 25, "source_path": f"{SNAPSHOT}/{PAIR}",
                "interpretation": "配对疾病覆盖及来源关系附加审查，不是新结合预测，不改变原排名或 SPR 放行状态"}
