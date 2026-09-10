"""Attach the frozen disease-review appendix without changing SPR design rows."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

REVIEW_DIR = "outputs/biomaster_disease_evidence_720x888_20260909"
DISEASE_REVIEW = f"{REVIEW_DIR}/SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv"
RELATION_REVIEW = f"{REVIEW_DIR}/SPR512_ECKG_RELATION_REVIEW.csv"
KEY_FIELDS = ("ligand_inchikey", "target_chembl_id", "selection_role")


def _read_unique(path: Path, required: tuple[str, ...]) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(KEY_FIELDS + required).issubset(reader.fieldnames or ()):
            raise ValueError(f"Disease review is missing required columns: {path.name}")
        records = {}
        for row in reader:
            if None in row:
                raise ValueError(f"Malformed disease-review CSV row: {path.name}")
            key = tuple((row.get(field) or "").strip() for field in KEY_FIELDS)
            if not all(key) or key in records:
                raise ValueError(f"Missing or duplicate disease-review identity in {path.name}: {key}")
            records[key] = row
    return records


def _boolean(value: Any, field: str) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text not in {"true", "false"}:
        raise ValueError(f"Invalid {field} in disease review: {value!r}")
    return text == "true"


def _count(value: Any, field: str) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in disease review") from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Invalid {field} in disease review: {value!r}")
    return int(number)


def load_disease_reviews(root: Path | str, drugs: dict, targets: dict) -> dict[str, Any]:
    """Strictly join both appendices by exact drug, target and frozen role.

    The files remain unchanged. A compound absent from the 720-drug catalog is
    attached only to its target. No name, salt or connectivity fallback is used.
    """
    root = Path(root)
    disease_path, relation_path = DISEASE_REVIEW, RELATION_REVIEW
    integrated = "outputs/spr512_integrated_disease_20260909"
    validation = root / integrated / "VALIDATION.json"
    if validation.is_file():
        if json.loads(validation.read_text()).get("all_pass") is not True:
            raise ValueError("Integrated SPR disease review has not passed validation")
        disease_path = f"{integrated}/SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv"
        relation_path = f"{integrated}/SPR512_ECKG_RELATION_REVIEW.csv"
    result: dict[str, Any] = {
        "drugs": {key: {"disease_review": {"spr_records": []}} for key in drugs},
        "targets": {key: {"disease_review": {"spr_records": []}} for key in targets},
        "sources": [
            {"name": "SPR512 disease hypotheses · append-only review · 2026-09-09",
             "path": disease_path, "available": (root / disease_path).is_file()},
            {"name": "SPR512 EC-KG source-relation review · 2026-09-09",
             "path": relation_path, "available": (root / relation_path).is_file()},
        ],
    }
    available = [item["available"] for item in result["sources"]]
    if not any(available):
        return result
    if not all(available):
        raise ValueError("SPR disease appendix requires both disease and EC-KG review files")
    disease_rows = _read_unique(root / disease_path, (
        "drug_names", "gene_symbol", "txgnn_available", "ot_status",
        "treatment_direction_established", "top50_ot_overlap", "disease_hypotheses_json",
    ))
    relation_rows = _read_unique(root / relation_path, (
        "drug_names", "gene_symbol", "ec_relation_rows", "non_text_unambiguous_assertion_rows",
        "predicates", "primary_sources", "review_status",
    ))
    if disease_rows.keys() != relation_rows.keys():
        raise ValueError("Disease and EC-KG SPR appendices have inconsistent drug/target/role identities")
    for key, disease in disease_rows.items():
        relation = relation_rows[key]
        for field in disease.keys() & relation.keys():
            if (disease[field] or "").strip() != (relation[field] or "").strip():
                raise ValueError(f"Conflicting SPR appendix field {field!r} for {key}")
        raw_hypotheses = disease.get("disease_hypotheses_json") or ""
        try:
            hypotheses = json.loads(raw_hypotheses) if raw_hypotheses.strip() else []
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid SPR disease hypotheses JSON for {key}") from exc
        if not isinstance(hypotheses, list) or any(not isinstance(row, dict) for row in hypotheses):
            raise ValueError(f"SPR disease hypotheses must be a list of records for {key}")
        for hypothesis in hypotheses:
            for field in ("txgnn_logit", "ot_score"):
                value = hypothesis.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f"Invalid hypothesis {field} for {key}")
        item: dict[str, Any] = {**disease, **relation}
        for field in ("txgnn_available", "treatment_direction_established"):
            item[field] = _boolean(item.get(field), field)
        for field in ("top50_ot_overlap", "ec_relation_rows", "non_text_unambiguous_assertion_rows"):
            item[field] = _count(item.get(field), field)
        if hypotheses and item["txgnn_available"] is not True:
            raise ValueError(f"Disease hypotheses present without available TxGNN scores for {key}")
        if (item["ec_relation_rows"] is not None and item["non_text_unambiguous_assertion_rows"] is not None
                and item["non_text_unambiguous_assertion_rows"] > item["ec_relation_rows"]):
            raise ValueError(f"EC assertion count exceeds relation count for {key}")
        drug_id, target_id, role = key
        drug_name = disease.get("drug_names") or drugs.get(drug_id, {}).get("name") or drug_id
        target_name = disease.get("gene_symbol") or targets.get(target_id, {}).get("name") or target_id
        item.update({
            "id": "__".join(key), "name": f"{drug_name} → {target_name}",
            "drug_id": drug_id, "target_id": target_id,
            "drug_name": drug_name, "target_name": target_name,
            "drug_in_catalog": drug_id in drugs, "target_in_catalog": target_id in targets,
            "disease_hypotheses": hypotheses,
            "ot_top_diseases": json.loads(disease.get("ot_top3_json") or "[]"),
            "source": "SPR512 disease and EC-KG append-only review · 2026-09-09",
            "source_paths": [disease_path, relation_path],
            "interpretation": "附加疾病假设与来源审查，不改变原设计角色或实验放行状态；不是实测结果",
        })
        if drug_id in result["drugs"]:
            result["drugs"][drug_id]["disease_review"]["spr_records"].append(item)
        if target_id in result["targets"]:
            result["targets"][target_id]["disease_review"]["spr_records"].append(item)
    return result
