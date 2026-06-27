#!/usr/bin/env python3
"""Fetch child-level Open Targets disease evidence for strict895 candidates.

The broad mechanism layer used parent disease nodes such as cancer or immune
system disease. This script maps concrete disease hypotheses to Open Targets
IDs and checks whether each candidate target has direct child-level association
evidence through target.associatedDiseases.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "outputs/broad_mechanism_layer_v2/strict895_concrete_disease_completion/strict895_concrete_disease_annotations.csv"
BROAD_OT = ROOT / "outputs/broad_mechanism_layer_v2/opentargets_broad_current_target_hits.csv"
OUTDIR = ROOT / "outputs/broad_mechanism_layer_v2/strict895_opentargets_child_disease"
API = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query search($queryString: String!) {
  search(queryString: $queryString) {
    hits {
      id
      name
      entity
      category
    }
  }
}
"""

TARGET_DISEASES_QUERY = """
query targetDiseases($ensemblId: String!, $pageIndex: Int!, $pageSize: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: $pageIndex, size: $pageSize}) {
      count
      rows {
        score
        disease { id name }
        datatypeScores { id score }
      }
    }
  }
}
"""


SKIP_PATTERNS = [
    "待定",
    "优先按",
    "readout",
    "具体化",
    "不优先作为",
    "安全性反筛",
    "反筛",
]


PHRASE_QUERY_RULES: list[tuple[str, str]] = [
    ("KRAS突变NSCLC", "non-small cell lung carcinoma"),
    ("EGFR突变NSCLC", "non-small cell lung carcinoma"),
    ("RET融合NSCLC", "non-small cell lung carcinoma"),
    ("HER2突变肺癌", "non-small cell lung carcinoma"),
    ("NSCLC", "non-small cell lung carcinoma"),
    ("HER2阳性乳腺癌", "HER2-positive breast carcinoma"),
    ("HR阳性乳腺癌", "breast carcinoma"),
    ("ER阳性乳腺癌", "breast carcinoma"),
    ("AR阳性乳腺癌", "breast carcinoma"),
    ("CCND1扩增乳腺癌", "breast carcinoma"),
    ("乳腺癌", "breast carcinoma"),
    ("乳腺", "breast carcinoma"),
    ("前列腺癌", "prostate carcinoma"),
    ("胰腺癌", "pancreatic carcinoma"),
    ("结直肠癌", "colorectal carcinoma"),
    ("结直肠", "colorectal carcinoma"),
    ("头颈鳞癌", "head and neck squamous cell carcinoma"),
    ("甲状腺癌", "thyroid carcinoma"),
    ("子宫内膜癌", "endometrial carcinoma"),
    ("卵巢", "ovarian carcinoma"),
    ("黑色素瘤", "melanoma"),
    ("GIST", "gastrointestinal stromal tumor"),
    ("胃癌", "stomach carcinoma"),
    ("HCC", "hepatocellular carcinoma"),
    ("肝癌", "hepatocellular carcinoma"),
    ("AML", "acute myeloid leukemia"),
    ("急性髓系白血病", "acute myeloid leukemia"),
    ("髓系白血病", "acute myeloid leukemia"),
    ("T细胞白血病", "T-cell leukemia"),
    ("CLL", "chronic lymphocytic leukemia"),
    ("BCL2依赖淋巴瘤", "lymphoma"),
    ("套细胞淋巴瘤", "mantle cell lymphoma"),
    ("淋巴瘤", "lymphoma"),
    ("多发性骨髓瘤", "multiple myeloma"),
    ("骨髓增殖性肿瘤", "myeloproliferative neoplasm"),
    ("银屑病", "psoriasis"),
    ("特应性皮炎", "atopic dermatitis"),
    ("皮炎", "dermatitis"),
    ("Crohn", "Crohn disease"),
    ("IBD", "inflammatory bowel disease"),
    ("自身免疫", "autoimmune disease"),
    ("RA", "rheumatoid arthritis"),
    ("动脉粥样硬化", "atherosclerosis"),
    ("心肌纤维化", "cardiac fibrosis"),
    ("心律失常", "cardiac arrhythmia"),
    ("心脏传导异常", "cardiac conduction disease"),
    ("心衰", "heart failure"),
    ("扩张型心肌病", "dilated cardiomyopathy"),
    ("肺动脉高压", "pulmonary arterial hypertension"),
    ("肾纤维化", "chronic kidney disease"),
    ("CKD", "chronic kidney disease"),
    ("高尿酸血症", "hyperuricemia"),
    ("痛风", "gout"),
    ("肾结石", "nephrolithiasis"),
    ("尿酸性肾病", "hyperuricemic nephropathy"),
    ("癫痫", "epilepsy"),
    ("帕金森", "Parkinson disease"),
    ("阿尔茨海默", "Alzheimer disease"),
    ("抑郁", "depression"),
    ("焦虑", "anxiety disorder"),
    ("精神分裂", "schizophrenia"),
    ("骨质疏松", "osteoporosis"),
    ("肌强直", "myotonia"),
    ("周期性麻痹", "periodic paralysis"),
    ("NASH", "nonalcoholic steatohepatitis"),
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(text).lower()).strip()


def disease_to_query(name: str) -> str:
    name = clean(name)
    if not name:
        return ""
    for pattern, query in PHRASE_QUERY_RULES:
        if pattern.lower() in name.lower():
            return query
    if re.search(r"[\u4e00-\u9fff]", name):
        if any(p in name for p in SKIP_PATTERNS):
            return ""
        return ""
    # Remove parenthetical aliases that often hurt search less than help.
    name = re.sub(r"\([^)]*\)", "", name).strip()
    # Normalize a few common abbreviations.
    abbr = {
        "NSCLC": "non-small cell lung carcinoma",
        "HCC": "hepatocellular carcinoma",
        "AML": "acute myeloid leukemia",
        "CLL": "chronic lymphocytic leukemia",
    }
    return abbr.get(name, name)


def is_disease_hit(hit: dict[str, Any]) -> bool:
    return clean(hit.get("entity")) == "disease" and not clean(hit.get("id")).startswith("HP_")


def best_search_hit(query: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    disease_hits = [h for h in hits if is_disease_hit(h)]
    if not disease_hits:
        return None
    qn = normalize_name(query)
    # Exact normalized name first.
    for hit in disease_hits:
        if normalize_name(hit.get("name")) == qn:
            return hit
    # Prefer MONDO/EFO disease hits over studies/phenotypes, with matching token overlap.
    q_tokens = set(qn.split())
    def score(hit: dict[str, Any]) -> tuple[int, int, int]:
        name = normalize_name(hit.get("name"))
        tokens = set(name.split())
        overlap = len(q_tokens & tokens)
        id_bonus = 2 if clean(hit.get("id")).startswith(("MONDO_", "EFO_")) else 0
        length_penalty = -abs(len(tokens) - len(q_tokens))
        return (overlap, id_bonus, length_penalty)
    return sorted(disease_hits, key=score, reverse=True)[0]


def request_json(session: requests.Session, payload: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    last_error = ""
    for attempt in range(retries):
        try:
            response = session.post(API, json=payload, timeout=60)
            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = f"{response.status_code} {response.reason}"
                time.sleep(1.5 * (attempt + 1) ** 2)
                continue
            response.raise_for_status()
            data = response.json()
            if "errors" in data:
                last_error = str(data["errors"][:1])
                time.sleep(1.0 * (attempt + 1))
                continue
            return data
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1) ** 2)
    return {"error": last_error}


def build_gene_ensembl_map() -> dict[str, str]:
    raw = pd.read_csv(BROAD_OT, usecols=["gene_name", "ensembl_gene_id"], low_memory=False)
    raw = raw.dropna().drop_duplicates("gene_name")
    return {clean(r.gene_name).upper(): clean(r.ensembl_gene_id) for r in raw.itertuples(index=False)}


def search_targets_for_missing_genes(session: requests.Session, genes: set[str], gene_to_ensembl: dict[str, str], cache: dict[str, Any]) -> None:
    for gene in sorted(genes):
        if gene in gene_to_ensembl:
            continue
        key = f"target::{gene}"
        if key not in cache:
            data = request_json(session, {"query": SEARCH_QUERY, "variables": {"queryString": gene}})
            cache[key] = data
            time.sleep(0.05)
        hits = (((cache[key].get("data") or {}).get("search") or {}).get("hits") or [])
        for hit in hits:
            if clean(hit.get("entity")) == "target" and clean(hit.get("name")).upper() == gene:
                gene_to_ensembl[gene] = clean(hit.get("id"))
                break


def map_diseases(session: requests.Session, disease_names: set[str], cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for name in sorted(disease_names):
        query = disease_to_query(name)
        if not query:
            mapped[name] = {"query": "", "mapped_id": "", "mapped_name": "", "mapping_status": "skipped_no_specific_query"}
            continue
        key = f"disease::{query}"
        if key not in cache:
            data = request_json(session, {"query": SEARCH_QUERY, "variables": {"queryString": query}})
            cache[key] = data
            time.sleep(0.05)
        hits = (((cache[key].get("data") or {}).get("search") or {}).get("hits") or [])
        hit = best_search_hit(query, hits)
        if hit:
            mapped[name] = {
                "query": query,
                "mapped_id": clean(hit.get("id")),
                "mapped_name": clean(hit.get("name")),
                "mapping_status": "mapped",
            }
        else:
            mapped[name] = {"query": query, "mapped_id": "", "mapped_name": "", "mapping_status": "not_mapped"}
    return mapped


def datatype_scores_to_dict(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {clean(item.get("id")): float(item.get("score") or 0.0) for item in rows or []}


def fetch_target_diseases(
    session: requests.Session,
    ensembl_id: str,
    cache: dict[str, Any],
    *,
    page_size: int = 500,
    max_pages: int = 8,
) -> dict[str, Any]:
    key = f"target_diseases::{ensembl_id}::ps{page_size}::mp{max_pages}"
    if key in cache:
        return cache[key]
    rows: list[dict[str, Any]] = []
    total_count = None
    approved_symbol = ""
    approved_name = ""
    for page in range(max_pages):
        data = request_json(
            session,
            {
                "query": TARGET_DISEASES_QUERY,
                "variables": {"ensemblId": ensembl_id, "pageIndex": page, "pageSize": page_size},
            },
        )
        if "error" in data:
            cache[key] = {"ok": False, "error": data["error"], "rows": rows, "count": total_count}
            return cache[key]
        target = ((data.get("data") or {}).get("target") or {})
        if not target:
            cache[key] = {"ok": False, "error": "target_not_found", "rows": rows, "count": total_count}
            return cache[key]
        approved_symbol = clean(target.get("approvedSymbol"))
        approved_name = clean(target.get("approvedName"))
        assoc = target.get("associatedDiseases") or {}
        total_count = int(assoc.get("count") or 0)
        page_rows = assoc.get("rows") or []
        for i, row in enumerate(page_rows):
            disease = row.get("disease") or {}
            dts = datatype_scores_to_dict(row.get("datatypeScores") or [])
            rows.append(
                {
                    "rank": page * page_size + i + 1,
                    "disease_id": clean(disease.get("id")),
                    "disease_name": clean(disease.get("name")),
                    "score": float(row.get("score") or 0.0),
                    "datatype_scores": dts,
                }
            )
        time.sleep(0.05)
        if len(rows) >= total_count or len(page_rows) < page_size:
            break
    cache[key] = {
        "ok": True,
        "ensembl_id": ensembl_id,
        "approved_symbol": approved_symbol,
        "approved_name": approved_name,
        "count": total_count,
        "rows": rows,
        "fetched_rows": len(rows),
    }
    return cache[key]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(INFILE, low_memory=False)
    cache_path = OUTDIR / "opentargets_child_disease_cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    session = requests.Session()
    session.headers.update({"User-Agent": "BioMaster strict895 child disease evidence"})

    gene_to_ensembl = build_gene_ensembl_map()
    genes = set(candidates["target_gene"].dropna().astype(str).str.upper())
    search_targets_for_missing_genes(session, genes, gene_to_ensembl, cache)

    disease_cols = [f"concrete_disease_{i}" for i in range(1, 6)]
    disease_names = set()
    for col in disease_cols:
        disease_names.update(candidates[col].dropna().astype(str).map(clean).loc[lambda s: s.ne("")].tolist())
    disease_map = map_diseases(session, disease_names, cache)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    target_cache: dict[str, dict[str, Any]] = {}
    for gene in sorted(genes):
        ensembl_id = gene_to_ensembl.get(gene, "")
        if not ensembl_id:
            continue
        target_cache[gene] = fetch_target_diseases(session, ensembl_id, cache, page_size=500, max_pages=8)
        if len(target_cache) % 20 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[opentargets-child] targets {len(target_cache)}/{len(genes)}", flush=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    long_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        review_id = clean(getattr(row, "review_id"))
        gene = clean(getattr(row, "target_gene")).upper()
        ensembl_id = gene_to_ensembl.get(gene, "")
        target_data = target_cache.get(gene, {})
        disease_by_id = {r["disease_id"]: r for r in target_data.get("rows", [])}
        disease_by_name = {normalize_name(r["disease_name"]): r for r in target_data.get("rows", [])}
        best: dict[str, Any] | None = None
        mapped_count = 0
        for idx, col in enumerate(disease_cols, 1):
            original = clean(getattr(row, col))
            mapped = disease_map.get(original, {"mapping_status": "not_seen"})
            mapped_id = clean(mapped.get("mapped_id"))
            mapped_name = clean(mapped.get("mapped_name"))
            query = clean(mapped.get("query"))
            status = clean(mapped.get("mapping_status"))
            assoc = None
            if mapped_id:
                mapped_count += 1
                assoc = disease_by_id.get(mapped_id) or disease_by_name.get(normalize_name(mapped_name))
            dt = assoc.get("datatype_scores", {}) if assoc else {}
            rec = {
                "review_id": review_id,
                "drug_name": clean(getattr(row, "drug_name")),
                "target_gene": gene,
                "ensembl_id": ensembl_id,
                "candidate_rank": idx,
                "candidate_disease_original": original,
                "opentargets_search_query": query,
                "mapped_disease_id": mapped_id,
                "mapped_disease_name": mapped_name,
                "mapping_status": status,
                "target_disease_assoc_found": bool(assoc),
                "target_disease_score": assoc.get("score", 0.0) if assoc else 0.0,
                "target_disease_rank_within_target": assoc.get("rank", "") if assoc else "",
                "genetic_association_score": dt.get("genetic_association", 0.0),
                "somatic_mutation_score": dt.get("somatic_mutation", 0.0),
                "clinical_score": dt.get("clinical", 0.0),
                "known_drug_score": dt.get("known_drug", 0.0),
                "affected_pathway_score": dt.get("affected_pathway", 0.0),
                "literature_score": dt.get("literature", 0.0),
                "animal_model_score": dt.get("animal_model", 0.0),
                "fetched_target_disease_rows": target_data.get("fetched_rows", 0),
                "target_total_associated_disease_count": target_data.get("count", 0),
            }
            long_rows.append(rec)
            if assoc and (best is None or rec["target_disease_score"] > best["target_disease_score"]):
                best = rec
        if best:
            child_status = "child_ot_match"
        elif mapped_count > 0 and ensembl_id:
            child_status = "mapped_but_not_in_fetched_target_diseases"
        elif not ensembl_id:
            child_status = "target_ensembl_unmapped"
        else:
            child_status = "no_specific_disease_mapped"
        row_summaries.append(
            {
                "review_id": review_id,
                "drug_name": clean(getattr(row, "drug_name")),
                "target_gene": gene,
                "ensembl_id": ensembl_id,
                "direction": clean(getattr(row, "direction")),
                "concrete_disease_evidence_level": clean(getattr(row, "concrete_disease_evidence_level")),
                "child_ot_status": child_status,
                "mapped_candidate_disease_count": mapped_count,
                "best_child_ot_disease_id": best.get("mapped_disease_id", "") if best else "",
                "best_child_ot_disease_name": best.get("mapped_disease_name", "") if best else "",
                "best_child_ot_score": best.get("target_disease_score", 0.0) if best else 0.0,
                "best_child_ot_rank_within_target": best.get("target_disease_rank_within_target", "") if best else "",
                "best_child_genetic_score": best.get("genetic_association_score", 0.0) if best else 0.0,
                "best_child_somatic_score": best.get("somatic_mutation_score", 0.0) if best else 0.0,
                "best_child_clinical_score": best.get("clinical_score", 0.0) if best else 0.0,
                "best_child_known_drug_score": best.get("known_drug_score", 0.0) if best else 0.0,
                "best_child_literature_score": best.get("literature_score", 0.0) if best else 0.0,
                "concrete_disease_1": clean(getattr(row, "concrete_disease_1")),
                "concrete_disease_2": clean(getattr(row, "concrete_disease_2")),
                "concrete_disease_3": clean(getattr(row, "concrete_disease_3")),
                "recommended_disease_model_zh": clean(getattr(row, "recommended_disease_model_zh")),
                "missing_for_final_disease_call_zh": clean(getattr(row, "missing_for_final_disease_call_zh")),
            }
        )

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(row_summaries)
    long_df.to_csv(OUTDIR / "strict895_opentargets_child_disease_long.csv", index=False)
    summary_df.to_csv(OUTDIR / "strict895_opentargets_child_disease_summary.csv", index=False)

    mapping_df = pd.DataFrame(
        [
            {"candidate_disease_original": k, **v}
            for k, v in sorted(disease_map.items(), key=lambda kv: kv[0].lower())
        ]
    )
    mapping_df.to_csv(OUTDIR / "strict895_concrete_disease_name_mapping.csv", index=False)

    summary = {
        "rows": int(len(summary_df)),
        "unique_targets": int(len(genes)),
        "targets_with_ensembl": int(sum(1 for g in genes if g in gene_to_ensembl)),
        "candidate_disease_names": int(len(disease_names)),
        "mapped_candidate_disease_names": int(mapping_df["mapped_id"].astype(str).str.len().gt(0).sum()),
        "child_ot_status_counts": summary_df["child_ot_status"].value_counts().to_dict(),
        "rows_with_child_ot_match": int(summary_df["child_ot_status"].eq("child_ot_match").sum()),
        "rows_with_child_ot_score_ge_0_2": int(summary_df["best_child_ot_score"].ge(0.2).sum()),
        "rows_with_child_ot_score_ge_0_5": int(summary_df["best_child_ot_score"].ge(0.5).sum()),
        "outputs": {
            "summary": str(OUTDIR / "strict895_opentargets_child_disease_summary.csv"),
            "long": str(OUTDIR / "strict895_opentargets_child_disease_long.csv"),
            "mapping": str(OUTDIR / "strict895_concrete_disease_name_mapping.csv"),
        },
        "note": "Associated diseases fetched through target.associatedDiseases, first 4000 diseases per target by Open Targets score.",
    }
    (OUTDIR / "strict895_opentargets_child_disease_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
