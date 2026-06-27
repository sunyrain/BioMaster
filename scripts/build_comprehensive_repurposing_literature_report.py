#!/usr/bin/env python3
"""Build a comprehensive literature-aware FDA repurposing report.

This script intentionally moves beyond the old 384-molecule wet-lab panel.
It treats the all-direction mechanism evidence cards as the detailed
candidate universe, checks drug-target reporting in PubMed from 2000 to the
current project date, and adds a post-FDA-approval window as a practical proxy
for "old drug, new target" chronology.

The PubMed layer is an automated reporting/novelty audit. A co-occurrence hit
does not prove direct binding or therapeutic efficacy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CARDS = ROOT / "outputs/all_directions_mechanism_panel/candidate_evidence_cards.csv"
DEFAULT_RECALL = ROOT / "outputs/all_directions_mechanism_panel/target_recall_evidence.csv"
DEFAULT_FDA = ROOT / "FDA_approved_small_molecules_2005_2026_with_structures.xlsx"
DEFAULT_OUT_DIR = ROOT / "outputs/comprehensive_repurposing_literature"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_START_DATE = "2000/01/01"
DEFAULT_END_DATE = "2026/06/24"

DRUG_SALT_WORDS = (
    "hydrochloride|hydrochrloride|phosphate|bromide|succinate|mesylate|maleate|"
    "dimaleate|dihydrochloride|acetate|sodium|potassium|calcium|camsylate|tosylate|"
    "d-tartrate|tartrate|s-malate|malate|esylate|fumarate|arginine|pamoate|"
    "citrate|bitartrate|oxalate|dimesylate|monohydrate|dihydrate|sulfate|"
    "sulphate|besylate|benzoate|lactate|nitrate"
)

TARGET_SYNONYMS: dict[str, list[str]] = {
    "ADORA1": ["ADORA1", "adenosine receptor A1", "adenosine A1 receptor"],
    "ADORA2A": ["ADORA2A", "adenosine receptor A2A", "adenosine A2A receptor"],
    "ADRB1": ["ADRB1", "beta-1 adrenergic receptor", "beta1 adrenergic receptor"],
    "ADRB2": ["ADRB2", "beta-2 adrenergic receptor", "beta2 adrenergic receptor", "β2-AR"],
    "BCL2": ["BCL2", "B-cell lymphoma 2"],
    "CDK4": ["CDK4", "cyclin-dependent kinase 4"],
    "CDK6": ["CDK6", "cyclin-dependent kinase 6"],
    "CHRM2": ["CHRM2", "muscarinic acetylcholine receptor M2", "M2 muscarinic receptor"],
    "CHRM3": ["CHRM3", "muscarinic acetylcholine receptor M3", "M3 muscarinic receptor"],
    "CRHR1": ["CRHR1", "corticotropin-releasing factor receptor 1", "CRF1 receptor"],
    "DRD2": ["DRD2", "dopamine D2 receptor"],
    "DRD3": ["DRD3", "dopamine D3 receptor"],
    "EDNRA": ["EDNRA", "endothelin receptor type A", "endothelin-1 receptor", "ETA receptor"],
    "EDNRB": ["EDNRB", "endothelin receptor type B", "ETB receptor"],
    "EGFR": ["EGFR", "epidermal growth factor receptor", "ERBB1"],
    "ERBB2": ["ERBB2", "HER2", "receptor tyrosine-protein kinase erbB-2"],
    "ESR1": ["ESR1", "estrogen receptor alpha", "ER alpha"],
    "F2R": ["F2R", "PAR1", "protease-activated receptor 1", "proteinase-activated receptor 1"],
    "GNRHR": ["GNRHR", "gonadotropin-releasing hormone receptor", "GnRH receptor"],
    "HTR1A": ["HTR1A", "5-HT1A receptor", "serotonin 1A receptor"],
    "HTR2A": ["HTR2A", "5-HT2A receptor", "serotonin 2A receptor"],
    "HTR3A": ["HTR3A", "5-HT3 receptor", "serotonin 3A receptor"],
    "JAK1": ["JAK1", "Janus kinase 1"],
    "JAK2": ["JAK2", "Janus kinase 2"],
    "KIT": ["KIT", "c-KIT", "CD117", "mast/stem cell growth factor receptor"],
    "MET": ["MET receptor", "hepatocyte growth factor receptor", "c-Met"],
    "NFE2L2": ["NFE2L2", "Nrf2", "nuclear factor erythroid 2-related factor 2"],
    "NR3C1": ["NR3C1", "glucocorticoid receptor"],
    "NR3C2": ["NR3C2", "mineralocorticoid receptor"],
    "OPRM1": ["OPRM1", "mu opioid receptor", "mu-opioid receptor"],
    "PARP1": ["PARP1", "poly ADP-ribose polymerase 1", "poly(ADP-ribose) polymerase 1"],
    "PDE4A": ["PDE4A", "phosphodiesterase 4A"],
    "PDE5A": ["PDE5A", "phosphodiesterase type 5"],
    "PPIA": ["PPIA", "cyclophilin A", "peptidyl-prolyl cis-trans isomerase A"],
    "PTGS2": ["PTGS2", "COX-2", "cyclooxygenase-2", "prostaglandin-endoperoxide synthase 2"],
    "RXRA": ["RXRA", "retinoid X receptor alpha", "RXR alpha"],
    "S1PR1": ["S1PR1", "sphingosine-1-phosphate receptor 1", "S1P receptor 1"],
    "S1PR4": ["S1PR4", "sphingosine-1-phosphate receptor 4", "S1P receptor 4"],
    "TSHR": ["TSHR", "thyrotropin receptor", "thyroid stimulating hormone receptor"],
}

DISEASE_TERMS = {
    "oncology": "cancer OR tumor OR tumour OR neoplasm OR oncology OR carcinoma OR leukemia OR lymphoma",
    "neurology_psychiatry": (
        "brain OR neuron OR neurological OR psychiatric OR depression OR anxiety OR Alzheimer "
        "OR Parkinson OR epilepsy OR sleep"
    ),
    "infectious_disease": "infection OR infectious OR virus OR viral OR bacterial OR antimicrobial OR antiviral OR pathogen",
    "cardiovascular": "cardiovascular OR vascular OR heart OR hypertension OR cardiology OR platelet OR thrombosis",
    "immunology_inflammation": "immune OR immunology OR inflammation OR inflammatory OR autoimmune OR cytokine OR arthritis",
    "endocrinology_metabolic": "diabetes OR metabolic OR obesity OR insulin OR glucose OR lipid OR endocrine",
    "gastroenterology": "intestinal OR gastrointestinal OR liver OR hepatic OR bowel OR colitis OR Crohn",
    "respiratory": "lung OR pulmonary OR asthma OR COPD OR respiratory",
    "rare_disease": "rare disease OR orphan OR genetic disorder",
    "urology_nephrology": "kidney OR renal OR nephrology OR urology OR bladder",
    "ophthalmology": "eye OR retina OR ophthalmology OR ocular OR glaucoma",
    "musculoskeletal": "bone OR muscle OR arthritis OR musculoskeletal OR osteoporosis",
    "dermatology": "skin OR dermatitis OR psoriasis OR dermatology",
    "hematology": "blood OR hematology OR anemia OR platelet OR coagulation",
    "anesthesia_pain": "pain OR analgesia OR anesthesia OR anaesthesia OR nociception",
    "diagnostic_imaging": "diagnostic OR imaging OR contrast OR ultrasound OR radiology",
    "womens_health": "pregnancy OR endometriosis OR contraception OR ovary OR uterus OR women's health",
    "other": "disease OR therapy OR treatment OR mechanism",
}

MECHANISM_WORDS = re.compile(
    r"\b(inhibit|inhibits|inhibited|inhibition|inhibitor|antagonist|agonist|activate|"
    r"activated|activation|modulate|modulates|modulator|bind|binds|binding|bound|"
    r"block|blocks|blocker|target|targets|kinase|receptor|IC50|EC50|Ki|Kd|CETSA|"
    r"NanoBRET|knockdown|CRISPR|siRNA)\b",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def as_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def base_drug_name(drug: str) -> str:
    base = re.sub(rf"\b({DRUG_SALT_WORDS})\b", "", clean(drug), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", base).strip(" ,-")


def unique_terms(terms: list[str], max_terms: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = re.sub(r"\s+", " ", clean(term))
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def terms_for_drug(drug: str) -> list[str]:
    return unique_terms([drug, base_drug_name(drug)], 2)


def terms_for_target(gene: str, protein_name: str = "") -> list[str]:
    gene = clean(gene).upper()
    protein_name = clean(protein_name)
    terms = list(TARGET_SYNONYMS.get(gene, [gene]))
    if protein_name and 6 <= len(protein_name) <= 90:
        terms.append(protein_name)
    return unique_terms(terms, 5)


def field_term(term: str) -> str:
    escaped = term.replace('"', "")
    return f'"{escaped}"[Title/Abstract]'


def date_clause(start_date: str, end_date: str) -> str:
    return f'("{start_date}"[PDAT] : "{end_date}"[PDAT])'


def build_pair_query(
    drug: str,
    gene: str,
    protein_name: str,
    *,
    start_date: str,
    end_date: str,
    disease_direction: str | None = None,
) -> str:
    drug_part = " OR ".join(field_term(t) for t in terms_for_drug(drug))
    target_part = " OR ".join(field_term(t) for t in terms_for_target(gene, protein_name))
    query = f"({drug_part}) AND ({target_part}) AND {date_clause(start_date, end_date)}"
    if disease_direction:
        query += f" AND ({DISEASE_TERMS.get(disease_direction, DISEASE_TERMS['other'])})"
    return query


def eutils_esearch(session: requests.Session, query: str, delay_s: float, retmax: int = 3) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub+date",
    }
    url = f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    last_error = ""
    for attempt in range(5):
        time.sleep(delay_s)
        try:
            response = session.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=35)
            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = f"{response.status_code} {response.reason}"
                time.sleep(min(12.0, 1.5 * (attempt + 1) ** 2))
                continue
            response.raise_for_status()
            data = response.json().get("esearchresult", {})
            return {
                "ok": True,
                "count": int(data.get("count", 0)),
                "pmids": data.get("idlist", [])[:retmax],
                "url": url,
            }
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(12.0, 1.5 * (attempt + 1) ** 2))
    return {"ok": False, "count": None, "pmids": [], "url": url, "error": last_error}


def eutils_esummary(session: requests.Session, pmids: list[str], delay_s: float) -> list[dict[str, str]]:
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids[:20]), "retmode": "json"}
    try:
        time.sleep(delay_s)
        response = session.get(f"{EUTILS}/esummary.fcgi", params=params, timeout=35)
        response.raise_for_status()
        result = response.json().get("result", {})
    except Exception:
        return []
    articles: list[dict[str, str]] = []
    for pmid in result.get("uids", []):
        item = result.get(pmid, {})
        articles.append(
            {
                "pmid": pmid,
                "year": clean(item.get("pubdate"))[:4],
                "title": clean(item.get("title")),
                "journal": clean(item.get("fulljournalname")),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return articles


def load_fda_profile(path: Path) -> pd.DataFrame:
    fda = pd.read_excel(path)
    fda = fda.rename(
        columns={
            "ChEMBL ID": "drug_chembl_id",
            "Generic Name (INN)": "generic_name",
            "Approval Year": "approval_year",
            "Target Name": "known_target_name",
            "Target ChEMBL ID": "known_target_chembl_id",
            "Action Type": "known_action_type",
            "Mechanism of Action": "known_moa",
            "Therapeutic Area": "therapeutic_area",
            "Indication": "indication",
            "Route": "route",
        }
    )
    fda["approval_year"] = pd.to_numeric(fda["approval_year"], errors="coerce")

    def concat(values: pd.Series, limit: int = 12) -> str:
        out = []
        seen = set()
        for value in values:
            value = clean(value)
            if value and value.lower() not in seen:
                seen.add(value.lower())
                out.append(value)
            if len(out) >= limit:
                break
        return "; ".join(out)

    grouped = (
        fda.groupby("drug_chembl_id", dropna=False)
        .agg(
            fda_generic_name=("generic_name", lambda s: concat(s, 3)),
            first_approval_year=("approval_year", "min"),
            fda_therapeutic_area=("therapeutic_area", lambda s: concat(s, 6)),
            fda_indication=("indication", lambda s: concat(s, 6)),
            fda_route=("route", lambda s: concat(s, 6)),
            fda_known_targets=("known_target_name", lambda s: concat(s, 12)),
            fda_known_target_ids=("known_target_chembl_id", lambda s: concat(s, 12)),
            fda_known_action_types=("known_action_type", lambda s: concat(s, 10)),
            fda_known_moa=("known_moa", lambda s: concat(s, 12)),
        )
        .reset_index()
    )
    grouped["first_approval_year"] = grouped["first_approval_year"].astype("Int64")
    return grouped


def union_semicolon(values: pd.Series, limit: int = 30) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in clean(value).split(";"):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
            if len(out) >= limit:
                return ";".join(out)
    return ";".join(out)


def fda_text_known_target_match(row: pd.Series) -> bool:
    """Catch known pharmacology missed by exact target-edge mapping."""

    blob = " ".join(
        clean(row.get(col))
        for col in ["fda_known_targets", "fda_known_moa", "fda_known_action_types"]
    ).lower()
    if not blob:
        return False
    gene = clean(row.get("candidate_anchor_gene")).upper()
    if gene in {"CDK4", "CDK6"} and ("cdk4" in blob or "cdk6" in blob or "cyclin-dependent kinase 4" in blob or "cyclin-dependent kinase 6" in blob):
        return True
    terms = terms_for_target(gene, clean(row.get("candidate_anchor_name")))
    for term in terms:
        term_l = term.lower()
        if len(term_l) >= 4 and term_l in blob:
            return True
    return False


def build_pair_table(cards: pd.DataFrame, fda_profile: pd.DataFrame) -> pd.DataFrame:
    cards = cards.copy()
    for col in ["direct_known_label_flag", "known_drug_target_pair", "core_discovery_eligible"]:
        if col in cards.columns:
            cards[col] = cards[col].map(as_bool)
    cards["candidate_total_score"] = pd.to_numeric(cards.get("candidate_total_score"), errors="coerce")
    cards["non_table_evidence_count"] = pd.to_numeric(cards.get("non_table_evidence_count"), errors="coerce")
    cards["de_leaked_rank_proxy"] = pd.to_numeric(cards.get("de_leaked_rank_proxy"), errors="coerce")

    grouped = (
        cards.groupby(["drug_chembl_id", "generic_name", "candidate_anchor_gene"], dropna=False)
        .agg(
            candidate_anchor_name=("candidate_anchor_name", "first"),
            candidate_anchor_uniprot=("candidate_anchor_uniprot", lambda s: union_semicolon(s, 6)),
            directions=("disease_direction", lambda s: union_semicolon(s, 20)),
            direction_count=("disease_direction", "nunique"),
            mechanism_buckets=("mechanism_bucket", lambda s: union_semicolon(s, 20)),
            mechanism_bucket_count=("mechanism_bucket", "nunique"),
            source_tables=("source_table", lambda s: union_semicolon(s, 12)),
            lanes=("lane", lambda s: union_semicolon(s, 12)),
            any_direct_known_label=("direct_known_label_flag", "max"),
            any_known_drug_target_pair=("known_drug_target_pair", "max"),
            max_candidate_total_score=("candidate_total_score", "max"),
            median_candidate_total_score=("candidate_total_score", "median"),
            max_non_table_evidence_count=("non_table_evidence_count", "max"),
            min_de_leaked_rank_proxy=("de_leaked_rank_proxy", "min"),
            best_recall_sources=("recall_evidence_sources", lambda s: union_semicolon(s, 20)),
            best_non_table_sources=("non_table_evidence_sources", lambda s: union_semicolon(s, 30)),
            representative_rationale=("rationale", "first"),
            representative_assay_family=("assay_family", "first"),
            representative_readout=("primary_readout", "first"),
            representative_counterscreen=("required_counterscreens", "first"),
        )
        .reset_index()
    )
    grouped = grouped.merge(fda_profile, on="drug_chembl_id", how="left")
    grouped["generic_name"] = grouped["generic_name"].where(grouped["generic_name"].map(clean).ne(""), grouped["fda_generic_name"])

    grouped["fda_text_known_target_match"] = grouped.apply(fda_text_known_target_match, axis=1)
    grouped["pair_priority_class"] = grouped.apply(classify_pair_priority, axis=1)
    grouped["automated_novel_target_candidate"] = ~(
        grouped["any_direct_known_label"].fillna(False)
        | grouped["any_known_drug_target_pair"].fillna(False)
        | grouped["fda_text_known_target_match"].fillna(False)
    )
    return grouped


def classify_pair_priority(row: pd.Series) -> str:
    if (
        bool(row.get("any_direct_known_label"))
        or bool(row.get("any_known_drug_target_pair"))
        or bool(row.get("fda_text_known_target_match"))
    ):
        return "known_or_control"
    score = float(row.get("max_candidate_total_score") or 0)
    non_table = float(row.get("max_non_table_evidence_count") or 0)
    directions = int(row.get("direction_count") or 0)
    if score >= 80 and non_table >= 4 and directions >= 2:
        return "A_high_priority_multi_context"
    if score >= 75 and non_table >= 3:
        return "B_high_priority_single_or_focused_context"
    if score >= 65 and non_table >= 2:
        return "C_mechanism_review"
    return "D_exploratory_or_low_support"


def literature_class(row: pd.Series) -> str:
    if not bool(row.get("lit_ok", True)):
        return "query_failed"
    if (
        bool(row.get("any_direct_known_label"))
        or bool(row.get("any_known_drug_target_pair"))
        or bool(row.get("fda_text_known_target_match"))
    ):
        return "known_pharmacology_or_control_not_new"
    post_count = safe_int(row.get("post_approval_pair_pubmed_count"))
    full_count = safe_int(row.get("pair_pubmed_count_2000_2026"))
    if post_count >= 20:
        return "reported_after_approval_heavy_overlap"
    if post_count >= 3:
        return "reported_after_approval_moderate_overlap"
    if post_count >= 1:
        return "reported_after_approval_sparse_signal"
    if full_count >= 1:
        return "reported_only_before_or_without_approval_window"
    return "no_pubmed_pair_report_2000_2026"


def run_pair_literature_audit(
    pairs: pd.DataFrame,
    out_dir: Path,
    *,
    start_date: str,
    end_date: str,
    delay_s: float,
    refresh: bool,
    max_pairs: int | None,
    workers: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "pubmed_pair_audit_cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    work = pairs.copy()
    work = work.sort_values(
        ["pair_priority_class", "max_candidate_total_score", "max_non_table_evidence_count"],
        ascending=[True, False, False],
    )
    if max_pairs is not None:
        work = work.head(max_pairs).copy()

    def query_item(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        drug_id = clean(item.get("drug_chembl_id"))
        gene = clean(item.get("candidate_anchor_gene"))
        cache_key = f"{drug_id}__{gene}"
        approval_year = item.get("first_approval_year")
        try:
            approval_year_int = int(approval_year) if not pd.isna(approval_year) else 2000
        except Exception:
            approval_year_int = 2000
        post_start = f"{max(2000, approval_year_int):04d}/01/01"

        session = requests.Session()
        session.headers.update({"User-Agent": "BioMaster-comprehensive-repurposing-literature/1.0"})
        pair_query = build_pair_query(
            clean(item.get("generic_name")),
            gene,
            clean(item.get("candidate_anchor_name")),
            start_date=start_date,
            end_date=end_date,
        )
        post_query = build_pair_query(
            clean(item.get("generic_name")),
            gene,
            clean(item.get("candidate_anchor_name")),
            start_date=post_start,
            end_date=end_date,
        )
        pair = eutils_esearch(session, pair_query, delay_s=delay_s, retmax=3)
        post = eutils_esearch(session, post_query, delay_s=delay_s, retmax=3)
        return cache_key, {
            "pair_query": pair_query,
            "post_approval_query": post_query,
            "post_approval_start_date": post_start,
            "pair": pair,
            "post_approval": post,
        }

    def row_from_result(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        pair_count = result.get("pair", {}).get("count")
        post_count = result.get("post_approval", {}).get("count")
        out = dict(item)
        out.update(
            {
                "literature_window": f"{start_date}..{end_date}",
                "post_approval_window_start": result.get("post_approval_start_date", ""),
                "lit_ok": bool(result.get("pair", {}).get("ok")) and bool(result.get("post_approval", {}).get("ok")),
                "pair_pubmed_count_2000_2026": pair_count,
                "pair_pubmed_pmids_2000_2026": ";".join(result.get("pair", {}).get("pmids", [])),
                "pair_pubmed_url_2000_2026": result.get("pair", {}).get("url", ""),
                "post_approval_pair_pubmed_count": post_count,
                "post_approval_pair_pubmed_pmids": ";".join(result.get("post_approval", {}).get("pmids", [])),
                "post_approval_pair_pubmed_url": result.get("post_approval", {}).get("url", ""),
            }
        )
        out["literature_class"] = literature_class(pd.Series(out))
        out["counts_as_reported_old_drug_new_target_signal"] = (
            out["literature_class"].startswith("reported_after_approval") and bool(out["automated_novel_target_candidate"])
        )
        return out

    items = work.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    item_by_key: dict[str, dict[str, Any]] = {}

    for item in items:
        cache_key = f"{clean(item.get('drug_chembl_id'))}__{clean(item.get('candidate_anchor_gene'))}"
        item_by_key[cache_key] = item
        if cache_key in cache:
            rows.append(row_from_result(item, cache[cache_key]))
        else:
            pending.append(item)

    if pending:
        if workers <= 1:
            for idx, item in enumerate(pending, start=1):
                cache_key, result = query_item(item)
                cache[cache_key] = result
                rows.append(row_from_result(item, result))
                if idx % 100 == 0:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    print(f"[pair-literature] queried {idx}/{len(pending)} new; total rows {len(rows)}/{len(items)}", flush=True)
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(query_item, item): item for item in pending}
                for future in as_completed(future_map):
                    cache_key, result = future.result()
                    cache[cache_key] = result
                    rows.append(row_from_result(future_map[future], result))
                    completed += 1
                    if completed % 100 == 0:
                        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                        print(
                            f"[pair-literature] queried {completed}/{len(pending)} new; total rows {len(rows)}/{len(items)}",
                            flush=True,
                        )

    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return pd.DataFrame(rows).sort_values(
        ["pair_priority_class", "max_candidate_total_score", "drug_chembl_id", "candidate_anchor_gene"],
        ascending=[True, False, True, True],
    )


def annotate_combinations(cards: pd.DataFrame, pair_audit: pd.DataFrame, fda_profile: pd.DataFrame) -> pd.DataFrame:
    cards = cards.copy()
    for col in ["direct_known_label_flag", "known_drug_target_pair", "core_discovery_eligible"]:
        if col in cards.columns:
            cards[col] = cards[col].map(as_bool)
    cards = cards.merge(fda_profile, on="drug_chembl_id", how="left")
    key_cols = ["drug_chembl_id", "candidate_anchor_gene"]
    lit_cols = [
        "drug_chembl_id",
        "candidate_anchor_gene",
        "literature_window",
        "post_approval_window_start",
        "lit_ok",
        "pair_pubmed_count_2000_2026",
        "pair_pubmed_pmids_2000_2026",
        "pair_pubmed_url_2000_2026",
        "post_approval_pair_pubmed_count",
        "post_approval_pair_pubmed_pmids",
        "post_approval_pair_pubmed_url",
        "literature_class",
        "counts_as_reported_old_drug_new_target_signal",
        "pair_priority_class",
        "automated_novel_target_candidate",
        "fda_text_known_target_match",
    ]
    annotated = cards.merge(pair_audit[lit_cols], on=key_cols, how="left")
    annotated["combination_literature_status"] = annotated.apply(classify_combination_status, axis=1)
    return annotated


def classify_combination_status(row: pd.Series) -> str:
    if pd.isna(row.get("pair_pubmed_count_2000_2026")):
        return "not_checked_in_pair_literature_run"
    if (
        as_bool(row.get("direct_known_label_flag"))
        or as_bool(row.get("known_drug_target_pair"))
        or as_bool(row.get("fda_text_known_target_match"))
    ):
        return "known_label_or_control_combination"
    if safe_int(row.get("post_approval_pair_pubmed_count")) > 0:
        return "drug_target_reported_after_approval_needs_manual_validation"
    if safe_int(row.get("pair_pubmed_count_2000_2026")) > 0:
        return "drug_target_reported_in_2000_2026_but_not_post_approval_window"
    return "no_pubmed_drug_target_pair_found_2000_2026"


def summarize_recall_universe(recall_path: Path) -> dict[str, Any]:
    recall = pd.read_csv(recall_path, low_memory=False)
    return {
        "target_recall_rows": int(len(recall)),
        "target_recall_unique_drugs": int(recall["drug_chembl_id"].nunique()),
        "target_recall_unique_drug_target_pairs": int(
            recall[["drug_chembl_id", "gene_name"]].drop_duplicates().shape[0]
        ),
        "target_recall_direct_known_rows": int(recall["direct_known_label_flag"].map(as_bool).sum()),
        "target_recall_top_rank_max": int(pd.to_numeric(recall["production_rank"], errors="coerce").max()),
    }


def choose_examples(pair_audit: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    reported = pair_audit[
        pair_audit["counts_as_reported_old_drug_new_target_signal"].eq(True)
    ].copy()
    if reported.empty:
        return reported
    reported["post_approval_pair_pubmed_count"] = pd.to_numeric(
        reported["post_approval_pair_pubmed_count"], errors="coerce"
    ).fillna(0)
    reported["max_candidate_total_score"] = pd.to_numeric(
        reported["max_candidate_total_score"], errors="coerce"
    ).fillna(0)
    return reported.sort_values(
        ["pair_priority_class", "post_approval_pair_pubmed_count", "max_candidate_total_score"],
        ascending=[True, False, False],
    ).head(n)


def fetch_example_article_summaries(examples: pd.DataFrame, out_dir: Path, delay_s: float) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame()
    session = requests.Session()
    session.headers.update({"User-Agent": "BioMaster-comprehensive-repurposing-literature/1.0"})
    rows: list[dict[str, Any]] = []
    for row in examples.itertuples(index=False):
        item = row._asdict()
        pmids = [p for p in clean(item.get("post_approval_pair_pubmed_pmids")).split(";") if p]
        articles = eutils_esummary(session, pmids, delay_s=delay_s)
        rows.append(
            {
                "drug_chembl_id": item.get("drug_chembl_id"),
                "generic_name": item.get("generic_name"),
                "candidate_anchor_gene": item.get("candidate_anchor_gene"),
                "post_approval_pair_pubmed_count": item.get("post_approval_pair_pubmed_count"),
                "pmids": ";".join(a["pmid"] for a in articles),
                "years": ";".join(a["year"] for a in articles),
                "titles": " || ".join(a["title"] for a in articles),
                "urls": " || ".join(a["url"] for a in articles),
                "literature_class": item.get("literature_class"),
                "pair_priority_class": item.get("pair_priority_class"),
                "directions": item.get("directions"),
                "representative_rationale": item.get("representative_rationale"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "representative_reported_examples_with_pubmed_titles.csv", index=False)
    return df


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "NA"
    return f"{100 * numerator / denominator:.2f}%"


def write_report(
    out_dir: Path,
    *,
    recall_summary: dict[str, Any],
    cards: pd.DataFrame,
    pair_audit: pd.DataFrame,
    combinations: pd.DataFrame,
    examples_with_titles: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    total_combos = len(combinations)
    checked_combos = combinations["combination_literature_status"].ne("not_checked_in_pair_literature_run").sum()
    unique_pairs = len(pair_audit)
    novel_pairs = pair_audit["automated_novel_target_candidate"].eq(True).sum()
    reported_pairs = pair_audit["counts_as_reported_old_drug_new_target_signal"].eq(True).sum()
    no_pair_hits = pair_audit["literature_class"].eq("no_pubmed_pair_report_2000_2026").sum()
    known_pairs = pair_audit["literature_class"].eq("known_pharmacology_or_control_not_new").sum()
    fda_text_known_pairs = pair_audit["fda_text_known_target_match"].eq(True).sum()
    query_failed = pair_audit["literature_class"].eq("query_failed").sum()

    combo_status_counts = combinations["combination_literature_status"].value_counts().to_dict()
    pair_class_counts = pair_audit["literature_class"].value_counts().to_dict()
    priority_counts = pair_audit["pair_priority_class"].value_counts().to_dict()
    direction_counts = combinations.groupby("disease_direction").size().sort_values(ascending=False).to_dict()
    reported_combo_rows = combinations[
        combinations["combination_literature_status"].eq("drug_target_reported_after_approval_needs_manual_validation")
    ]

    summary = {
        "createdUtc": now_utc(),
        "literatureWindow": f"{start_date}..{end_date}",
        "recallUniverse": recall_summary,
        "detailedCombinationRows": int(total_combos),
        "detailedUniqueDrugTargetPairsChecked": int(unique_pairs),
        "detailedCombinationRowsWithPairLiteratureCheck": int(checked_combos),
        "automatedNovelTargetPairs": int(novel_pairs),
        "reportedOldDrugNewTargetPairsPostApproval": int(reported_pairs),
        "reportedOldDrugNewTargetPairRatioAmongNovelPairs": pct(int(reported_pairs), int(novel_pairs)),
        "reportedOldDrugNewTargetPairRatioAmongAllCheckedPairs": pct(int(reported_pairs), int(unique_pairs)),
        "reportedCombinationRowsPostApproval": int(len(reported_combo_rows)),
        "reportedCombinationRowRatio": pct(int(len(reported_combo_rows)), int(checked_combos)),
        "noPubmedPairHitRows": int(no_pair_hits),
        "knownOrControlPairs": int(known_pairs),
        "fdaTextKnownTargetMatchPairs": int(fda_text_known_pairs),
        "queryFailedPairs": int(query_failed),
        "pairLiteratureClassCounts": pair_class_counts,
        "combinationLiteratureStatusCounts": combo_status_counts,
        "pairPriorityClassCounts": priority_counts,
        "combinationRowsByDirection": direction_counts,
    }
    (out_dir / "comprehensive_repurposing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    direction_summary = (
        combinations.groupby(["disease_direction", "combination_literature_status"])
        .size()
        .reset_index(name="rows")
        .sort_values(["disease_direction", "rows"], ascending=[True, False])
    )
    direction_summary.to_csv(out_dir / "combination_literature_status_by_direction.csv", index=False)

    high_unreported = combinations[
        combinations["combination_literature_status"].eq("no_pubmed_drug_target_pair_found_2000_2026")
    ].copy()
    high_unreported["candidate_total_score"] = pd.to_numeric(high_unreported["candidate_total_score"], errors="coerce")
    high_unreported.sort_values(
        ["candidate_total_score", "non_table_evidence_count"], ascending=[False, False]
    ).head(300).to_csv(out_dir / "top_unreported_high_priority_combinations.csv", index=False)

    reported_combo_rows.sort_values(
        ["candidate_total_score", "post_approval_pair_pubmed_count"], ascending=[False, False]
    ).head(500).to_csv(out_dir / "top_reported_post_approval_combinations.csv", index=False)

    example_lines: list[str] = []
    for row in examples_with_titles.head(8).itertuples(index=False):
        item = row._asdict()
        title = clean(item.get("titles")).split(" || ")[0]
        url = clean(item.get("urls")).split(" || ")[0]
        example_lines.extend(
            [
                f"- {item.get('generic_name')} - {item.get('candidate_anchor_gene')}: "
                f"post-approval PubMed pair count={item.get('post_approval_pair_pubmed_count')}; "
                f"方向={item.get('directions')}; 代表 PMID/标题：{clean(item.get('pmids')).split(';')[0] if clean(item.get('pmids')) else 'NA'} "
                f"{title} {url}",
            ]
        )
    if not example_lines:
        example_lines = ["- 本轮没有拿到可展示的 post-approval reported pair 示例。"]

    lines = [
        "# FDA 老药新用全量机制组合与文献审计最终汇报",
        "",
        f"Generated UTC: {summary['createdUtc']}",
        "",
        "## 这轮真正做了什么",
        "",
        "这轮不再把 384 分子面板作为项目边界。384 只保留为旧版湿实验容量假设；本轮主分母改成两个层次：",
        "",
        f"- Recall universe：{recall_summary['target_recall_rows']:,} 条 target-recall rows，"
        f"{recall_summary['target_recall_unique_drug_target_pairs']:,} 个唯一 FDA drug-target pair，覆盖 "
        f"{recall_summary['target_recall_unique_drugs']:,} 个 FDA/ChEMBL 药物。",
        f"- Detailed mechanism universe：{total_combos:,} 条 drug-target-disease/mechanism evidence cards，"
        f"折叠为 {unique_pairs:,} 个唯一 drug-target pair；这一层有机制桶、疾病方向、action/readout、反筛和候选总分，因此作为本轮详细分析分母。",
        "",
        "## 文献审计口径",
        "",
        f"- PubMed 检索窗口：{start_date} 到 {end_date}。",
        "- 对每个唯一 drug-target pair 查询两次：2000-2026 全窗口，以及 FDA `Approval Year` 之后窗口。",
        "- 若候选不是 FDA/label known target，且在批准年之后出现 drug-target PubMed 命中，标记为 `reported_after_approval_*`。",
        "- 为避免把已知多组分/家族靶点误判为 novel，额外用 FDA target/MoA 文本匹配候选 anchor；匹配到的 pair 归为 known/control。",
        "- 这只是“是否已有报道”的自动审计，不等同于直接结合、功能验证或疗效验证；gene symbol 歧义、综述、组合用药和下游 readout 都需要人工二次判读。",
        "",
        "## 核心数量和比例",
        "",
        f"- 已检查唯一 drug-target pair：{unique_pairs:,}。",
        f"- 其中自动判定为非已知靶点候选：{novel_pairs:,}。",
        f"- post-approval 后已有 PubMed drug-target 报道信号：{reported_pairs:,}/{novel_pairs:,} = {pct(int(reported_pairs), int(novel_pairs))}，"
        f"按全部检查 pair 为 {reported_pairs:,}/{unique_pairs:,} = {pct(int(reported_pairs), int(unique_pairs))}。",
        f"- 映射回详细机制组合后，post-approval 已报道信号组合：{len(reported_combo_rows):,}/{checked_combos:,} = {pct(int(len(reported_combo_rows)), int(checked_combos))}。",
        f"- 2000-2026 未见 PubMed drug-target pair 直接命中的唯一 pair：{no_pair_hits:,}/{unique_pairs:,} = {pct(int(no_pair_hits), int(unique_pairs))}。",
        f"- known/control pair：{known_pairs:,}，其中 FDA target/MoA 文本补充识别 {fda_text_known_pairs:,} 个；query failed：{query_failed:,}。",
        "",
        "## Pair 文献分类分布",
        "",
        json.dumps(pair_class_counts, ensure_ascii=False, indent=2),
        "",
        "## 组合状态分布",
        "",
        json.dumps(combo_status_counts, ensure_ascii=False, indent=2),
        "",
        "## 代表性已报道例子",
        "",
        *example_lines,
        "",
        "## 应该如何向老师解释",
        "",
        "本轮结论不能说“模型发现已经被文献证明”。更准确的说法是：我们把 FDA 小分子和 ChEMBL 成药锚点组合成一个高召回候选宇宙，再对所有有机制解释的组合做 2000-2026 文献去重。"
        "结果把候选分成三类：已知/对照、post-approval 后已有报道的可复现或可排重候选、以及 PubMed 未见直接 pair 的潜在新颖候选。"
        "下一步应该从未报道但机制/实验可行性强的组合里选 wet-lab wave，而不是固定 384，也不是把亲和分数当最终答案。",
        "",
        "## 交付文件",
        "",
        "- `unique_drug_target_pairs_literature_audit.csv`：唯一 drug-target pair 文献审计。",
        "- `all_candidate_combinations_literature_audit.csv`：39,328 条机制组合回填文献状态。",
        "- `top_reported_post_approval_combinations.csv`：已有报道、适合排重/复现/机制核查的组合。",
        "- `top_unreported_high_priority_combinations.csv`：未见 PubMed 直接 pair、优先人工审阅的新颖候选。",
        "- `combination_literature_status_by_direction.csv`：各疾病方向的已报道/未报道分布。",
    ]
    (out_dir / "FINAL_COMPREHENSIVE_REPURPOSING_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--target-recall", type=Path, default=DEFAULT_RECALL)
    parser.add_argument("--fda", type=Path, default=DEFAULT_FDA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--delay-s", type=float, default=0.12)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fda_profile = load_fda_profile(args.fda)
    cards = pd.read_csv(args.cards, low_memory=False)
    recall_summary = summarize_recall_universe(args.target_recall)

    pair_table = build_pair_table(cards, fda_profile)
    pair_table.to_csv(args.out_dir / "unique_drug_target_pairs_for_literature_audit.csv", index=False)

    pair_audit = run_pair_literature_audit(
        pair_table,
        args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        delay_s=args.delay_s,
        refresh=args.refresh,
        max_pairs=args.max_pairs,
        workers=args.workers,
    )
    pair_audit.to_csv(args.out_dir / "unique_drug_target_pairs_literature_audit.csv", index=False)

    combinations = annotate_combinations(cards, pair_audit, fda_profile)
    combinations.to_csv(args.out_dir / "all_candidate_combinations_literature_audit.csv", index=False)

    examples = choose_examples(pair_audit, n=20)
    examples.to_csv(args.out_dir / "representative_reported_examples.csv", index=False)
    examples_with_titles = fetch_example_article_summaries(examples, args.out_dir, delay_s=args.delay_s)

    summary = write_report(
        args.out_dir,
        recall_summary=recall_summary,
        cards=cards,
        pair_audit=pair_audit,
        combinations=combinations,
        examples_with_titles=examples_with_titles,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out_dir / 'FINAL_COMPREHENSIVE_REPURPOSING_REPORT_ZH.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
