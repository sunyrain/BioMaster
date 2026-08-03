#!/usr/bin/env python3
"""Fetch PubMed titles/abstracts for pair-screen hits and assign review tiers.

The resulting tiers are triage signals only. Keyword language in an abstract
does not prove that the named drug was directly measured against the named
human target.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests


EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DIRECT_PATTERN = re.compile(
    r"\b(bind(?:s|ing)?|binding|affinit(?:y|ies)|dissociation constant|"
    r"surface plasmon resonance|isothermal titration|thermal shift|"
    r"radioligand|competition assay|k[i,d]\b|ic50\b|ec50\b)",
    re.IGNORECASE,
)
FUNCTIONAL_PATTERN = re.compile(
    r"\b(inhibit(?:s|ed|ion)?|activat(?:e|es|ed|ion)|agonis[tm]|"
    r"antagonis[tm]|modulat(?:e|es|ed|ion)|enzyme activity|uptake assay)",
    re.IGNORECASE,
)
COMPUTATIONAL_PATTERN = re.compile(
    r"\b(docking|in silico|molecular dynamics|virtual screening|computational)\b",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def split_pmids(*values: Any) -> list[str]:
    return sorted(
        {
            token
            for value in values
            for token in re.split(r"[;,|\s]+", clean(value))
            if token.isdigit()
        }
    )


def node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def fetch_batch(pmids: list[str], retries: int = 5) -> str:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    last_error = ""
    for attempt in range(retries):
        try:
            response = requests.get(EFETCH_URL, params=params, timeout=90)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"PubMed efetch failed: {last_error}")


def parse_articles(xml_text: str) -> dict[str, dict[str, str]]:
    root = ET.fromstring(xml_text)
    records: dict[str, dict[str, str]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = node_text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue
        title = node_text(article.find(".//Article/ArticleTitle"))
        abstract_parts = [node_text(node) for node in article.findall(".//Article/Abstract/AbstractText")]
        abstract = " ".join(part for part in abstract_parts if part)
        doi = ""
        for node in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if node.attrib.get("IdType") == "doi":
                doi = node_text(node)
                break
        records[pmid] = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "doi": doi,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--articles-output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--delay-s", type=float, default=0.4)
    args = parser.parse_args()

    audit = pd.read_csv(args.audit, low_memory=False).fillna("")
    cache_path = Path(args.cache)
    articles: dict[str, dict[str, str]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    all_pmids = sorted(
        {
            pmid
            for _, row in audit.iterrows()
            for pmid in split_pmids(
                row.get("pair_pubmed_pmids_2000_2026"),
                row.get("post_approval_pair_pubmed_pmids"),
            )
        }
    )
    missing = [pmid for pmid in all_pmids if pmid not in articles]
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        articles.update(parse_articles(fetch_batch(batch)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
        if args.delay_s and start + args.batch_size < len(missing):
            time.sleep(args.delay_s)

    annotations: list[dict[str, Any]] = []
    for _, row in audit.iterrows():
        pmids = split_pmids(
            row.get("pair_pubmed_pmids_2000_2026"),
            row.get("post_approval_pair_pubmed_pmids"),
        )
        records = [articles[pmid] for pmid in pmids if pmid in articles]
        text = " ".join(f"{record['title']} {record['abstract']}" for record in records)
        direct = bool(DIRECT_PATTERN.search(text))
        functional = bool(FUNCTIONAL_PATTERN.search(text))
        computational = bool(COMPUTATIONAL_PATTERN.search(text))
        if not pmids:
            tier = "P0_no_pair_cooccurrence"
        elif direct:
            tier = "P1_direct_measurement_language_manual_validation"
        elif functional:
            tier = "P2_functional_language_manual_validation"
        else:
            tier = "P3_cooccurrence_only_manual_validation"
        annotations.append(
            {
                "pubmed_abstracts_retrieved": len(records),
                "pubmed_screen_tier_v6": tier,
                "pubmed_direct_measurement_language": direct,
                "pubmed_functional_language": functional,
                "pubmed_computational_language": computational,
                "pubmed_screen_titles_v6": " | ".join(record["title"] for record in records),
                "pubmed_screen_dois_v6": ";".join(
                    record["doi"] for record in records if record["doi"]
                ),
                "pubmed_screen_note_v6": (
                    "Automated abstract-language triage only; exact drug-target evidence requires original-paper review."
                ),
            }
        )

    result = pd.concat([audit.reset_index(drop=True), pd.DataFrame(annotations)], axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    article_output = Path(args.articles_output)
    pd.DataFrame(list(articles.values())).sort_values("pmid").to_csv(article_output, index=False)
    summary = {
        "rows": int(len(result)),
        "unique_pmids": int(len(all_pmids)),
        "articles_retrieved": int(sum(pmid in articles for pmid in all_pmids)),
        "tier_counts": result["pubmed_screen_tier_v6"].value_counts().to_dict(),
        "warning": "Abstract keyword tiers are triage signals, not exact-pair validation.",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
