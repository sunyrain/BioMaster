#!/usr/bin/env python3
"""Complete Open Targets disease annotations for the refined final1000 package.

This script fetches target -> associated diseases through the Open Targets
GraphQL API for every unique target in the current refined final1000 table.
It writes a long target-disease table, target-level summaries, and a final1000
table enriched with top disease annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

TARGET_DISEASES_QUERY = """
query targetDiseases($ensemblId: String!, $pageIndex: Int!, $pageSize: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: $pageIndex, size: $pageSize}) {
      count
      rows {
        disease {
          id
          name
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""

META_QUERY = """
query platformMeta {
  meta {
    name
    apiVersion { x y z suffix }
    dataVersion { year month iteration }
  }
}
"""

DATATYPE_COLUMNS = [
    "genetic_association",
    "somatic_mutation",
    "known_drug",
    "clinical",
    "affected_pathway",
    "literature",
    "animal_model",
    "rna_expression",
    "genetic_literature",
]


def clean_token(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "target"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def post_graphql(
    query: str,
    variables: dict[str, Any],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
            return payload
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"Open Targets GraphQL failed after {retries} attempts: {last_error}")


def datatype_scores(row: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in row.get("datatypeScores") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "")
        if not key:
            continue
        try:
            scores[key] = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            scores[key] = 0.0
    return scores


def fetch_target(
    ensembl_id: str,
    page_size: int,
    max_pages: int,
    sleep_seconds: float,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    first = post_graphql(
        TARGET_DISEASES_QUERY,
        {"ensemblId": ensembl_id, "pageIndex": 0, "pageSize": page_size},
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    target = ((first.get("data") or {}).get("target") or None)
    if not target:
        return {"target": None, "count": 0, "rows": []}

    associated = target.get("associatedDiseases") or {}
    count = int(associated.get("count") or 0)
    rows = list(associated.get("rows") or [])
    pages = math.ceil(count / page_size) if page_size else 1
    if max_pages > 0:
        pages = min(pages, max_pages)

    for page_index in range(1, pages):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        payload = post_graphql(
            TARGET_DISEASES_QUERY,
            {"ensemblId": ensembl_id, "pageIndex": page_index, "pageSize": page_size},
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        page_target = ((payload.get("data") or {}).get("target") or {})
        page_assoc = page_target.get("associatedDiseases") or {}
        rows.extend(page_assoc.get("rows") or [])

    return {"target": {k: target.get(k) for k in ["id", "approvedSymbol", "approvedName"]}, "count": count, "rows": rows}


def target_cache_path(cache_dir: Path, gene: str, ensembl_id: str) -> Path:
    return cache_dir / f"{clean_token(gene)}__{clean_token(ensembl_id)}.json"


def load_or_fetch_target(
    cache_dir: Path,
    gene: str,
    ensembl_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = target_cache_path(cache_dir, gene, ensembl_id)
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))
    payload = fetch_target(
        ensembl_id=ensembl_id,
        page_size=args.page_size,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    payload["fetched_utc"] = now_utc()
    payload["query_ensembl_id"] = ensembl_id
    payload["query_gene"] = gene
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def rows_from_payload(gene: str, ensembl_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    target = payload.get("target") or {}
    output: list[dict[str, Any]] = []
    for idx, item in enumerate(payload.get("rows") or [], start=1):
        disease = item.get("disease") or {}
        scores = datatype_scores(item)
        row = {
            "target_gene": gene,
            "query_ensembl_id": ensembl_id,
            "opentargets_target_id": target.get("id") or ensembl_id,
            "approved_symbol": target.get("approvedSymbol") or gene,
            "approved_name": target.get("approvedName") or "",
            "disease_rank_for_target": idx,
            "disease_id": disease.get("id") or "",
            "disease_name": disease.get("name") or "",
            "overall_score": float(item.get("score") or 0.0),
            "datatype_scores_json": json.dumps(scores, ensure_ascii=False, sort_keys=True),
            "source": "OpenTargets Platform GraphQL target.associatedDiseases",
        }
        for col in DATATYPE_COLUMNS:
            row[f"{col}_score"] = float(scores.get(col, 0.0))
        output.append(row)
    return output


def top_join(group: pd.DataFrame, name_col: str, score_col: str, n: int) -> str:
    if group.empty:
        return ""
    part = group.sort_values(score_col, ascending=False).head(n)
    return "; ".join(
        f"{str(row[name_col])}({float(row[score_col]):.3f})"
        for _, row in part.iterrows()
        if str(row.get(name_col, "")).strip()
    )


def build_target_summary(long_df: pd.DataFrame, targets: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if long_df.empty:
        summary = targets.copy()
        summary["ot_full_disease_count"] = 0
        return summary
    grouped = []
    for (gene, ensembl_id), group in long_df.groupby(["target_gene", "query_ensembl_id"], dropna=False):
        top = group.sort_values("overall_score", ascending=False).head(top_n)
        grouped.append(
            {
                "primary_gene": gene,
                "anchor_opentargets_id": ensembl_id,
                "ot_full_disease_count": int(len(group)),
                "ot_full_top_diseases": top_join(top, "disease_name", "overall_score", top_n),
                "ot_full_top_disease_ids": "; ".join(top["disease_id"].astype(str).tolist()),
                "ot_full_max_disease_score": float(pd.to_numeric(group["overall_score"], errors="coerce").max()),
                "ot_full_top_genetic_diseases": top_join(group, "disease_name", "genetic_association_score", min(10, top_n)),
                "ot_full_top_clinical_diseases": top_join(group, "disease_name", "clinical_score", min(10, top_n)),
                "ot_full_top_known_drug_diseases": top_join(group, "disease_name", "known_drug_score", min(10, top_n)),
                "ot_full_top_literature_diseases": top_join(group, "disease_name", "literature_score", min(10, top_n)),
            }
        )
    summary = pd.DataFrame(grouped)
    return targets.merge(summary, on=["primary_gene", "anchor_opentargets_id"], how="left").fillna(
        {
            "ot_full_disease_count": 0,
            "ot_full_top_diseases": "",
            "ot_full_top_disease_ids": "",
            "ot_full_max_disease_score": 0.0,
            "ot_full_top_genetic_diseases": "",
            "ot_full_top_clinical_diseases": "",
            "ot_full_top_known_drug_diseases": "",
            "ot_full_top_literature_diseases": "",
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Open Targets diseases for refined final1000 targets.")
    parser.add_argument(
        "--final1000",
        default="outputs/chembl_moa_enhanced_information_package_v1/boltz_refined_3000_final_package/refined_final_1000_candidates.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/chembl_moa_enhanced_information_package_v1/opentargets_final1000_full_disease_completion",
    )
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=0, help="0 fetches all pages per target.")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    final_path = ROOT / args.final1000
    out_dir = ROOT / args.out_dir
    cache_dir = out_dir / "target_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = post_graphql(
        META_QUERY,
        {},
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    platform_meta = ((meta_payload.get("data") or {}).get("meta") or {})

    final = pd.read_csv(final_path, low_memory=False)
    required = {"primary_gene", "anchor_opentargets_id"}
    missing = required - set(final.columns)
    if missing:
        raise ValueError(f"Missing columns in final1000: {sorted(missing)}")

    target_cols = [
        "primary_gene",
        "anchor_opentargets_id",
        "protein_names",
        "target_assay_family",
        "anchor_project_assay_family",
    ]
    targets = (
        final[[c for c in target_cols if c in final.columns]]
        .drop_duplicates(["primary_gene", "anchor_opentargets_id"])
        .sort_values(["primary_gene", "anchor_opentargets_id"])
        .reset_index(drop=True)
    )
    targets.to_csv(out_dir / "final1000_unique_targets_for_opentargets.csv", index=False)

    all_rows: list[dict[str, Any]] = []
    target_status: list[dict[str, Any]] = []
    started = now_utc()
    for i, row in targets.iterrows():
        gene = str(row["primary_gene"])
        ensembl_id = str(row["anchor_opentargets_id"])
        print(f"[{i + 1}/{len(targets)}] {gene} {ensembl_id}", flush=True)
        try:
            payload = load_or_fetch_target(cache_dir, gene, ensembl_id, args)
            rows = rows_from_payload(gene, ensembl_id, payload)
            all_rows.extend(rows)
            target_status.append(
                {
                    "primary_gene": gene,
                    "anchor_opentargets_id": ensembl_id,
                    "status": "ok" if payload.get("target") else "target_not_found",
                    "reported_count": int(payload.get("count") or 0),
                    "fetched_rows": len(rows),
                    "cache_file": str(target_cache_path(cache_dir, gene, ensembl_id).relative_to(ROOT)),
                }
            )
        except Exception as error:  # noqa: BLE001
            target_status.append(
                {
                    "primary_gene": gene,
                    "anchor_opentargets_id": ensembl_id,
                    "status": "failed",
                    "reported_count": 0,
                    "fetched_rows": 0,
                    "error": f"{type(error).__name__}: {error}",
                    "cache_file": str(target_cache_path(cache_dir, gene, ensembl_id).relative_to(ROOT)),
                }
            )

    fieldnames = [
        "target_gene",
        "query_ensembl_id",
        "opentargets_target_id",
        "approved_symbol",
        "approved_name",
        "disease_rank_for_target",
        "disease_id",
        "disease_name",
        "overall_score",
        *[f"{col}_score" for col in DATATYPE_COLUMNS],
        "datatype_scores_json",
        "source",
    ]
    long_path = out_dir / "opentargets_final1000_target_disease_long.csv"
    write_csv(long_path, all_rows, fieldnames)

    status_df = pd.DataFrame(target_status)
    status_df.to_csv(out_dir / "opentargets_final1000_target_fetch_status.csv", index=False)
    long_df = pd.DataFrame(all_rows, columns=fieldnames)
    summary = build_target_summary(long_df, targets, top_n=args.top_n)
    summary.to_csv(out_dir / "opentargets_final1000_target_disease_summary.csv", index=False)

    enriched = final.merge(
        summary[
            [
                "primary_gene",
                "anchor_opentargets_id",
                "ot_full_disease_count",
                "ot_full_top_diseases",
                "ot_full_top_disease_ids",
                "ot_full_max_disease_score",
                "ot_full_top_genetic_diseases",
                "ot_full_top_clinical_diseases",
                "ot_full_top_known_drug_diseases",
                "ot_full_top_literature_diseases",
            ]
        ],
        on=["primary_gene", "anchor_opentargets_id"],
        how="left",
    )
    enriched.to_csv(out_dir / "refined_final_1000_candidates_with_full_opentargets_diseases.csv", index=False)

    readable_cols = [
        c
        for c in [
            "drug_names",
            "primary_gene",
            "protein_names",
            "target_assay_family",
            "boltz_support_tier_refined",
            "refined_enhanced_selection_score",
            "manual_review_priority",
            "risk_notes",
            "ot_full_disease_count",
            "ot_full_top_diseases",
            "ot_full_top_genetic_diseases",
            "ot_full_top_clinical_diseases",
            "ot_full_top_known_drug_diseases",
            "txgnn_top_diseases",
        ]
        if c in enriched.columns
    ]
    readable = enriched[readable_cols].copy()
    readable.to_csv(out_dir / "refined_final_1000_teacher_readable_with_full_opentargets_diseases.csv", index=False)
    with pd.ExcelWriter(out_dir / "refined_final_1000_with_full_opentargets_diseases.xlsx", engine="openpyxl") as writer:
        readable.to_excel(writer, index=False, sheet_name="final1000_readable")
        summary.to_excel(writer, index=False, sheet_name="target_disease_summary")
        status_df.to_excel(writer, index=False, sheet_name="fetch_status")

    ok_targets = int(status_df["status"].eq("ok").sum())
    failed_targets = int(status_df["status"].eq("failed").sum())
    found_targets = int((summary["ot_full_disease_count"].fillna(0).astype(float) > 0).sum())
    pair_coverage = int((enriched["ot_full_disease_count"].fillna(0).astype(float) > 0).sum())
    metadata = {
        "created_utc": now_utc(),
        "started_utc": started,
        "source": "OpenTargets Platform GraphQL API target.associatedDiseases",
        "graphql_url": GRAPHQL_URL,
        "platform_meta": platform_meta,
        "final1000_input": str(final_path.relative_to(ROOT)),
        "unique_targets": int(len(targets)),
        "ok_targets": ok_targets,
        "failed_targets": failed_targets,
        "targets_with_disease_rows": found_targets,
        "final1000_rows": int(len(final)),
        "final1000_rows_with_full_ot_diseases": pair_coverage,
        "long_rows": int(len(long_df)),
        "page_size": args.page_size,
        "max_pages": args.max_pages,
        "outputs": {
            "unique_targets": str((out_dir / "final1000_unique_targets_for_opentargets.csv").relative_to(ROOT)),
            "long": str(long_path.relative_to(ROOT)),
            "summary": str((out_dir / "opentargets_final1000_target_disease_summary.csv").relative_to(ROOT)),
            "status": str((out_dir / "opentargets_final1000_target_fetch_status.csv").relative_to(ROOT)),
            "enriched_final1000": str(
                (out_dir / "refined_final_1000_candidates_with_full_opentargets_diseases.csv").relative_to(ROOT)
            ),
            "readable_csv": str(
                (out_dir / "refined_final_1000_teacher_readable_with_full_opentargets_diseases.csv").relative_to(ROOT)
            ),
            "xlsx": str((out_dir / "refined_final_1000_with_full_opentargets_diseases.xlsx").relative_to(ROOT)),
        },
    }
    (out_dir / "opentargets_final1000_full_disease_completion_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "OPENTARGETS_FINAL1000_FULL_DISEASE_COMPLETION_REPORT_ZH.md").write_text(
        "\n".join(
            [
                "# final1000 Open Targets 疾病关联补全报告",
                "",
                f"- 生成时间：{metadata['created_utc']}",
                f"- 输入 final1000：{metadata['final1000_input']}",
                f"- unique target：{metadata['unique_targets']}",
                f"- 成功查询靶点：{metadata['ok_targets']}",
                f"- 失败靶点：{metadata['failed_targets']}",
                f"- 有疾病关联行的靶点：{metadata['targets_with_disease_rows']}",
                f"- final1000 有 full Open Targets 疾病补全的行：{metadata['final1000_rows_with_full_ot_diseases']} / {metadata['final1000_rows']}",
                f"- target-disease long table 行数：{metadata['long_rows']}",
                "",
                "## 输出文件",
                "",
                *[f"- `{v}`" for v in metadata["outputs"].values()],
                "",
                "## 口径",
                "",
                "该结果来自 Open Targets GraphQL `target.associatedDiseases`，按 final1000 的 198 个 unique target 逐靶点分页拉取。",
                "这些疾病关联用于疾病/机制辅助解释，不参与当前物理主排序。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
