#!/usr/bin/env python3
"""Merge strict895 chunk agent reviews and build summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs/broad_mechanism_layer_v2/strict895_agent_review"
FIRST_PASS = OUTDIR / "strict_top_ready_895_first_pass_analysis.csv"
REVIEWS = OUTDIR / "agent_reviews"
MERGED = OUTDIR / "strict_top_ready_895_agent_review_merged.csv"
SUMMARY = OUTDIR / "strict895_agent_review_merged_summary.json"
REPORT = OUTDIR / "STRICT895_AGENT_REVIEW_REPORT_ZH.md"


EXPECTED_COLS = [
    "review_id",
    "drug_name",
    "target_gene",
    "final_candidate_diseases_zh",
    "final_mechanism_assessment_zh",
    "final_feasibility_assessment_zh",
    "assay_recommendation_zh",
    "counterscreen_recommendation_zh",
    "risk_level",
    "priority_after_agent",
    "agent_decision",
    "agent_notes_zh",
]


def read_reviews() -> pd.DataFrame:
    frames = []
    for path in sorted(REVIEWS.glob("strict895_chunk_*_agent_review.csv")):
        df = pd.read_csv(path, low_memory=False)
        df["agent_review_file"] = path.name
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = ""
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLS + ["agent_review_file"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    first = pd.read_csv(FIRST_PASS, low_memory=False)
    reviews = read_reviews()
    duplicate_ids = reviews["review_id"][reviews["review_id"].duplicated()].dropna().astype(str).tolist()
    reviews = reviews.drop_duplicates("review_id", keep="last")

    merged = first.merge(
        reviews[EXPECTED_COLS + ["agent_review_file"]],
        on="review_id",
        how="left",
        suffixes=("_auto", "_agent"),
    )
    merged["has_agent_review"] = merged["agent_decision"].notna() & merged["agent_decision"].astype(str).str.len().gt(0)

    missing_ids = merged.loc[~merged["has_agent_review"], "review_id"].tolist()
    merged.to_csv(MERGED, index=False)

    def vc(col: str) -> dict[str, int]:
        if col not in merged.columns:
            return {}
        return {str(k): int(v) for k, v in merged[col].fillna("NA").value_counts().items()}

    summary = {
        "expected_rows": int(len(first)),
        "agent_review_files": sorted(p.name for p in REVIEWS.glob("strict895_chunk_*_agent_review.csv")),
        "agent_review_rows_raw": int(len(read_reviews())),
        "agent_review_rows_unique": int(reviews["review_id"].nunique()),
        "merged_rows": int(len(merged)),
        "reviewed_rows": int(merged["has_agent_review"].sum()),
        "missing_agent_review_rows": int((~merged["has_agent_review"]).sum()),
        "missing_review_ids": missing_ids,
        "duplicate_review_ids_in_agent_outputs": duplicate_ids,
        "agent_decision_counts": vc("agent_decision"),
        "risk_level_counts": vc("risk_level"),
        "priority_after_agent_counts": vc("priority_after_agent"),
        "auto_priority_bucket_counts": vc("priority_bucket_auto"),
        "direction_counts": vc("direction"),
        "assay_lane_counts": vc("assay_lane"),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top_cols = [
        "review_id",
        "drug_name_auto",
        "target_gene_auto",
        "direction",
        "conplex_score",
        "review_priority_score",
        "agent_decision",
        "risk_level",
        "final_candidate_diseases_zh",
        "agent_notes_zh",
    ]
    high = merged[merged.get("agent_decision", "").eq("keep_high")].head(20)
    low = merged[merged.get("agent_decision", "").isin(["deprioritize", "review_low"])].head(20)

    report = [
        "# strict_top_ready 895 条 agent 机制与可行性审计报告",
        "",
        "## 合并状态",
        "",
        f"- 预期行数：{summary['expected_rows']}",
        f"- 已收到 agent review 文件：{len(summary['agent_review_files'])}",
        f"- 已审计行数：{summary['reviewed_rows']}",
        f"- 未收到 agent 审计行数：{summary['missing_agent_review_rows']}",
        "",
        "## Agent 决策分布",
        "",
        pd.Series(summary["agent_decision_counts"]).to_markdown(),
        "",
        "## 风险等级分布",
        "",
        pd.Series(summary["risk_level_counts"]).to_markdown(),
        "",
        "## 方向分布",
        "",
        pd.Series(summary["direction_counts"]).to_markdown(),
        "",
        "## Assay lane 分布",
        "",
        pd.Series(summary["assay_lane_counts"]).to_markdown(),
        "",
        "## keep_high 示例 Top20",
        "",
        high[[c for c in top_cols if c in high.columns]].to_markdown(index=False) if len(high) else "暂无 keep_high。",
        "",
        "## 降级/低优先级示例 Top20",
        "",
        low[[c for c in top_cols if c in low.columns]].to_markdown(index=False) if len(low) else "暂无降级项。",
        "",
        "## 输出文件",
        "",
        f"- 合并表：`{MERGED}`",
        f"- 汇总 JSON：`{SUMMARY}`",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
