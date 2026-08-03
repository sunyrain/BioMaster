#!/usr/bin/env python3
"""Create an independent adjudication queue for consequential V6 deep-review calls."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ADJUDICATION_COLUMNS = [
    "pair_id",
    "adjudication_decision",
    "adjudicated_feasibility_grade",
    "adjudicated_literature_class",
    "adjudicated_active_species_status",
    "adjudicated_database_query_resolution",
    "adjudicated_exposure_feasibility",
    "adjudicated_verdict",
    "adjudication_rationale",
    "adjudication_confidence",
    "adjudication_sources",
    "adjudicated_utc",
]


def reasons(row: pd.Series) -> str:
    values: list[str] = []
    grade = str(row["agent_feasibility_grade"])
    literature = str(row["agent_literature_class"])
    species = str(row["agent_active_species_status"])
    database = str(row["agent_database_query_resolution"])
    if grade in {"A", "B"}:
        values.append("potential_final_selection")
    if grade == "D":
        values.append("hard_exclusion_false_negative_check")
    if literature == "exact_pair_validated":
        values.append("exact_pair_claim")
    if literature == "contradictory":
        values.append("counterevidence_claim")
    if species == "prodrug_active_metabolite_requires_rerun":
        values.append("prodrug_or_metabolite_rerun")
    if species == "active_species_uncertain":
        values.append("active_species_uncertain")
    if database == "unresolved":
        values.append("database_unresolved")
    return ";".join(values)


def priority(reason: str) -> int:
    if "exact_pair_claim" in reason or "counterevidence_claim" in reason:
        return 1
    if "prodrug_or_metabolite_rerun" in reason or "database_unresolved" in reason:
        return 2
    if "potential_final_selection" in reason:
        return 3
    if "hard_exclusion_false_negative_check" in reason:
        return 4
    return 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage2", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()

    stage1 = pd.read_csv(args.stage1, low_memory=False).fillna("")
    stage2 = pd.read_csv(args.stage2, low_memory=False).fillna("")
    stage1["first_review_stage"] = "stage1_current_top500"
    stage2["first_review_stage"] = "stage2_reserve"
    combined = pd.concat([stage1, stage2], ignore_index=True)
    if combined["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id across stage1 and stage2")
    if len(combined) != 625:
        raise ValueError(f"Expected 625 first-pass rows, observed {len(combined)}")

    combined["adjudication_reason"] = combined.apply(reasons, axis=1)
    queue = combined.loc[combined["adjudication_reason"].ne("")].copy()
    queue["adjudication_priority"] = queue["adjudication_reason"].map(priority)
    queue = queue.sort_values(
        ["adjudication_priority", "first_review_stage", "v5_pair_physics_score", "pair_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    queue.insert(0, "adjudication_queue_rank", range(1, len(queue) + 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "CRITICAL_ADJUDICATION_QUEUE_V6.csv"
    queue.to_csv(all_path, index=False)

    batch_count = max(1, min(args.batches, len(queue)))
    batch_size = math.ceil(len(queue) / batch_count)
    manifest: list[dict[str, object]] = []
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        stop = min(len(queue), start + batch_size)
        if start >= stop:
            continue
        batch = queue.iloc[start:stop].copy()
        name = f"batch_{batch_index + 1:02d}"
        input_path = out_dir / f"{name}_input.csv"
        output_path = out_dir / f"{name}_adjudication.csv"
        batch.to_csv(input_path, index=False)
        template = pd.DataFrame({column: "" for column in ADJUDICATION_COLUMNS}, index=batch.index)
        template["pair_id"] = batch["pair_id"].values
        template.to_csv(output_path, index=False)
        manifest.append(
            {
                "batch": name,
                "rows": int(len(batch)),
                "start_queue_rank": int(batch["adjudication_queue_rank"].min()),
                "end_queue_rank": int(batch["adjudication_queue_rank"].max()),
                "input": str(input_path),
                "output": str(output_path),
            }
        )

    contract = """# V6 关键结论独立裁决合同

这是第二位审阅者的独立裁决，不是文字润色。必须重新核查第一审阅者据以作出结论的原始来源。

- 重点裁决：A/B 入选判断、D 硬排除、精确 pair、明确反证、前药/活性代谢物和数据库异常。
- 不得把同家族、通路、表达、疾病共现、文献共现或计算结构分当作直接互作事实。
- `adjudication_decision`: confirm / upgrade / downgrade / revise_non_grade。
- `adjudicated_feasibility_grade`: A / B / C / D。
- `adjudicated_literature_class`: exact_pair_validated / functional_only / indirect_or_family_only / no_exact_report_found / contradictory。
- `adjudicated_active_species_status`: parent_drug_relevant / salt_normalization_adequate / active_species_uncertain / prodrug_active_metabolite_requires_rerun。
- `adjudicated_database_query_resolution`: not_needed / resolved_manually / unresolved。
- 暴露判断优先使用人体游离暴露；不能把总 Cmax 与体外总浓度直接等同。
- 代谢酶底物/抑制剂关系只有在精确人源靶点及直接 assay 语境成立时才可归类，且不能自动作为创新新靶点。
- 每行必须给出可解析 PMID、DOI 或稳定 URL；不得照抄第一审阅者来源而不核查。
- 所有字段非空，pair_id 与输入一一对应，UTC 使用 ISO-8601。
"""
    (out_dir / "ADJUDICATION_INSTRUCTIONS_ZH.md").write_text(contract, encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first_pass_rows": int(len(combined)),
        "adjudication_rows": int(len(queue)),
        "reason_counts": {
            key: int(queue["adjudication_reason"].str.contains(key, regex=False).sum())
            for key in [
                "potential_final_selection",
                "hard_exclusion_false_negative_check",
                "exact_pair_claim",
                "counterevidence_claim",
                "prodrug_or_metabolite_rerun",
                "active_species_uncertain",
                "database_unresolved",
            ]
        },
        "batches": manifest,
    }
    (out_dir / "adjudication_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
