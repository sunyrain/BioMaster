#!/usr/bin/env python3
"""Audit the fully deep-reviewed V7 Top500 and write a compact Chinese report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "yes", "y"})


def counts(series: pd.Series) -> dict[str, int]:
    return {clean(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all1000", required=True)
    parser.add_argument("--top500", required=True)
    parser.add_argument("--review625", required=True)
    parser.add_argument("--adjudications-dir", required=True)
    parser.add_argument("--old-top500", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    all_path = Path(args.all1000)
    top_path = Path(args.top500)
    all1000 = pd.read_csv(all_path, low_memory=False).fillna("")
    top500 = pd.read_csv(top_path, low_memory=False).fillna("")
    reviewed625 = pd.read_csv(args.review625, low_memory=False).fillna("")
    old = pd.read_csv(args.old_top500, low_memory=False).fillna("")

    required_agent = [
        "agent_feasibility_grade",
        "agent_verdict",
        "agent_literature_class",
        "agent_exposure_feasibility",
        "agent_active_species_status",
        "agent_assay_plan",
        "agent_key_risks",
        "agent_sources",
        "agent_reviewed_utc",
    ]
    checks = {
        "all1000_exact_unique": len(all1000) == 1000 and all1000["pair_id"].nunique() == 1000,
        "top500_exact_unique": len(top500) == 500 and top500["pair_id"].nunique() == 500,
        "top500_subset_of_all1000": set(top500["pair_id"]).issubset(set(all1000["pair_id"])),
        "all1000_deep_review_fields_complete": all(
            column in all1000
            and all1000[column].astype(str).str.strip().ne("").all()
            for column in required_agent
        ),
        "top500_no_grade_D": not top500["feasibility_grade_v6"].eq("D").any(),
        "top500_no_contradictory": not top500["agent_literature_class"].eq("contradictory").any(),
        "top500_no_exact_validated": not top500["agent_literature_class"].eq("exact_pair_validated").any(),
        "top500_active_species_resolved": top500["active_species_status_v6"].isin(
            ["parent_drug_relevant", "salt_normalization_adequate"]
        ).all(),
        "top500_database_identity_resolved": not top500["agent_database_query_resolution"].eq("unresolved").any(),
        "top500_no_known_fda_pair": not truthy(top500["is_known_fda_target_pair"]).any(),
        "top500_no_family_or_rediscovery_risk": not truthy(top500["family_or_rediscovery_risk_v2"]).any(),
        "top500_no_severe_liability": not truthy(top500["severe_compound_liability"]).any(),
        "top500_no_structure_sequence_mismatch": not truthy(top500["structure_sequence_mismatch_v4"]).any(),
        "top500_pose_stability_A_or_B": top500["pose_stability_tier"].str.startswith(("A_", "B_")).all(),
        "top500_collapsed_conplex_rank_le_100": (pd.to_numeric(top500["rank_within_drug"], errors="coerce") <= 100).all(),
        "top500_physics_strength_A_or_B": top500["v5_strength_tier"].str.startswith(("A_", "B_")).all(),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"V7 audit failed: {failed}")

    adjudication_frames = []
    audit_paths = sorted(Path(args.adjudications_dir).glob("batch_*_adjudication.audit.json"))
    for audit_path in audit_paths:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise ValueError(f"Adjudication audit not passed: {audit_path}")
        csv_path = audit_path.with_suffix("").with_suffix(".csv")
        adjudication_frames.append(pd.read_csv(csv_path, low_memory=False).fillna(""))
    adjudication = pd.concat(adjudication_frames, ignore_index=True)
    if len(adjudication) != 384 or adjudication["pair_id"].nunique() != 384:
        raise ValueError("Expected 384 unique audited adjudications")

    selected_ids = set(top500["pair_id"])
    old_ids = set(old["pair_id"])
    reviewed625_ids = set(reviewed625["pair_id"])
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "deep_review": {
            "all1000_rows": int(len(all1000)),
            "newly_deep_reviewed_rows": int(len(reviewed625)),
            "adjudicated_critical_rows": int(len(adjudication)),
            "adjudication_decision_counts": counts(adjudication["adjudication_decision"]),
            "adjudication_grade_changes": int(
                adjudication["adjudication_decision"].isin(["upgrade", "downgrade"]).sum()
            ),
            "all1000_grade_counts": counts(all1000["feasibility_grade_v6"]),
            "all1000_literature_counts": counts(all1000["agent_literature_class"]),
            "all1000_active_species_counts": counts(all1000["active_species_status_v6"]),
        },
        "top500": {
            "rows": int(len(top500)),
            "unique_drugs": int(top500["drug_chembl_id"].nunique()),
            "unique_targets": int(top500["primary_gene"].nunique()),
            "unique_scaffolds": int(top500["murcko_scaffold"].nunique()),
            "from_newly_reviewed625": int(top500["pair_id"].isin(reviewed625_ids).sum()),
            "feasibility_grade_counts": counts(top500["feasibility_grade_v6"]),
            "execution_tier_counts": counts(top500["experimental_execution_tier_v6"]),
            "physics_strength_counts": counts(top500["v5_strength_tier"]),
            "assay_family_counts": counts(top500["target_assay_family"]),
            "candidate_role_counts": counts(top500["candidate_role_v6"]),
            "hot_target_rows": int(truthy(top500["is_hot_target_2026_v6"]).sum()),
            "hot_target_unique_genes": int(
                top500.loc[truthy(top500["is_hot_target_2026_v6"]), "primary_gene"].nunique()
            ),
            "hot_target_genes": sorted(
                set(top500.loc[truthy(top500["is_hot_target_2026_v6"]), "primary_gene"])
            ),
        },
        "old_v6_comparison": {
            "overlap": int(len(selected_ids & old_ids)),
            "newly_entered": int(len(selected_ids - old_ids)),
            "removed": int(len(old_ids - selected_ids)),
        },
        "sha256": {"all1000": sha256(all_path), "top500": sha256(top_path)},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FINAL500_FULL_DEEP_REVIEW_V7_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    top = summary["top500"]
    deep = summary["deep_review"]
    compare = summary["old_v6_comparison"]
    report = [
        "# FDA老药新靶点 Top500 全深审结果（V7）",
        "",
        f"- 全1000条候选均完成逐条深审；本轮补审{deep['newly_deep_reviewed_rows']}条。",
        f"- {deep['adjudicated_critical_rows']}条关键结论完成第二位审阅者独立裁决，等级变化{deep['adjudication_grade_changes']}条。",
        f"- 正式Top500覆盖{top['unique_drugs']}个药物、{top['unique_targets']}个靶点、{top['unique_scaffolds']}个Murcko骨架。",
        f"- 与V6初版重叠{compare['overlap']}条，新进入{compare['newly_entered']}条，移出{compare['removed']}条。",
        "",
        "## 正式硬门",
        "",
        "Top500不含已知FDA pair、同家族/标签泄露、D级、直接反证、精确已验证pair、严重化合物责任、结构序列错配、前药重跑、活性实体不确定或数据库身份未解决项。",
        "全部候选满足ConPLEx药内折叠Top100、Boltz双构象稳定性A/B，并处于已知阳性校准的物理A/B档。",
        "",
        "## 结果分层",
        "",
        f"- 实验可行性：{top['feasibility_grade_counts']}。",
        f"- 实验执行层：{top['execution_tier_counts']}。",
        f"- 物理强度：{top['physics_strength_counts']}。",
        f"- 靶点实验类型：{top['assay_family_counts']}。",
        f"- 热门靶点：{top['hot_target_rows']}条、{top['hot_target_unique_genes']}个基因（{', '.join(top['hot_target_genes'])}）。",
        "",
        "C级不等于计算无效，而是暴露、选择性或assay仍有主要不确定性；因此500条是亲和发现包，不是500条同等成熟的转化候选。优先实验应先看T1，再看T2，T3使用专门膜蛋白或复杂体系。",
    ]
    (out_dir / "FINAL500_FULL_DEEP_REVIEW_V7_REPORT_ZH.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
