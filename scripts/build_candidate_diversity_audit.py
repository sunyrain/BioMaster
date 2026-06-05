from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def hhi(values: list[str]) -> float:
    if not values:
        return 0.0
    total = len(values)
    counts = Counter(values)
    return sum((count / total) ** 2 for count in counts.values())


def target_family(gene: str, protein_name: str = "") -> str:
    gene = str(gene or "").upper()
    name = str(protein_name or "").lower()
    if gene.startswith("ADRA") or gene.startswith("ADRB"):
        return "adrenergic_receptor"
    if gene.startswith("DRD"):
        return "dopamine_receptor"
    if gene.startswith("HTR"):
        return "serotonin_receptor"
    if gene.startswith("OPR"):
        return "opioid_receptor"
    if gene.startswith("EDNR"):
        return "endothelin_receptor"
    if gene.startswith("EPH") or gene in {"SRC", "FYN", "FRK", "HCK", "YES1", "ABL1", "KIT", "EGFR", "FLT3", "CSF1R"}:
        return "kinase_or_receptor_tyrosine_kinase"
    if gene.startswith("CA"):
        return "carbonic_anhydrase"
    if gene.startswith("CHRM"):
        return "muscarinic_receptor"
    if gene.startswith("SSTR"):
        return "somatostatin_receptor"
    if gene.startswith("GABR"):
        return "gaba_receptor"
    if gene.startswith("PTGER"):
        return "prostaglandin_receptor"
    if "kinase" in name:
        return "kinase_or_receptor_tyrosine_kinase"
    if "receptor" in name:
        return "other_receptor"
    if "enzyme" in name:
        return "enzyme"
    return "other"


def concentration_rows(df: pd.DataFrame, cutoffs: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, group in df.groupby("direction"):
        group = group.sort_values("finalRankWithinDirection")
        for cutoff in cutoffs:
            top = group.head(cutoff)
            if top.empty:
                continue
            drug_counts = Counter(top["drug"])
            target_counts = Counter(top["target"])
            family_counts = Counter(top["targetFamily"])
            track_counts = Counter(top["reviewTrack"])
            rows.append(
                {
                    "direction": direction,
                    "labelZh": DIRECTION_LABELS.get(direction, direction),
                    "cutoff": cutoff,
                    "rows": len(top),
                    "uniqueDrugs": int(top["drug"].nunique()),
                    "uniqueTargets": int(top["target"].nunique()),
                    "uniqueTargetFamilies": int(top["targetFamily"].nunique()),
                    "drugHHI": round(hhi(top["drug"].tolist()), 4),
                    "targetHHI": round(hhi(top["target"].tolist()), 4),
                    "targetFamilyHHI": round(hhi(top["targetFamily"].tolist()), 4),
                    "topDrug": drug_counts.most_common(1)[0][0],
                    "topDrugPct": round(pct(drug_counts.most_common(1)[0][1], len(top)), 2),
                    "topTarget": target_counts.most_common(1)[0][0],
                    "topTargetPct": round(pct(target_counts.most_common(1)[0][1], len(top)), 2),
                    "topTargetFamily": family_counts.most_common(1)[0][0],
                    "topTargetFamilyPct": round(pct(family_counts.most_common(1)[0][1], len(top)), 2),
                    "knownTargetRows": int(pd.to_numeric(top["knownDrugTargetPair"], errors="coerce").fillna(0).sum()),
                    "novelReviewRows": int(top["reviewTrack"].isin(["A_repurposing_mechanism_review", "B_novel_pair_disease_context_review"]).sum()),
                    "safetyReviewRows": int(top["reviewTrack"].eq("C_safety_or_contraindication_review").sum()),
                    "deprioritizedRows": int(top["reviewTrack"].eq("D_deprioritize_until_issue_resolved").sum()),
                    "reviewTrackCounts": dict(track_counts),
                }
            )
    return rows


def select_diverse_shortlist(df: pd.DataFrame, per_direction: int, max_per_drug: int, max_per_target_family: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for direction, group in df.groupby("direction"):
        drug_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        direction_selected: list[pd.Series] = []

        # Pass 1: favor clean A/B candidates with a cap on repeated drugs and families.
        ranked = group.sort_values(["finalPriorityScore", "rank"], ascending=[False, True])
        for _, row in ranked.iterrows():
            if len(direction_selected) >= per_direction:
                break
            drug = str(row["drug"])
            family = str(row["targetFamily"])
            target = str(row["target"])
            if drug_counts[drug] >= max_per_drug:
                continue
            if family_counts[family] >= max_per_target_family:
                continue
            if target_counts[target] >= 2:
                continue
            if row.get("hardFlags") != "none":
                continue
            direction_selected.append(row)
            drug_counts[drug] += 1
            family_counts[family] += 1
            target_counts[target] += 1

        # Pass 2: backfill if a direction has too few clean/diverse candidates.
        for _, row in ranked.iterrows():
            if len(direction_selected) >= per_direction:
                break
            pair_id = row["pairId"]
            if any(existing["pairId"] == pair_id for existing in direction_selected):
                continue
            drug = str(row["drug"])
            if drug_counts[drug] >= max_per_drug + 1:
                continue
            if row.get("hardFlags") != "none":
                continue
            direction_selected.append(row)
            drug_counts[drug] += 1
            family_counts[str(row["targetFamily"])] += 1
            target_counts[str(row["target"])] += 1

        selected.extend(direction_selected)

    if not selected:
        return pd.DataFrame(columns=df.columns)
    result = pd.DataFrame(selected)
    result = result.sort_values(["direction", "finalPriorityScore", "rank"], ascending=[True, False, True]).copy()
    result["diverseRankWithinDirection"] = result.groupby("direction").cumcount() + 1
    return result


def table_columns() -> list[str]:
    return [
        "diverseRankWithinDirection",
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "directionLabelZhFinal",
        "drug",
        "target",
        "protein",
        "targetFamily",
        "finalPriorityScore",
        "finalPriorityTier",
        "reviewTrack",
        "knownDrugTargetPair",
        "noveltyLabelZh",
        "admetTier",
        "kgEvidenceScore",
        "poseAuditStatus",
        "poseAuditReason",
        "hardFlags",
        "softFlags",
        "kgExplanationZh",
    ]


def summarize(df: pd.DataFrame, concentration: list[dict[str, Any]], diverse: pd.DataFrame) -> dict[str, Any]:
    by_direction: dict[str, dict[str, Any]] = {}
    for direction, group in df.groupby("direction"):
        top20 = group.sort_values("finalRankWithinDirection").head(20)
        diverse_dir = diverse[diverse["direction"].eq(direction)]
        by_direction[direction] = {
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "rows": int(len(group)),
            "top20UniqueDrugs": int(top20["drug"].nunique()),
            "top20UniqueTargets": int(top20["target"].nunique()),
            "top20UniqueFamilies": int(top20["targetFamily"].nunique()),
            "top20DrugHHI": round(hhi(top20["drug"].tolist()), 4),
            "top20FamilyHHI": round(hhi(top20["targetFamily"].tolist()), 4),
            "diverseRows": int(len(diverse_dir)),
            "diverseUniqueDrugs": int(diverse_dir["drug"].nunique()),
            "diverseUniqueTargets": int(diverse_dir["target"].nunique()),
            "diverseUniqueFamilies": int(diverse_dir["targetFamily"].nunique()),
        }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "diverseRows": int(len(diverse)),
        "concentrationRows": int(len(concentration)),
        "overallTop20UniqueDrugsMean": float(pd.Series([item["top20UniqueDrugs"] for item in by_direction.values()]).mean()),
        "overallTop20UniqueFamiliesMean": float(pd.Series([item["top20UniqueFamilies"] for item in by_direction.values()]).mean()),
        "byDirection": by_direction,
        "methodNote": "Concentration audit quantifies whether final ranked candidates are dominated by a few drugs, targets, or target families. Diverse shortlist enforces caps per drug and target family for expert review breadth.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build diversity and concentration audit for final candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--per-direction", type=int, default=20)
    parser.add_argument("--max-per-drug", type=int, default=3)
    parser.add_argument("--max-per-target-family", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / args.final_table).fillna("")
    df["targetFamily"] = [target_family(gene, name) for gene, name in zip(df["target"], df.get("proteinName", ""))]
    for col in ["finalPriorityScore", "rank", "finalRankWithinDirection"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    concentration = concentration_rows(df, [20, 50, 100])
    diverse = select_diverse_shortlist(df, args.per_direction, args.max_per_drug, args.max_per_target_family)
    summary = summarize(df, concentration, diverse)
    summary["inputs"] = {"finalTable": args.final_table}
    summary["outputs"] = {
        "concentrationAudit": str((out_dir / "final_candidate_concentration_audit.csv").resolve()),
        "diverseShortlist": str((out_dir / "final_candidate_diverse_expert_shortlist.csv").resolve()),
        "summary": str((out_dir / "final_candidate_diversity_summary.json").resolve()),
    }
    write_csv(out_dir / "final_candidate_concentration_audit.csv", concentration)
    cols = [col for col in table_columns() if col in diverse.columns]
    diverse[cols].to_csv(out_dir / "final_candidate_diverse_expert_shortlist.csv", index=False)
    write_json(out_dir / "final_candidate_diversity_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
