from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFFS = [20, 50, 100, 200, 300, 500, 1000]
DEFAULT_WAVE1_SIZE = 96
DEFAULT_DIRECTION_MIN = 12
DEFAULT_GATE_MIN = {"novel_candidate": 48, "positive_control": 12, "mechanism_extension": 12}
DEFAULT_ASSAY_MIN = {
    "biochemical_kinase_or_enzyme_assay": 8,
    "electrophysiology_or_channel_function_assay": 6,
    "transporter_activity_assay": 6,
    "target_engagement_and_cell_phenotype_assay": 6,
}


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def hhi(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return round(sum((count / total) ** 2 for count in counts.values()), 6)


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    score = 0.0
    for count in counts.values():
        p = count / total
        score -= p * math.log(p)
    return round(score, 6)


def effective_n(values: list[str]) -> float:
    if not values:
        return 0.0
    return round(1.0 / hhi(values), 4) if hhi(values) else 0.0


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["validationRankGlobal"] = pd.to_numeric(ordered.get("validationRankGlobal"), errors="coerce")
    ordered["validationRankWithinDirection"] = pd.to_numeric(ordered.get("validationRankWithinDirection"), errors="coerce")
    ordered["validationScore"] = pd.to_numeric(ordered.get("validationScore"), errors="coerce").fillna(0.0)
    return ordered.sort_values(["validationRankGlobal", "pairId"], ascending=[True, True]).reset_index(drop=True)


def is_ab_tier(df: pd.DataFrame) -> pd.Series:
    return df.get("validationTier", pd.Series("", index=df.index)).astype(str).str.startswith(("A_", "B_"))


def is_interpretable(df: pd.DataFrame) -> pd.Series:
    return df.get("poseInterpretabilityTier", pd.Series("", index=df.index)).astype(str).str.startswith(("A_", "B_"))


def is_known(df: pd.DataFrame) -> pd.Series:
    return df.get("knownDrugTargetPair", pd.Series(False, index=df.index)).map(truthy)


def is_novel(df: pd.DataFrame) -> pd.Series:
    novelty = df.get("noveltyGroup", pd.Series("", index=df.index)).astype(str)
    strict = df.get("strictNovelPairFlag", pd.Series(False, index=df.index)).map(truthy)
    return strict | novelty.eq("novel_pair_or_new_target")


def col_value(row: pd.Series, column: str) -> str:
    value = clean(row.get(column, ""))
    return value if value else "NA"


def concentration_row(df: pd.DataFrame, group_type: str, group_value: str, cutoff: int | str) -> dict[str, Any]:
    rows = len(df)
    known = is_known(df)
    novel = is_novel(df)
    interpretable = is_interpretable(df)
    ab = is_ab_tier(df)
    out: dict[str, Any] = {
        "groupType": group_type,
        "groupValue": group_value,
        "cutoff": cutoff,
        "rows": rows,
        "abRows": int(ab.sum()),
        "abPct": round(pct(int(ab.sum()), rows), 4),
        "knownRows": int(known.sum()),
        "knownPct": round(pct(int(known.sum()), rows), 4),
        "novelRows": int(novel.sum()),
        "novelPct": round(pct(int(novel.sum()), rows), 4),
        "interpretableRows": int(interpretable.sum()),
        "interpretablePct": round(pct(int(interpretable.sum()), rows), 4),
    }
    for column, label in [
        ("direction", "direction"),
        ("validationGate", "gate"),
        ("assayModality", "assay"),
        ("drugId", "drug"),
        ("protein", "target"),
        ("murckoScaffold", "scaffold"),
        ("chemotypeClusterId", "cluster"),
    ]:
        if column not in df.columns or rows == 0:
            continue
        values = df[column].astype(str).fillna("").map(lambda value: value if value else "NA").tolist()
        counts = Counter(values)
        top_value, top_count = counts.most_common(1)[0]
        out[f"unique{label.title()}s"] = len(counts)
        out[f"top{label.title()}"] = top_value
        out[f"top{label.title()}Rows"] = int(top_count)
        out[f"top{label.title()}Pct"] = round(pct(top_count, rows), 4)
        out[f"{label}Hhi"] = hhi(values)
        out[f"{label}Entropy"] = entropy(values)
        out[f"{label}EffectiveN"] = effective_n(values)
    return out


def topk_concentration(df: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = sort_panel(df)
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        n = min(cutoff, len(ordered))
        rows.append(concentration_row(ordered.head(n), "global_topk", "all", n))
    rows.append(concentration_row(ordered, "global_topk", "all", len(ordered)))
    for direction, group in ordered.groupby("direction", sort=True):
        group = group.sort_values(["validationRankWithinDirection", "pairId"], ascending=[True, True]).reset_index(drop=True)
        for cutoff in [20, 50, 100, 200, len(group)]:
            n = min(cutoff, len(group))
            if n > 0:
                rows.append(concentration_row(group.head(n), "direction_topk", str(direction), n))
    return rows


def queue_frames(root: Path, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paths = {
        "full_panel": root / "outputs/sota_validation/experimental_validation/experimental_validation_panel.csv",
        "balanced_shortlist": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_balanced_shortlist.csv",
        "novel_shortlist": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_novel_shortlist.csv",
        "positive_controls": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_positive_controls.csv",
        "risk_review": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_risk_review.csv",
    }
    frames = {"sorted_full_panel": sort_panel(df)}
    for name, path in paths.items():
        if path.exists():
            frames[name] = sort_panel(pd.read_csv(path).fillna(""))
    return frames


def queue_concentration(root: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    return [concentration_row(frame, "queue", name, len(frame)) for name, frame in queue_frames(root, df).items()]


def cap_ok(row: pd.Series, selected: list[pd.Series], caps: dict[str, int]) -> bool:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for item in selected:
        for column in caps:
            counters[column][col_value(item, column)] += 1
    for column, cap in caps.items():
        if counters[column][col_value(row, column)] >= cap:
            return False
    return True


def add_candidates(
    selected: list[pd.Series],
    pool: pd.DataFrame,
    target_count: int,
    caps: dict[str, int],
    seen_pairs: set[str],
) -> None:
    for _, row in pool.iterrows():
        if len(selected) >= target_count:
            break
        pair_id = col_value(row, "pairId")
        if pair_id in seen_pairs:
            continue
        if not cap_ok(row, selected, caps):
            continue
        selected.append(row)
        seen_pairs.add(pair_id)


def select_wave1(df: pd.DataFrame, target_size: int, direction_min: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ordered = sort_panel(df)
    eligible = ordered[is_ab_tier(ordered) & is_interpretable(ordered)].copy()
    strict_caps = {"drugId": 3, "protein": 2, "murckoScaffold": 3, "chemotypeClusterId": 3}
    relaxed_caps = {"drugId": 5, "protein": 4, "murckoScaffold": 5, "chemotypeClusterId": 5}
    loose_caps = {"drugId": 8, "protein": 6, "murckoScaffold": 8, "chemotypeClusterId": 8}
    selected: list[pd.Series] = []
    seen_pairs: set[str] = set()
    selection_log: list[dict[str, Any]] = []

    for direction, group in eligible.groupby("direction", sort=True):
        before = len(selected)
        add_candidates(selected, group, len(selected) + direction_min, strict_caps, seen_pairs)
        selection_log.append(
            {
                "selectionPass": "direction_minimum_strict_caps",
                "groupType": "direction",
                "groupValue": direction,
                "requestedRows": direction_min,
                "selectedRows": len(selected) - before,
            }
        )

    for gate, minimum in DEFAULT_GATE_MIN.items():
        before = len(selected)
        current = sum(1 for row in selected if col_value(row, "validationGate") == gate)
        if current < minimum:
            pool = eligible[eligible["validationGate"].astype(str).eq(gate)]
            add_candidates(selected, pool, len(selected) + (minimum - current), strict_caps, seen_pairs)
        selection_log.append(
            {
                "selectionPass": "gate_minimum_strict_caps",
                "groupType": "validationGate",
                "groupValue": gate,
                "requestedRows": max(minimum - current, 0),
                "selectedRows": len(selected) - before,
            }
        )

    for assay, minimum in DEFAULT_ASSAY_MIN.items():
        before = len(selected)
        current = sum(1 for row in selected if col_value(row, "assayModality") == assay)
        if current < minimum:
            pool = eligible[eligible["assayModality"].astype(str).eq(assay)]
            add_candidates(selected, pool, len(selected) + (minimum - current), strict_caps, seen_pairs)
        selection_log.append(
            {
                "selectionPass": "assay_minimum_strict_caps",
                "groupType": "assayModality",
                "groupValue": assay,
                "requestedRows": max(minimum - current, 0),
                "selectedRows": len(selected) - before,
            }
        )

    before = len(selected)
    add_candidates(selected, eligible, target_size, strict_caps, seen_pairs)
    selection_log.append(
        {
            "selectionPass": "global_fill_strict_caps",
            "groupType": "global",
            "groupValue": "all",
            "requestedRows": max(target_size - before, 0),
            "selectedRows": len(selected) - before,
        }
    )

    if len(selected) < target_size:
        before = len(selected)
        add_candidates(selected, eligible, target_size, relaxed_caps, seen_pairs)
        selection_log.append(
            {
                "selectionPass": "global_fill_relaxed_caps",
                "groupType": "global",
                "groupValue": "all",
                "requestedRows": max(target_size - before, 0),
                "selectedRows": len(selected) - before,
            }
        )

    if len(selected) < target_size:
        before = len(selected)
        add_candidates(selected, ordered[is_ab_tier(ordered)], target_size, loose_caps, seen_pairs)
        selection_log.append(
            {
                "selectionPass": "ab_fallback_loose_caps",
                "groupType": "global",
                "groupValue": "all",
                "requestedRows": max(target_size - before, 0),
                "selectedRows": len(selected) - before,
            }
        )

    if len(selected) < target_size:
        before = len(selected)
        add_candidates(selected, ordered, target_size, {"drugId": 12, "protein": 10, "murckoScaffold": 12, "chemotypeClusterId": 12}, seen_pairs)
        selection_log.append(
            {
                "selectionPass": "any_candidate_fallback_loose_caps",
                "groupType": "global",
                "groupValue": "all",
                "requestedRows": max(target_size - before, 0),
                "selectedRows": len(selected) - before,
            }
        )

    wave = pd.DataFrame([row.to_dict() for row in selected]).reset_index(drop=True)
    if not wave.empty:
        wave.insert(0, "wave1Rank", range(1, len(wave) + 1))
        wave["wave1Rationale"] = wave.apply(wave1_rationale, axis=1)
    return wave, selection_log


def wave1_rationale(row: pd.Series) -> str:
    gate = clean(row.get("validationGate"))
    direction = clean(row.get("direction"))
    assay = clean(row.get("assayModality"))
    structure = clean(row.get("poseInterpretabilityTier"))
    target = clean(row.get("targetDruggabilityTier"))
    novelty = clean(row.get("noveltyGroup"))
    return (
        f"{direction} {gate} candidate with {assay}; "
        f"structure={structure}, target={target}, novelty={novelty}."
    )


def group_summary(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in df.columns:
            continue
        counts = df[column].astype(str).fillna("NA").value_counts()
        for value, count in counts.items():
            rows.append(
                {
                    "groupType": column,
                    "groupValue": value,
                    "rows": int(count),
                    "pct": round(pct(int(count), len(df)), 4),
                }
            )
    return rows


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    panel_path = root / args.panel
    panel = sort_panel(pd.read_csv(panel_path).fillna(""))
    topk_rows = topk_concentration(panel)
    queue_rows = queue_concentration(root, panel)
    wave, selection_log = select_wave1(panel, args.wave1_size, args.direction_min)
    wave_concentration = concentration_row(wave, "queue", "wave1_diverse_validation_panel", len(wave))
    wave_groups = group_summary(
        wave,
        ["direction", "validationGate", "assayModality", "validationTier", "poseInterpretabilityTier", "targetDruggabilityTier"],
    )

    full_ab_interpretable = panel[is_ab_tier(panel) & is_interpretable(panel)]
    balanced = next((row for row in queue_rows if row["groupValue"] == "balanced_shortlist"), {})
    full = next((row for row in queue_rows if row["groupValue"] == "full_panel"), {})
    top100 = next((row for row in topk_rows if row["groupType"] == "global_topk" and row["cutoff"] == 100), {})
    top300 = next((row for row in topk_rows if row["groupType"] == "global_topk" and row["cutoff"] == 300), {})

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(panel)),
        "abInterpretableEligibleRows": int(len(full_ab_interpretable)),
        "top100UniqueDrugs": top100.get("uniqueDrugs"),
        "top100UniqueTargets": top100.get("uniqueTargets"),
        "top100UniqueScaffolds": top100.get("uniqueScaffolds"),
        "top100TopDrugPct": top100.get("topDrugPct"),
        "top100TopTargetPct": top100.get("topTargetPct"),
        "top100TopScaffoldPct": top100.get("topScaffoldPct"),
        "top300UniqueDrugs": top300.get("uniqueDrugs"),
        "top300UniqueTargets": top300.get("uniqueTargets"),
        "top300UniqueScaffolds": top300.get("uniqueScaffolds"),
        "balancedRows": balanced.get("rows"),
        "balancedUniqueDrugs": balanced.get("uniqueDrugs"),
        "balancedUniqueTargets": balanced.get("uniqueTargets"),
        "balancedUniqueScaffolds": balanced.get("uniqueScaffolds"),
        "balancedTopDrugPct": balanced.get("topDrugPct"),
        "balancedTopTargetPct": balanced.get("topTargetPct"),
        "balancedTopScaffoldPct": balanced.get("topScaffoldPct"),
        "wave1Rows": int(len(wave)),
        "wave1UniqueDrugs": wave_concentration.get("uniqueDrugs"),
        "wave1UniqueTargets": wave_concentration.get("uniqueTargets"),
        "wave1UniqueScaffolds": wave_concentration.get("uniqueScaffolds"),
        "wave1TopDrugPct": wave_concentration.get("topDrugPct"),
        "wave1TopTargetPct": wave_concentration.get("topTargetPct"),
        "wave1TopScaffoldPct": wave_concentration.get("topScaffoldPct"),
        "wave1KnownRows": wave_concentration.get("knownRows"),
        "wave1NovelRows": wave_concentration.get("novelRows"),
        "wave1InterpretableRows": wave_concentration.get("interpretableRows"),
        "wave1DirectionCount": wave["direction"].nunique() if "direction" in wave else 0,
        "wave1GateCounts": dict(Counter(wave.get("validationGate", pd.Series(dtype=str)).astype(str))),
        "wave1AssayCounts": dict(Counter(wave.get("assayModality", pd.Series(dtype=str)).astype(str))),
        "methodNote": (
            "This audit measures concentration and practical validation coverage after score-based ranking. "
            "It separates ranking quality from assay-panel design by enforcing diversity across disease direction, "
            "drug, target, scaffold, mechanism gate, and assay modality."
        ),
    }

    out_dir = root / args.out_dir
    final_dir = root / "outputs/sota_validation/final_prioritization"
    write_csv(out_dir / "validation_panel_topk_diversity.csv", topk_rows)
    write_csv(out_dir / "validation_panel_queue_diversity.csv", queue_rows + [wave_concentration])
    write_csv(out_dir / "validation_panel_wave1_selection_log.csv", selection_log)
    write_csv(out_dir / "validation_panel_wave1_group_summary.csv", wave_groups)
    write_csv(out_dir / "validation_panel_wave1_diverse_panel.csv", wave.to_dict("records"))
    write_json(out_dir / "validation_panel_diversity_summary.json", summary)

    write_csv(final_dir / "final_priority_validation_panel_topk_diversity.csv", topk_rows)
    write_csv(final_dir / "final_priority_validation_panel_queue_diversity.csv", queue_rows + [wave_concentration])
    write_csv(final_dir / "final_priority_validation_panel_wave1_selection_log.csv", selection_log)
    write_csv(final_dir / "final_priority_validation_panel_wave1_group_summary.csv", wave_groups)
    write_csv(final_dir / "final_priority_validation_panel_wave1_diverse_panel.csv", wave.to_dict("records"))
    write_json(final_dir / "final_priority_validation_panel_diversity_summary.json", summary)
    (final_dir / "FINAL_PRIORITY_VALIDATION_PANEL_DIVERSITY_AUDIT.md").write_text(
        markdown(summary, selection_log, wave_groups),
        encoding="utf-8",
    )
    return {"summary": summary}


def markdown(summary: dict[str, Any], selection_log: list[dict[str, Any]], wave_groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Validation Panel Diversity and Coverage Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["methodNote"],
        "",
        "## Headline Metrics",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- A/B interpretable eligible rows: {summary['abInterpretableEligibleRows']}",
        f"- Global Top100 diversity: {summary['top100UniqueDrugs']} drugs, {summary['top100UniqueTargets']} targets, "
        f"{summary['top100UniqueScaffolds']} scaffolds; top drug/target/scaffold concentration "
        f"{summary['top100TopDrugPct']}%/{summary['top100TopTargetPct']}%/{summary['top100TopScaffoldPct']}%",
        f"- Balanced shortlist diversity: {summary['balancedUniqueDrugs']} drugs, {summary['balancedUniqueTargets']} targets, "
        f"{summary['balancedUniqueScaffolds']} scaffolds; top drug/target/scaffold concentration "
        f"{summary['balancedTopDrugPct']}%/{summary['balancedTopTargetPct']}%/{summary['balancedTopScaffoldPct']}%",
        f"- Wave-1 diverse validation panel: {summary['wave1Rows']} rows across {summary['wave1DirectionCount']} directions; "
        f"{summary['wave1UniqueDrugs']} drugs, {summary['wave1UniqueTargets']} targets, {summary['wave1UniqueScaffolds']} scaffolds",
        f"- Wave-1 concentration caps achieved: top drug {summary['wave1TopDrugPct']}%, top target "
        f"{summary['wave1TopTargetPct']}%, top scaffold {summary['wave1TopScaffoldPct']}%",
        f"- Wave-1 composition: known {summary['wave1KnownRows']}, novel {summary['wave1NovelRows']}, "
        f"interpretable {summary['wave1InterpretableRows']}; gates {summary['wave1GateCounts']}",
        "",
        "## Selection Passes",
        "",
    ]
    for row in selection_log:
        lines.append(
            f"- {row['selectionPass']} {row['groupType']}={row['groupValue']}: requested "
            f"{row['requestedRows']}, selected {row['selectedRows']}."
        )
    lines.extend(["", "## Wave-1 Group Coverage", ""])
    for row in wave_groups:
        lines.append(f"- {row['groupType']}={row['groupValue']}: {row['rows']} rows ({row['pct']}%).")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit validation-panel diversity and build wave-1 diverse validation panel.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--panel", default="outputs/sota_validation/experimental_validation/experimental_validation_panel.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/validation_panel_diversity")
    parser.add_argument("--wave1-size", type=int, default=DEFAULT_WAVE1_SIZE)
    parser.add_argument("--direction-min", type=int, default=DEFAULT_DIRECTION_MIN)
    args = parser.parse_args()
    result = build(Path(args.root).resolve(), args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
