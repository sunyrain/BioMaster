from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_vina_consensus_rescoring_audit import direction_summary, fmt, json_safe, pct, rank_column, write_json


MERGE_COLS = [
    "direction",
    "pairId",
    "vinaStatus",
    "vinaError",
    "pdbqtReady",
    "vinaScoringFunction",
    "vinaScoreKcalMol",
    "vinaIntermolecularKcalMol",
    "vinaOptimizedScoreKcalMol",
    "vinaRelaxationImprovementKcalMol",
    "vinaHeavyAtomCount",
    "vinaBoxCenterX",
    "vinaBoxCenterY",
    "vinaBoxCenterZ",
    "vinaBoxSizeX",
    "vinaBoxSizeY",
    "vinaBoxSizeZ",
    "vinaConsensusScore",
    "vinaConsensusTier",
    "vinaConsensusAction",
    "vinaConsensusReason",
    "_vinaAuditSourcePath",
]


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def round_float(value: Any, digits: int = 4) -> float | str:
    parsed = number(value)
    return "" if parsed is None else round(parsed, digits)


def audit_priority(row: pd.Series) -> tuple[int, int, float]:
    status = str(row.get("vinaStatus", ""))
    tier = str(row.get("vinaConsensusTier", ""))
    score = number(row.get("vinaConsensusScore"))
    if status == "ok":
        status_rank = 50
    elif truthy(row.get("pdbqtReady")):
        status_rank = 30
    elif status == "not_ready":
        status_rank = 20
    elif status:
        status_rank = 10
    else:
        status_rank = 0
    tier_rank = 0
    if tier.startswith("A_"):
        tier_rank = 4
    elif tier.startswith("B_"):
        tier_rank = 3
    elif tier.startswith("C_"):
        tier_rank = 2
    elif tier.startswith("D_"):
        tier_rank = 1
    return status_rank, tier_rank, score if score is not None else -1.0


def load_audits(root: Path, patterns: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches and (root / pattern).exists():
            matches = [root / pattern]
        for path in matches:
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not path.exists() or path.stat().st_size == 0:
                continue
            frame = pd.read_csv(path, low_memory=False).fillna("")
            if "direction" not in frame.columns or "pairId" not in frame.columns:
                continue
            rel = str(path.relative_to(root))
            frame["_vinaAuditSourcePath"] = rel
            frame["_vinaAuditSourceMtime"] = path.stat().st_mtime
            frames.append(frame)
            manifest.append({"path": rel, "rows": int(len(frame)), "sizeBytes": int(path.stat().st_size)})
    if not frames:
        return pd.DataFrame(), manifest
    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    combined = combined[
        combined["direction"].astype(str).str.len().gt(0) & combined["pairId"].astype(str).str.len().gt(0)
    ].copy()
    if combined.empty:
        return combined, manifest
    priorities = combined.apply(audit_priority, axis=1, result_type="expand")
    combined["_statusPriority"] = priorities[0]
    combined["_tierPriority"] = priorities[1]
    combined["_scorePriority"] = priorities[2]
    combined = combined.sort_values(
        ["direction", "pairId", "_statusPriority", "_tierPriority", "_scorePriority", "_vinaAuditSourceMtime"],
        ascending=[True, True, False, False, False, False],
    )
    deduped = combined.drop_duplicates(["direction", "pairId"], keep="first").copy()
    return deduped.drop(columns=["_statusPriority", "_tierPriority", "_scorePriority"], errors="ignore"), manifest


def build_summary(
    audit_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    final_df: pd.DataFrame,
    audit_manifest: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_keys = set(zip(final_df["direction"].astype(str), final_df["pairId"].astype(str)))
    audit_keys = set(zip(audit_df["direction"].astype(str), audit_df["pairId"].astype(str)))
    covered = len(source_keys & audit_keys)
    scored = int(sum(audit_df.get("vinaStatus", pd.Series(dtype=str)).astype(str) == "ok"))
    supported = int(sum(audit_df.get("vinaConsensusTier", pd.Series(dtype=str)).astype(str).str.startswith(("A_", "B_"))))
    ready = int(sum(audit_df.get("structureInputReady", pd.Series(dtype=bool)).astype(bool))) if "structureInputReady" in audit_df else 0
    pdbqt_ready = int(sum(audit_df.get("pdbqtReady", pd.Series(dtype=str)).apply(truthy))) if "pdbqtReady" in audit_df else 0
    tiers = Counter(audit_df.get("vinaConsensusTier", pd.Series(dtype=str)).astype(str))
    median_score = pd.to_numeric(audit_df.get("vinaScoreKcalMol"), errors="coerce").median()
    median_optimized = pd.to_numeric(audit_df.get("vinaOptimizedScoreKcalMol"), errors="coerce").median()
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.source,
        "sourceRows": int(len(final_df)),
        "candidateRows": int(len(final_df)),
        "mergedAuditRows": int(len(audit_df)),
        "coveredFinalRows": int(covered),
        "coveredFinalRowsPct": round(pct(covered, len(final_df)), 4),
        "missingFinalRows": int(len(final_df) - covered),
        "mergeStatus": "complete" if int(len(final_df) - covered) == 0 else "partial",
        "structureInputReadyRowsInMergedAudit": ready,
        "structureInputReadyRows": ready,
        "structureInputReadyPct": round(pct(ready, len(audit_df)), 4),
        "pdbqtReadyRows": pdbqt_ready,
        "pdbqtReadyPct": round(pct(pdbqt_ready, len(audit_df)), 4),
        "vinaScoredRows": scored,
        "vinaScoredPctOfMergedAudit": round(pct(scored, len(audit_df)), 4),
        "vinaScoredPct": round(pct(scored, len(audit_df)), 4),
        "vinaConsensusSupportedRows": supported,
        "vinaConsensusSupportedPctOfMergedAudit": round(pct(supported, len(audit_df)), 4),
        "vinaConsensusSupportedPct": round(pct(supported, len(audit_df)), 4),
        "medianVinaScoreKcalMol": round_float(median_score),
        "medianVinaOptimizedScoreKcalMol": round_float(median_optimized),
        "vinaConsensusTierCounts": dict(tiers),
        "directionRows": direction_df.to_dict(orient="records"),
        "auditInputs": audit_manifest,
        "methodNote": "Merged AutoDock Vina shard outputs are used as an independent structural consensus layer for DiffDock poses. Rows absent from completed shard audits retain a neutral Vina component until their shard finishes.",
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, audit_df: pd.DataFrame, out_paths: dict[str, Path], root: Path) -> str:
    lines = [
        "# Merged AutoDock Vina Consensus Rescoring Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["methodNote"],
        "",
        "## Coverage",
        "",
        f"- Source candidate rows: {summary['sourceRows']}.",
        f"- Merged audited rows: {summary['mergedAuditRows']} ({fmt(summary['coveredFinalRowsPct'])}% of source rows).",
        f"- Missing source rows not yet present in completed audits: {summary['missingFinalRows']}.",
        f"- Vina scored rows: {summary['vinaScoredRows']} ({fmt(summary['vinaScoredPctOfMergedAudit'])}% of merged audit).",
        f"- A/B Vina consensus-supported rows: {summary['vinaConsensusSupportedRows']} ({fmt(summary['vinaConsensusSupportedPctOfMergedAudit'])}% of merged audit).",
        f"- Median Vina score-only affinity: {summary['medianVinaScoreKcalMol']} kcal/mol.",
        f"- Median Vina locally optimized score: {summary['medianVinaOptimizedScoreKcalMol']} kcal/mol.",
        f"- Tier counts: {summary['vinaConsensusTierCounts']}.",
        "",
        "## Direction Summary",
        "",
        "| Direction | Rows | Scored | A/B Supported | Median Vina | Tier counts |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in direction_df.iterrows():
        lines.append(
            f"| {row['direction']} | {row['rows']} | {row['vinaScoredRows']} | "
            f"{row['vinaConsensusSupportedRows']} | {fmt(row['medianVinaScoreKcalMol'])} | {row['tierCounts']} |"
        )
    lines.extend(
        [
            "",
            "## Top Merged Examples",
            "",
            "| Direction | Pair | Drug | Target | Vina | Optimized | Tier | Source |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in audit_df.head(20).iterrows():
        lines.append(
            f"| {row.get('direction', '')} | {row.get('pairId', '')} | {row.get('drug', '')} | "
            f"{row.get('target', '')} | {fmt(row.get('vinaScoreKcalMol'))} | "
            f"{fmt(row.get('vinaOptimizedScoreKcalMol'))} | {row.get('vinaConsensusTier', '')} | "
            f"{row.get('_vinaAuditSourcePath', '')} |"
        )
    lines.extend(["", "## Output Files", ""])
    for label, path in out_paths.items():
        lines.append(f"- {label}: `{path.relative_to(root)}`")
    return "\n".join(lines) + "\n"


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_path = root / args.source
    final_df = pd.read_csv(source_path, low_memory=False).fillna("")
    audit_df, audit_manifest = load_audits(root, args.audit_glob)
    if audit_df.empty:
        raise RuntimeError("No completed Vina candidate audit CSVs were found.")

    sort_col = rank_column(final_df)
    final_df["_rankSort"] = pd.to_numeric(final_df[sort_col], errors="coerce").fillna(999999999)
    final_df = final_df.sort_values("_rankSort").drop(columns=["_rankSort"], errors="ignore").copy()
    audit_df["_sourceRankSort"] = pd.to_numeric(audit_df.get(sort_col), errors="coerce").fillna(999999999)
    audit_df = audit_df.sort_values("_sourceRankSort").drop(columns=["_sourceRankSort"], errors="ignore").reset_index(drop=True)

    direction_df = direction_summary(audit_df)
    summary = build_summary(audit_df, direction_df, final_df, audit_manifest, args)

    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "candidate audit": out_dir / "vina_consensus_candidate_audit.csv",
        "direction summary": out_dir / "vina_consensus_direction_summary.csv",
        "summary json": out_dir / "vina_consensus_summary.json",
        "audit markdown": out_dir / "VINA_CONSENSUS_RESCORING_AUDIT.md",
        "final matrix": final_dir / "final_priority_vina_consensus_matrix.csv",
        "top300 shortlist": final_dir / "final_priority_vina_consensus_top300_expert_shortlist.csv",
        "review queue": final_dir / "final_priority_vina_consensus_review_queue.csv",
        "final markdown": final_dir / "FINAL_PRIORITY_VINA_CONSENSUS_AUDIT.md",
    }

    audit_df.to_csv(out_paths["candidate audit"], index=False)
    direction_df.to_csv(out_paths["direction summary"], index=False)
    write_json(out_paths["summary json"], summary)

    merge_cols = [column for column in MERGE_COLS if column in audit_df.columns]
    augmented = final_df.merge(audit_df[merge_cols], on=["direction", "pairId"], how="left")
    base_score = pd.to_numeric(
        augmented.get("sotaStandardStructureScore", augmented.get("sotaPoseQualityScore", 0)), errors="coerce"
    ).fillna(0)
    vina_score = pd.to_numeric(augmented.get("vinaConsensusScore"), errors="coerce").fillna(40)
    augmented["sotaVinaConsensusScore"] = (0.75 * base_score + 0.25 * vina_score).round(4)
    augmented = augmented.sort_values("sotaVinaConsensusScore", ascending=False).reset_index(drop=True)
    augmented.insert(0, "sotaVinaConsensusRankGlobal", np.arange(1, len(augmented) + 1))
    augmented.to_csv(out_paths["final matrix"], index=False)

    supported = augmented[augmented["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"), na=False)].copy()
    supported.head(300).to_csv(out_paths["top300 shortlist"], index=False)
    review = augmented[~augmented["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"), na=False)].copy()
    review.head(300).to_csv(out_paths["review queue"], index=False)

    md_text = markdown(summary, direction_df, audit_df, out_paths, root)
    out_paths["audit markdown"].write_text(md_text, encoding="utf-8")
    out_paths["final markdown"].write_text(md_text, encoding="utf-8")
    return {"summary": summary, "outputs": {key: str(path.relative_to(root)) for key, path in out_paths.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge completed AutoDock Vina consensus rescoring shards.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source",
        default="outputs/sota_validation/final_prioritization/final_priority_standard_pose_validation_matrix.csv",
    )
    parser.add_argument(
        "--audit-glob",
        action="append",
        default=[
            "outputs/sota_validation/vina_consensus_rescoring_final3921/vina_consensus_candidate_audit.csv",
            "outputs/sota_validation/vina_consensus_rescoring_final3921_shards/shard_*/vina_consensus_candidate_audit.csv",
        ],
        help="Glob or path to completed Vina candidate audit CSV. Can be repeated.",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/vina_consensus_rescoring_final3921_merged")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization_vina_final3921_merged")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build(root, args)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
