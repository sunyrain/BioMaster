from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


AFFINITY_RE = re.compile(r"Affinity:\s*([-+]?\d+(?:\.\d+)?)\s*\(kcal/mol\)")
INTRAMOL_RE = re.compile(r"Intramolecular energy:\s*([-+]?\d+(?:\.\d+)?)")


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def fmt(value: Any, digits: int = 2) -> str:
    parsed = number(value)
    return "NA" if parsed is None else f"{parsed:.{digits}f}"


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def abs_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def is_known(row: dict[str, Any]) -> bool:
    return truthy(row.get("knownDrugTargetPair"))


def is_novel(row: dict[str, Any]) -> bool:
    novelty = str(row.get("noveltyClass") or "").lower()
    return "new_pair" in novelty or "model_priority" in novelty


def rank_column(df: pd.DataFrame) -> str:
    for column in [
        "externalQueueRank",
        "sotaVinaConsensusRankGlobal",
        "sotaStandardStructureRankGlobal",
        "sotaPoseQualityRankGlobal",
        "sotaMlAdmetRankGlobal",
        "sotaContextRankGlobal",
        "sotaReadyRankGlobal",
        "finalRankGlobal",
    ]:
        if column in df.columns:
            return column
    return df.columns[0]


def parse_smina_output(stdout: str, stderr: str) -> dict[str, Any]:
    text = "\n".join([stdout or "", stderr or ""])
    affinity = None
    intramol = None
    affinity_match = AFFINITY_RE.search(text)
    intramol_match = INTRAMOL_RE.search(text)
    if affinity_match:
        affinity = float(affinity_match.group(1))
    if intramol_match:
        intramol = float(intramol_match.group(1))
    return {
        "sminaAffinityKcalMol": round(affinity, 4) if affinity is not None else "",
        "sminaIntramolecularEnergy": round(intramol, 4) if intramol is not None else "",
    }


def safe_token(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned[:120] or "item"


def path_digest(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "stdout": (exc.stdout or "")[:500],
            "stderr": (exc.stderr or "")[:500],
            "error": "timeout",
        }
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[:500],
        "stderr": result.stderr[:500],
        "error": "" if result.returncode == 0 else "nonzero_exit",
    }


def receptor_pdbqt_from_cache(row: dict[str, Any], args: argparse.Namespace, obabel: str, out_dir: Path) -> dict[str, Any]:
    receptor_pdbqt_text = str(row.get("receptorPdbqtPath") or "").strip()
    receptor_pdbqt = Path(receptor_pdbqt_text) if receptor_pdbqt_text else None
    if receptor_pdbqt is not None and receptor_pdbqt.is_file() and receptor_pdbqt.stat().st_size > 0:
        return {"ok": True, "receptorPdbqtPath": str(receptor_pdbqt), "receptorPdbqtGenerated": False, "error": ""}

    receptor_pdb_text = str(row.get("receptorPdbPath") or "").strip()
    receptor_pdb = Path(receptor_pdb_text) if receptor_pdb_text else None
    if receptor_pdb is None or not receptor_pdb.is_file() or receptor_pdb.stat().st_size == 0:
        return {"ok": False, "receptorPdbqtPath": "", "receptorPdbqtGenerated": False, "error": "receptor_pdb_missing"}

    cache_dir = out_dir / "pdbqt_cache" / "receptors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{safe_token(receptor_pdb.stem)}_{path_digest(receptor_pdb)}.pdbqt"
    if destination.exists() and destination.stat().st_size > 0:
        return {"ok": True, "receptorPdbqtPath": str(destination), "receptorPdbqtGenerated": False, "error": ""}

    command = [obabel, str(receptor_pdb), "-O", str(destination), "-xr", "--partialcharge", "gasteiger"]
    result = run_command(command, args.conversion_timeout)
    ok = result["ok"] and destination.exists() and destination.stat().st_size > 0
    return {
        "ok": ok,
        "receptorPdbqtPath": str(destination) if ok else "",
        "receptorPdbqtGenerated": ok,
        "error": "" if ok else f"{result['error']}: {result['stderr'][:300]}",
    }


def classify_smina(row: dict[str, Any]) -> tuple[float, str, str]:
    if row.get("sminaStatus") != "ok":
        return 0.0, "D_smina_not_scored", row.get("sminaError") or "smina_not_scored"

    affinity = number(row.get("sminaAffinityKcalMol"))
    standard_tier = str(row.get("standardPoseValidationTier") or "")
    pose_supported = standard_tier.startswith(("A_", "B_"))

    if affinity is None:
        return 0.0, "D_smina_unparsed", "smina_affinity_unparsed"

    # Conservative score mapping: -4 to 0, -9 to 100.
    score = max(0.0, min(100.0, ((-affinity - 4.0) / 5.0) * 100.0))
    if pose_supported and affinity <= -7.0:
        return max(score, 85.0), "A_smina_structural_support", "standard_pose_supported_and_smina_affinity_strong"
    if pose_supported and affinity <= -5.5:
        return max(score, 70.0), "B_smina_structural_support", "standard_pose_supported_and_smina_affinity_moderate"
    if affinity <= -5.5:
        return max(score, 55.0), "C_smina_score_pose_conflict", "smina_affinity_moderate_but_standard_pose_not_supported"
    if pose_supported:
        return max(score, 45.0), "C_smina_weak_score_review", "standard_pose_supported_but_smina_affinity_weak"
    return max(score, 20.0), "D_smina_low_support_review", "weak_smina_score_or_pose_validation_conflict"


def base_row(root: Path, row: pd.Series) -> dict[str, Any]:
    ligand = abs_path(root, row.get("confidenceSdfPath"))
    receptor_pdbqt = abs_path(root, row.get("receptorPdbqtPath"))
    receptor_pdb = abs_path(root, row.get("receptorPdbPath"))
    queue_rank = (
        row.get("externalQueueRank")
        or row.get("sotaVinaConsensusRankGlobal")
        or row.get("sotaStandardStructureRankGlobal")
        or row.get("sotaPoseQualityRankGlobal")
        or row.get("sotaMlAdmetRankGlobal")
        or row.get("sotaContextRankGlobal")
        or row.get("sotaReadyRankGlobal")
        or row.get("finalRankGlobal")
        or ""
    )
    return {
        "externalQueueRank": queue_rank,
        "sotaVinaConsensusRankGlobal": row.get("sotaVinaConsensusRankGlobal", ""),
        "sotaStandardStructureRankGlobal": row.get("sotaStandardStructureRankGlobal", ""),
        "sotaPoseQualityRankGlobal": row.get("sotaPoseQualityRankGlobal", ""),
        "sotaMlAdmetRankGlobal": row.get("sotaMlAdmetRankGlobal", ""),
        "sotaContextRankGlobal": row.get("sotaContextRankGlobal", ""),
        "sotaReadyRankGlobal": row.get("sotaReadyRankGlobal", ""),
        "finalRankGlobal": row.get("finalRankGlobal", ""),
        "direction": row.get("direction", ""),
        "pairId": row.get("pairId", ""),
        "drugId": row.get("drugId", ""),
        "drug": row.get("drug", ""),
        "target": row.get("target", ""),
        "protein": row.get("protein", ""),
        "proteinName": row.get("proteinName", ""),
        "knownDrugTargetPair": row.get("knownDrugTargetPair", ""),
        "noveltyClass": row.get("noveltyClass", ""),
        "structureConfidenceTier": row.get("structureConfidenceTier", ""),
        "targetDruggabilityTier": row.get("targetDruggabilityTier", ""),
        "poseQualityTier": row.get("poseQualityTier", ""),
        "standardPoseValidationTier": row.get("standardPoseValidationTier", ""),
        "vinaScoreKcalMol": row.get("vinaScoreKcalMol", ""),
        "vinaOptimizedScoreKcalMol": row.get("vinaOptimizedScoreKcalMol", ""),
        "vinaConsensusTier": row.get("vinaConsensusTier", ""),
        "confidenceSdfPath": str(ligand) if ligand else "",
        "receptorPdbPath": str(receptor_pdb) if receptor_pdb else "",
        "receptorPdbqtPath": str(receptor_pdbqt) if receptor_pdbqt else "",
        "sminaInputReady": bool(
            ligand is not None
            and ligand.exists()
            and (
                (receptor_pdbqt is not None and receptor_pdbqt.exists())
                or (receptor_pdb is not None and receptor_pdb.exists())
            )
        ),
    }


def score_row(row: dict[str, Any], args: argparse.Namespace, smina: str, obabel: str, out_dir: Path) -> dict[str, Any]:
    if not row["sminaInputReady"]:
        merged = {**row, "sminaStatus": "not_ready", "sminaError": "ligand_sdf_or_receptor_missing"}
        score, tier, reason = classify_smina(merged)
        return {**merged, "sminaConsensusScore": round(score, 4), "sminaConsensusTier": tier, "sminaConsensusReason": reason}

    receptor = receptor_pdbqt_from_cache(row, args, obabel, out_dir)
    if not receptor["ok"]:
        merged = {
            **row,
            "sminaStatus": "conversion_failed",
            "sminaError": receptor["error"],
            "receptorPdbqtGenerated": receptor.get("receptorPdbqtGenerated", False),
        }
        score, tier, reason = classify_smina(merged)
        return {**merged, "sminaConsensusScore": round(score, 4), "sminaConsensusTier": tier, "sminaConsensusReason": reason}

    row = {**row, "receptorPdbqtPath": receptor["receptorPdbqtPath"], "receptorPdbqtGenerated": receptor["receptorPdbqtGenerated"]}

    command = [
        smina,
        "--score_only",
        "-r",
        row["receptorPdbqtPath"],
        "-l",
        row["confidenceSdfPath"],
        "--autobox_ligand",
        row["confidenceSdfPath"],
        "--autobox_add",
        str(args.autobox_add),
        "--cpu",
        str(args.cpu_per_job),
    ]
    start = time.time()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        merged = {
            **row,
            "sminaStatus": "timeout",
            "sminaError": "timeout",
            "sminaReturnCode": 124,
            "sminaRuntimeSec": round(time.time() - start, 4),
            "sminaStdoutPreview": (exc.stdout or "")[:500],
            "sminaStderrPreview": (exc.stderr or "")[:500],
        }
        score, tier, reason = classify_smina(merged)
        return {**merged, "sminaConsensusScore": round(score, 4), "sminaConsensusTier": tier, "sminaConsensusReason": reason}

    parsed = parse_smina_output(result.stdout, result.stderr)
    status = "ok" if result.returncode == 0 and parsed.get("sminaAffinityKcalMol") != "" else "failed"
    merged = {
        **row,
        **parsed,
        "sminaStatus": status,
        "sminaError": "" if status == "ok" else (result.stderr or result.stdout)[:500],
        "sminaReturnCode": result.returncode,
        "sminaRuntimeSec": round(time.time() - start, 4),
        "sminaStdoutPreview": result.stdout[:500],
        "sminaStderrPreview": result.stderr[:500],
    }
    score, tier, reason = classify_smina(merged)
    return {**merged, "sminaConsensusScore": round(score, 4), "sminaConsensusTier": tier, "sminaConsensusReason": reason}


def direction_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, group in audit_df.groupby("direction", dropna=False):
        scored = int(sum(group["sminaStatus"].astype(str) == "ok"))
        supported = int(sum(group["sminaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
        rows.append(
            {
                "direction": direction,
                "rows": int(len(group)),
                "inputReadyRows": int(sum(group["sminaInputReady"].apply(truthy))),
                "sminaScoredRows": scored,
                "sminaScoredPct": round(pct(scored, len(group)), 4),
                "sminaSupportedRows": supported,
                "sminaSupportedPct": round(pct(supported, len(group)), 4),
                "knownRows": int(sum(group.apply(lambda item: is_known(item.to_dict()), axis=1))),
                "novelRows": int(sum(group.apply(lambda item: is_novel(item.to_dict()), axis=1))),
                "medianSminaAffinityKcalMol": (
                    round(pd.to_numeric(group["sminaAffinityKcalMol"], errors="coerce").median(), 4)
                    if scored
                    else ""
                ),
                "tierCounts": json.dumps(dict(Counter(group["sminaConsensusTier"].astype(str))), ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values("direction")


def summarize(audit_df: pd.DataFrame, direction_df: pd.DataFrame, source_rows: int, args: argparse.Namespace, smina_path: str) -> dict[str, Any]:
    scored = int(sum(audit_df["sminaStatus"].astype(str) == "ok"))
    ready = int(sum(audit_df["sminaInputReady"].apply(truthy)))
    supported = int(sum(audit_df["sminaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": f"smina score-only structural rescoring over Top{len(audit_df)} external structural queue rows.",
        "source": args.source,
        "sourceRows": int(source_rows),
        "candidateRows": int(len(audit_df)),
        "sminaBinary": smina_path,
        "inputReadyRows": ready,
        "inputReadyPct": round(pct(ready, len(audit_df)), 4),
        "sminaScoredRows": scored,
        "sminaScoredPct": round(pct(scored, len(audit_df)), 4),
        "sminaSupportedRows": supported,
        "sminaSupportedPct": round(pct(supported, len(audit_df)), 4),
        "knownRows": int(sum(audit_df.apply(lambda row: is_known(row.to_dict()), axis=1))),
        "novelRows": int(sum(audit_df.apply(lambda row: is_novel(row.to_dict()), axis=1))),
        "medianSminaAffinityKcalMol": (
            round(pd.to_numeric(audit_df["sminaAffinityKcalMol"], errors="coerce").median(), 4) if scored else ""
        ),
        "tierCounts": dict(Counter(audit_df["sminaConsensusTier"].astype(str))),
        "directionRows": direction_df.to_dict(orient="records"),
        "methodNote": "smina is used as an additional score-only structural rescoring layer over the existing DiffDock poses and Vina PDBQT cache. It is not GNINA CNN scoring and should not be interpreted as an experimental affinity measurement.",
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    out_dir = str(summary.get("outDir", "outputs/sota_validation/smina_rescoring")).rstrip("/")
    lines = [
        "# smina Structural Rescoring Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["scope"],
        "",
        summary["methodNote"],
        "",
        "## Headline Results",
        "",
        f"- Input-ready rows: {summary['inputReadyRows']}/{summary['candidateRows']} ({fmt(summary['inputReadyPct'])}%).",
        f"- smina-scored rows: {summary['sminaScoredRows']}/{summary['candidateRows']} ({fmt(summary['sminaScoredPct'])}%).",
        f"- A/B smina-supported rows: {summary['sminaSupportedRows']}/{summary['candidateRows']} ({fmt(summary['sminaSupportedPct'])}%).",
        f"- Median smina affinity: {fmt(summary['medianSminaAffinityKcalMol'])} kcal/mol.",
        f"- Tier counts: {summary['tierCounts']}.",
        "",
        "## Direction Summary",
        "",
        "| Direction | Rows | Scored | A/B Supported | Median smina | Tier counts |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in direction_df.iterrows():
        lines.append(
            f"| {row['direction']} | {row['rows']} | {row['sminaScoredRows']} | "
            f"{row['sminaSupportedRows']} | {fmt(row['medianSminaAffinityKcalMol'])} | {row['tierCounts']} |"
        )
    lines.extend(
        [
            "",
            "## Top Rescored Examples",
            "",
            "| Queue rank | Direction | Pair | Drug | Target | smina | Tier | Reason |",
            "| ---: | --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for _, row in audit_df.head(20).iterrows():
        lines.append(
            f"| {row.get('externalQueueRank', '')} | {row.get('direction', '')} | {row.get('pairId', '')} | "
            f"{row.get('drug', '')} | {row.get('target', '')} | {fmt(row.get('sminaAffinityKcalMol'))} | "
            f"{row.get('sminaConsensusTier', '')} | {row.get('sminaConsensusReason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/B rows have standard pose support and favorable smina score-only affinity.",
            "- C rows are structurally reviewable but have either weak score or a conflict with the standard pose-validation layer.",
            "- D rows should not be promoted from structure alone; they require manual pose review or an independent complex-generation model.",
            "",
            "## Output Files",
            "",
            f"- Candidate audit: `{out_dir}/smina_rescoring_candidate_audit.csv`",
            f"- Direction summary: `{out_dir}/smina_rescoring_direction_summary.csv`",
            f"- Summary JSON: `{out_dir}/smina_rescoring_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    smina = shutil.which(args.smina) or args.smina
    if not Path(smina).exists() and shutil.which(smina) is None:
        raise RuntimeError(f"smina executable not found: {args.smina}")
    obabel = shutil.which(args.obabel) or args.obabel
    if not Path(obabel).exists() and shutil.which(obabel) is None:
        raise RuntimeError(f"OpenBabel executable not found: {args.obabel}")

    source_path = root / args.source
    df = pd.read_csv(source_path, low_memory=False).fillna("")
    selected_rank_column = rank_column(df)
    selected = df.sort_values(selected_rank_column).copy()
    if args.top_n > 0:
        selected = selected.head(args.top_n).copy()
    base_rows = [base_row(root, row) for _, row in selected.iterrows()]

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(score_row, row, args, smina, obabel, out_dir) for row in base_rows]
        for index, future in enumerate(as_completed(futures), start=1):
            audit_rows.append(future.result())
            if args.progress_every and index % args.progress_every == 0:
                print(json.dumps({"processed": index}, ensure_ascii=False))

    audit_sort_column = "externalQueueRank" if "externalQueueRank" in pd.DataFrame(audit_rows).columns else selected_rank_column
    audit_df = pd.DataFrame(audit_rows).sort_values(audit_sort_column).fillna("")
    direction_df = direction_summary(audit_df)
    summary = summarize(audit_df, direction_df, len(df), args, smina)
    summary["rankColumn"] = selected_rank_column
    summary["outDir"] = args.out_dir

    paths = {
        "audit": out_dir / "smina_rescoring_candidate_audit.csv",
        "direction": out_dir / "smina_rescoring_direction_summary.csv",
        "summary": out_dir / "smina_rescoring_summary.json",
        "md": out_dir / "SMINA_RESCORING_AUDIT.md",
    }
    audit_df.to_csv(paths["audit"], index=False)
    direction_df.to_csv(paths["direction"], index=False)
    write_json(paths["summary"], summary)
    paths["md"].write_text(markdown(summary, direction_df, audit_df), encoding="utf-8")
    return {"summary": summary, **{key: str(value.relative_to(root)) for key, value in paths.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smina score-only structural rescoring over the external Top100 queue.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/sota_validation/external_sota_model_inputs/gnina_top100_rescoring_queue.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/smina_rescoring")
    parser.add_argument("--smina", default="/root/autodl-tmp/conda_envs/smina/bin/smina")
    parser.add_argument("--obabel", default="/root/autodl-tmp/conda_envs/smina/bin/obabel")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-per-job", type=int, default=1)
    parser.add_argument("--autobox-add", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--conversion-timeout", type=int, default=180)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build(root, args)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
