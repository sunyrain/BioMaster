from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_LABEL = "保守单复合物 round5 补跑"
SOURCE_CODE = "conservative_round5"


def number(value: Any) -> float | None:
    if value in ("", None, "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv_maybe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def extract_log_reason(log_path: Path) -> str:
    if not log_path.exists():
        return "log_missing"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"Failed to read molecule.*?reason is the exception:\s*(.+)",
        r"Failed on \[[^\]]+\]:\s*(.+)",
        r"The test dataset did not contain .*? We are skipping this complex\.",
        r"No edges and no nodes",
    ]
    reasons: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            reason = re.sub(r"\s+", " ", match.group(1) if match.groups() else match.group(0)).strip()
            if reason and reason not in reasons:
                reasons.append(reason)
    if not reasons:
        if "rank1_confidence_sdf_missing" in text:
            return "rank1_confidence_sdf_missing"
        return "no_explicit_error_found"
    return " | ".join(reasons[:4])


def load_round5(root: Path, score_dir: Path, manifest_path: Path, job_index_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    job_index = pd.read_csv(job_index_path, dtype=str).fillna("")
    manifest = manifest.reset_index(drop=True).copy()
    manifest["round5_attempt_idx"] = range(len(manifest))

    attempt_rows: list[dict[str, Any]] = []
    attempt_idx = 0
    for _, job in job_index.sort_values("job_id", key=lambda s: pd.to_numeric(s, errors="coerce")).iterrows():
        chunk = read_csv_maybe(root / str(job["chunk_csv"]))
        score = read_csv_maybe(root / str(job["score_csv"]))
        for row_offset, chunk_row in chunk.reset_index(drop=True).iterrows():
            score_row: dict[str, Any] = {}
            if row_offset < len(score):
                score_row = score.iloc[row_offset].to_dict()
            attempt_rows.append(
                {
                    "round5_attempt_idx": attempt_idx,
                    "pair_id": chunk_row.get("complex_name", ""),
                    "diffdock_confidence": score_row.get("diffdock_confidence", ""),
                    "status": score_row.get("status", "not_scored"),
                    "error": score_row.get("error", ""),
                    "confidence_sdf_path": score_row.get("confidence_sdf_path", ""),
                    "round5_score_file": str(job["score_csv"]),
                    "round5_log_file": str(job["log_file"]),
                }
            )
            attempt_idx += 1

    scores = pd.DataFrame(attempt_rows)
    merged = manifest.merge(scores, on="round5_attempt_idx", how="left", suffixes=("", "_score"))
    pair_mismatch = merged["pairId"].astype(str).ne(merged["pair_id"].astype(str))
    if pair_mismatch.any():
        bad = merged.loc[pair_mismatch, ["round5_attempt_idx", "direction", "pairId", "pair_id"]].to_dict("records")
        raise ValueError(f"round5 manifest/chunk order mismatch: {bad[:5]}")
    merged["round5_status"] = merged["status"].where(merged["status"].astype(str).str.len() > 0, "not_scored")
    merged["round5_diffdock_confidence"] = merged["diffdock_confidence"]
    merged["round5_confidence_sdf_path"] = merged["confidence_sdf_path"]
    merged["round5_error"] = merged["error"]
    merged["round5_failure_reason"] = [
        extract_log_reason(root / path) if str(path).strip() else ""
        for path in merged["round5_log_file"].astype(str)
    ]
    return merged


def update_integrated(root: Path, integrated: pd.DataFrame, round5: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = integrated.copy()
    out["_key"] = out["direction"].astype(str) + "||" + out["pairId"].astype(str)
    round5 = round5.copy()
    round5["_key"] = round5["direction"].astype(str) + "||" + round5["pairId"].astype(str)
    attempts = round5.drop_duplicates("_key").set_index("_key").to_dict("index")

    updates: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        key = row["_key"]
        attempt = attempts.get(key)
        if not attempt:
            continue
        status = str(attempt.get("round5_status", ""))
        if status == "completed":
            confidence = number(attempt.get("round5_diffdock_confidence"))
            out.at[idx, "diffdock"] = fmt(confidence)
            out.at[idx, "status"] = "completed"
            out.at[idx, "scoreSource"] = SOURCE_CODE
            out.at[idx, "scoreSourceLabelZh"] = SOURCE_LABEL
            out.at[idx, "scoreFile"] = rel(root=root, path=root / str(attempt.get("round5_score_file", "")))
            out.at[idx, "diffdockError"] = ""
            out.at[idx, "confidenceSdfPath"] = attempt.get("round5_confidence_sdf_path", "")
            if "rank1SdfPath" in out.columns:
                out.at[idx, "rank1SdfPath"] = ""
            if "credibilityTierZh" in out.columns:
                # Keep original evidence category/score, but move it out of the structural-rerun-only bucket.
                score = number(out.at[idx, "credibilityScore"]) or 0
                if score >= 38:
                    out.at[idx, "credibilityTierZh"] = "B｜补跑结构已完成"
                else:
                    out.at[idx, "credibilityTierZh"] = "C｜补跑结构已完成"
            if "evidencePathZh" in out.columns:
                path_text = str(out.at[idx, "evidencePathZh"] or "")
                out.at[idx, "evidencePathZh"] = path_text.replace("DiffDock 缺失输出审计", "DiffDock round5 结构姿态")
            if "rationaleZh" in out.columns:
                out.at[idx, "rationaleZh"] = (
                    re.sub(
                        r"本轮 DiffDock 未产出可解析的 rank-1 confidence SDF，属于结构计算缺失输出；这不是药效否定证据，应作为受体/配体准备或参数补跑的优先审计对象。",
                        f"DiffDock round5 保守单复合物补跑已产生可审阅的 rank-1 结合姿态，confidence 为 {fmt(confidence)}；该值用于结构姿态优先审阅，不能直接等同于结合自由能或药效强度。",
                        str(out.at[idx, "rationaleZh"] or ""),
                    )
                )
            updates.append(
                {
                    "direction": row["direction"],
                    "pairId": row["pairId"],
                    "rank": row.get("rank", ""),
                    "drug": row.get("drug", ""),
                    "target": row.get("target", ""),
                    "protein": row.get("protein", ""),
                    "round5_status": status,
                    "round5_diffdock_confidence": fmt(confidence),
                    "round5_confidence_sdf_path": attempt.get("round5_confidence_sdf_path", ""),
                    "round5_score_file": attempt.get("round5_score_file", ""),
                }
            )
        else:
            out.at[idx, "diffdockError"] = "round5_" + str(attempt.get("round5_error") or "missing_output")
    return out.drop(columns=["_key"]), updates


def layer_summary(before: pd.DataFrame, round5: pd.DataFrame, updated: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if Path("outputs/sota_validation/final_diffdock_layer_summary_after_round4.csv").exists():
        rows = pd.read_csv("outputs/sota_validation/final_diffdock_layer_summary_after_round4.csv").to_dict("records")
    completed_keys = set(
        (round5[round5["round5_status"].eq("completed")]["direction"].astype(str) + "||" + round5[round5["round5_status"].eq("completed")]["pairId"].astype(str)).tolist()
    )
    rows.append(
        {
            "layer": SOURCE_CODE,
            "raw_score_rows": int(len(round5)),
            "completed_score_rows": int(round5["round5_status"].eq("completed").sum()),
            "candidate_rows_final_completed_from_layer": int(updated.assign(_key=updated["direction"].astype(str) + "||" + updated["pairId"].astype(str)).query("_key in @completed_keys").shape[0]),
        }
    )
    return rows


def build_failure_audit(round5: pd.DataFrame, updated: pd.DataFrame) -> pd.DataFrame:
    failures = round5[~round5["round5_status"].eq("completed")].copy()
    if failures.empty:
        return failures
    cols = [
        "direction",
        "rank",
        "pairId",
        "drug",
        "target",
        "protein",
        "round5_status",
        "round5_error",
        "round5_failure_reason",
        "round5_score_file",
        "round5_log_file",
    ]
    return failures[[col for col in cols if col in failures.columns]].copy()


def write_outputs(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    integrated_path = root / args.integrated
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    integrated = pd.read_csv(integrated_path, dtype=str).fillna("")
    round5 = load_round5(root, root / args.score_dir, root / args.manifest, root / args.job_index)
    updated, update_rows = update_integrated(root, integrated, round5)
    missing = updated[updated["status"].ne("completed")].copy()
    completion = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "integrated_rows": int(len(updated)),
        "completed": int(updated["status"].eq("completed").sum()),
        "missing": int(updated["status"].ne("completed").sum()),
        "completion_pct": float(updated["status"].eq("completed").sum() / len(updated) * 100) if len(updated) else 0.0,
        "previous_summary": args.previous_summary,
        "round5": {
            "queued": int(len(round5)),
            "completed": int(round5["round5_status"].eq("completed").sum()),
            "missing": int(round5["round5_status"].ne("completed").sum()),
            "statusCounts": dict(Counter(round5["round5_status"])),
            "completedByDirection": dict(Counter(round5[round5["round5_status"].eq("completed")]["direction"])),
            "missingByDirection": dict(Counter(round5[round5["round5_status"].ne("completed")]["direction"])),
        },
        "completed_by_score_source": dict(Counter(updated[updated["status"].eq("completed")]["scoreSource"])),
        "rows_by_direction": dict(Counter(updated["direction"])),
        "missing_by_direction": dict(Counter(missing["direction"])),
        "outputs": {
            "integratedCandidates": args.integrated,
            "round5UpdatedCandidates": f"{args.out_dir}/disease_direction_integrated_candidates_after_round5.csv",
            "completionSummary": f"{args.out_dir}/final_diffdock_completion_after_round5.json",
            "missingRows": f"{args.out_dir}/final_diffdock_missing_after_round5.csv",
            "failureAudit": f"{args.out_dir}/final_diffdock_round5_failure_audit.csv",
            "successAudit": f"{args.out_dir}/final_diffdock_round5_success_audit.csv",
        },
    }

    updated.to_csv(root / args.integrated, index=False)
    updated.to_csv(out_dir / "disease_direction_integrated_candidates_after_round5.csv", index=False)
    pd.DataFrame(update_rows).to_csv(out_dir / "final_diffdock_round5_success_audit.csv", index=False)
    build_failure_audit(round5, updated).to_csv(out_dir / "final_diffdock_round5_failure_audit.csv", index=False)
    missing.to_csv(out_dir / "final_diffdock_missing_after_round5.csv", index=False)
    pd.DataFrame(layer_summary(integrated, round5, updated)).to_csv(out_dir / "final_diffdock_layer_summary_after_round5.csv", index=False)
    (out_dir / "final_diffdock_completion_after_round5.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# DiffDock Round5 Conservative Rerun Audit",
        "",
        f"Generated: {completion['created_utc']}",
        "",
        "Round5 reran the stable missing DiffDock cases one complex per job with conservative inference settings.",
        "",
        f"- Previous stable missing rows: {completion['round5']['queued']}",
        f"- Round5 recovered completed structures: {completion['round5']['completed']}",
        f"- Remaining stable missing rows: {completion['round5']['missing']}",
        f"- Final structural completion: {completion['completed']} / {completion['integrated_rows']} ({completion['completion_pct']:.4f}%)",
        "",
        "Remaining failures are retained as structural-computation boundary cases rather than interpreted as biological negatives.",
    ]
    (out_dir / "FINAL_DIFFDOCK_ROUND5_CONSERVATIVE_RERUN_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge conservative round5 DiffDock rescues into disease-direction candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--integrated", default="outputs/disease_directions/disease_direction_integrated_candidates.csv")
    parser.add_argument("--score-dir", default="outputs/disease_directions/missing_output_conservative_round5/scores")
    parser.add_argument("--manifest", default="outputs/disease_directions/missing_output_conservative_round5/missing_priority_manifest.csv")
    parser.add_argument("--job-index", default="outputs/disease_directions/missing_output_conservative_round5/diffdock_missing_priority_job_index.csv")
    parser.add_argument("--previous-summary", default="outputs/sota_validation/final_diffdock_completion_after_round4.json")
    parser.add_argument("--out-dir", default="outputs/sota_validation")
    args = parser.parse_args()
    payload = write_outputs(Path(args.root).resolve(), args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
