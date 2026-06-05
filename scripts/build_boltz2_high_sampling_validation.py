from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd


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


def as_float(value: Any) -> float | None:
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


def median(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_result_files(run_dir: Path) -> dict[str, dict[str, Path]]:
    results: dict[str, dict[str, Path]] = {}
    if not run_dir.exists():
        return results
    for path in run_dir.rglob("confidence_*_model_0.json"):
        stem = path.name.replace("confidence_", "").replace("_model_0.json", "")
        results.setdefault(stem, {})["confidencePath"] = path
    for path in run_dir.rglob("affinity_*.json"):
        stem = path.name.replace("affinity_", "").replace(".json", "")
        results.setdefault(stem, {})["affinityPath"] = path
    for path in run_dir.rglob("*_model_0.cif"):
        stem = path.name.replace("_model_0.cif", "")
        results.setdefault(stem, {})["cifPath"] = path
    return results


def support_tier(row: dict[str, Any]) -> tuple[str, str, float]:
    confidence = as_float(row.get("boltzHighConfidenceScore"))
    ligand_iptm = as_float(row.get("boltzHighLigandIptm"))
    complex_iplddt = as_float(row.get("boltzHighComplexIplddt"))
    affinity_prob = as_float(row.get("boltzHighAffinityProbabilityBinary"))
    completed = bool(row.get("boltzHighCompleted"))
    if not completed:
        return "U_boltz_high_not_completed", "missing_high_sampling_boltz_output", 0.0

    terms = [confidence or 0.0, ligand_iptm or 0.0, complex_iplddt or 0.0, affinity_prob or 0.0]
    score = 100.0 * (0.30 * terms[0] + 0.25 * terms[1] + 0.20 * terms[2] + 0.25 * terms[3])

    if (confidence or 0.0) >= 0.50 and (ligand_iptm or 0.0) >= 0.60 and (affinity_prob or 0.0) >= 0.80:
        return "A_boltz_high_sampling_supported", "high_sampling_confirms_complex_interface_and_affinity", score
    if (confidence or 0.0) >= 0.40 and (ligand_iptm or 0.0) >= 0.45 and (affinity_prob or 0.0) >= 0.60:
        return "B_boltz_high_sampling_supported", "high_sampling_moderate_second_model_support", score
    if (affinity_prob or 0.0) >= 0.50 or (ligand_iptm or 0.0) >= 0.40 or (confidence or 0.0) >= 0.35:
        return "C_boltz_high_sampling_review", "partial_high_sampling_signal_requires_review", score
    return "D_boltz_high_sampling_low_support", "low_high_sampling_structure_or_affinity_support", score


def select_queue(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    source = pd.read_csv(root / args.source, low_memory=False).fillna("")
    source = source[source["boltzCompleted"].map(truthy)].copy()
    source = source[source["boltzLigandInputMode"].astype(str).eq("original_smiles")].copy()
    source = source[pd.to_numeric(source["boltzCompositeScore"], errors="coerce").notna()].copy()
    source["_scoreSort"] = pd.to_numeric(source["boltzCompositeScore"], errors="coerce")
    source["_rankSort"] = pd.to_numeric(source["externalQueueRank"], errors="coerce").fillna(999999)
    source = source.sort_values(["_scoreSort", "_rankSort"], ascending=[False, True])

    selected: list[pd.Series] = []
    selected_pairs: set[str] = set()
    per_drug: dict[str, int] = {}
    per_target: dict[str, int] = {}
    per_direction: dict[str, int] = {}

    def add_row(row: pd.Series, relaxed: bool = False) -> None:
        pair_id = str(row.get("pairId") or "")
        drug_id = str(row.get("drugId") or row.get("drug") or "")
        target = str(row.get("target") or "")
        direction = str(row.get("direction") or "")
        if pair_id in selected_pairs:
            return
        if not relaxed:
            if per_drug.get(drug_id, 0) >= args.max_per_drug:
                return
            if per_target.get(target, 0) >= args.max_per_target:
                return
            if per_direction.get(direction, 0) >= args.max_per_direction:
                return
        selected.append(row)
        selected_pairs.add(pair_id)
        per_drug[drug_id] = per_drug.get(drug_id, 0) + 1
        per_target[target] = per_target.get(target, 0) + 1
        per_direction[direction] = per_direction.get(direction, 0) + 1

    for _, row in source.iterrows():
        if len(selected) >= args.top_n:
            break
        add_row(row)
    if len(selected) < args.top_n:
        for _, row in source.iterrows():
            if len(selected) >= args.top_n:
                break
            add_row(row, relaxed=True)

    if not selected:
        return pd.DataFrame()
    queue = pd.DataFrame([row.to_dict() for row in selected]).reset_index(drop=True)
    queue.insert(0, "boltzHighQueueRank", range(1, len(queue) + 1))
    queue["boltzHighSelectionReason"] = (
        "top_original_ligand_boltz2_finalist_for_higher_sampling_second_model_validation"
    )
    return queue.drop(columns=[col for col in ["_scoreSort", "_rankSort"] if col in queue.columns])


def prepare(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / args.out_dir
    input_dir = out_dir / "inputs"
    shard_root = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    queue = select_queue(root, args)
    if queue.empty:
        raise ValueError("No high-sampling finalist rows were selected.")

    copied_yaml_paths: list[str] = []
    shard_paths: list[str] = []
    for _, row in queue.iterrows():
        source_yaml = Path(str(row.get("yamlPath") or ""))
        if not source_yaml.is_absolute():
            source_yaml = root / source_yaml
        if not source_yaml.exists():
            raise FileNotFoundError(f"Missing Boltz YAML input: {source_yaml}")
        dest_yaml = input_dir / str(row["yamlFile"])
        shutil.copy2(source_yaml, dest_yaml)
        copied_yaml_paths.append(str(dest_yaml))

        shard_idx = (int(row["boltzHighQueueRank"]) - 1) % args.shards
        shard_dir = shard_root / f"shard_{shard_idx}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_yaml = shard_dir / str(row["yamlFile"])
        shutil.copy2(source_yaml, shard_yaml)
        shard_paths.append(str(shard_yaml))

    queue["boltzHighYamlPath"] = copied_yaml_paths
    queue["boltzHighShardYamlPath"] = shard_paths
    queue_path = out_dir / "boltz2_high_sampling_queue.csv"
    queue.to_csv(queue_path, index=False)

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Boltz-2 high-sampling finalist complex validation input package.",
        "source": args.source,
        "selectedRows": int(len(queue)),
        "shards": int(args.shards),
        "selectionPolicy": (
            "Select completed original-ligand Boltz-2 Top50 finalists by prior Boltz composite score, "
            "with caps by drug, target, and disease direction, then run stronger sampling in a separate directory."
        ),
        "samplingPlan": {
            "model": "boltz2",
            "msaMode": "empty_single_sequence",
            "recyclingSteps": args.recycling_steps,
            "samplingSteps": args.sampling_steps,
            "diffusionSamples": args.diffusion_samples,
            "samplingStepsAffinity": args.sampling_steps_affinity,
            "diffusionSamplesAffinity": args.diffusion_samples_affinity,
            "noKernels": bool(args.no_kernels),
        },
        "selectedPairs": queue[["boltzHighQueueRank", "direction", "pairId", "drug", "target", "knownDrugTargetPair", "boltzCompositeScore", "boltzSupportTier"]].to_dict(orient="records"),
        "artifacts": {
            "queueCsv": str(queue_path),
            "inputDir": str(input_dir),
            "shardRoot": str(shard_root),
        },
    }
    write_json(out_dir / "boltz2_high_sampling_input_summary.json", summary)
    return summary


def audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / args.out_dir
    queue_path = out_dir / "boltz2_high_sampling_queue.csv"
    queue = pd.read_csv(queue_path, low_memory=False).fillna("")
    results = collect_result_files(out_dir / "runs")

    records: list[dict[str, Any]] = []
    for _, row in queue.iterrows():
        data = row.to_dict()
        stem = Path(str(data["yamlFile"])).stem
        result = results.get(stem, {})
        confidence = read_json(result.get("confidencePath"))
        affinity = read_json(result.get("affinityPath"))
        completed = bool(confidence and affinity)
        record = {
            **data,
            "boltzHighCompleted": completed,
            "boltzHighConfidencePath": str(result.get("confidencePath", "")),
            "boltzHighAffinityPath": str(result.get("affinityPath", "")),
            "boltzHighCifPath": str(result.get("cifPath", "")),
            "boltzHighConfidenceScore": confidence.get("confidence_score", ""),
            "boltzHighPtm": confidence.get("ptm", ""),
            "boltzHighIptm": confidence.get("iptm", ""),
            "boltzHighLigandIptm": confidence.get("ligand_iptm", ""),
            "boltzHighComplexPlddt": confidence.get("complex_plddt", ""),
            "boltzHighComplexIplddt": confidence.get("complex_iplddt", ""),
            "boltzHighComplexPde": confidence.get("complex_pde", ""),
            "boltzHighComplexIpde": confidence.get("complex_ipde", ""),
            "boltzHighAffinityPredValue": affinity.get("affinity_pred_value", ""),
            "boltzHighAffinityProbabilityBinary": affinity.get("affinity_probability_binary", ""),
            "boltzHighAffinityPredValue1": affinity.get("affinity_pred_value1", ""),
            "boltzHighAffinityPredValue2": affinity.get("affinity_pred_value2", ""),
            "boltzHighAffinityProbabilityBinary1": affinity.get("affinity_probability_binary1", ""),
            "boltzHighAffinityProbabilityBinary2": affinity.get("affinity_probability_binary2", ""),
            "boltzHighMsaMode": "empty_single_sequence",
            "boltzHighSamplingSteps": args.sampling_steps,
            "boltzHighAffinitySamplingSteps": args.sampling_steps_affinity,
            "boltzHighRecyclingSteps": args.recycling_steps,
            "boltzHighDiffusionSamples": args.diffusion_samples,
            "boltzHighAffinityDiffusionSamples": args.diffusion_samples_affinity,
        }
        tier, reason, score = support_tier(record)
        record["boltzHighSupportTier"] = tier
        record["boltzHighSupportReason"] = reason
        record["boltzHighCompositeScore"] = round(score, 4)
        old_score = as_float(record.get("boltzCompositeScore"))
        record["boltzHighMinusFastCompositeDelta"] = round(score - old_score, 4) if old_score is not None else ""
        records.append(record)

    df = pd.DataFrame(records).fillna("")
    completed = df[df["boltzHighCompleted"].map(truthy)].copy()
    ab_supported = int(df["boltzHighSupportTier"].astype(str).str.startswith(("A_", "B_")).sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Boltz-2 high-sampling finalist complex validation audit.",
        "candidateRows": int(len(df)),
        "completedRows": int(len(completed)),
        "completedPct": pct(int(len(completed)), int(len(df))),
        "knownRows": int(df["knownDrugTargetPair"].map(truthy).sum()) if "knownDrugTargetPair" in df else 0,
        "novelOrExtensionRows": int(len(df) - df["knownDrugTargetPair"].map(truthy).sum()) if "knownDrugTargetPair" in df else int(len(df)),
        "abSupportedRows": ab_supported,
        "abSupportedPct": pct(ab_supported, int(len(df))),
        "tierCounts": df["boltzHighSupportTier"].value_counts(dropna=False).to_dict(),
        "directionTierCounts": {
            direction: sub["boltzHighSupportTier"].value_counts(dropna=False).to_dict()
            for direction, sub in df.groupby("direction", dropna=False)
        },
        "medianHighConfidenceScore": median([as_float(v) for v in completed["boltzHighConfidenceScore"].tolist()]) if not completed.empty else None,
        "medianHighLigandIptm": median([as_float(v) for v in completed["boltzHighLigandIptm"].tolist()]) if not completed.empty else None,
        "medianHighAffinityProbabilityBinary": median([as_float(v) for v in completed["boltzHighAffinityProbabilityBinary"].tolist()]) if not completed.empty else None,
        "medianCompositeDeltaVsFast": median([as_float(v) for v in completed["boltzHighMinusFastCompositeDelta"].tolist()]) if not completed.empty else None,
        "samplingPlan": {
            "model": "boltz2",
            "msaMode": "empty_single_sequence",
            "recyclingSteps": args.recycling_steps,
            "samplingSteps": args.sampling_steps,
            "diffusionSamples": args.diffusion_samples,
            "samplingStepsAffinity": args.sampling_steps_affinity,
            "diffusionSamplesAffinity": args.diffusion_samples_affinity,
            "noKernels": bool(args.no_kernels),
        },
        "topSupportedRows": completed.sort_values("boltzHighCompositeScore", ascending=False).head(12)[
            [
                "boltzHighQueueRank",
                "direction",
                "pairId",
                "drug",
                "target",
                "knownDrugTargetPair",
                "boltzHighCompositeScore",
                "boltzHighSupportTier",
                "boltzHighConfidenceScore",
                "boltzHighLigandIptm",
                "boltzHighAffinityProbabilityBinary",
                "boltzHighMinusFastCompositeDelta",
            ]
        ].to_dict(orient="records")
        if not completed.empty
        else [],
        "methodNote": (
            "This layer reruns selected original-ligand Boltz-2 finalists with stronger sampling than the initial "
            "Top50 fast spot-check. It is an orthogonal second-model structural corroboration layer, not a DiffDock rerun "
            "and not an experimental binding assay. The `noKernels` flag records whether fused kernels were disabled "
            "for runtime compatibility."
        ),
    }

    candidate_path = out_dir / "boltz2_high_sampling_candidate_audit.csv"
    direction_path = out_dir / "boltz2_high_sampling_direction_summary.csv"
    summary_path = out_dir / "boltz2_high_sampling_summary.json"
    md_path = out_dir / "BOLTZ2_HIGH_SAMPLING_VALIDATION_AUDIT.md"
    df.to_csv(candidate_path, index=False)
    if not df.empty:
        direction = (
            df.groupby("direction", dropna=False)
            .agg(
                rows=("pairId", "count"),
                completedRows=("boltzHighCompleted", lambda s: int(s.map(truthy).sum())),
                medianBoltzHighCompositeScore=("boltzHighCompositeScore", "median"),
                medianBoltzHighConfidenceScore=("boltzHighConfidenceScore", lambda s: median([as_float(v) for v in s])),
                medianBoltzHighLigandIptm=("boltzHighLigandIptm", lambda s: median([as_float(v) for v in s])),
                medianBoltzHighAffinityProbabilityBinary=("boltzHighAffinityProbabilityBinary", lambda s: median([as_float(v) for v in s])),
            )
            .reset_index()
        )
        direction.to_csv(direction_path, index=False)
    else:
        pd.DataFrame().to_csv(direction_path, index=False)
    summary["artifacts"] = {
        "candidateAuditCsv": str(candidate_path),
        "directionSummaryCsv": str(direction_path),
        "summaryJson": str(summary_path),
        "markdown": str(md_path),
    }
    write_json(summary_path, summary)
    md_path.write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(summary: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 2) -> str:
        parsed = as_float(value)
        return "NA" if parsed is None else f"{parsed:.{digits}f}"

    lines = [
        "# Boltz-2 High-Sampling Validation Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["scope"],
        "",
        "## Headline",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Completed rows: {summary['completedRows']}/{summary['candidateRows']} ({fmt(summary['completedPct'])}%)",
        f"- A/B high-sampling supported rows: {summary['abSupportedRows']} ({fmt(summary['abSupportedPct'])}%)",
        f"- Tier counts: {summary['tierCounts']}",
        f"- Median high-sampling confidence: {fmt(summary['medianHighConfidenceScore'], 4)}",
        f"- Median high-sampling ligand iPTM: {fmt(summary['medianHighLigandIptm'], 4)}",
        f"- Median high-sampling affinity probability: {fmt(summary['medianHighAffinityProbabilityBinary'], 4)}",
        f"- Median composite delta vs fast Boltz-2: {fmt(summary['medianCompositeDeltaVsFast'], 4)}",
        "",
        "## Top Supported Rows",
        "",
        "| Rank | Direction | Pair | Drug | Target | Known | Score | Tier | Confidence | ligand iPTM | Affinity prob. | Delta vs fast |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["topSupportedRows"]:
        lines.append(
            f"| {row['boltzHighQueueRank']} | {row['direction']} | {row['pairId']} | {row['drug']} | {row['target']} | "
            f"{row['knownDrugTargetPair']} | {fmt(row['boltzHighCompositeScore'])} | {row['boltzHighSupportTier']} | "
            f"{fmt(row['boltzHighConfidenceScore'], 3)} | {fmt(row['boltzHighLigandIptm'], 3)} | "
            f"{fmt(row['boltzHighAffinityProbabilityBinary'], 3)} | {fmt(row['boltzHighMinusFastCompositeDelta'], 2)} |"
        )
    lines.extend(["", "## Method Note", "", summary["methodNote"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or audit Boltz-2 high-sampling finalist validation.")
    parser.add_argument("action", choices=["prepare", "audit"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation_candidate_audit.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/boltz2_high_sampling_validation")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--max-per-drug", type=int, default=3)
    parser.add_argument("--max-per-target", type=int, default=2)
    parser.add_argument("--max-per-direction", type=int, default=5)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=60)
    parser.add_argument("--diffusion-samples", type=int, default=2)
    parser.add_argument("--sampling-steps-affinity", type=int, default=60)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=3)
    parser.add_argument("--no-kernels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    summary = prepare(root, args) if args.action == "prepare" else audit(root, args)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
