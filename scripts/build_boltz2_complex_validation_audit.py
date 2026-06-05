from __future__ import annotations

import argparse
import json
import math
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def median(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def collect_result_files(run_dirs: list[Path]) -> dict[str, dict[str, Path]]:
    results: dict[str, dict[str, Path]] = {}
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
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
    confidence = as_float(row.get("boltzConfidenceScore"))
    ligand_iptm = as_float(row.get("boltzLigandIptm"))
    complex_iplddt = as_float(row.get("boltzComplexIplddt"))
    affinity_prob = as_float(row.get("boltzAffinityProbabilityBinary"))
    completed = bool(row.get("boltzCompleted"))
    if not completed:
        return "U_boltz_not_completed", "missing_boltz_confidence_or_affinity_output", 0.0

    terms = [confidence or 0.0, ligand_iptm or 0.0, complex_iplddt or 0.0, affinity_prob or 0.0]
    score = 100.0 * (0.30 * terms[0] + 0.25 * terms[1] + 0.20 * terms[2] + 0.25 * terms[3])

    if (confidence or 0.0) >= 0.50 and (ligand_iptm or 0.0) >= 0.60 and (affinity_prob or 0.0) >= 0.80:
        return "A_boltz_second_model_supported", "high_complex_confidence_high_interface_confidence_high_affinity_probability", score
    if (confidence or 0.0) >= 0.40 and (ligand_iptm or 0.0) >= 0.45 and (affinity_prob or 0.0) >= 0.60:
        return "B_boltz_review_supported", "moderate_complex_interface_and_affinity_support", score
    if (affinity_prob or 0.0) >= 0.50 or (ligand_iptm or 0.0) >= 0.40 or (confidence or 0.0) >= 0.35:
        return "C_boltz_partial_signal_review", "partial_second_model_signal_requires_review", score
    return "D_boltz_low_support_review", "low_second_model_structure_or_affinity_support", score


def enrich_records(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    input_manifest = pd.read_csv(root / args.input_manifest).fillna("")
    external_queue_path = root / args.external_queue
    external_queue = pd.read_csv(external_queue_path).fillna("") if external_queue_path.exists() else pd.DataFrame()
    repair_path = root / args.repair_manifest
    repair = pd.read_csv(repair_path).fillna("") if repair_path.exists() else pd.DataFrame()

    external_cols = [
        "pairId",
        "finalRankGlobal",
        "finalPriorityScore",
        "sotaContextScore",
        "sotaReadyScore",
        "sotaVinaConsensusScore",
        "structureConfidenceTier",
        "targetDruggabilityTier",
        "poseQualityTier",
        "standardPoseValidationTier",
        "confidenceSdfPath",
    ]
    if not external_queue.empty:
        external_queue = external_queue[[col for col in external_cols if col in external_queue.columns]].drop_duplicates("pairId")
        input_manifest = input_manifest.merge(external_queue, on="pairId", how="left")

    repair_cols = ["yamlFile", "originalSmiles", "repairedSmiles", "repairOk", "repairReason", "fragmentCount"]
    if not repair.empty:
        repair = repair[[col for col in repair_cols if col in repair.columns]].drop_duplicates("yamlFile")
        repair = repair.rename(columns={"yamlFile": "repairYamlFile"})
    else:
        repair = pd.DataFrame(columns=["repairYamlFile", "originalSmiles", "repairedSmiles", "repairOk", "repairReason", "fragmentCount"])

    initial_results = collect_result_files([root / args.initial_run_dir])
    repaired_results = collect_result_files([root / args.repaired_run_dir])

    records: list[dict[str, Any]] = []
    repair_by_file = {str(row["repairYamlFile"]): row.to_dict() for _, row in repair.iterrows()}
    for _, row in input_manifest.iterrows():
        data = row.to_dict()
        yaml_file = str(data["yamlFile"])
        stem = Path(yaml_file).stem
        result_source = "initial_original_ligand"
        result = initial_results.get(stem, {})
        repair_data = repair_by_file.get(yaml_file, {})
        ligand_input_mode = "original_smiles"
        boltz_ligand_smiles = data.get("canonicalSmiles", "")
        if stem in repaired_results:
            result_source = "repaired_parent_ligand"
            result = repaired_results.get(stem, {})
            ligand_input_mode = "largest_organic_parent_fragment"
            boltz_ligand_smiles = repair_data.get("repairedSmiles", "")

        confidence = read_json(result.get("confidencePath", Path(""))) if result.get("confidencePath") else {}
        affinity = read_json(result.get("affinityPath", Path(""))) if result.get("affinityPath") else {}
        completed = bool(confidence and affinity)
        record = {
            **data,
            "boltzRunSource": result_source if completed else "missing_result",
            "boltzLigandInputMode": ligand_input_mode,
            "boltzLigandSmiles": boltz_ligand_smiles,
            "boltzOriginalSmiles": repair_data.get("originalSmiles", data.get("canonicalSmiles", "")),
            "boltzRepairReason": repair_data.get("repairReason", ""),
            "boltzRepairFragmentCount": repair_data.get("fragmentCount", ""),
            "boltzCompleted": completed,
            "boltzConfidencePath": str(result.get("confidencePath", "")),
            "boltzAffinityPath": str(result.get("affinityPath", "")),
            "boltzCifPath": str(result.get("cifPath", "")),
            "boltzConfidenceScore": confidence.get("confidence_score"),
            "boltzPtm": confidence.get("ptm"),
            "boltzIptm": confidence.get("iptm"),
            "boltzLigandIptm": confidence.get("ligand_iptm"),
            "boltzComplexPlddt": confidence.get("complex_plddt"),
            "boltzComplexIplddt": confidence.get("complex_iplddt"),
            "boltzComplexPde": confidence.get("complex_pde"),
            "boltzComplexIpde": confidence.get("complex_ipde"),
            "boltzAffinityPredValue": affinity.get("affinity_pred_value"),
            "boltzAffinityProbabilityBinary": affinity.get("affinity_probability_binary"),
            "boltzAffinityPredValue1": affinity.get("affinity_pred_value1"),
            "boltzAffinityPredValue2": affinity.get("affinity_pred_value2"),
            "boltzAffinityProbabilityBinary1": affinity.get("affinity_probability_binary1"),
            "boltzAffinityProbabilityBinary2": affinity.get("affinity_probability_binary2"),
            "boltzMsaMode": "empty_single_sequence",
            "boltzSamplingSteps": 10,
            "boltzAffinitySamplingSteps": 10,
            "boltzRecyclingSteps": 1,
            "boltzDiffusionSamples": 1,
        }
        tier, reason, score = support_tier(record)
        record["boltzSupportTier"] = tier
        record["boltzSupportReason"] = reason
        record["boltzCompositeScore"] = round(score, 4)
        records.append(record)
    return pd.DataFrame(records).fillna("")


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    completed = df[df["boltzCompleted"].astype(bool)].copy()
    top = completed.sort_values("boltzCompositeScore", ascending=False).head(15)
    known_completed = int(completed["knownDrugTargetPair"].astype(str).str.lower().isin(["1", "1.0", "true"]).sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Boltz-2 Top50 second-model protein-ligand complex validation audit.",
        "candidateRows": int(len(df)),
        "completedRows": int(len(completed)),
        "completedPct": pct(int(len(completed)), int(len(df))),
        "initialOriginalLigandRows": int((df["boltzRunSource"] == "initial_original_ligand").sum()),
        "repairedParentLigandRows": int((df["boltzRunSource"] == "repaired_parent_ligand").sum()),
        "knownCompletedRows": known_completed,
        "novelOrExtensionCompletedRows": int(len(completed) - known_completed),
        "tierCounts": df["boltzSupportTier"].value_counts(dropna=False).to_dict(),
        "directionTierCounts": {
            direction: sub["boltzSupportTier"].value_counts(dropna=False).to_dict()
            for direction, sub in df.groupby("direction", dropna=False)
        },
        "medianConfidenceScore": median([as_float(v) for v in completed["boltzConfidenceScore"].tolist()]),
        "medianLigandIptm": median([as_float(v) for v in completed["boltzLigandIptm"].tolist()]),
        "medianComplexIplddt": median([as_float(v) for v in completed["boltzComplexIplddt"].tolist()]),
        "medianAffinityProbabilityBinary": median([as_float(v) for v in completed["boltzAffinityProbabilityBinary"].tolist()]),
        "abSupportedRows": int(df["boltzSupportTier"].astype(str).str.startswith(("A_", "B_")).sum()),
        "abSupportedPct": pct(int(df["boltzSupportTier"].astype(str).str.startswith(("A_", "B_")).sum()), int(len(df))),
        "topSupportedRows": top[
            [
                "externalQueueRank",
                "direction",
                "pairId",
                "drug",
                "target",
                "knownDrugTargetPair",
                "boltzCompositeScore",
                "boltzSupportTier",
                "boltzConfidenceScore",
                "boltzLigandIptm",
                "boltzAffinityProbabilityBinary",
                "boltzLigandInputMode",
            ]
        ].to_dict(orient="records"),
        "methodNote": "Boltz-2 was run locally as a fast second-model spot-check with empty MSA, one recycling step, 10 structure sampling steps, and 10 affinity sampling steps. Salt/solvate/counter-ion ligands that failed parsing were rerun as largest organic parent fragments and are explicitly flagged.",
    }
    return summary


def markdown(summary: dict[str, Any], df: pd.DataFrame) -> str:
    lines = [
        "# Boltz-2 Complex Validation Audit",
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
        f"- Completed Boltz-2 rows: {summary['completedRows']}/{summary['candidateRows']} ({summary['completedPct']:.2f}%)",
        f"- Original-ligand rows: {summary['initialOriginalLigandRows']}",
        f"- Parent-fragment repaired rows: {summary['repairedParentLigandRows']}",
        f"- A/B second-model supported rows: {summary['abSupportedRows']} ({summary['abSupportedPct']:.2f}%)",
        f"- Tier counts: {summary['tierCounts']}",
        f"- Median confidence score: {summary['medianConfidenceScore']:.4f}",
        f"- Median ligand iPTM: {summary['medianLigandIptm']:.4f}",
        f"- Median affinity probability: {summary['medianAffinityProbabilityBinary']:.4f}",
        "",
        "## Top Boltz-2 Supported Rows",
        "",
        "| Queue rank | Direction | Pair | Drug | Target | Known | Score | Tier | Confidence | ligand iPTM | Affinity probability | Ligand input |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["topSupportedRows"]:
        lines.append(
            f"| {row['externalQueueRank']} | {row['direction']} | {row['pairId']} | {row['drug']} | {row['target']} | "
            f"{row['knownDrugTargetPair']} | {float(row['boltzCompositeScore']):.2f} | {row['boltzSupportTier']} | "
            f"{float(row['boltzConfidenceScore']):.3f} | {float(row['boltzLigandIptm']):.3f} | "
            f"{float(row['boltzAffinityProbabilityBinary']):.3f} | {row['boltzLigandInputMode']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an orthogonal structure-generation and affinity-probability layer over the selected Top50 candidate set, not a rerun of DiffDock.",
            "- Rows using parent-fragment repair should be discussed as parent-ligand checks because salts, solvates, and counter-ions were removed before the repaired Boltz-2 run.",
            "- The run used empty MSA and low sampling settings for throughput; high-confidence finalists can be rerun later with fuller MSA/sampling if needed.",
            "",
            "## Method Note",
            "",
            summary["methodNote"],
            "",
        ]
    )
    return "\n".join(lines)


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = enrich_records(root, args)
    summary = summarize(df)

    candidate_path = out_dir / "boltz2_complex_validation_candidate_audit.csv"
    direction_path = out_dir / "boltz2_complex_validation_direction_summary.csv"
    summary_path = out_dir / "boltz2_complex_validation_summary.json"
    md_path = out_dir / "BOLTZ2_COMPLEX_VALIDATION_AUDIT.md"

    df.to_csv(candidate_path, index=False)
    direction = (
        df.groupby("direction", dropna=False)
        .agg(
            rows=("pairId", "count"),
            completedRows=("boltzCompleted", lambda s: int(s.astype(bool).sum())),
            medianBoltzCompositeScore=("boltzCompositeScore", "median"),
            medianBoltzConfidenceScore=("boltzConfidenceScore", lambda s: median([as_float(v) for v in s])),
            medianBoltzLigandIptm=("boltzLigandIptm", lambda s: median([as_float(v) for v in s])),
            medianBoltzAffinityProbabilityBinary=("boltzAffinityProbabilityBinary", lambda s: median([as_float(v) for v in s])),
        )
        .reset_index()
    )
    direction.to_csv(direction_path, index=False)
    summary["artifacts"] = {
        "candidateAuditCsv": str(candidate_path),
        "directionSummaryCsv": str(direction_path),
        "summaryJson": str(summary_path),
        "markdown": str(md_path),
    }
    write_json(summary_path, summary)
    md_path.write_text(markdown(summary, df), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Boltz-2 complex validation audit from local prediction outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--input-manifest", default="outputs/sota_validation/boltz2_complex_validation/boltz2_input_manifest.csv")
    parser.add_argument("--external-queue", default="outputs/sota_validation/external_sota_model_inputs/boltz_chai_top50_complex_queue.csv")
    parser.add_argument("--repair-manifest", default="outputs/sota_validation/boltz2_complex_validation/ligand_repair_failed/boltz2_ligand_repair_manifest.csv")
    parser.add_argument("--initial-run-dir", default="outputs/sota_validation/boltz2_complex_validation/runs_top50")
    parser.add_argument("--repaired-run-dir", default="outputs/sota_validation/boltz2_complex_validation/runs_repaired_failed")
    parser.add_argument("--out-dir", default="outputs/sota_validation/boltz2_complex_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(Path(args.root).resolve(), args)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
