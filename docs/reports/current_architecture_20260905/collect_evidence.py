"""Collect version-specific evidence; read predictions without fitting or selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES: dict[str, str] = {}


def track(path: str | Path) -> Path:
    p = ROOT / path
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read_json(path: str | Path):
    return json.loads(track(path).read_text())


def read_csv(path: str | Path):
    return pd.read_csv(track(path))


def metrics(frame: pd.DataFrame, score: str):
    result = {"micro_auprc": average_precision_score(frame.binary_label, frame[score])}
    for prefix, key in [("drug", "parent_standard_inchi_key"), ("target", "target_chembl_id")]:
        groups = [g for _, g in frame.groupby(key) if g.binary_label.nunique() == 2]
        result[f"{prefix}_macro_auprc"] = np.mean([
            average_precision_score(g.binary_label, g[score]) for g in groups
        ])
        result[f"{prefix}_macro_auroc"] = np.mean([
            roc_auc_score(g.binary_label, g[score]) for g in groups
        ])
    return {k: float(v) for k, v in result.items()}


def main():
    base = Path("outputs/old_drug_target_sota_v1")
    comparison = base / "public_retrained_v1/dtiam_same_data_comparison_v1"
    suites = ["biomaster_odti_v2_v21_s1_formal", "biomaster_odti_v2_v21_s2_formal",
              "biomaster_odti_v2_v21_s3_formal", "biomaster_odti_v2_s4_esm2_formal",
              "biomaster_odti_v2_s5_esm2_formal"]
    rows, checks, architecture, fold_rows = [], [], [], []
    for suite in suites:
        source = base / suite / "V2_MULTI_SEED_SUMMARY.json"
        summary = read_json(source)
        aggregate = summary["aggregates"][0]
        protocol = aggregate["protocol"]
        ours = read_csv(aggregate["prediction_path"])
        dt = read_csv(comparison / f"{protocol}_ALIGNED_PREDICTIONS_V1.csv.gz")
        assert ours.calibration_pair_id.is_unique and dt.calibration_pair_id.is_unique
        joined = ours.merge(dt, on="calibration_pair_id", suffixes=("", "_dt"), validate="one_to_one")
        assert len(joined) == len(ours) == len(dt)
        for key in ["binary_label", "target_chembl_id", "parent_standard_inchi_key"]:
            assert (joined[key] == joined[f"{key}_dt"]).all(), (protocol, key)
        table = read_csv(comparison / f"{protocol}_DTIAM_SAME_DATA_COMPARISON_METRICS_V1.csv")
        dtiam = table.loc[table.model.eq("DTIAM_OFFICIAL_REPRESENTATION_COMPAT_RETRAIN")].iloc[0]
        for model, score, saved in [("BioMaster_V2", "score_mean", aggregate["metric"]),
                                     ("DTIAM_compat_retrain", "dtiam_probability", dtiam)]:
            calculated = metrics(joined, score)
            deltas = {k: abs(v - float(saved[k])) for k, v in calculated.items()}
            # Serialized CSV precision can change ties relative to in-memory scores.
            assert max(deltas.values()) < 2e-6, (protocol, model, deltas)
            rows.append({"protocol": protocol, "model": model, "rows": len(joined),
                         "positives": int(joined.binary_label.sum()),
                         "two_class_drugs": int(saved["drug_groups_with_both_classes"]),
                         **calculated, "drug_recall_at_20": saved["drug_macro_recall_at_20"],
                         "bio_suite": suite, "source": str(source) if model == "BioMaster_V2" else str(comparison)})
            checks.append({"protocol": protocol, "model": model, "pairs_and_labels_equal": True,
                           "recomputed_metric_max_abs_error": max(deltas.values())})
            if protocol.startswith("S2"):
                for fold, subset in joined.groupby("fold"):
                    fold_rows.append({"protocol": protocol, "model": model, "fold": int(fold),
                                      "rows": len(subset), **metrics(subset, score)})
        checkpoint = sorted((ROOT / base / suite).glob("*/BEST_MODEL_V2.pt"))[0]
        ckpt = torch.load(track(checkpoint), map_location="cpu", weights_only=False)
        architecture.append({"suite": suite, "seeds": summary["seeds"],
                             "runs": summary["task_count"], "representative_checkpoint": str(checkpoint.relative_to(ROOT)),
                             "config": ckpt["config"],
                             "state_tensor_elements": sum(v.numel() for v in ckpt["model_state_dict"].values())})
    pd.DataFrame(rows).to_csv(OUT / "V2_DTIAM_S1_S5_METRICS.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "S2_WITHIN_FOLD_DIAGNOSTIC.csv", index=False)

    frozen = read_json(base / "drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json")
    pd.DataFrame(frozen["s5_test"]["metrics"]).to_csv(OUT / "FROZEN_LINEAGE_S5_METRICS.csv", index=False)
    pd.DataFrame(frozen["strict_kirhub_post_audit"]["metrics"]).to_csv(OUT / "FROZEN_KIRHUB_METRICS.csv", index=False)
    temporal = read_json("outputs/retrain_20260901/bidirectional_720x745_stage_a_rows4/STAGE_A_TEMPORAL_2024_2025_EXPANDED_SCOPES_SUMMARY_V1.json")
    pd.DataFrame([{"scope": k, "evidence_status": "retrospective_temporal_aggregation_contamination_confirmed", **v}
                  for k, v in temporal["metrics"].items()]).to_csv(OUT / "TEMPORAL_SCOPES_DIAGNOSTIC.csv", index=False)
    affinity = []
    for p in sorted((ROOT / "outputs/retrain_20260901/comprehensive_balanced_full_fit_v2").glob("*/FULL_FIT_RUN_SUMMARY_V2.json")):
        d = read_json(p)
        affinity.append({"seed": d["seed"], **d["test_metrics"]["bindingdb_exact_target_cold"]})
    pd.DataFrame(affinity).to_csv(OUT / "BINDINGDB_COLD_TARGET_METRICS.csv", index=False)
    read_csv("docs/reports/v3_results_20260905/AGGREGATE_RESULTS_V3.csv").to_csv(OUT / "V3_METRICS.csv", index=False)
    for p in ["configs/biomaster_current_contract_v1.json",
              "docs/reports/project_audit_20260905/TEMPORAL_RAW_AUDIT.json",
              "docs/reports/project_audit_20260905/SCORE_AND_STRUCTURE_PROBES.json",
              "docs/reports/v3_results_20260905/RESULTS_VERIFICATION_V3.json",
              "outputs/retrain_20260901/kirhub_expanded_functional_adapter_v1/KIRHUB_EXPANDED_FUNCTIONAL_ADAPTER_SUMMARY_V1.json",
              "outputs/affinity_first_remote_discovery_v1/drugclip_inference_v1/DRUGCLIP_PROJECT_INFERENCE_V1_SUMMARY.json",
              "outputs/biomaster_odti_pretrained_formal_audit_v1/PRETRAINED_RESIDUAL_FORMAL_PAIRED_METRICS_V1.csv",
              "outputs/biomaster_odti_local_graph_formal_v1/formal_audit_v1/LOCAL_GRAPH_PAIRED_METRICS_V1.csv"]:
        track(p)
    (OUT / "ARCHITECTURE_CHECKPOINTS.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n")
    (OUT / "VERIFICATION.json").write_text(json.dumps({"status": "PASS", "method": "Existing predictions only; no fitting, selection or new training", "checks": checks}, indent=2) + "\n")
    (OUT / "SOURCE_MANIFEST.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    print(f"PASS: {len(checks)} model/protocol metric reproductions; {len(SOURCES)} hashed sources.")
    print(pd.DataFrame(rows)[["protocol", "model", "drug_macro_auprc", "micro_auprc"]].to_string(index=False))
    print("S2 per-fold drug AUROC (descriptive; different query subsets):")
    print(pd.DataFrame(fold_rows)[["model", "fold", "drug_macro_auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
