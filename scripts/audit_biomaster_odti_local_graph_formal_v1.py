#!/usr/bin/env python3
"""Aggregate and cluster-bootstrap the formal local graph paired suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_biomaster_odti_v2 import load_local_graph_features  # noqa: E402

PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DEFAULT_RUNS = ROOT / "outputs/biomaster_odti_local_graph_formal_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(y: np.ndarray, score: np.ndarray, name: str) -> float:
    if name == "auprc":
        return float(average_precision_score(y, score))
    if name == "auroc":
        return float(roc_auc_score(y, score))
    if name == "brier":
        return float(np.square(score - y).mean())
    raise ValueError(name)


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    metric_name: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    candidate = frame["candidate_score"].to_numpy(dtype=np.float64)
    baseline = frame["baseline_score"].to_numpy(dtype=np.float64)
    clusters = frame[cluster_column].astype(str).to_numpy()
    unique = np.unique(clusters)
    positions = {value: np.flatnonzero(clusters == value) for value in unique}
    generator = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([positions[value] for value in sampled])
        sampled_y = y[index]
        if sampled_y.min() == sampled_y.max():
            continue
        differences.append(
            metric(sampled_y, candidate[index], metric_name)
            - metric(sampled_y, baseline[index], metric_name)
        )
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < iterations * 0.95:
        raise RuntimeError("too many invalid bootstrap replicates")
    observed_candidate = metric(y, candidate, metric_name)
    observed_baseline = metric(y, baseline, metric_name)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "cluster_column": cluster_column,
        "metric": metric_name,
        "rows": len(frame),
        "clusters": len(unique),
        "candidate_score": observed_candidate,
        "baseline_score": observed_baseline,
        "observed_difference": observed_candidate - observed_baseline,
        "iterations_requested": iterations,
        "iterations_valid": len(values),
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "probability_difference_gt_zero": float((values > 0).mean()),
        "ci95_excludes_zero_positive": bool(low > 0),
        "ci95_excludes_zero_negative": bool(high < 0),
    }


def collect_variant(root: Path, variant: str, protocol: str) -> pd.DataFrame:
    paths = sorted((root / variant).glob(f"{protocol}__fold_*__seed_*/TEST_PREDICTIONS_V2.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"no predictions for {variant} {protocol}")
    frames = []
    for path in paths:
        run = path.parent.name
        seed = int(run.rsplit("__seed_", 1)[1])
        fold = int(run.split("__fold_", 1)[1].split("__seed_", 1)[0])
        frame = pd.read_csv(
            path,
            usecols=[
                "calibration_pair_id", "binary_label", "v2_probability_calibrated",
                "v2_base_logit", "v2_final_logit", "v2_local_pair_gate",
                "v2_local_pair_residual_logit",
            ],
        )
        frame["seed"] = seed
        frame["fold"] = fold
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["calibration_pair_id", "seed"]).any():
        raise RuntimeError(f"duplicate pair/seed predictions for {variant} {protocol}")
    grouped = combined.groupby("calibration_pair_id", as_index=False).agg(
        binary_label=("binary_label", "first"),
        score=("v2_probability_calibrated", "mean"),
        base_logit=("v2_base_logit", "mean"),
        final_logit=("v2_final_logit", "mean"),
        local_gate=("v2_local_pair_gate", "mean"),
        local_residual=("v2_local_pair_residual_logit", "mean"),
        seed_count=("seed", "nunique"),
    )
    return grouped


def grouped_average_precision(frame: pd.DataFrame, group: str, score: str) -> float:
    values=[]
    for _, part in frame.groupby(group, sort=False):
        y=part["binary_label"].to_numpy(dtype=np.int8)
        if len(y) and y.min()!=y.max():
            values.append(average_precision_score(y,part[score].to_numpy(dtype=np.float64)))
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--pairs", default=str(PAIRS))
    parser.add_argument("--graph-store", default=str(ROOT / "outputs/biomaster_odti_local_graph_features_v1"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026081907)
    parser.add_argument("--out-dir", default=None)
    args=parser.parse_args()
    root=Path(args.runs)
    out_dir=Path(args.out_dir) if args.out_dir else root / "formal_audit_v1"
    out_dir.mkdir(parents=True,exist_ok=True)
    pair_meta=pd.read_csv(
        args.pairs,
        usecols=["calibration_pair_id","binary_label","target_homology_cluster","scaffold_group",
                 "target_chembl_id","parent_standard_inchi_key","drug_feature_index","target_feature_index"],
        low_memory=False,
    )
    store=load_local_graph_features(
        args.graph_store,
        int(pair_meta.drug_feature_index.max())+1,
        int(pair_meta.target_feature_index.max())+1,
    )
    pair_meta["local_graph_available"]=(
        np.asarray(store["ligand_available"])[pair_meta.drug_feature_index.to_numpy(dtype=np.int64)]
        & np.asarray(store["pocket_available"])[pair_meta.target_feature_index.to_numpy(dtype=np.int64)]
    )
    protocols=["S2_HOMOLOGY_COLD_TARGET","S3_STRICT_DOUBLE_COLD","S5_OLD_DRUG_ENTITY_COLD"]
    metric_rows=[]; bootstrap_rows=[]; prediction_paths={}
    for protocol_index,protocol in enumerate(protocols):
        baseline=collect_variant(root,"GLOBAL_UPGRADE",protocol).rename(columns={
            "score":"baseline_score","base_logit":"baseline_base_logit",
            "final_logit":"baseline_final_logit","local_gate":"baseline_local_gate",
            "local_residual":"baseline_local_residual","seed_count":"baseline_seed_count",
        })
        candidate=collect_variant(root,"FULL_LOCAL_WARMSTART",protocol).rename(columns={
            "score":"candidate_score","base_logit":"candidate_base_logit",
            "final_logit":"candidate_final_logit","local_gate":"candidate_local_gate",
            "local_residual":"candidate_local_residual","seed_count":"candidate_seed_count",
        })
        candidate=candidate.drop(columns=["binary_label"])
        frame=baseline.merge(candidate,on="calibration_pair_id",validate="one_to_one")
        frame=frame.merge(pair_meta.drop(columns=["binary_label"]),on="calibration_pair_id",validate="one_to_one")
        if frame["baseline_seed_count"].min()!=2 or frame["candidate_seed_count"].min()!=2:
            raise RuntimeError(f"{protocol} is not a complete two-seed aggregate")
        unavailable=~frame.local_graph_available
        # The local branch is added on top of the global/structure path, so
        # exact fallback must be audited against the local contribution itself,
        # not against ``base_logit`` (which intentionally excludes structure).
        fallback_exact=(
            np.allclose(frame.loc[unavailable,"candidate_local_gate"], 0.0, rtol=0, atol=1e-6)
            and np.allclose(frame.loc[unavailable,"candidate_local_residual"], 0.0, rtol=0, atol=1e-6)
        )
        for slice_name,part in {
            "ALL":frame,
            "LOCAL_GRAPH_AVAILABLE":frame[frame.local_graph_available],
            "LOCAL_GRAPH_MISSING":frame[~frame.local_graph_available],
        }.items():
            if len(part)==0 or part.binary_label.nunique()<2: continue
            y=part.binary_label.to_numpy(dtype=np.int8)
            row={"protocol":protocol,"slice":slice_name,"rows":len(part),
                 "prevalence":float(y.mean()),"fallback_exact_if_missing":bool(fallback_exact)}
            for score_name in ["baseline_score","candidate_score"]:
                score=part[score_name].to_numpy(dtype=np.float64)
                row[f"{score_name}_micro_auprc"]=float(average_precision_score(y,score))
                row[f"{score_name}_micro_auroc"]=float(roc_auc_score(y,score))
                row[f"{score_name}_brier"]=float(np.square(score-y).mean())
                row[f"{score_name}_target_macro_auprc"]=grouped_average_precision(part,"target_chembl_id",score_name)
                row[f"{score_name}_drug_macro_auprc"]=grouped_average_precision(part,"parent_standard_inchi_key",score_name)
            for metric_name in ["micro_auprc","micro_auroc","brier","target_macro_auprc","drug_macro_auprc"]:
                row[f"delta_{metric_name}"]=row[f"candidate_score_{metric_name}"]-row[f"baseline_score_{metric_name}"]
            row["candidate_local_gate_mean"]=float(part.candidate_local_gate.mean())
            row["candidate_local_gate_max"]=float(part.candidate_local_gate.max())
            metric_rows.append(row)
        for cluster_index,cluster in enumerate(["target_homology_cluster","scaffold_group"]):
            for metric_index,metric_name in enumerate(["auprc","auroc","brier"]):
                row=paired_cluster_bootstrap(
                    frame,cluster,metric_name,args.iterations,
                    args.seed+protocol_index*10000+cluster_index*1000+metric_index,
                )
                row["protocol"]=protocol
                bootstrap_rows.append(row)
        output=out_dir/f"{protocol}_PAIR_ALIGNED_SEED_MEAN.csv.gz"
        frame.to_csv(output,index=False,compression="gzip")
        prediction_paths[protocol]={"path":str(output),"sha256":sha256(output)}
    metrics=pd.DataFrame(metric_rows); boots=pd.DataFrame(bootstrap_rows)
    metrics_path=out_dir/"LOCAL_GRAPH_PAIRED_METRICS_V1.csv"
    boots_path=out_dir/"LOCAL_GRAPH_CLUSTER_BOOTSTRAP_V1.csv"
    metrics.to_csv(metrics_path,index=False); boots.to_csv(boots_path,index=False)
    all_rows=metrics[metrics.slice.eq("ALL")].set_index("protocol")
    checks={
        "all_three_protocols_present":set(all_rows.index)==set(protocols),
        "two_seed_pair_alignment_complete":True,
        "missing_local_graph_exact_candidate_fallback":bool(metrics.fallback_exact_if_missing.all()),
        "all_bootstraps_complete":len(boots)==18 and boots.iterations_valid.ge(args.iterations*0.95).all(),
        "all_metrics_finite":bool(np.isfinite(metrics.select_dtypes(include=[np.number])).all().all()),
        "s2_micro_auprc_positive":bool(all_rows.loc["S2_HOMOLOGY_COLD_TARGET","delta_micro_auprc"]>0),
        "s3_micro_auprc_positive":bool(all_rows.loc["S3_STRICT_DOUBLE_COLD","delta_micro_auprc"]>0),
        "s5_micro_auprc_noninferior_minus_0_005":bool(all_rows.loc["S5_OLD_DRUG_ENTITY_COLD","delta_micro_auprc"]>=-0.005),
        "no_negative_auprc_cluster_ci":bool(
            ~boots[(boots.metric=="auprc")].ci95_excludes_zero_negative.any()
        ),
    }
    promotion={
        "micro_gate":bool(checks["s2_micro_auprc_positive"] and checks["s3_micro_auprc_positive"] and checks["s5_micro_auprc_noninferior_minus_0_005"]),
        "cluster_support_gate":bool(
            boots[(boots.metric=="auprc")].groupby("protocol").ci95_excludes_zero_positive.any().all()
        ),
        "calibration_gate":bool((all_rows.delta_brier<=0).all()),
    }
    promotion["pass"] = all(promotion.values())
    summary={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS" if all(checks.values()) else "FAIL",
        "iterations":args.iterations,
        "checks":{key:bool(value) for key,value in checks.items()},
        "promotion_gate":promotion,
        "decision":"PROMOTE" if promotion["pass"] else "DO_NOT_PROMOTE_KEEP_CANDIDATE",
        "all_slice_metrics":metrics[metrics.slice.eq("ALL")].to_dict(orient="records"),
        "bootstrap_summary":boots.to_dict(orient="records"),
        "artifacts":{
            "metrics":{"path":str(metrics_path),"sha256":sha256(metrics_path)},
            "bootstrap":{"path":str(boots_path),"sha256":sha256(boots_path)},
            "predictions":prediction_paths,
        },
        "claim_status":"PAIRED_INTERNAL_TWO_SEED_EVIDENCE; SOURCE_HELDOUT_AND_W1_STILL_REQUIRED",
    }
    path=out_dir/"LOCAL_GRAPH_FORMAL_AUDIT_V1.json"
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
