#!/usr/bin/env python3
"""Finalize the refined Boltz-2 Top3000 package.

The full refined run wrote most rows in one 5-row batch queue. A small number of
rows were later rerun as single-row batches. This script merges both result
directories, keeps one best Boltz result per pair, rebuilds refined scores, and
exports audit-ready tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


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


def pct(n: int | float, d: int | float) -> float | None:
    if not d:
        return None
    return 100.0 * float(n) / float(d)


def median(values: list[Any]) -> float | None:
    nums = [as_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    return float(statistics.median(nums)) if nums else None


def collect_result_files(run_dirs: list[tuple[str, Path]]) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    for source_label, run_dir in run_dirs:
        if not run_dir.exists():
            continue
        for path in run_dir.rglob("confidence_*_model_0.json"):
            stem = path.name.replace("confidence_", "").replace("_model_0.json", "")
            records.setdefault(stem, {})["confidencePath"] = str(path)
            records[stem]["boltzRunSource"] = source_label
        for path in run_dir.rglob("affinity_*.json"):
            stem = path.name.replace("affinity_", "").replace(".json", "")
            records.setdefault(stem, {})["affinityPath"] = str(path)
            records[stem]["boltzRunSource"] = source_label
        for path in run_dir.rglob("*_model_0.cif"):
            stem = path.name.replace("_model_0.cif", "")
            records.setdefault(stem, {})["cifPath"] = str(path)
            records[stem]["boltzRunSource"] = source_label
        for path in run_dir.rglob("*_model_1.cif"):
            stem = path.name.replace("_model_1.cif", "")
            records.setdefault(stem, {})["cifModel1Path"] = str(path)
            records[stem]["boltzRunSource"] = source_label
    rows: list[dict[str, Any]] = []
    for stem, record in records.items():
        confidence = read_json(Path(record.get("confidencePath", ""))) if record.get("confidencePath") else {}
        affinity = read_json(Path(record.get("affinityPath", ""))) if record.get("affinityPath") else {}
        required_values = {
            "confidence_score": as_float(confidence.get("confidence_score")),
            "ligand_iptm": as_float(confidence.get("ligand_iptm")),
            "complex_iplddt": as_float(confidence.get("complex_iplddt")),
            "affinity_probability_binary": as_float(affinity.get("affinity_probability_binary")),
        }
        missing_paths = [
            label
            for label, key in [
                ("confidence_model0", "confidencePath"),
                ("affinity", "affinityPath"),
                ("cif_model0", "cifPath"),
                ("cif_model1", "cifModel1Path"),
            ]
            if not clean(record.get(key)) or not Path(str(record.get(key))).is_file()
        ]
        invalid_values = [
            key for key, value in required_values.items() if value is None or not 0.0 <= value <= 1.0
        ]
        completed = not missing_paths and not invalid_values
        integrity_reason = ";".join(
            [
                *(f"missing:{label}" for label in missing_paths),
                *(f"invalid:{label}" for label in invalid_values),
            ]
        )
        tier, reason, composite = support_tier(
            confidence.get("confidence_score"),
            confidence.get("ligand_iptm"),
            confidence.get("complex_iplddt"),
            affinity.get("affinity_probability_binary"),
            completed,
        )
        rows.append(
            {
                "boltz_stem": stem,
                "boltz_run_source": record.get("boltzRunSource", ""),
                "boltz_completed_refined": completed,
                "boltz_confidence_path_refined": record.get("confidencePath", ""),
                "boltz_affinity_path_refined": record.get("affinityPath", ""),
                "boltz_cif_path_refined": record.get("cifPath", ""),
                "boltz_cif_model1_path_refined": record.get("cifModel1Path", ""),
                "boltz_output_integrity_reason": integrity_reason,
                "boltz_confidence_sha256_refined": file_sha256(
                    Path(record["confidencePath"]) if record.get("confidencePath") else None
                ),
                "boltz_affinity_sha256_refined": file_sha256(
                    Path(record["affinityPath"]) if record.get("affinityPath") else None
                ),
                "boltz_cif_model0_sha256_refined": file_sha256(
                    Path(record["cifPath"]) if record.get("cifPath") else None
                ),
                "boltz_cif_model1_sha256_refined": file_sha256(
                    Path(record["cifModel1Path"]) if record.get("cifModel1Path") else None
                ),
                "boltz_confidence_score_refined": confidence.get("confidence_score"),
                "boltz_ptm_refined": confidence.get("ptm"),
                "boltz_iptm_refined": confidence.get("iptm"),
                "boltz_ligand_iptm_refined": confidence.get("ligand_iptm"),
                "boltz_complex_plddt_refined": confidence.get("complex_plddt"),
                "boltz_complex_iplddt_refined": confidence.get("complex_iplddt"),
                "boltz_affinity_pred_value_refined": affinity.get("affinity_pred_value"),
                "boltz_affinity_probability_refined": affinity.get("affinity_probability_binary"),
                "boltz_support_tier_refined": tier,
                "boltz_support_reason_refined": reason,
                "boltz_composite_score_refined": round(composite, 4),
            }
        )
    return pd.DataFrame(rows)


def collect_skipped_inputs(log_dirs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"Failed to process .*?/([^/\s]+?\.yaml)\. Skipping\. Error: (.*?)(?:\n|$)")
    for source_label, log_dir in log_dirs:
        if not log_dir.exists():
            continue
        for path in sorted(log_dir.glob("batch_*.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for match in pattern.finditer(text):
                yaml_name = match.group(1)
                rows.append(
                    {
                        "boltz_stem": Path(yaml_name).stem,
                        "boltz_skip_source": source_label,
                        "boltz_skip_log_path": str(path),
                        "boltz_skip_reason": match.group(2).strip(),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["boltz_stem", "boltz_skip_source", "boltz_skip_log_path", "boltz_skip_reason"])
    return pd.DataFrame(rows).drop_duplicates("boltz_stem", keep="last")


def support_tier(
    confidence: Any,
    ligand_iptm: Any,
    complex_iplddt: Any,
    affinity_prob: Any,
    completed: bool,
) -> tuple[str, str, float]:
    confidence_f = as_float(confidence) or 0.0
    ligand_iptm_f = as_float(ligand_iptm) or 0.0
    complex_iplddt_f = as_float(complex_iplddt) or 0.0
    affinity_prob_f = as_float(affinity_prob) or 0.0
    if not completed:
        return "U_boltz_not_completed", "missing_boltz_confidence_or_affinity_output", 0.0
    score = 100.0 * (
        0.30 * confidence_f
        + 0.25 * ligand_iptm_f
        + 0.20 * complex_iplddt_f
        + 0.25 * affinity_prob_f
    )
    if confidence_f >= 0.50 and ligand_iptm_f >= 0.60 and affinity_prob_f >= 0.80:
        return "A_boltz_second_model_supported", "high_complex_confidence_high_interface_confidence_high_affinity_probability", score
    if confidence_f >= 0.40 and ligand_iptm_f >= 0.45 and affinity_prob_f >= 0.60:
        return "B_boltz_review_supported", "moderate_complex_interface_and_affinity_support", score
    if affinity_prob_f >= 0.50 or ligand_iptm_f >= 0.40 or confidence_f >= 0.35:
        return "C_boltz_partial_signal_review", "partial_second_model_signal_requires_review", score
    return "D_boltz_low_support_review", "low_second_model_structure_or_affinity_support", score


def refined_boltz_second_model_score(row: pd.Series) -> float:
    tier = clean(row.get("boltz_support_tier_refined"))
    score = {
        "A_boltz_second_model_supported": 6.0,
        "B_boltz_review_supported": 5.0,
        "C_boltz_partial_signal_review": 2.0,
        "D_boltz_low_support_review": 0.0,
        "U_boltz_not_completed": 0.0,
    }.get(tier, 0.0)
    prob = as_float(row.get("boltz_affinity_probability_refined"))
    conf = as_float(row.get("boltz_confidence_score_refined"))
    iptm = as_float(row.get("boltz_ligand_iptm_refined"))
    if prob is not None and prob >= 0.80:
        score += 1.0
    if conf is not None and conf >= 0.45:
        score += 0.5
    if iptm is not None and iptm >= 0.65:
        score += 0.5
    return max(0.0, min(8.0, score))


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin({"1", "1.0", "true", "yes"})


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tier = out["boltz_support_tier_refined"].astype(str)
    out["refined_boltz_ab"] = tier.str.startswith(("A_", "B_"), na=False)
    out["refined_boltz_a"] = tier.str.startswith("A_", na=False)
    out["refined_boltz_c"] = tier.str.startswith("C_", na=False)
    out["refined_boltz_u"] = tier.str.startswith("U_", na=False)
    out["refined_boltz_second_model_score"] = out.apply(refined_boltz_second_model_score, axis=1)
    old_score = pd.to_numeric(out.get("boltz_second_model_score", 0), errors="coerce").fillna(0)
    enhanced = pd.to_numeric(out.get("enhanced_selection_score", 0), errors="coerce").fillna(0)
    out["refined_enhanced_selection_score"] = enhanced - old_score + out["refined_boltz_second_model_score"]
    out["refined_evidence_class"] = "physics_only_or_boltz_uncompleted"
    out.loc[out["refined_boltz_c"], "refined_evidence_class"] = "physics_plus_refined_boltz_partial"
    out.loc[out["refined_boltz_ab"], "refined_evidence_class"] = "physics_plus_refined_boltz_AB"
    out["rediscovery_or_control_risk"] = False
    out["risk_notes"] = ""
    risk_parts: list[pd.Series] = []
    if "same_family_or_label_risk" in out.columns:
        risk_parts.append(bool_series(out, "same_family_or_label_risk").map(lambda x: "same_family_or_label" if x else ""))
    if "target" in out.columns:
        target = out["gene_names"].fillna("").astype(str)
    else:
        target = out.get("primary_gene", pd.Series("", index=out.index)).fillna("").astype(str)
    ca_mask = target.str.contains(r"\bCA(?:1|2|3|4|5A|5B|6|7|8|9|10|11|12|13|14)\b", regex=True)
    ion_mask = out.get("target_assay_family", pd.Series("", index=out.index)).astype(str).eq("ion_channel")
    kinase_family_mask = out.get("target_assay_family", pd.Series("", index=out.index)).astype(str).eq("kinase") & out.get(
        "fda_original_target_family", pd.Series("", index=out.index)
    ).astype(str).eq("kinase")
    risk_parts.extend(
        [
            ca_mask.map(lambda x: "carbonic_anhydrase_rediscovery_risk" if x else ""),
            ion_mask.map(lambda x: "ion_channel_harder_target_engagement" if x else ""),
            kinase_family_mask.map(lambda x: "kinase_to_kinase_family_extension" if x else ""),
        ]
    )
    if risk_parts:
        notes = []
        for idx in out.index:
            row_notes = [part.loc[idx] for part in risk_parts if clean(part.loc[idx])]
            notes.append(";".join(row_notes))
        out["risk_notes"] = notes
        out["rediscovery_or_control_risk"] = out["risk_notes"].astype(str).str.len() > 0
    out["manual_review_priority"] = "standard_review"
    out.loc[out["rediscovery_or_control_risk"], "manual_review_priority"] = "control_or_risk_review"
    out.loc[
        out["refined_boltz_ab"] & ~out["rediscovery_or_control_risk"],
        "manual_review_priority",
    ] = "high_priority_discovery_review"
    out.loc[out["refined_boltz_u"], "manual_review_priority"] = "incomplete_boltz_review"
    return out


def greedy_select(
    df: pd.DataFrame,
    n: int,
    score_column: str,
    caps: dict[str, int],
    family_minimums: dict[str, int],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    data = df.sort_values([score_column, "physics_first_pass_score"], ascending=False).copy()
    selected: list[int] = []
    counts: dict[str, dict[str, int]] = {"drug": {}, "target": {}, "scaffold": {}, "family": {}, "queue": {}}

    def can_add(row: pd.Series, enforce_minimums: bool = True) -> bool:
        drug = clean(row.get("drug_chembl_id")) or clean(row.get("drug_names"))
        target = clean(row.get("primary_gene")) or clean(row.get("gene_names"))
        scaffold = clean(row.get("murcko_scaffold")) or clean(row.get("canonical_smiles"))
        family = clean(row.get("target_assay_family")) or "unknown"
        queue = clean(row.get("discovery_queue_class"))
        if counts["drug"].get(drug, 0) >= caps.get("drug", 10**9):
            return False
        if counts["target"].get(target, 0) >= caps.get("target", 10**9):
            return False
        if counts["scaffold"].get(scaffold, 0) >= caps.get("scaffold", 10**9):
            return False
        if counts["family"].get(family, 0) >= caps.get("family", 10**9):
            return False
        qcap = caps.get(f"queue_class:{queue}")
        if qcap is not None and counts["queue"].get(queue, 0) >= qcap:
            return False
        if enforce_minimums:
            remaining_slots = n - len(selected)
            unmet = sum(max(0, family_minimums.get(fam, 0) - counts["family"].get(fam, 0)) for fam in family_minimums)
            fam_need = max(0, family_minimums.get(family, 0) - counts["family"].get(family, 0))
            if unmet >= remaining_slots and fam_need <= 0:
                return False
        return True

    def add(row_idx: int, row: pd.Series) -> None:
        selected.append(row_idx)
        drug = clean(row.get("drug_chembl_id")) or clean(row.get("drug_names"))
        target = clean(row.get("primary_gene")) or clean(row.get("gene_names"))
        scaffold = clean(row.get("murcko_scaffold")) or clean(row.get("canonical_smiles"))
        family = clean(row.get("target_assay_family")) or "unknown"
        queue = clean(row.get("discovery_queue_class"))
        for bucket, key in [("drug", drug), ("target", target), ("scaffold", scaffold), ("family", family), ("queue", queue)]:
            counts[bucket][key] = counts[bucket].get(key, 0) + 1

    for idx, row in data.iterrows():
        if len(selected) >= n:
            break
        if can_add(row, enforce_minimums=True):
            add(idx, row)
    if len(selected) < n:
        for idx, row in data.iterrows():
            if len(selected) >= n:
                break
            if idx in selected:
                continue
            if can_add(row, enforce_minimums=False):
                add(idx, row)
    out = df.loc[selected].copy()
    out["refined_selection_rank"] = range(1, len(out) + 1)
    return out


def summarize(df: pd.DataFrame, recommended: pd.DataFrame) -> dict[str, Any]:
    completed = df[df["boltz_completed_refined"].astype(bool)].copy()
    ab = completed[completed["refined_boltz_ab"]].copy()
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": int(len(df)),
        "completed_rows": int(len(completed)),
        "completed_pct": pct(len(completed), len(df)),
        "incomplete_rows": int(len(df) - len(completed)),
        "tier_counts": df["boltz_support_tier_refined"].value_counts(dropna=False).to_dict(),
        "ab_rows": int(len(ab)),
        "ab_pct_all": pct(len(ab), len(df)),
        "ab_pct_completed": pct(len(ab), len(completed)),
        "unique_drugs": int(df["drug_chembl_id"].nunique()),
        "unique_targets": int(df["primary_gene"].nunique() if "primary_gene" in df.columns else df["gene_names"].nunique()),
        "ab_unique_drugs": int(ab["drug_chembl_id"].nunique()),
        "ab_unique_targets": int(ab["primary_gene"].nunique() if "primary_gene" in ab.columns else ab["gene_names"].nunique()),
        "median_confidence": median(completed["boltz_confidence_score_refined"].tolist()),
        "median_ligand_iptm": median(completed["boltz_ligand_iptm_refined"].tolist()),
        "median_complex_iplddt": median(completed["boltz_complex_iplddt_refined"].tolist()),
        "median_affinity_probability": median(completed["boltz_affinity_probability_refined"].tolist()),
        "source_counts": df["boltz_run_source"].value_counts(dropna=False).to_dict(),
        "manual_review_priority_counts": df["manual_review_priority"].value_counts(dropna=False).to_dict(),
        "target_assay_family_counts": df["target_assay_family"].value_counts(dropna=False).to_dict(),
        "ab_target_assay_family_counts": ab["target_assay_family"].value_counts(dropna=False).to_dict(),
        "recommended_rows": int(len(recommended)),
        "recommended_ab_rows": int(recommended["refined_boltz_ab"].sum()) if not recommended.empty else 0,
        "recommended_unique_drugs": int(recommended["drug_chembl_id"].nunique()) if not recommended.empty else 0,
        "recommended_unique_targets": int(recommended["primary_gene"].nunique()) if not recommended.empty and "primary_gene" in recommended.columns else 0,
    }
    return summary


def group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            rows=("pair_id", "count"),
            completed=("boltz_completed_refined", lambda s: int(s.astype(bool).sum())),
            AB=("refined_boltz_ab", lambda s: int(s.astype(bool).sum())),
            A=("refined_boltz_a", lambda s: int(s.astype(bool).sum())),
            median_refined_score=("boltz_composite_score_refined", "median"),
            median_affinity_probability=("boltz_affinity_probability_refined", "median"),
        )
        .reset_index()
    )
    out["AB_pct"] = (100.0 * out["AB"] / out["rows"]).round(2)
    return out.sort_values(["AB", "A", "median_refined_score"], ascending=False)


def top_join(group: pd.DataFrame, name_col: str, score_col: str, n: int = 3) -> str:
    if group.empty or name_col not in group.columns or score_col not in group.columns:
        return ""
    part = group.sort_values(score_col, ascending=False).head(n)
    rows = []
    for _, row in part.iterrows():
        name = clean(row.get(name_col))
        score = as_float(row.get(score_col))
        if name and score is not None:
            rows.append(f"{name}({score:.3f})")
    return "; ".join(rows)


def join_unique(values: pd.Series, limit: int = 5) -> str:
    seen: list[str] = []
    for value in values:
        text = clean(value)
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return ";".join(seen)


def add_disease_auxiliary(df: pd.DataFrame, ot_path: Path, txgnn_path: Path) -> pd.DataFrame:
    out = df.copy()
    drop_cols = [
        "ot_top_diseases",
        "ot_max_disease_score",
        "ot_top_disease_ids",
        "ot_top_known_drug_score",
        "ot_top_genetic_score",
        "txgnn_top_diseases",
        "txgnn_max_score",
        "disease_evidence_role",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    if ot_path.exists():
        ot = pd.read_csv(ot_path, low_memory=False).fillna("")
        if {"approved_symbol", "overall_score", "disease_name"}.issubset(ot.columns):
            ot["overall_score"] = pd.to_numeric(ot["overall_score"], errors="coerce").fillna(0)
            rows = []
            for gene, group in ot.groupby("approved_symbol"):
                top = group.sort_values("overall_score", ascending=False).head(3)
                rows.append(
                    {
                        "primary_gene": gene,
                        "ot_top_diseases": top_join(top, "disease_name", "overall_score", 3),
                        "ot_max_disease_score": float(top["overall_score"].max()) if len(top) else 0.0,
                        "ot_top_disease_ids": join_unique(top.get("disease_id", pd.Series(dtype=str))),
                        "ot_top_known_drug_score": float(pd.to_numeric(top.get("known_drug_score", 0), errors="coerce").max()) if len(top) else 0.0,
                        "ot_top_genetic_score": float(pd.to_numeric(top.get("genetic_association_score", 0), errors="coerce").max()) if len(top) else 0.0,
                    }
                )
            out = out.merge(pd.DataFrame(rows), on="primary_gene", how="left")
    if txgnn_path.exists():
        tx = pd.read_csv(txgnn_path, low_memory=False).fillna("")
        if {"drug_id", "txgnn_indication_score", "disease_name"}.issubset(tx.columns):
            tx["txgnn_indication_score"] = pd.to_numeric(tx["txgnn_indication_score"], errors="coerce").fillna(0)
            rows = []
            for drug, group in tx.groupby("drug_id"):
                top = group.sort_values("txgnn_indication_score", ascending=False).head(3)
                rows.append(
                    {
                        "drug_chembl_id": drug,
                        "txgnn_top_diseases": top_join(top, "disease_name", "txgnn_indication_score", 3),
                        "txgnn_max_score": float(top["txgnn_indication_score"].max()) if len(top) else 0.0,
                    }
                )
            out = out.merge(pd.DataFrame(rows), on="drug_chembl_id", how="left")
    for col in ["ot_top_diseases", "ot_top_disease_ids", "txgnn_top_diseases"]:
        if col in out.columns:
            out[col] = out[col].fillna("")
    for col in ["ot_max_disease_score", "ot_top_known_drug_score", "ot_top_genetic_score", "txgnn_max_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["disease_evidence_role"] = "auxiliary_annotation_only_not_used_for_main_ranking"
    return out


def readable_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "refined_selection_rank": "推荐顺序",
        "manual_review_priority": "审计优先级",
        "refined_evidence_class": "精修证据分层",
        "drug_names": "药物",
        "drug_chembl_id": "药物ChEMBL",
        "gene_names": "候选靶点",
        "protein_names": "靶点蛋白",
        "target_assay_family": "实验类型",
        "anchor_availability_tier": "靶点锚点分层",
        "anchor_project_assay_family": "OT实验家族",
        "conplex_score": "ConPLEx分数",
        "rank_within_drug": "药物内排名",
        "target_rank": "靶点内排名",
        "structure_consensus_tier": "结构口袋分层",
        "p2rank_pocketability_tier": "P2Rank等级",
        "puresnet_tier": "PUResNet等级",
        "p2rank_puresnet_overlap_fraction": "口袋重叠比例",
        "boltz_support_tier_refined": "Boltz精修等级",
        "boltz_composite_score_refined": "Boltz精修复合分",
        "boltz_affinity_probability_refined": "Boltz结合概率",
        "boltz_confidence_score_refined": "Boltz复合体置信度",
        "boltz_ligand_iptm_refined": "Boltz配体界面iPTM",
        "refined_boltz_second_model_score": "Boltz证据分",
        "refined_enhanced_selection_score": "精修增强选择分",
        "fda_therapeutic_area": "原FDA治疗领域",
        "fda_indication": "原FDA适应症",
        "fda_moa": "原FDA MoA",
        "fda_action_type": "原FDA作用类型",
        "fda_target_names": "原FDA靶点",
        "same_family_or_label_risk": "同家族/标签风险",
        "risk_notes": "审计风险备注",
        "canonical_smiles": "SMILES",
        "murcko_scaffold": "Murcko骨架",
        "ot_top_diseases": "OT靶点相关疾病Top",
        "txgnn_top_diseases": "TxGNN药物相关疾病Top",
    }
    available = [c for c in cols if c in df.columns]
    out = df[available].rename(columns=cols).copy()
    for col in ["ConPLEx分数", "Boltz精修复合分", "Boltz结合概率", "Boltz复合体置信度", "Boltz配体界面iPTM", "精修增强选择分"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(ROOT / args.queue, low_memory=False).fillna("")
    manifest = pd.read_csv(ROOT / args.manifest, low_memory=False).fillna("")
    if "pairId" in manifest.columns:
        manifest_small = manifest[["pairId", "externalQueueRank", "yamlFile"]].drop_duplicates("pairId")
        queue = queue.merge(manifest_small, left_on="boltz_pair_id", right_on="pairId", how="left")
    if "pair_id" not in queue.columns:
        queue["pair_id"] = queue.get("boltz_pair_id", "")
    run_dirs = [
        ("full_refined_batch5", ROOT / args.full_run_dir),
        ("missing24_singleton_rerun", ROOT / args.rerun_dir),
    ]
    log_dirs = [
        ("full_refined_batch5", ROOT / args.full_log_dir),
        ("missing24_singleton_rerun", ROOT / args.rerun_log_dir),
    ]
    results = collect_result_files(run_dirs)
    if results.empty:
        raise RuntimeError("No Boltz result files found.")
    skipped = collect_skipped_inputs(log_dirs)
    queue["boltz_stem"] = queue["yamlFile"].astype(str).map(lambda x: Path(x).stem)
    merged = queue.merge(results, on="boltz_stem", how="left")
    if not skipped.empty:
        merged = merged.merge(skipped, on="boltz_stem", how="left")
    else:
        merged["boltz_skip_reason"] = ""
        merged["boltz_skip_source"] = ""
        merged["boltz_skip_log_path"] = ""
    missing_mask = merged["boltz_support_tier_refined"].isna()
    merged.loc[missing_mask, "boltz_support_tier_refined"] = "U_boltz_not_completed"
    merged.loc[missing_mask, "boltz_support_reason_refined"] = "missing_boltz_confidence_or_affinity_output"
    skipped_mask = missing_mask & merged["boltz_skip_reason"].fillna("").astype(str).str.len().gt(0)
    merged.loc[skipped_mask, "boltz_support_reason_refined"] = "boltz_input_skipped_" + merged.loc[
        skipped_mask,
        "boltz_skip_reason",
    ].astype(str)
    merged.loc[skipped_mask, "boltz_run_source"] = merged.loc[skipped_mask, "boltz_skip_source"]
    merged.loc[missing_mask & merged["boltz_run_source"].fillna("").astype(str).eq(""), "boltz_run_source"] = "missing_or_unparsed"
    merged.loc[missing_mask, "boltz_completed_refined"] = False
    merged = add_flags(merged)
    merged = add_disease_auxiliary(merged, ROOT / args.ot_disease, ROOT / args.txgnn)
    merged["refined_preclinical_role"] = "review_pool"
    merged.loc[merged["refined_boltz_ab"] & ~merged["rediscovery_or_control_risk"], "refined_preclinical_role"] = "core_discovery_review"
    merged.loc[merged["rediscovery_or_control_risk"], "refined_preclinical_role"] = "positive_control_or_risk_review"
    merged.loc[merged["refined_boltz_u"], "refined_preclinical_role"] = "incomplete_boltz_review"

    final_base = merged[
        (~bool_series(merged, "is_known_fda_target_pair"))
        & (~bool_series(merged, "same_family_or_label_risk"))
        & merged["discovery_queue_class"].eq("novel_high_physics")
    ].copy()
    recommended = greedy_select(
        final_base,
        n=min(1000, len(final_base)),
        caps={
            "drug": 6,
            "target": 18,
            "scaffold": 30,
            "family": 520,
            "queue_class:positive_control_or_family_extension": 120,
        },
        family_minimums={
            "kinase": 90,
            "enzyme": 300,
            "ion_channel": 60,
            "transporter": 140,
            "nuclear_epigenetic": 60,
            "other_assayable": 60,
        },
        score_column="refined_enhanced_selection_score",
    )
    recommended["refined_wetlab_priority_band"] = "P3_rank701_1000_reserve_review"
    recommended.loc[recommended["refined_selection_rank"] <= 700, "refined_wetlab_priority_band"] = "P2_rank301_700_balanced_physics"
    recommended.loc[recommended["refined_selection_rank"] <= 300, "refined_wetlab_priority_band"] = "P1_rank1_300_first_wave"

    merged.sort_values(["refined_enhanced_selection_score", "boltz_composite_score_refined"], ascending=False).to_csv(
        out_dir / "boltz_refined_3000_merged_enriched.csv",
        index=False,
    )
    merged[merged["refined_boltz_ab"]].sort_values(
        ["boltz_support_tier_refined", "boltz_composite_score_refined"],
        ascending=[True, False],
    ).to_csv(out_dir / "boltz_refined_AB_candidates.csv", index=False)
    merged[merged["refined_boltz_u"]].sort_values("externalQueueRank").to_csv(out_dir / "boltz_refined_incomplete_rows.csv", index=False)
    merged[merged["rediscovery_or_control_risk"]].sort_values(
        ["risk_notes", "boltz_composite_score_refined"],
        ascending=[True, False],
    ).to_csv(out_dir / "boltz_refined_risk_review_rows.csv", index=False)
    recommended.to_csv(out_dir / "refined_final_1000_candidates.csv", index=False)
    readable_table(recommended).to_csv(out_dir / "refined_final_1000_teacher_readable_zh.csv", index=False)
    with pd.ExcelWriter(out_dir / "refined_final_1000_teacher_readable_zh.xlsx", engine="openpyxl") as writer:
        readable_table(recommended).to_excel(writer, index=False, sheet_name="final_1000")
        readable_table(merged[merged["refined_boltz_ab"]].sort_values("boltz_composite_score_refined", ascending=False).head(500)).to_excel(
            writer,
            index=False,
            sheet_name="AB_top500",
        )
        readable_table(merged[merged["refined_boltz_u"]].sort_values("externalQueueRank")).to_excel(
            writer,
            index=False,
            sheet_name="incomplete",
        )
    group_summary(merged, "target_assay_family").to_csv(out_dir / "audit_by_target_assay_family.csv", index=False)
    group_summary(merged, "primary_gene").to_csv(out_dir / "audit_by_target_gene.csv", index=False)
    group_summary(merged, "drug_names").to_csv(out_dir / "audit_by_drug.csv", index=False)
    if "externalQueueRank" in merged.columns:
        merged["_rank_num"] = pd.to_numeric(merged["externalQueueRank"], errors="coerce")
        merged["rank_band_300"] = pd.cut(
            merged["_rank_num"],
            bins=[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000],
            labels=["1-300", "301-600", "601-900", "901-1200", "1201-1500", "1501-1800", "1801-2100", "2101-2400", "2401-2700", "2701-3000"],
        )
        group_summary(merged, "rank_band_300").to_csv(out_dir / "audit_by_rank_band_300.csv", index=False)
        merged = merged.drop(columns=["_rank_num"], errors="ignore")

    summary = summarize(merged, recommended)
    summary["artifacts"] = {
        "merged_3000": str(out_dir / "boltz_refined_3000_merged_enriched.csv"),
        "AB_candidates": str(out_dir / "boltz_refined_AB_candidates.csv"),
        "incomplete_rows": str(out_dir / "boltz_refined_incomplete_rows.csv"),
        "risk_review_rows": str(out_dir / "boltz_refined_risk_review_rows.csv"),
        "refined_final_1000": str(out_dir / "refined_final_1000_candidates.csv"),
        "teacher_csv": str(out_dir / "refined_final_1000_teacher_readable_zh.csv"),
        "teacher_xlsx": str(out_dir / "refined_final_1000_teacher_readable_zh.xlsx"),
    }
    write_json(out_dir / "boltz_refined_3000_final_audit_summary.json", summary)
    (out_dir / "README.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Boltz-2 Refined Top3000 Final Package",
            "",
            f"Generated: {summary['created_utc']}",
            "",
            "## Scope",
            "",
            "Merged the full refined Boltz-2 Top3000 run with singleton reruns for rows that failed inside 5-row batches.",
            "",
            "## Headline",
            "",
            f"- Rows: {summary['rows']}",
            f"- Completed rows: {summary['completed_rows']} ({summary['completed_pct']:.2f}%)",
            f"- A/B rows: {summary['ab_rows']} ({summary['ab_pct_all']:.2f}% of all rows; {summary['ab_pct_completed']:.2f}% of completed rows)",
            f"- Tier counts: {summary['tier_counts']}",
            f"- A/B unique drugs: {summary['ab_unique_drugs']}",
            f"- A/B unique targets: {summary['ab_unique_targets']}",
            f"- Recommended final rows: {summary['recommended_rows']}",
            f"- Recommended A/B rows: {summary['recommended_ab_rows']}",
            "",
            "## Interpretation",
            "",
            "- Boltz refined A/B is structural second-model support, not wet-lab binding proof.",
            "- Carbonic anhydrase, same-family kinase, and ion-channel rows are explicitly flagged for control/risk review.",
            "- The refined final 1000 keeps the existing physics-first scoring framework and replaces fast Boltz evidence with refined Boltz evidence.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="outputs/chembl_moa_enhanced_information_package_v1/pre_boltz_shortlist_3000_enhanced.csv")
    parser.add_argument("--manifest", default="outputs/final_1000_funnel_v1/boltz_pre3000_input_package/boltz2_input_manifest.csv")
    parser.add_argument("--full-run-dir", default="outputs/chembl_moa_enhanced_information_package_v1/boltz_pre3000_refined_full_run/batch_runs")
    parser.add_argument("--rerun-dir", default="outputs/chembl_moa_enhanced_information_package_v1/boltz_pre3000_refined_missing24_rerun/batch_runs")
    parser.add_argument("--full-log-dir", default="outputs/chembl_moa_enhanced_information_package_v1/boltz_pre3000_refined_full_run/logs")
    parser.add_argument("--rerun-log-dir", default="outputs/chembl_moa_enhanced_information_package_v1/boltz_pre3000_refined_missing24_rerun/logs")
    parser.add_argument("--ot-disease", default="data/processed/opentargets_target_disease_scores.csv")
    parser.add_argument("--txgnn", default="data/processed/txgnn_drug_disease_scores.csv")
    parser.add_argument("--out-dir", default="outputs/chembl_moa_enhanced_information_package_v1/boltz_refined_3000_final_package")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
