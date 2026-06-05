from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
import prolif as plf
from posebusters import PoseBusters
from rdkit import Chem


NOVEL_CLASSES = {
    "disease_context_supported_new_pair",
    "model_priority_without_txgnn_kg_path",
}
CRITICAL_POSEBUSTERS_CHECKS = {
    "mol_pred_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "bond_lengths",
    "minimum_distance_to_protein",
    "volume_overlap_with_protein",
}
POSEBUSTERS_METADATA_COLUMNS = {
    "file",
    "molecule",
    "position",
    "mol_pred",
    "mol_cond",
}


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def round_float(value: Any, digits: int = 4) -> float | str:
    parsed = number(value)
    return "" if parsed is None else round(parsed, digits)


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


def abs_or_root(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def rank_column(df: pd.DataFrame) -> str:
    for column in ["sotaPoseQualityRankGlobal", "sotaMlAdmetRankGlobal", "sotaContextRankGlobal", "sotaReadyRankGlobal", "finalRankGlobal"]:
        if column in df.columns:
            return column
    return df.columns[0]


def is_novel(row: dict[str, Any]) -> bool:
    novelty = str(row.get("noveltyClass") or "").lower()
    return truthy(row.get("strictNovelPairFlag")) or str(row.get("noveltyClass") or "") in NOVEL_CLASSES or "new_pair" in novelty


def is_known(row: dict[str, Any]) -> bool:
    return truthy(row.get("knownDrugTargetPair"))


def bool_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def make_base_row(root: Path, row: pd.Series) -> dict[str, Any]:
    ligand_path = abs_or_root(root, row.get("confidenceSdfPath"))
    receptor_path = abs_or_root(root, row.get("receptorPdbPath"))
    return {
        "sotaPoseQualityRankGlobal": row.get("sotaPoseQualityRankGlobal", ""),
        "sotaMlAdmetRankGlobal": row.get("sotaMlAdmetRankGlobal", ""),
        "sotaContextRankGlobal": row.get("sotaContextRankGlobal", ""),
        "sotaReadyRankGlobal": row.get("sotaReadyRankGlobal", ""),
        "finalRankGlobal": row.get("finalRankGlobal", ""),
        "direction": row.get("direction", ""),
        "directionLabelZhFinal": row.get("directionLabelZhFinal", ""),
        "pairId": row.get("pairId", ""),
        "drugId": row.get("drugId", ""),
        "drug": row.get("drug", ""),
        "target": row.get("target", ""),
        "protein": row.get("protein", ""),
        "proteinName": row.get("proteinName", ""),
        "knownDrugTargetPair": row.get("knownDrugTargetPair", ""),
        "strictNovelPairFlag": row.get("strictNovelPairFlag", ""),
        "noveltyClass": row.get("auditNoveltyClass") or row.get("noveltyClass", ""),
        "candidateStatus": row.get("status", ""),
        "diffdock": row.get("diffdock", ""),
        "poseQualityTier": row.get("poseQualityTier", ""),
        "poseQualityScore": row.get("poseQualityScore", ""),
        "confidenceSdfPath": str(ligand_path) if ligand_path else "",
        "receptorPdbPath": str(receptor_path) if receptor_path else "",
        "structureInputReady": bool(
            str(row.get("status", "")) == "completed"
            and ligand_path is not None
            and receptor_path is not None
            and ligand_path.exists()
            and receptor_path.exists()
        ),
    }


def run_posebusters(rows: list[dict[str, Any]], config: str, max_workers: int | None) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    ready = [row for row in rows if row["structureInputReady"]]
    if not ready:
        return pd.DataFrame(), {}
    table = pd.DataFrame(
        [{"mol_pred": row["confidenceSdfPath"], "mol_cond": row["receptorPdbPath"]} for row in ready]
    )
    buster = PoseBusters(config=config, max_workers=max_workers)
    result = buster.bust_table(table, full_report=False).reset_index()
    if "file" in result.columns:
        result["mol_pred"] = result["file"].astype(str)
    elif "mol_pred" not in result.columns:
        result["mol_pred"] = table["mol_pred"].values[: len(result)]
    by_path: dict[str, dict[str, Any]] = {}
    for _, item in result.iterrows():
        by_path[str(item.get("mol_pred"))] = item.to_dict()
    return result, by_path


def summarize_posebusters(pb: dict[str, Any]) -> dict[str, Any]:
    if not pb:
        return {
            "posebustersStatus": "not_run",
            "posebustersPass": False,
            "posebustersFailedCheckCount": "",
            "posebustersFailedChecks": "posebusters_not_run",
            "posebustersCriticalFailedChecks": "posebusters_not_run",
        }
    check_cols = []
    failed = []
    critical_failed = []
    for key, value in pb.items():
        if key in POSEBUSTERS_METADATA_COLUMNS:
            continue
        parsed = bool_value(value)
        if parsed is None:
            continue
        check_cols.append(key)
        if not parsed:
            failed.append(key)
            if key in CRITICAL_POSEBUSTERS_CHECKS:
                critical_failed.append(key)
    return {
        "posebustersStatus": "ok",
        "posebustersPass": len(failed) == 0,
        "posebustersCheckCount": len(check_cols),
        "posebustersFailedCheckCount": len(failed),
        "posebustersFailedChecks": "; ".join(failed) if failed else "none",
        "posebustersCriticalFailedChecks": "; ".join(critical_failed) if critical_failed else "none",
    }


def load_protein(path: str, cache: dict[str, Any]) -> Any:
    if path not in cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            universe = mda.Universe(path)
            cache[path] = plf.Molecule.from_mda(
                universe,
                "protein",
                implicit_hydrogens=False,
                inferrer=None,
            )
    return cache[path]


def load_ligand(path: str) -> Any:
    supplier = Chem.SDMolSupplier(path, removeHs=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        raise ValueError("ligand_sdf_unreadable")
    return plf.Molecule.from_rdkit(mol)


def run_prolif_row(row: dict[str, Any], protein_cache: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not row["structureInputReady"]:
        return {
            "prolifStatus": "not_applicable",
            "prolifInteractionCount": 0,
            "prolifUniqueResidueCount": 0,
            "prolifInteractionTypes": "",
            "prolifTopInteractions": "",
            "prolifError": "structure_input_not_ready",
        }, []
    try:
        ligand = load_ligand(row["confidenceSdfPath"])
        protein = load_protein(row["receptorPdbPath"], protein_cache)
        fp = plf.Fingerprint()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp.run_from_iterable([ligand], protein, progress=False)
        frame = fp.to_dataframe()
    except Exception as exc:  # noqa: BLE001 - record row-level tool failure.
        return {
            "prolifStatus": "failed",
            "prolifInteractionCount": 0,
            "prolifUniqueResidueCount": 0,
            "prolifInteractionTypes": "",
            "prolifTopInteractions": "",
            "prolifError": f"{type(exc).__name__}: {str(exc)[:400]}",
        }, []

    interactions: list[dict[str, Any]] = []
    if frame.empty:
        return {
            "prolifStatus": "ok",
            "prolifInteractionCount": 0,
            "prolifUniqueResidueCount": 0,
            "prolifInteractionTypes": "",
            "prolifTopInteractions": "",
            "prolifError": "",
        }, []
    first = frame.iloc[0]
    for col in frame.columns:
        if bool_value(first[col]) is not True:
            continue
        if isinstance(col, tuple) and len(col) >= 3:
            ligand_residue, protein_residue, interaction_type = col[:3]
        else:
            ligand_residue, protein_residue, interaction_type = "", str(col), ""
        interactions.append(
            {
                "direction": row.get("direction", ""),
                "pairId": row.get("pairId", ""),
                "drug": row.get("drug", ""),
                "target": row.get("target", ""),
                "protein": row.get("protein", ""),
                "ligandResidue": str(ligand_residue),
                "proteinResidue": str(protein_residue),
                "interactionType": str(interaction_type),
            }
        )
    residue_count = len({item["proteinResidue"] for item in interactions})
    type_counts = Counter(item["interactionType"] for item in interactions)
    top = [f"{item['proteinResidue']}:{item['interactionType']}" for item in interactions[:20]]
    return {
        "prolifStatus": "ok",
        "prolifInteractionCount": len(interactions),
        "prolifUniqueResidueCount": residue_count,
        "prolifInteractionTypes": "; ".join(f"{key}:{type_counts[key]}" for key in sorted(type_counts)),
        "prolifTopInteractions": "; ".join(top),
        "prolifError": "",
    }, interactions


def classify_standard(row: dict[str, Any]) -> tuple[float, str, str, str]:
    failed = int(number(row.get("posebustersFailedCheckCount")) or 0)
    critical_failed = str(row.get("posebustersCriticalFailedChecks") or "")
    interactions = int(number(row.get("prolifInteractionCount")) or 0)
    score = 100.0 - failed * 7.0 + min(interactions, 12) * 1.5
    if row.get("structureInputReady") is not True:
        return 0.0, "D_standard_pose_not_ready", "structure_input_missing", "structure_input_not_ready"
    if row.get("posebustersStatus") != "ok":
        return 20.0, "D_standard_pose_not_ready", "posebusters_not_run", "posebusters_not_run"
    if critical_failed != "none":
        score = min(score, 50.0 if "volume_overlap_with_protein" in critical_failed else 60.0)
        tier = "D_standard_pose_fail"
        action = "standard_pose_resolution_required"
        reason = critical_failed
    elif failed:
        score = min(score, 78.0)
        tier = "C_standard_pose_review"
        action = "standard_pose_review"
        reason = row.get("posebustersFailedChecks") or "posebusters_failed_checks"
    elif row.get("prolifStatus") != "ok":
        score = min(score, 72.0)
        tier = "C_standard_interaction_review"
        action = "prolif_interaction_review"
        reason = row.get("prolifError") or "prolif_failed"
    elif interactions >= 5:
        tier = "A_standard_structure_supported"
        action = "standard_structure_supported"
        reason = "posebusters_pass_with_interaction_fingerprint"
    elif interactions > 0:
        tier = "B_standard_structure_acceptable"
        action = "standard_structure_acceptable"
        reason = "posebusters_pass_with_sparse_interactions"
    else:
        tier = "C_standard_interaction_review"
        action = "prolif_no_interaction_review"
        reason = "posebusters_pass_but_no_prolif_interactions"
    return round(max(0.0, min(100.0, score)), 4), tier, action, reason


def build_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_path = root / args.source
    final_df = pd.read_csv(source_path).fillna("")
    sort_col = rank_column(final_df)
    final_df["_rankSort"] = pd.to_numeric(final_df[sort_col], errors="coerce").fillna(999999999)
    selected = final_df.sort_values("_rankSort").copy()
    if args.top_n > 0:
        selected = selected.head(args.top_n).copy()
    base_rows = [make_base_row(root, row) for _, row in selected.iterrows()]

    posebusters_raw, posebusters_by_path = run_posebusters(base_rows, args.posebusters_config, args.max_workers)
    protein_cache: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    for base in base_rows:
        pb_summary = summarize_posebusters(posebusters_by_path.get(base["confidenceSdfPath"], {}))
        prolif_summary, interactions = run_prolif_row(base, protein_cache)
        combined = {**base, **pb_summary, **prolif_summary}
        score, tier, action, reason = classify_standard(combined)
        combined.update(
            {
                "standardPoseValidationScore": score,
                "standardPoseValidationTier": tier,
                "standardPoseValidationAction": action,
                "standardPoseValidationReason": reason,
            }
        )
        audit_rows.append(combined)
        interaction_rows.extend(interactions)

    audit_df = pd.DataFrame(audit_rows).fillna("")
    interaction_df = pd.DataFrame(interaction_rows)
    direction_df = direction_summary(audit_df)
    summary = summarize(audit_df, direction_df, interaction_df, len(final_df), len(selected), len(protein_cache), args)

    merge_cols = [
        "direction",
        "pairId",
        "standardPoseValidationScore",
        "standardPoseValidationTier",
        "standardPoseValidationAction",
        "standardPoseValidationReason",
        "posebustersPass",
        "posebustersFailedCheckCount",
        "posebustersFailedChecks",
        "posebustersCriticalFailedChecks",
        "prolifStatus",
        "prolifInteractionCount",
        "prolifUniqueResidueCount",
        "prolifInteractionTypes",
        "prolifTopInteractions",
        "prolifError",
    ]
    augmented = final_df.drop(columns=["_rankSort"], errors="ignore").merge(
        audit_df[[column for column in merge_cols if column in audit_df.columns]],
        on=["direction", "pairId"],
        how="left",
    )
    base_score = pd.to_numeric(augmented.get("sotaPoseQualityScore", augmented.get("sotaMlAdmetScore", 0)), errors="coerce").fillna(0)
    standard_score = pd.to_numeric(augmented.get("standardPoseValidationScore"), errors="coerce").fillna(40)
    augmented["sotaStandardStructureScore"] = (0.82 * base_score + 0.18 * standard_score).round(4)
    augmented = augmented.sort_values("sotaStandardStructureScore", ascending=False).reset_index(drop=True)
    augmented.insert(0, "sotaStandardStructureRankGlobal", np.arange(1, len(augmented) + 1))

    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "audit": out_dir / "standard_pose_validation_candidate_audit.csv",
        "posebusters": out_dir / "posebusters_raw.csv",
        "interactions": out_dir / "prolif_interaction_fingerprints.csv",
        "direction": out_dir / "standard_pose_validation_direction_summary.csv",
        "summary": out_dir / "standard_pose_validation_summary.json",
        "md": out_dir / "STANDARD_POSE_VALIDATION_AUDIT.md",
        "matrix": final_dir / "final_priority_standard_pose_validation_matrix.csv",
        "shortlist": final_dir / "final_priority_standard_pose_validation_top300_expert_shortlist.csv",
        "review": final_dir / "final_priority_standard_pose_validation_review_queue.csv",
        "final_md": final_dir / "FINAL_PRIORITY_STANDARD_POSE_VALIDATION_AUDIT.md",
    }
    audit_df.to_csv(out_paths["audit"], index=False)
    posebusters_raw.to_csv(out_paths["posebusters"], index=False)
    interaction_df.to_csv(out_paths["interactions"], index=False)
    direction_df.to_csv(out_paths["direction"], index=False)
    write_json(out_paths["summary"], summary)
    md_text = markdown(summary, direction_df, audit_df)
    out_paths["md"].write_text(md_text, encoding="utf-8")
    augmented.to_csv(out_paths["matrix"], index=False)
    supported = augmented[augmented["standardPoseValidationTier"].astype(str).str.startswith(("A_", "B_"))].copy()
    supported.head(300).to_csv(out_paths["shortlist"], index=False)
    review = augmented[~augmented["standardPoseValidationTier"].astype(str).str.startswith(("A_", "B_"))].copy()
    review.head(300).to_csv(out_paths["review"], index=False)
    out_paths["final_md"].write_text(md_text, encoding="utf-8")
    return {"summary": summary, **{key: str(path.relative_to(root)) for key, path in out_paths.items()}}


def direction_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, group in audit_df.groupby("direction", dropna=False):
        tiers = Counter(group["standardPoseValidationTier"].astype(str))
        supported = int(sum(group["standardPoseValidationTier"].astype(str).str.startswith(("A_", "B_"))))
        rows.append(
            {
                "direction": direction,
                "rows": int(len(group)),
                "structureInputReadyRows": int(sum(group["structureInputReady"].astype(bool))),
                "standardSupportedRows": supported,
                "standardSupportedPct": round(pct(supported, len(group)), 4),
                "knownRows": int(sum(group.apply(lambda row: is_known(row.to_dict()), axis=1))),
                "novelRows": int(sum(group.apply(lambda row: is_novel(row.to_dict()), axis=1))),
                "tierCounts": json.dumps(dict(tiers), ensure_ascii=False, sort_keys=True),
                "posebustersPassRows": int(sum(group["posebustersPass"].apply(truthy))),
                "prolifOkRows": int(sum(group["prolifStatus"].astype(str) == "ok")),
                "medianProlifInteractionCount": round_float(pd.to_numeric(group["prolifInteractionCount"], errors="coerce").median()),
                "medianStandardPoseValidationScore": round_float(
                    pd.to_numeric(group["standardPoseValidationScore"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("direction")


def summarize(
    audit_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    source_rows: int,
    audited_rows: int,
    protein_cache_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tiers = Counter(audit_df["standardPoseValidationTier"].astype(str))
    supported = int(sum(audit_df["standardPoseValidationTier"].astype(str).str.startswith(("A_", "B_"))))
    ready = int(sum(audit_df["structureInputReady"].astype(bool)))
    posebusters_pass = int(sum(audit_df["posebustersPass"].apply(truthy)))
    prolif_ok = int(sum(audit_df["prolifStatus"].astype(str) == "ok"))
    known_rows = int(sum(audit_df.apply(lambda row: is_known(row.to_dict()), axis=1)))
    novel_rows = int(sum(audit_df.apply(lambda row: is_novel(row.to_dict()), axis=1)))
    top100 = audit_df.head(100).copy()
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": f"Standard PoseBusters/ProLIF structural validation over Top{audited_rows} of {source_rows} final candidates.",
        "source": args.source,
        "sourceRows": int(source_rows),
        "candidateRows": int(audited_rows),
        "structureInputReadyRows": ready,
        "structureInputReadyPct": round(pct(ready, audited_rows), 4),
        "uniqueProteinsConverted": int(protein_cache_size),
        "posebustersConfig": args.posebusters_config,
        "posebustersPassRows": posebusters_pass,
        "posebustersPassPct": round(pct(posebusters_pass, audited_rows), 4),
        "prolifOkRows": prolif_ok,
        "prolifOkPct": round(pct(prolif_ok, audited_rows), 4),
        "prolifInteractionRows": int(len(interaction_df)),
        "standardSupportedRows": supported,
        "standardSupportedPct": round(pct(supported, audited_rows), 4),
        "knownRows": known_rows,
        "novelRows": novel_rows,
        "standardPoseValidationTierCounts": dict(tiers),
        "medianProlifInteractionCount": round_float(pd.to_numeric(audit_df["prolifInteractionCount"], errors="coerce").median()),
        "medianStandardPoseValidationScore": round_float(
            pd.to_numeric(audit_df["standardPoseValidationScore"], errors="coerce").median()
        ),
        "top100": {
            "rows": int(len(top100)),
            "standardSupportedRows": int(sum(top100["standardPoseValidationTier"].astype(str).str.startswith(("A_", "B_")))),
            "posebustersPassRows": int(sum(top100["posebustersPass"].apply(truthy))),
            "prolifOkRows": int(sum(top100["prolifStatus"].astype(str) == "ok")),
            "knownRows": int(sum(top100.apply(lambda row: is_known(row.to_dict()), axis=1))) if len(top100) else 0,
            "novelRows": int(sum(top100.apply(lambda row: is_novel(row.to_dict()), axis=1))) if len(top100) else 0,
            "tierCounts": dict(Counter(top100["standardPoseValidationTier"].astype(str))),
        },
        "directionRows": json_safe(direction_df.to_dict(orient="records")),
        "methodNote": (
            "PoseBusters dock configuration checks ligand chemistry and ligand-protein plausibility; ProLIF generates "
            "protein-ligand interaction fingerprints from the same docked SDF/PDB files. This layer is a standard-tool "
            "validation extension over high-priority candidates, not a new affinity model."
        ),
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    examples = audit_df.sort_values("standardPoseValidationScore", ascending=False).head(10)
    lines = [
        "# Standard Pose Validation Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["scope"],
        "",
        "## Headline Metrics",
        "",
        f"- Candidate rows audited: {summary['candidateRows']}",
        f"- Structure-input-ready rows: {summary['structureInputReadyRows']} ({summary['structureInputReadyPct']:.2f}%)",
        f"- PoseBusters pass rows: {summary['posebustersPassRows']} ({summary['posebustersPassPct']:.2f}%)",
        f"- ProLIF successful rows: {summary['prolifOkRows']} ({summary['prolifOkPct']:.2f}%)",
        f"- ProLIF interaction rows: {summary['prolifInteractionRows']}",
        f"- A/B standard-supported rows: {summary['standardSupportedRows']} ({summary['standardSupportedPct']:.2f}%)",
        f"- Tier counts: {summary['standardPoseValidationTierCounts']}",
        f"- Median ProLIF interaction count: {summary['medianProlifInteractionCount']}",
        f"- Top100 standard-supported rows: {summary['top100']['standardSupportedRows']}/{summary['top100']['rows']}",
        "",
        "## Direction Summary",
        "",
    ]
    for _, row in direction_df.iterrows():
        lines.append(
            f"- {row['direction']}: {row['standardSupportedRows']}/{row['rows']} A/B supported "
            f"({row['standardSupportedPct']:.2f}%); PoseBusters pass {row['posebustersPassRows']}; "
            f"ProLIF OK {row['prolifOkRows']}."
        )
    lines.extend(["", "## Representative Standard-Supported Poses", ""])
    for _, row in examples.iterrows():
        lines.append(
            f"- {row.get('drug')} - {row.get('target')} ({row.get('direction')}): "
            f"{row.get('standardPoseValidationTier')}, score {row.get('standardPoseValidationScore')}, "
            f"PoseBusters failed checks {row.get('posebustersFailedCheckCount')}, "
            f"ProLIF interactions {row.get('prolifInteractionCount')}."
        )
    lines.extend(["", "## Method Note", "", summary["methodNote"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build standard PoseBusters/ProLIF validation audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/sota_validation/final_prioritization/final_priority_pose_quality_matrix.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/standard_pose_validation")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--top-n", type=int, default=1000, help="0 means audit all source rows.")
    parser.add_argument("--posebusters-config", default="dock")
    parser.add_argument("--max-workers", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build_audit(root, args)
    print(json.dumps(json_safe(result["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
