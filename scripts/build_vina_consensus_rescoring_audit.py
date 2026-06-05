from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from vina import Vina


NOVEL_CLASSES = {
    "disease_context_supported_new_pair",
    "model_priority_without_txgnn_kg_path",
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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def round_float(value: Any, digits: int = 4) -> float | str:
    parsed = number(value)
    return "" if parsed is None else round(parsed, digits)


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
    for column in [
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


def safe_token(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned[:120] or "item"


def path_digest(path: Path) -> str:
    text = str(path.resolve())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def is_novel(row: dict[str, Any]) -> bool:
    novelty = str(row.get("noveltyClass") or row.get("auditNoveltyClass") or "").lower()
    return truthy(row.get("strictNovelPairFlag")) or novelty in NOVEL_CLASSES or "new_pair" in novelty


def is_known(row: dict[str, Any]) -> bool:
    return truthy(row.get("knownDrugTargetPair"))


def read_ligand_geometry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "ligand_sdf_missing"}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        return {"ok": False, "error": "ligand_sdf_unreadable"}
    if mol.GetNumConformers() == 0:
        return {"ok": False, "error": "ligand_has_no_conformer"}
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
    if not coords:
        return {"ok": False, "error": "ligand_has_no_heavy_atoms"}
    coord_arr = np.asarray(coords, dtype=float)
    if not np.isfinite(coord_arr).all():
        return {"ok": False, "error": "ligand_coordinates_invalid"}
    bbox = coord_arr.max(axis=0) - coord_arr.min(axis=0)
    return {
        "ok": True,
        "center": coord_arr.mean(axis=0),
        "bbox": bbox,
        "heavyAtomCount": int(coord_arr.shape[0]),
        "error": "",
    }


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
            "stdout": (exc.stdout or "")[:1000],
            "stderr": (exc.stderr or "")[:1000],
            "error": "timeout",
        }
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[:1000],
        "stderr": result.stderr[:1000],
        "error": "" if result.returncode == 0 else "nonzero_exit",
    }


def convert_with_obabel(
    source: Path,
    destination: Path,
    kind: str,
    obabel: str,
    timeout: int,
    force: bool,
) -> dict[str, Any]:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return {"ok": True, "pdbqtPath": str(destination), "cached": True, "returncode": 0, "stderr": "", "error": ""}
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [obabel, str(source), "-O", str(destination), "--partialcharge", "gasteiger"]
    if kind == "receptor":
        command = [obabel, str(source), "-O", str(destination), "-xr", "--partialcharge", "gasteiger"]
    result = run_command(command, timeout)
    ok = result["ok"] and destination.exists() and destination.stat().st_size > 0
    return {
        "ok": ok,
        "pdbqtPath": str(destination) if ok else "",
        "cached": False,
        "returncode": result["returncode"],
        "stderr": result["stderr"],
        "error": "" if ok else f"{result['error']}: {result['stderr'][:400]}",
    }


def make_base_row(root: Path, row: pd.Series) -> dict[str, Any]:
    ligand_path = abs_or_root(root, row.get("confidenceSdfPath"))
    receptor_path = abs_or_root(root, row.get("receptorPdbPath"))
    return {
        "sotaStandardStructureRankGlobal": row.get("sotaStandardStructureRankGlobal", ""),
        "sotaPoseQualityRankGlobal": row.get("sotaPoseQualityRankGlobal", ""),
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
        "sotaStandardStructureScore": row.get("sotaStandardStructureScore", ""),
        "standardPoseValidationTier": row.get("standardPoseValidationTier", ""),
        "standardPoseValidationScore": row.get("standardPoseValidationScore", ""),
        "posebustersPass": row.get("posebustersPass", ""),
        "prolifInteractionCount": row.get("prolifInteractionCount", ""),
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


def row_cache_paths(row: dict[str, Any], cache_dir: Path) -> tuple[Path, Path]:
    ligand_source = Path(row["confidenceSdfPath"])
    receptor_source = Path(row["receptorPdbPath"])
    pair = safe_token(row.get("pairId"))
    ligand_name = f"{pair}_{path_digest(ligand_source)}.pdbqt"
    receptor_name = f"{safe_token(receptor_source.stem)}_{path_digest(receptor_source)}.pdbqt"
    return cache_dir / "ligands" / ligand_name, cache_dir / "receptors" / receptor_name


def vina_score_row(row: dict[str, Any], args: argparse.Namespace, cache_dir: Path, obabel: str) -> dict[str, Any]:
    if not row["structureInputReady"]:
        return {
            "vinaStatus": "not_ready",
            "vinaError": "structure_input_not_ready",
            "pdbqtReady": False,
        }
    ligand_path = Path(row["confidenceSdfPath"])
    receptor_path = Path(row["receptorPdbPath"])
    geometry = read_ligand_geometry(ligand_path)
    if not geometry.get("ok"):
        return {
            "vinaStatus": "failed",
            "vinaError": geometry.get("error", "ligand_geometry_failed"),
            "pdbqtReady": False,
        }

    ligand_pdbqt, receptor_pdbqt = row_cache_paths(row, cache_dir)
    ligand_conv = convert_with_obabel(ligand_path, ligand_pdbqt, "ligand", obabel, args.conversion_timeout, args.force_reconvert)
    receptor_conv = convert_with_obabel(
        receptor_path, receptor_pdbqt, "receptor", obabel, args.conversion_timeout, args.force_reconvert
    )
    if not ligand_conv["ok"] or not receptor_conv["ok"]:
        return {
            "vinaStatus": "conversion_failed",
            "vinaError": f"ligand={ligand_conv['error']}; receptor={receptor_conv['error']}",
            "pdbqtReady": False,
            "ligandPdbqtPath": ligand_conv.get("pdbqtPath", ""),
            "receptorPdbqtPath": receptor_conv.get("pdbqtPath", ""),
            "ligandPdbqtCached": ligand_conv.get("cached", False),
            "receptorPdbqtCached": receptor_conv.get("cached", False),
        }

    center = np.asarray(geometry["center"], dtype=float)
    bbox = np.asarray(geometry["bbox"], dtype=float)
    box = np.clip(bbox + float(args.box_padding), float(args.min_box_size), float(args.max_box_size))
    try:
        v = Vina(sf_name=args.scoring_function, cpu=args.cpu, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(center=center.tolist(), box_size=box.tolist())
        score = np.asarray(v.score(), dtype=float)
        optimized = np.asarray(v.optimize(max_steps=args.optimize_steps), dtype=float) if args.optimize else np.asarray([])
    except Exception as exc:  # noqa: BLE001 - preserve row-level scoring failures.
        return {
            "vinaStatus": "scoring_failed",
            "vinaError": f"{type(exc).__name__}: {str(exc)[:500]}",
            "pdbqtReady": True,
            "ligandPdbqtPath": str(ligand_pdbqt),
            "receptorPdbqtPath": str(receptor_pdbqt),
            "ligandPdbqtCached": ligand_conv.get("cached", False),
            "receptorPdbqtCached": receptor_conv.get("cached", False),
            "vinaBoxCenterX": round(float(center[0]), 4),
            "vinaBoxCenterY": round(float(center[1]), 4),
            "vinaBoxCenterZ": round(float(center[2]), 4),
            "vinaBoxSizeX": round(float(box[0]), 4),
            "vinaBoxSizeY": round(float(box[1]), 4),
            "vinaBoxSizeZ": round(float(box[2]), 4),
        }

    total = float(score[0]) if len(score) else None
    intermolecular = float(score[1]) if len(score) > 1 else None
    optimized_total = float(optimized[0]) if len(optimized) else None
    return {
        "vinaStatus": "ok",
        "vinaError": "",
        "pdbqtReady": True,
        "ligandPdbqtPath": str(ligand_pdbqt),
        "receptorPdbqtPath": str(receptor_pdbqt),
        "ligandPdbqtCached": ligand_conv.get("cached", False),
        "receptorPdbqtCached": receptor_conv.get("cached", False),
        "vinaScoringFunction": args.scoring_function,
        "vinaScoreKcalMol": round_float(total),
        "vinaIntermolecularKcalMol": round_float(intermolecular),
        "vinaOptimizedScoreKcalMol": round_float(optimized_total),
        "vinaRelaxationImprovementKcalMol": round_float((total - optimized_total) if total is not None and optimized_total is not None else None),
        "vinaRawEnergyVector": ";".join(f"{item:.4f}" for item in score),
        "vinaOptimizedEnergyVector": ";".join(f"{item:.4f}" for item in optimized),
        "vinaHeavyAtomCount": geometry.get("heavyAtomCount", ""),
        "vinaBoxCenterX": round(float(center[0]), 4),
        "vinaBoxCenterY": round(float(center[1]), 4),
        "vinaBoxCenterZ": round(float(center[2]), 4),
        "vinaBoxSizeX": round(float(box[0]), 4),
        "vinaBoxSizeY": round(float(box[1]), 4),
        "vinaBoxSizeZ": round(float(box[2]), 4),
    }


def vina_score_component(vina_score: Any, optimized_score: Any) -> float:
    score = number(vina_score)
    optimized = number(optimized_score)
    values = []
    for item in [score, optimized]:
        if item is None:
            continue
        # Maps approximately -3 kcal/mol to 0 and -9 kcal/mol to 100.
        values.append(max(0.0, min(100.0, ((-item - 3.0) / 6.0) * 100.0)))
    if not values:
        return 0.0
    return float(np.mean(values))


def classify_vina(row: dict[str, Any]) -> tuple[float, str, str, str]:
    if row.get("vinaStatus") != "ok":
        return 0.0, "D_vina_not_scored", "vina_input_or_scoring_resolution_required", row.get("vinaError") or "vina_not_scored"

    score = number(row.get("vinaScoreKcalMol"))
    optimized = number(row.get("vinaOptimizedScoreKcalMol"))
    standard_tier = str(row.get("standardPoseValidationTier") or "")
    standard_supported = standard_tier.startswith(("A_", "B_"))
    score_component = vina_score_component(score, optimized)
    if not standard_supported:
        if score is not None and score <= -5.5:
            return min(72.0, score_component), "C_vina_pose_validation_conflict", "vina_supported_but_pose_review", "standard_pose_validation_not_A_or_B"
        return min(55.0, score_component), "D_vina_pose_validation_conflict", "pose_geometry_resolution_required", "standard_pose_validation_not_A_or_B"
    if score is not None and score <= -7.0 and (optimized is None or optimized <= -7.0):
        return max(85.0, score_component), "A_vina_consensus_supported", "vina_consensus_supported", "score_only_and_standard_pose_supported"
    if score is not None and score <= -5.5:
        return max(72.0, score_component), "B_vina_consensus_acceptable", "vina_consensus_acceptable", "score_only_supports_diffdock_pose"
    if score is not None and score <= -4.0:
        return max(55.0, score_component), "C_vina_weak_support_review", "vina_weak_support_review", "weak_vina_score_for_standard_pose"
    return max(25.0, score_component), "D_vina_low_affinity_review", "vina_low_affinity_review", "weak_or_positive_vina_score"


def direction_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, group in audit_df.groupby("direction", dropna=False):
        tiers = Counter(group["vinaConsensusTier"].astype(str))
        scored = int(sum(group["vinaStatus"].astype(str) == "ok"))
        supported = int(sum(group["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
        rows.append(
            {
                "direction": direction,
                "rows": int(len(group)),
                "structureInputReadyRows": int(sum(group["structureInputReady"].astype(bool))),
                "pdbqtReadyRows": int(sum(group["pdbqtReady"].apply(truthy))),
                "vinaScoredRows": scored,
                "vinaScoredPct": round(pct(scored, len(group)), 4),
                "vinaConsensusSupportedRows": supported,
                "vinaConsensusSupportedPct": round(pct(supported, len(group)), 4),
                "knownRows": int(sum(group.apply(lambda row: is_known(row.to_dict()), axis=1))),
                "novelRows": int(sum(group.apply(lambda row: is_novel(row.to_dict()), axis=1))),
                "tierCounts": json.dumps(dict(tiers), ensure_ascii=False, sort_keys=True),
                "medianVinaScoreKcalMol": round_float(pd.to_numeric(group["vinaScoreKcalMol"], errors="coerce").median()),
                "medianVinaOptimizedScoreKcalMol": round_float(
                    pd.to_numeric(group["vinaOptimizedScoreKcalMol"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("direction")


def summarize(audit_df: pd.DataFrame, direction_df: pd.DataFrame, source_rows: int, audited_rows: int, args: argparse.Namespace) -> dict[str, Any]:
    tiers = Counter(audit_df["vinaConsensusTier"].astype(str))
    scored = int(sum(audit_df["vinaStatus"].astype(str) == "ok"))
    supported = int(sum(audit_df["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
    ready = int(sum(audit_df["structureInputReady"].astype(bool)))
    pdbqt_ready = int(sum(audit_df["pdbqtReady"].apply(truthy)))
    known_rows = int(sum(audit_df.apply(lambda row: is_known(row.to_dict()), axis=1)))
    novel_rows = int(sum(audit_df.apply(lambda row: is_novel(row.to_dict()), axis=1)))
    top100 = audit_df.head(100).copy()
    top_supported = int(sum(top100["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": f"AutoDock Vina score-only structural consensus rescoring over Top{audited_rows} of {source_rows} final candidates.",
        "source": args.source,
        "sourceRows": int(source_rows),
        "candidateRows": int(audited_rows),
        "structureInputReadyRows": ready,
        "structureInputReadyPct": round(pct(ready, audited_rows), 4),
        "pdbqtReadyRows": pdbqt_ready,
        "pdbqtReadyPct": round(pct(pdbqt_ready, audited_rows), 4),
        "vinaScoredRows": scored,
        "vinaScoredPct": round(pct(scored, audited_rows), 4),
        "vinaConsensusSupportedRows": supported,
        "vinaConsensusSupportedPct": round(pct(supported, audited_rows), 4),
        "knownRows": known_rows,
        "novelRows": novel_rows,
        "vinaConsensusTierCounts": dict(tiers),
        "medianVinaScoreKcalMol": round_float(pd.to_numeric(audit_df["vinaScoreKcalMol"], errors="coerce").median()),
        "medianVinaOptimizedScoreKcalMol": round_float(
            pd.to_numeric(audit_df["vinaOptimizedScoreKcalMol"], errors="coerce").median()
        ),
        "top100": {
            "rows": int(len(top100)),
            "vinaScoredRows": int(sum(top100["vinaStatus"].astype(str) == "ok")),
            "vinaConsensusSupportedRows": top_supported,
            "knownRows": int(sum(top100.apply(lambda row: is_known(row.to_dict()), axis=1))),
            "novelRows": int(sum(top100.apply(lambda row: is_novel(row.to_dict()), axis=1))),
            "tierCounts": dict(Counter(top100["vinaConsensusTier"].astype(str))),
        },
        "directionRows": direction_df.to_dict(orient="records"),
        "boxPaddingAngstrom": args.box_padding,
        "minBoxSizeAngstrom": args.min_box_size,
        "maxBoxSizeAngstrom": args.max_box_size,
        "optimize": bool(args.optimize),
        "optimizeSteps": args.optimize_steps,
        "methodNote": "Vina is used here as an independent score-only/rescoring layer for DiffDock poses. It is a structural consensus check, not a replacement for affinity prediction, PoseBusters geometry validation, or experimental binding assays.",
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    lines = [
        "# AutoDock Vina Consensus Rescoring Audit",
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
        f"- Structure input ready: {summary['structureInputReadyRows']}/{summary['candidateRows']} ({fmt(summary['structureInputReadyPct'])}%).",
        f"- PDBQT conversion ready: {summary['pdbqtReadyRows']}/{summary['candidateRows']} ({fmt(summary['pdbqtReadyPct'])}%).",
        f"- Vina scored rows: {summary['vinaScoredRows']}/{summary['candidateRows']} ({fmt(summary['vinaScoredPct'])}%).",
        f"- A/B Vina consensus-supported rows: {summary['vinaConsensusSupportedRows']}/{summary['candidateRows']} ({fmt(summary['vinaConsensusSupportedPct'])}%).",
        f"- Median Vina score-only affinity: {fmt(summary['medianVinaScoreKcalMol'])} kcal/mol.",
        f"- Median Vina locally optimized score: {fmt(summary['medianVinaOptimizedScoreKcalMol'])} kcal/mol.",
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

    preview_cols = [
        "direction",
        "pairId",
        "drug",
        "target",
        "vinaScoreKcalMol",
        "vinaOptimizedScoreKcalMol",
        "vinaConsensusTier",
        "vinaConsensusReason",
    ]
    lines.extend(
        [
            "",
            "## Top Rescored Examples",
            "",
            "| Direction | Pair | Drug | Target | Vina | Optimized | Tier | Reason |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in audit_df.head(20).iterrows():
        lines.append(
            f"| {row.get('direction', '')} | {row.get('pairId', '')} | {row.get('drug', '')} | "
            f"{row.get('target', '')} | {fmt(row.get('vinaScoreKcalMol'))} | "
            f"{fmt(row.get('vinaOptimizedScoreKcalMol'))} | {row.get('vinaConsensusTier', '')} | "
            f"{row.get('vinaConsensusReason', '')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/B rows have both a standard structure-supported pose and an independently favorable Vina score.",
            "- C rows usually mean either Vina support is weak or Vina support conflicts with the PoseBusters/ProLIF geometry layer.",
            "- D rows should not be promoted without manual structural review or a second structure-generation method.",
            "",
            "## Output Files",
            "",
            "- Candidate audit: `outputs/sota_validation/vina_consensus_rescoring/vina_consensus_candidate_audit.csv`",
            "- Direction summary: `outputs/sota_validation/vina_consensus_rescoring/vina_consensus_direction_summary.csv`",
            "- Final matrix: `outputs/sota_validation/final_prioritization/final_priority_vina_consensus_matrix.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    obabel = shutil.which(args.obabel)
    if not obabel:
        raise RuntimeError(f"OpenBabel executable not found: {args.obabel}")

    source_path = root / args.source
    final_df = pd.read_csv(source_path, low_memory=False).fillna("")
    sort_col = rank_column(final_df)
    final_df["_rankSort"] = pd.to_numeric(final_df[sort_col], errors="coerce").fillna(999999999)
    selected = final_df.sort_values("_rankSort").copy()
    if args.top_n > 0:
        selected = selected.head(args.top_n).copy()
    if args.start_index > 1:
        selected = selected.iloc[args.start_index - 1 :].copy()
    if args.row_count > 0:
        selected = selected.head(args.row_count).copy()

    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    cache_dir = root / args.cache_dir if args.cache_dir else out_dir / "pdbqt_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    base_rows = [make_base_row(root, row) for _, row in selected.iterrows()]
    for index, base in enumerate(base_rows, start=1):
        scored = vina_score_row(base, args, cache_dir, obabel)
        combined = {**base, **scored}
        score, tier, action, reason = classify_vina(combined)
        combined.update(
            {
                "vinaConsensusScore": round(score, 4),
                "vinaConsensusTier": tier,
                "vinaConsensusAction": action,
                "vinaConsensusReason": reason,
            }
        )
        audit_rows.append(combined)
        if args.progress_every and index % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
                        "globalIndex": int(args.start_index + index - 1),
                        "pairId": base.get("pairId"),
                        "vinaStatus": scored.get("vinaStatus"),
                    }
                ),
                flush=True,
            )

    audit_df = pd.DataFrame(audit_rows).fillna("")
    direction_df = direction_summary(audit_df)
    summary = summarize(audit_df, direction_df, len(final_df), len(selected), args)

    merge_cols = [
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
    ]
    augmented = final_df.drop(columns=["_rankSort"], errors="ignore").merge(
        audit_df[[column for column in merge_cols if column in audit_df.columns]],
        on=["direction", "pairId"],
        how="left",
    )
    base_score = pd.to_numeric(
        augmented.get("sotaStandardStructureScore", augmented.get("sotaPoseQualityScore", 0)), errors="coerce"
    ).fillna(0)
    vina_score = pd.to_numeric(augmented.get("vinaConsensusScore"), errors="coerce").fillna(40)
    augmented["sotaVinaConsensusScore"] = (0.75 * base_score + 0.25 * vina_score).round(4)
    augmented = augmented.sort_values("sotaVinaConsensusScore", ascending=False).reset_index(drop=True)
    augmented.insert(0, "sotaVinaConsensusRankGlobal", np.arange(1, len(augmented) + 1))

    out_paths = {
        "audit": out_dir / "vina_consensus_candidate_audit.csv",
        "direction": out_dir / "vina_consensus_direction_summary.csv",
        "summary": out_dir / "vina_consensus_summary.json",
        "md": out_dir / "VINA_CONSENSUS_RESCORING_AUDIT.md",
        "matrix": final_dir / "final_priority_vina_consensus_matrix.csv",
        "shortlist": final_dir / "final_priority_vina_consensus_top300_expert_shortlist.csv",
        "review": final_dir / "final_priority_vina_consensus_review_queue.csv",
        "final_md": final_dir / "FINAL_PRIORITY_VINA_CONSENSUS_AUDIT.md",
    }
    audit_df.to_csv(out_paths["audit"], index=False)
    direction_df.to_csv(out_paths["direction"], index=False)
    write_json(out_paths["summary"], summary)
    md_text = markdown(summary, direction_df, audit_df)
    out_paths["md"].write_text(md_text, encoding="utf-8")
    augmented.to_csv(out_paths["matrix"], index=False)
    supported = augmented[augmented["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"))].copy()
    supported.head(300).to_csv(out_paths["shortlist"], index=False)
    review = augmented[~augmented["vinaConsensusTier"].astype(str).str.startswith(("A_", "B_"))].copy()
    review.head(300).to_csv(out_paths["review"], index=False)
    out_paths["final_md"].write_text(md_text, encoding="utf-8")
    return {"summary": summary, **{key: str(path.relative_to(root)) for key, path in out_paths.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an AutoDock Vina consensus rescoring audit for BioMaster candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source",
        default="outputs/sota_validation/final_prioritization/final_priority_standard_pose_validation_matrix.csv",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based row offset after ranking/sorting and optional top-n selection. Used for CPU-only sharded runs.",
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=0,
        help="Maximum number of sorted rows to score after start-index. 0 means no additional shard limit.",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/vina_consensus_rescoring")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--obabel", default="obabel")
    parser.add_argument("--scoring-function", default="vina")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--box-padding", type=float, default=12.0)
    parser.add_argument("--min-box-size", type=float, default=18.0)
    parser.add_argument("--max-box-size", type=float, default=34.0)
    parser.add_argument("--conversion-timeout", type=int, default=180)
    parser.add_argument("--optimize", action="store_true", default=True)
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.add_argument("--optimize-steps", type=int, default=0)
    parser.add_argument("--force-reconvert", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build_audit(root, args)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
