#!/usr/bin/env python3
"""Build an orthogonal pocket-prediction package for targets without known pockets.

The package is deliberately restricted to the 151 ChEMBL 37 targets for which
the frozen known-pocket atlas contains no experimental pocket.  Exact-sequence
AlphaFold structures are evaluated with fpocket and compared with the existing
P2Rank 2.6-alpha predictions at residue and pocket-centre level.  Targets that
do not have an exact-sequence structure remain explicit structure-first tasks;
they are never silently treated as negative pocket predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
POCKET_SUMMARY = (
    ROOT
    / "outputs/chembl37_known_pocket_atlas/final_atlas/"
    "TARGET_KNOWN_POCKET_AND_P2RANK_SUMMARY_888.csv"
)
FPOCKET = ROOT / ".conda_envs/pocket_tools/bin/fpocket"
FPOCKET_ENV = ROOT / ".conda_envs/pocket_tools"
DEFAULT_OUTDIR = ROOT / "outputs/no_experimental_pocket_prediction_ch37_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).lower() in {"true", "1", "yes", "y"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scope() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE, low_memory=False)
    pocket = pd.read_csv(POCKET_SUMMARY, low_memory=False)
    if len(universe) != 888 or universe["target_chembl_id"].nunique() != 888:
        raise RuntimeError("Official target universe is not the frozen 888-target set")
    if len(pocket) != 888 or pocket["target_chembl_id"].nunique() != 888:
        raise RuntimeError("Known-pocket summary is not the frozen 888-target set")
    no_known = pocket[pocket["known_unique_pocket_count"].fillna(0).eq(0)].copy()
    if len(no_known) != 151:
        raise RuntimeError(f"Expected 151 targets without known pockets, observed {len(no_known)}")

    # Merge only fields not already frozen in the atlas table.  Joining on all
    # duplicated floating-point fields would make a harmless serialization
    # difference look like a missing target and silently drop structure paths.
    universe_columns = [
        "target_chembl_id",
        "target_name",
        "sequence",
        "sequence_length",
        "sequence_sha256",
        "target_class_l2",
        "target_class_leaf",
        "assay_lane",
        "evidence_class",
        "small_molecule_moa",
        "approved_small_molecule_moa",
        "calibration_status",
        "af_selected_accession",
        "af_pdb_path",
        "af_mean_plddt",
        "af_low_plddt_pct",
        "p2rank_file",
        "p2rank_top_rank",
        "p2rank_center_x",
        "p2rank_center_y",
        "p2rank_center_z",
        "p2rank_residue_ids",
    ]
    merged = no_known.merge(
        universe[universe_columns],
        on="target_chembl_id",
        how="left",
        validate="one_to_one",
    )
    merged["af_exact_sequence_model"] = merged["af_exact_sequence_model"].map(as_bool)
    return merged.sort_values(["target_class_l1", "gene_symbol", "target_chembl_id"])


def prepare(outdir: Path) -> None:
    scope = load_scope()
    inputs = outdir / "inputs"
    logs = outdir / "logs"
    inputs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for row in scope.to_dict(orient="records"):
        exact = as_bool(row.get("af_exact_sequence_model"))
        source_text = clean(row.get("af_pdb_path"))
        source = Path(source_text) if source_text else None
        input_path: Path | None = None
        source_hash = ""
        if exact:
            if source is None or not source.is_file():
                raise FileNotFoundError(f"Missing exact AlphaFold structure for {row['target_chembl_id']}")
            if source.name in seen_names:
                raise RuntimeError(f"Duplicate AlphaFold basename in scope: {source.name}")
            seen_names.add(source.name)
            input_path = inputs / source.name
            if input_path.exists() or input_path.is_symlink():
                if not input_path.is_symlink() or input_path.resolve() != source.resolve():
                    input_path.unlink()
            if not input_path.exists():
                input_path.symlink_to(source.resolve())
            source_hash = file_sha256(source)

        rows.append(
            {
                "target_chembl_id": row["target_chembl_id"],
                "gene_symbol": row["gene_symbol"],
                "uniprot_accession": row["uniprot_accession"],
                "target_class_l1": row["target_class_l1"],
                "assay_lane": row["assay_lane"],
                "evidence_class": row["evidence_class"],
                "sequence_length": row["sequence_length"],
                "sequence_sha256": row["sequence_sha256"],
                "af_exact_sequence_model": exact,
                "source_pdb_path": str(source.resolve()) if source else "",
                "source_pdb_sha256": source_hash,
                "fpocket_input_path": str(input_path.resolve()) if input_path else "",
                "pdb_basename": input_path.name if input_path else "",
                "fpocket_output_dir": (
                    str(inputs / f"{input_path.stem}_out") if input_path else ""
                ),
                "p2rank_status": row["p2rank_status"],
                "p2rank_tier": row["p2rank_tier"],
                "p2rank_file": clean(row.get("p2rank_file")),
            }
        )

    manifest = pd.DataFrame(rows)
    if int(manifest["af_exact_sequence_model"].sum()) != 146:
        raise RuntimeError("Expected 146 exact structures among the 151 no-pocket targets")
    manifest_path = outdir / "NO_EXPERIMENTAL_POCKET_TARGET_MANIFEST_151_V1.csv"
    manifest.to_csv(manifest_path, index=False)
    write_json(
        outdir / "NO_EXPERIMENTAL_POCKET_PREPARATION_SUMMARY_V1.json",
        {
            "created_utc": utc_now(),
            "status": "PASS",
            "targets_without_known_experimental_pocket": int(len(manifest)),
            "exact_sequence_structures_ready": int(manifest["af_exact_sequence_model"].sum()),
            "structure_first_targets": int((~manifest["af_exact_sequence_model"]).sum()),
            "fpocket_tool": str(FPOCKET.resolve()),
            "input_strategy": "output-local symlinks; source structures are not modified",
            "manifest": str(manifest_path.resolve()),
        },
    )


def run_one(task: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    target_id, pdb_basename, inputs_text, executable, force = task
    inputs = Path(inputs_text)
    stem = Path(pdb_basename).stem
    output = inputs / f"{stem}_out"
    info = output / f"{stem}_info.txt"
    if info.is_file() and not force:
        return {
            "target_chembl_id": target_id,
            "pdb_basename": pdb_basename,
            "return_code": 0,
            "status": "REUSED_COMPLETE",
            "info_exists": True,
            "stdout_tail": "",
        }
    if output.exists():
        shutil.rmtree(output)
    environment = os.environ.copy()
    environment["PATH"] = str(FPOCKET_ENV / "bin") + os.pathsep + environment.get("PATH", "")
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [executable, "-f", pdb_basename],
        cwd=inputs,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=900,
    )
    return {
        "target_chembl_id": target_id,
        "pdb_basename": pdb_basename,
        "return_code": int(completed.returncode),
        "status": "SUCCESS" if completed.returncode == 0 and info.is_file() else "FAILED",
        "info_exists": info.is_file(),
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-30:]),
    }


def run(outdir: Path, workers: int, force: bool) -> None:
    if not FPOCKET.is_file():
        raise FileNotFoundError(FPOCKET)
    manifest = pd.read_csv(outdir / "NO_EXPERIMENTAL_POCKET_TARGET_MANIFEST_151_V1.csv")
    exact = manifest[manifest["af_exact_sequence_model"].map(as_bool)].copy()
    tasks = [
        (
            str(row.target_chembl_id),
            str(row.pdb_basename),
            str((outdir / "inputs").resolve()),
            str(FPOCKET.resolve()),
            force,
        )
        for row in exact.itertuples(index=False)
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed_count % 10 == 0 or completed_count == len(futures):
                print(f"fpocket completed {completed_count}/{len(futures)}", flush=True)
    run_log = pd.DataFrame(results).sort_values("target_chembl_id")
    run_log.to_csv(outdir / "FPOCKET_NO_EXPERIMENTAL_POCKET_RUN_LOG_V1.csv", index=False)
    failed = run_log[~run_log["status"].isin(["SUCCESS", "REUSED_COMPLETE"])]
    if not failed.empty:
        raise RuntimeError(f"fpocket failed for {len(failed)} targets")


def metric_key(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(".", "")
        .replace("-", "")
        .replace(":", "")
        .replace(" ", "_")
        .replace("__", "_")
    )


def parse_info(path: Path) -> list[dict[str, Any]]:
    pockets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    if not path.is_file():
        return pockets
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Pocket ") and line.endswith(":"):
            if current:
                pockets.append(current)
            current = {"fpocket_rank": int(line.replace("Pocket", "").replace(":", "").strip())}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            current[metric_key(key)] = float(value.strip())
        except ValueError:
            current[metric_key(key)] = value.strip()
    if current:
        pockets.append(current)
    return pockets


def parse_pocket_atoms(path: Path) -> tuple[set[str], tuple[float, float, float] | None]:
    residues: set[str] = set()
    coordinates: list[tuple[float, float, float]] = []
    if not path.is_file():
        return residues, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            chain = line[21].strip() or "_"
            residue_number = int(line[22:26])
            insertion = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        residues.add(f"{chain}_{residue_number}{insertion}")
        coordinates.append((x, y, z))
    if not coordinates:
        return residues, None
    count = float(len(coordinates))
    centre = tuple(sum(point[index] for point in coordinates) / count for index in range(3))
    return residues, centre  # type: ignore[return-value]


def read_p2rank(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text) if path_text else None
    if path is None or not path.is_file():
        return []
    try:
        frame = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")
    except (pd.errors.ParserError, UnicodeDecodeError):
        return []
    frame.columns = [str(column).strip() for column in frame.columns]
    if "rank" not in frame.columns:
        return []
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame = frame[frame["rank"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        residues = {token.strip() for token in clean(row.get("residue_ids")).split() if token.strip()}
        rows.append(
            {
                "p2rank_rank": int(row["rank"]),
                "p2rank_score": pd.to_numeric(row.get("score"), errors="coerce"),
                "p2rank_probability": pd.to_numeric(row.get("probability"), errors="coerce"),
                "p2rank_center_x": pd.to_numeric(row.get("center_x"), errors="coerce"),
                "p2rank_center_y": pd.to_numeric(row.get("center_y"), errors="coerce"),
                "p2rank_center_z": pd.to_numeric(row.get("center_z"), errors="coerce"),
                "p2rank_residues": residues,
            }
        )
    return sorted(rows, key=lambda row: row["p2rank_rank"])


def fpocket_tier(pockets: list[dict[str, Any]]) -> tuple[str, str]:
    if not pockets:
        return "D_NO_FPOCKET_POCKET", "fpocket returned no pocket"
    max_druggability = max(float(row.get("druggability_score", 0) or 0) for row in pockets)
    max_volume = max(float(row.get("volume", 0) or 0) for row in pockets)
    reviewable = any(
        float(row.get("druggability_score", 0) or 0) >= 0.20
        or (
            float(row.get("volume", 0) or 0) >= 250
            and float(row.get("number_of_alpha_spheres", 0) or 0) >= 20
        )
        for row in pockets
    )
    if max_druggability >= 0.50 and max_volume >= 250:
        return "A_HIGH_DRUGGABILITY_GEOMETRY", "druggability >=0.50 and volume >=250 A^3"
    if reviewable:
        return "B_REVIEWABLE_GEOMETRY", "reviewable fpocket druggability or pocket geometry"
    return "C_WEAK_GEOMETRY", "fpocket pockets have weak geometry/druggability descriptors"


def finite_point(row: dict[str, Any], prefix: str) -> tuple[float, float, float] | None:
    values = tuple(float(row.get(f"{prefix}_{axis}", math.nan)) for axis in ("x", "y", "z"))
    return values if all(math.isfinite(value) for value in values) else None


def consensus_class(
    exact: bool,
    p2rank_tier: str,
    fpocket_status: str,
    fpocket_class: str,
    matching_site: bool,
) -> tuple[str, str, str]:
    if not exact:
        return (
            "C6_STRUCTURE_FIRST_NO_EXACT_MODEL",
            "结构优先：先获得精确序列的全长或功能域结构，再运行口袋预测",
            "NO_PREDICTION_WITHOUT_EXACT_STRUCTURE",
        )
    p2_ab = p2rank_tier.startswith(("A_", "B_"))
    fp_ab = fpocket_class.startswith(("A_", "B_"))
    if p2_ab and fp_ab and matching_site:
        return (
            "C1_DUAL_METHOD_SAME_SITE",
            "P2Rank与fpocket支持同一位点；进入多构象对接和口袋稳定性复核",
            "P2RANK_PRIMARY_FPOCKET_CONFIRMED",
        )
    if p2_ab and fp_ab:
        return (
            "C2_DUAL_METHOD_DIFFERENT_SITE",
            "两种方法均检出口袋但位点不同；保留为多口袋ensemble，不强行合并",
            "MULTI_POCKET_ENSEMBLE_REVIEW",
        )
    if p2_ab:
        return (
            "C3_P2RANK_ONLY_AB",
            "采用P2Rank探索口袋并保留单方法不确定性；优先做构象ensemble复核",
            "P2RANK_ONLY_UNCERTAIN",
        )
    if fp_ab and fpocket_status == "completed":
        return (
            "C4_FPOCKET_RESCUE",
            "fpocket提供几何口袋补救；需经结构质量和第二构象复核后方可对接",
            "FPOCKET_RESCUE_UNCERTAIN",
        )
    return (
        "C5_WEAK_OR_NO_POCKET",
        "未获得稳健口袋共识；转入功能域建模、构象采样或界面/变构位点路线",
        "NO_ROBUST_STATIC_POCKET",
    )


def parse(outdir: Path) -> None:
    scope = load_scope()
    manifest = pd.read_csv(outdir / "NO_EXPERIMENTAL_POCKET_TARGET_MANIFEST_151_V1.csv")
    run_log_path = outdir / "FPOCKET_NO_EXPERIMENTAL_POCKET_RUN_LOG_V1.csv"
    run_log = pd.read_csv(run_log_path) if run_log_path.is_file() else pd.DataFrame()
    run_map = (
        run_log.set_index("target_chembl_id").to_dict(orient="index")
        if not run_log.empty
        else {}
    )
    manifest_map = manifest.set_index("target_chembl_id").to_dict(orient="index")

    target_rows: list[dict[str, Any]] = []
    fpocket_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for source in scope.to_dict(orient="records"):
        target_id = str(source["target_chembl_id"])
        exact = as_bool(source.get("af_exact_sequence_model"))
        manifest_row = manifest_map[target_id]
        run_row = run_map.get(target_id, {})
        output_text = clean(manifest_row.get("fpocket_output_dir"))
        output = Path(output_text) if output_text else None
        stem = Path(clean(manifest_row.get("pdb_basename"))).stem
        info = output / f"{stem}_info.txt" if output else None
        completed = bool(
            exact
            and info is not None
            and info.is_file()
            and clean(run_row.get("status")) in {"SUCCESS", "REUSED_COMPLETE"}
        )
        raw_pockets = parse_info(info) if completed and info else []
        enriched_pockets: list[dict[str, Any]] = []
        for pocket in raw_pockets:
            rank = int(pocket["fpocket_rank"])
            atoms = output / "pockets" / f"pocket{rank}_atm.pdb" if output else Path("")
            residues, centre = parse_pocket_atoms(atoms)
            enriched = dict(pocket)
            enriched["fpocket_residues"] = residues
            enriched["fpocket_residue_count"] = len(residues)
            if centre:
                enriched.update(
                    {
                        "fpocket_center_x": centre[0],
                        "fpocket_center_y": centre[1],
                        "fpocket_center_z": centre[2],
                    }
                )
            enriched_pockets.append(enriched)
            public = {key: value for key, value in enriched.items() if key != "fpocket_residues"}
            public.update(
                {
                    "target_chembl_id": target_id,
                    "gene_symbol": source["gene_symbol"],
                    "fpocket_residue_ids": ";".join(sorted(residues)),
                    "fpocket_atom_file": str(atoms.resolve()) if atoms.is_file() else "",
                }
            )
            fpocket_rows.append(public)

        p2rank_pockets = read_p2rank(clean(source.get("p2rank_file")))
        best_match: dict[str, Any] | None = None
        top3_matching = False
        for p2 in p2rank_pockets:
            p2_centre = finite_point(p2, "p2rank_center")
            for fp in enriched_pockets:
                fp_centre = finite_point(fp, "fpocket_center")
                p2_residues = p2["p2rank_residues"]
                fp_residues = fp["fpocket_residues"]
                intersection = len(p2_residues & fp_residues)
                union = len(p2_residues | fp_residues)
                jaccard = intersection / union if union else 0.0
                p2_recall = intersection / len(p2_residues) if p2_residues else 0.0
                fp_recall = intersection / len(fp_residues) if fp_residues else 0.0
                distance = math.nan
                if p2_centre and fp_centre:
                    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(p2_centre, fp_centre)))
                same_site = bool(
                    jaccard >= 0.10
                    or p2_recall >= 0.25
                    or fp_recall >= 0.25
                    or (math.isfinite(distance) and distance <= 8.0)
                )
                payload = {
                    "target_chembl_id": target_id,
                    "gene_symbol": source["gene_symbol"],
                    "p2rank_rank": p2["p2rank_rank"],
                    "fpocket_rank": fp["fpocket_rank"],
                    "residue_intersection": intersection,
                    "residue_jaccard": jaccard,
                    "p2rank_residue_recall": p2_recall,
                    "fpocket_residue_recall": fp_recall,
                    "center_distance_A": distance,
                    "same_site": same_site,
                }
                match_rows.append(payload)
                if int(p2["p2rank_rank"]) <= 3 and same_site:
                    top3_matching = True
                sort_key = (
                    same_site,
                    intersection,
                    jaccard,
                    -distance if math.isfinite(distance) else -1e9,
                )
                if best_match is None or sort_key > best_match["_sort_key"]:
                    best_match = {**payload, "_sort_key": sort_key}

        fp_class, fp_reason = (
            fpocket_tier(enriched_pockets)
            if completed
            else (
                "NOT_RUN_NO_EXACT_STRUCTURE" if not exact else "FAILED_OR_MISSING",
                "no exact-sequence structure" if not exact else "fpocket output failed or missing",
            )
        )
        fp_status = "completed" if completed else ("not_run_no_exact_structure" if not exact else "failed_or_missing")
        consensus, action, policy = consensus_class(
            exact,
            clean(source.get("p2rank_tier")),
            fp_status,
            fp_class,
            top3_matching,
        )
        top_fp = (
            sorted(enriched_pockets, key=lambda row: float(row.get("score", 0) or 0), reverse=True)[0]
            if enriched_pockets
            else {}
        )
        public_match = (
            {key: value for key, value in best_match.items() if key != "_sort_key"}
            if best_match
            else {}
        )
        target_rows.append(
            {
                "target_chembl_id": target_id,
                "gene_symbol": source["gene_symbol"],
                "target_name": source["target_name"],
                "uniprot_accession": source["uniprot_accession"],
                "target_class_l1": source["target_class_l1"],
                "target_class_l2": source["target_class_l2"],
                "target_class_leaf": source["target_class_leaf"],
                "assay_lane": source["assay_lane"],
                "evidence_class": source["evidence_class"],
                "small_molecule_moa": source["small_molecule_moa"],
                "approved_small_molecule_moa": source["approved_small_molecule_moa"],
                "calibration_status": source["calibration_status"],
                "sequence_length": source["sequence_length"],
                "sequence_sha256": source["sequence_sha256"],
                "known_experimental_pocket_count": 0,
                "af_exact_sequence_model": exact,
                "af_pdb_path": clean(source.get("af_pdb_path")),
                "af_mean_plddt": source.get("af_mean_plddt"),
                "p2rank_status": source["p2rank_status"],
                "p2rank_tier": source["p2rank_tier"],
                "p2rank_top_score": source["p2rank_top_score"],
                "p2rank_top_probability": source["p2rank_top_probability"],
                "p2rank_pocket_count": len(p2rank_pockets),
                "fpocket_status": fp_status,
                "fpocket_pocket_count": len(enriched_pockets),
                "fpocket_tier": fp_class,
                "fpocket_tier_reason": fp_reason,
                "fpocket_top_score": top_fp.get("score"),
                "fpocket_top_druggability_score": top_fp.get("druggability_score"),
                "fpocket_top_volume_A3": top_fp.get("volume"),
                "p2rank_top3_matches_any_fpocket": top3_matching,
                "best_method_match_p2rank_rank": public_match.get("p2rank_rank"),
                "best_method_match_fpocket_rank": public_match.get("fpocket_rank"),
                "best_method_match_residue_jaccard": public_match.get("residue_jaccard"),
                "best_method_match_center_distance_A": public_match.get("center_distance_A"),
                "pocket_consensus_class": consensus,
                "primary_prediction_policy": policy,
                "next_compute_action_zh": action,
                "predicted_pocket_identified": bool(p2rank_pockets or enriched_pockets),
            }
        )

    targets = pd.DataFrame(target_rows).sort_values(
        ["pocket_consensus_class", "target_class_l1", "gene_symbol"]
    )
    fpockets = pd.DataFrame(fpocket_rows)
    matches = pd.DataFrame(match_rows)
    target_output = outdir / "NO_EXPERIMENTAL_POCKET_CONSENSUS_TARGETS_151_V1.csv"
    fpocket_output = outdir / "FPOCKET_POCKET_CANDIDATES_NO_EXPERIMENTAL_146_V1.csv.gz"
    match_output = outdir / "P2RANK_FPOCKET_SITE_COMPARISON_NO_EXPERIMENTAL_V1.csv.gz"
    targets.to_csv(target_output, index=False)
    fpockets.to_csv(fpocket_output, index=False, compression="gzip")
    matches.to_csv(match_output, index=False, compression="gzip")

    summary = {
        "created_utc": utc_now(),
        "status": "PASS" if targets["fpocket_status"].eq("completed").sum() == 146 else "FAIL",
        "targets_without_known_experimental_pocket": int(len(targets)),
        "targets_with_exact_sequence_structure": int(targets["af_exact_sequence_model"].sum()),
        "fpocket_completed_targets": int(targets["fpocket_status"].eq("completed").sum()),
        "p2rank_completed_with_pocket_targets": int(targets["p2rank_status"].eq("completed").sum()),
        "targets_with_any_predicted_pocket": int(targets["predicted_pocket_identified"].sum()),
        "structure_first_targets": int(targets["pocket_consensus_class"].eq("C6_STRUCTURE_FIRST_NO_EXACT_MODEL").sum()),
        "consensus_class_counts": targets["pocket_consensus_class"].value_counts().to_dict(),
        "target_class_counts": targets["target_class_l1"].value_counts().to_dict(),
        "outputs": {
            "target_consensus": str(target_output.resolve()),
            "fpocket_candidates": str(fpocket_output.resolve()),
            "method_site_comparison": str(match_output.resolve()),
        },
        "interpretation_boundaries": [
            "Predicted pockets are computational hypotheses, not experimental pocket evidence.",
            "A negative static-structure pocket call does not exclude cryptic, induced, interface, or state-specific pockets.",
            "Targets without an exact-sequence structure remain structure-first tasks and are not counted as negative predictions.",
        ],
    }
    write_json(outdir / "NO_EXPERIMENTAL_POCKET_CONSENSUS_SUMMARY_V1.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("No-experimental-pocket consensus package is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "parse", "all"])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    if args.mode in {"prepare", "all"}:
        prepare(outdir)
    if args.mode in {"run", "all"}:
        run(outdir, args.workers, args.force)
    if args.mode in {"parse", "all"}:
        parse(outdir)


if __name__ == "__main__":
    main()
