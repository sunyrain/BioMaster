"""Frozen, label-free P2Rank regions matched to full canonical sequences.

The cache contains predicted pockets only. Experimental ligand contact sets,
activity labels, docking scores and query compounds are never consulted.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POCKET_STORE = ROOT / "outputs/biomaster_v3_20260905/local_pockets/POCKET_REGIONS_V3.json"
DEFAULT_POCKET_MASTER = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
POCKET_CACHE_VERSION = "LABEL_FREE_P2RANK_REGIONS_V3_1"
THREE_TO_ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL SEC PYL MSE".split(),
    "ARNDCQEGHILKMFPSTWYVUOM",
))


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256("".join(sequence.split()).upper().encode()).hexdigest()


def parse_ca_residues(path: str | Path) -> dict[tuple[str, int, str], str]:
    """Use explicit chain, PDB number and insertion code; no row-position map."""
    result: dict[tuple[str, int, str], str] = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[16:17] not in {" ", "A"}:
                continue
            key = (line[21:22].strip(), int(line[22:26]), line[26:27].strip())
            code = THREE_TO_ONE.get(line[17:20].strip(), "X")
            if key in result and result[key] != code:
                raise ValueError(f"ambiguous CA residue: {key}")
            result[key] = code
    return result


def validate_candidate(candidate: dict[str, Any], sequence: str,
                       ca_residues: dict[tuple[str, int, str], str],
                       *, min_probability: float = 0.2, min_residues: int = 5) -> tuple[dict[str, Any] | None, str]:
    """Map a raw P2Rank row to zero-based canonical indices, fail the whole site."""
    probability = float(candidate.get("probability", float("nan")))
    if not min_probability <= probability <= 1:
        return None, "probability_below_threshold_or_invalid"
    records = str(candidate.get("residue_ids", "")).strip().split()
    indices = set()
    for record in records:
        match = re.fullmatch(r"([^_\s]+)_(\d+)([A-Za-z]?)", record)
        if match is None:
            return None, "invalid_residue_identifier"
        chain, number, insertion = match[1], int(match[2]), match[3]
        if chain != "A" or insertion:
            return None, "noncanonical_chain_or_insertion_code"
        index = number - 1
        if not 0 <= index < len(sequence):
            return None, "residue_outside_canonical_sequence"
        if ca_residues.get((chain, number, insertion)) != sequence[index]:
            return None, "residue_CA_missing_or_AA_mismatch"
        indices.add(index)
    if len(indices) < min_residues:
        return None, "insufficient_residues"
    score = float(candidate.get("score", 0.0))
    if not float("-inf") < score < float("inf"):
        return None, "invalid_pocket_score"
    return {"pocket_id": str(candidate.get("name", "")).strip(), "rank": int(candidate.get("rank", 0)),
            "probability": probability, "score": score, "residue_indices": sorted(indices),
            "residue_count": len(indices), "source_kind": "P2RANK_PREDICTED_POCKET"}, ""


def select_diverse_pockets(candidates: Iterable[dict[str, Any]], *, max_pockets: int = 3,
                           jaccard_threshold: float = 0.5) -> list[dict[str, Any]]:
    if max_pockets < 1 or not 0 < jaccard_threshold <= 1:
        raise ValueError("invalid pocket-selection limits")
    ordered = sorted(candidates, key=lambda x: (-x["probability"], -x.get("score", 0), x.get("rank", 0), x.get("pocket_id", ""), tuple(x["residue_indices"])))
    selected, sets = [], []
    for item in ordered:
        indices = frozenset(item["residue_indices"])
        if not indices:
            continue
        if any(len(indices & old) / len(indices | old) >= jaccard_threshold for old in sets):
            continue
        selected.append(item)
        sets.append(indices)
        if len(selected) == max_pockets:
            break
    return selected


def validate_pocket_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("format") != POCKET_CACHE_VERSION or payload.get("label_dependency") != "NONE":
        raise ValueError("unsupported or label-dependent pocket cache")
    for digest, item in payload["targets"].items():
        if len(digest) != 64 or item.get("sequence_sha256") != digest:
            raise ValueError("pocket sequence identity mismatch")
        length = int(item["sequence_length"])
        for pocket in item["pockets"]:
            indices = pocket["residue_indices"]
            if indices != sorted(set(indices)) or not indices or not all(isinstance(i, int) and 0 <= i < length for i in indices):
                raise ValueError("invalid canonical residue indices in pocket cache")
            if pocket.get("source_kind") != "P2RANK_PREDICTED_POCKET":
                raise ValueError("only predicted pockets are admitted in the V3 cache")
    return payload


def load_pocket_store(path: str | Path) -> dict[str, Any]:
    return validate_pocket_payload(json.loads(Path(path).read_text()))


def build_pocket_store(master_path: str | Path, target_index_paths: list[str | Path],
                       *, min_probability: float = 0.2, max_pockets: int = 3,
                       jaccard_threshold: float = 0.5, min_residues: int = 5) -> dict[str, Any]:
    # Import here to keep LocalFeatureStore's optional pocket import acyclic.
    from biomaster.odti_local_features_v3 import target_maps
    indices = pd.concat([pd.read_csv(path, low_memory=False) for path in target_index_paths], ignore_index=True)
    target_ids, _, lengths = target_maps(indices)
    masters = pd.read_csv(master_path, low_memory=False)
    current = set(target_ids.values())
    source_manifest = {"master": file_identity(master_path),
                       "target_indices": [file_identity(path) for path in target_index_paths]}
    targets: dict[str, Any] = {}
    rejected: Counter[str] = Counter()
    source_files = {}
    candidate_count = 0
    valid_count = 0
    for row in masters.itertuples(index=False):
        digest, sequence = str(row.sequence_sha256), str(row.sequence)
        if digest not in current:
            continue
        if sequence_hash(sequence) != digest:
            raise ValueError("master sequence SHA256 mismatch")
        if not bool(row.af_exact_sequence_model):
            rejected["target_without_exact_AF"] += 1
            continue
        receptor = Path(str(row.af_pdb_path))
        prediction = Path(str(row.p2rank_file))
        if not receptor.is_file() or not prediction.is_file():
            rejected["target_source_file_missing"] += 1
            continue
        residues = parse_ca_residues(receptor)
        predicted = pd.read_csv(prediction)
        predicted.columns = predicted.columns.str.strip()
        candidates = []
        for candidate in predicted.to_dict("records"):
            candidate_count += 1
            valid, reason = validate_candidate(candidate, sequence, residues, min_probability=min_probability, min_residues=min_residues)
            if valid is None:
                rejected[reason] += 1
            else:
                candidates.append(valid)
                valid_count += 1
        selected = select_diverse_pockets(candidates, max_pockets=max_pockets, jaccard_threshold=jaccard_threshold)
        if not selected:
            continue
        if digest in targets:
            raise ValueError("ambiguous duplicate canonical target in pocket master")
        sources = {"receptor": file_identity(receptor), "p2rank_predictions": file_identity(prediction)}
        for source in sources.values():
            source_files[source["sha256"]] = source
        targets[digest] = {"sequence_sha256": digest, "sequence_length": len(sequence),
                           "target_chembl_id": row.target_chembl_id, "uniprot_accession": row.uniprot_accession,
                           "mapping": "CHAIN_A_CANONICAL_NUMBER_AND_RESIDUE_AA_EXACT", "sources": sources,
                           "pockets": selected}
    missing = sorted(i for i, digest in target_ids.items() if digest not in targets)
    report = {"indexed_targets": len(target_ids), "targets_with_selected_pockets": len(targets),
              "targets_with_at_least_two_pockets": sum(len(x["pockets"]) >= 2 for x in targets.values()),
              "selected_pocket_count": sum(len(x["pockets"]) for x in targets.values()),
              "candidate_rows_examined": candidate_count, "candidate_rows_valid_before_deduplication": valid_count,
              "sequence_region_fallback_target_count": len(missing), "sequence_region_fallback_target_ids": missing,
              "rejection_counts": dict(rejected)}
    payload = {"format": POCKET_CACHE_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
               "label_dependency": "NONE", "coordinate_indexing": "zero_based_canonical_residue_indices",
               "selection": {"min_probability": min_probability, "max_pockets": max_pockets,
                             "jaccard_threshold": jaccard_threshold, "min_residues": min_residues,
                             "order": "probability_desc_score_desc_rank_asc", "full_sequence_region": "always retained at runtime"},
               "provenance": source_manifest, "source_files": source_files, "coverage": report, "targets": targets}
    return validate_pocket_payload(payload)
