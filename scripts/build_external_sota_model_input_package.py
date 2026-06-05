from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CORE_COLUMNS = [
    "direction",
    "pairId",
    "drugId",
    "drug",
    "target",
    "protein",
    "proteinName",
    "canonical_smiles",
    "confidenceSdfPath",
    "receptorPdbPath",
    "sotaVinaConsensusRankGlobal",
    "sotaContextRankGlobal",
    "sotaReadyRankGlobal",
    "finalRankGlobal",
    "finalPriorityScore",
    "sotaContextScore",
    "sotaReadyScore",
    "sotaVinaConsensusScore",
    "knownDrugTargetPair",
    "noveltyClass",
    "structureConfidenceTier",
    "targetDruggabilityTier",
    "poseQualityTier",
    "standardPoseValidationTier",
    "vinaStatus",
    "pdbqtReady",
]


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


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def truthy(value: Any) -> bool:
    text = clean_str(value).lower()
    return text in {"1", "true", "t", "yes", "y", "ok"}


def binary_status(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"available": bool(path), "detail": path or ""}


def import_status(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "detail": "module_not_found"}
    return {"available": True, "detail": str(spec.origin or "")}


def resolve_path(value: Any, root: Path) -> Path | None:
    text = clean_str(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path


def path_exists(value: Any, root: Path) -> bool:
    path = resolve_path(value, root)
    return bool(path and path.exists())


def normalize_rank(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def choose_rank(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any():
            return column
    return "_sourceOrder"


def short_text(value: Any, max_len: int = 120) -> str:
    text = clean_str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def safe_identifier(*parts: Any) -> str:
    text = "__".join(clean_str(part) for part in parts if clean_str(part))
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text.strip("_") or "item"


def file_map_by_pair(root: Path) -> dict[str, str]:
    ligand_root = root / "outputs/sota_validation/vina_consensus_rescoring/pdbqt_cache/ligands"
    mapping: dict[str, str] = {}
    if not ligand_root.exists():
        return mapping
    for path in sorted(ligand_root.glob("*.pdbqt")):
        stem = path.stem
        parts = stem.split("__")
        if len(parts) < 2:
            continue
        target_and_hash = parts[1].split("_")
        if not target_and_hash:
            continue
        pair_id = f"{parts[0]}__{target_and_hash[0]}"
        mapping.setdefault(pair_id, str(path))
    return mapping


def file_map_by_receptor(root: Path) -> dict[str, str]:
    receptor_root = root / "outputs/sota_validation/vina_consensus_rescoring/pdbqt_cache/receptors"
    mapping: dict[str, str] = {}
    if not receptor_root.exists():
        return mapping
    for path in sorted(receptor_root.glob("*.pdbqt")):
        stem = path.stem
        if "_" in stem:
            stem = "_".join(stem.split("_")[:-1])
        mapping.setdefault(stem, str(path))
    return mapping


def receptor_pdbqt_for(path_value: Any, receptor_map: dict[str, str]) -> str:
    text = clean_str(path_value)
    if not text:
        return ""
    stem = Path(text).stem
    if stem in receptor_map:
        return receptor_map[stem]
    for key, value in receptor_map.items():
        if stem.startswith(key) or key.startswith(stem):
            return value
    return ""


def read_candidates(path: Path, root: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["_sourceOrder"] = range(1, len(df) + 1)
    for column in [
        "sotaVinaConsensusRankGlobal",
        "sotaContextRankGlobal",
        "sotaReadyRankGlobal",
        "finalRankGlobal",
        "finalPriorityScore",
        "sotaContextScore",
        "sotaReadyScore",
        "sotaVinaConsensusScore",
    ]:
        normalize_rank(df, column)

    for column in CORE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df["canonicalSmiles"] = df["canonical_smiles"].map(clean_str)
    df["smilesAvailable"] = df["canonicalSmiles"].ne("")
    df["confidenceSdfPath"] = df["confidenceSdfPath"].map(clean_str)
    df["receptorPdbPath"] = df["receptorPdbPath"].map(clean_str)
    df["confidenceSdfExists"] = df["confidenceSdfPath"].map(lambda value: path_exists(value, root))
    df["receptorPdbExists"] = df["receptorPdbPath"].map(lambda value: path_exists(value, root))
    df["knownDrugTargetPairFlag"] = df["knownDrugTargetPair"].map(truthy)
    return df


def read_proteins(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["protein_id", "proteinSequence", "proteinSequenceLength", "libraryGeneName", "libraryProteinName"])
    columns = ["protein_id", "sequence", "gene_name", "protein_name", "length", "sequence_key"]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in columns if column in available]
    proteins = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "protein_id" not in proteins.columns:
        return pd.DataFrame(columns=["protein_id", "proteinSequence", "proteinSequenceLength", "libraryGeneName", "libraryProteinName"])
    if "sequence" not in proteins.columns:
        proteins["sequence"] = ""
    proteins = proteins.drop_duplicates("protein_id").copy()
    proteins["proteinSequence"] = proteins["sequence"].map(lambda value: re.sub(r"\s+", "", clean_str(value)))
    proteins["proteinSequenceLength"] = proteins["proteinSequence"].str.len()
    proteins["libraryGeneName"] = proteins["gene_name"].map(clean_str) if "gene_name" in proteins.columns else ""
    proteins["libraryProteinName"] = proteins["protein_name"].map(clean_str) if "protein_name" in proteins.columns else ""
    keep = ["protein_id", "proteinSequence", "proteinSequenceLength", "libraryGeneName", "libraryProteinName"]
    if "sequence_key" in proteins.columns:
        proteins["sequenceKey"] = proteins["sequence_key"].map(clean_str)
        keep.append("sequenceKey")
    return proteins[keep]


def attach_protein_sequences(candidates: pd.DataFrame, protein_library: pd.DataFrame) -> pd.DataFrame:
    df = candidates.merge(protein_library, left_on="protein", right_on="protein_id", how="left")
    df["proteinSequence"] = df["proteinSequence"].map(clean_str)
    df["sequenceAvailable"] = df["proteinSequence"].ne("")
    df["proteinSequenceLength"] = pd.to_numeric(df["proteinSequenceLength"], errors="coerce")
    if "sequenceKey" not in df.columns:
        df["sequenceKey"] = ""
    return df


def add_pdbqt_paths(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    pair_map = file_map_by_pair(root)
    receptor_map = file_map_by_receptor(root)
    df["ligandPdbqtPath"] = df["pairId"].map(lambda value: pair_map.get(clean_str(value), ""))
    df["ligandPdbqtExists"] = df["ligandPdbqtPath"].map(lambda value: path_exists(value, root))
    df["receptorPdbqtPath"] = df["receptorPdbPath"].map(lambda value: receptor_pdbqt_for(value, receptor_map))
    df["receptorPdbqtExists"] = df["receptorPdbqtPath"].map(lambda value: path_exists(value, root))
    return df


def missing_reason(row: pd.Series, required: list[str]) -> str:
    reasons: list[str] = []
    for item in required:
        if item == "smiles" and not bool(row.get("smilesAvailable")):
            reasons.append("missing_smiles")
        elif item == "sequence" and not bool(row.get("sequenceAvailable")):
            reasons.append("missing_protein_sequence")
        elif item == "sdf" and not bool(row.get("confidenceSdfExists")):
            reasons.append("missing_diffdock_ligand_sdf")
        elif item == "receptor_pdb" and not bool(row.get("receptorPdbExists")):
            reasons.append("missing_receptor_pdb")
    return "ready" if not reasons else ";".join(reasons)


def select_top(df: pd.DataFrame, rank_column: str, top_n: int, dedupe_pair: bool = True) -> pd.DataFrame:
    selected = df.sort_values([rank_column, "_sourceOrder"], ascending=[True, True], na_position="last").copy()
    if dedupe_pair and "pairId" in selected.columns:
        selected = selected.drop_duplicates("pairId", keep="first")
    return selected.head(top_n).copy()


def select_diverse(
    df: pd.DataFrame,
    rank_column: str,
    top_n: int,
    max_per_drug: int = 3,
    max_per_target: int = 3,
    max_per_direction: int = 15,
) -> pd.DataFrame:
    ranked = df.sort_values([rank_column, "_sourceOrder"], ascending=[True, True], na_position="last")
    ranked = ranked.drop_duplicates("pairId", keep="first")
    selected_indices: list[int] = []
    drug_counts: defaultdict[str, int] = defaultdict(int)
    target_counts: defaultdict[str, int] = defaultdict(int)
    direction_counts: defaultdict[str, int] = defaultdict(int)

    for index, row in ranked.iterrows():
        drug = clean_str(row.get("drugId")) or clean_str(row.get("drug"))
        target = clean_str(row.get("protein")) or clean_str(row.get("target"))
        direction = clean_str(row.get("direction"))
        if drug_counts[drug] >= max_per_drug:
            continue
        if target_counts[target] >= max_per_target:
            continue
        if direction_counts[direction] >= max_per_direction:
            continue
        selected_indices.append(index)
        drug_counts[drug] += 1
        target_counts[target] += 1
        direction_counts[direction] += 1
        if len(selected_indices) >= top_n:
            break

    if len(selected_indices) < top_n:
        seen = set(selected_indices)
        for index in ranked.index:
            if index in seen:
                continue
            selected_indices.append(index)
            if len(selected_indices) >= top_n:
                break

    return ranked.loc[selected_indices].copy()


def compact_columns(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    base = [
        "externalQueueRank",
        "externalQueue",
        "externalModelInputReady",
        "externalModelMissingInputReasons",
        "selectionRankColumn",
        "selectionRankValue",
    ]
    columns = [column for column in base + CORE_COLUMNS + extra if column in df.columns]
    return df[columns].copy()


def prepare_gnina_queue(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, str]:
    rank_column = choose_rank(df, ["sotaVinaConsensusRankGlobal", "sotaStandardStructureRankGlobal", "sotaPoseQualityRankGlobal", "finalRankGlobal"])
    queue = select_top(df, rank_column, top_n=top_n, dedupe_pair=True)
    queue["externalQueue"] = "gnina_cnn_rescoring_top_structural"
    queue["selectionRankColumn"] = rank_column
    queue["selectionRankValue"] = queue[rank_column]
    queue["externalModelMissingInputReasons"] = queue.apply(lambda row: missing_reason(row, ["sdf", "receptor_pdb"]), axis=1)
    queue["externalModelInputReady"] = queue["externalModelMissingInputReasons"].eq("ready")
    queue["externalQueueRank"] = range(1, len(queue) + 1)
    extra = [
        "confidenceSdfExists",
        "receptorPdbExists",
        "ligandPdbqtPath",
        "ligandPdbqtExists",
        "receptorPdbqtPath",
        "receptorPdbqtExists",
        "vinaScoreKcalMol",
        "vinaOptimizedScoreKcalMol",
        "vinaConsensusTier",
    ]
    return compact_columns(queue, extra), rank_column


def prepare_boltz_chai_queue(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, str]:
    rank_column = choose_rank(df, ["sotaContextRankGlobal", "sotaReadyRankGlobal", "finalRankGlobal"])
    queue = select_diverse(df, rank_column, top_n=top_n)
    queue["externalQueue"] = "boltz_chai_complex_spotcheck_diverse"
    queue["selectionRankColumn"] = rank_column
    queue["selectionRankValue"] = queue[rank_column]
    queue["externalModelMissingInputReasons"] = queue.apply(lambda row: missing_reason(row, ["smiles", "sequence"]), axis=1)
    queue["externalModelInputReady"] = queue["externalModelMissingInputReasons"].eq("ready")
    queue["externalQueueRank"] = range(1, len(queue) + 1)
    extra = [
        "canonicalSmiles",
        "proteinSequence",
        "proteinSequenceLength",
        "sequenceKey",
        "libraryGeneName",
        "libraryProteinName",
        "confidenceSdfExists",
        "receptorPdbExists",
    ]
    return compact_columns(queue, extra), rank_column


def prepare_dti_queue(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, str]:
    rank_column = choose_rank(df, ["sotaReadyRankGlobal", "sotaContextRankGlobal", "finalRankGlobal"])
    queue = select_top(df, rank_column, top_n=top_n, dedupe_pair=True)
    queue["externalQueue"] = "independent_dti_ensemble_top_pairs"
    queue["selectionRankColumn"] = rank_column
    queue["selectionRankValue"] = queue[rank_column]
    queue["externalModelMissingInputReasons"] = queue.apply(lambda row: missing_reason(row, ["smiles", "sequence"]), axis=1)
    queue["externalModelInputReady"] = queue["externalModelMissingInputReasons"].eq("ready")
    queue["externalQueueRank"] = range(1, len(queue) + 1)
    extra = [
        "canonicalSmiles",
        "proteinSequence",
        "proteinSequenceLength",
        "sequenceKey",
        "libraryGeneName",
        "libraryProteinName",
        "confidenceSdfExists",
        "receptorPdbExists",
    ]
    return compact_columns(queue, extra), rank_column


def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if df.empty or "externalModelMissingInputReasons" not in df.columns:
        return {}
    for value in df["externalModelMissingInputReasons"]:
        text = clean_str(value)
        if not text:
            continue
        if text == "ready":
            counter["ready"] += 1
            continue
        for reason in text.split(";"):
            counter[reason] += 1
    return dict(sorted(counter.items()))


def queue_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "inputReadyRows": int(df["externalModelInputReady"].fillna(False).astype(bool).sum()) if "externalModelInputReady" in df.columns else 0,
        "uniquePairs": int(df["pairId"].nunique()) if "pairId" in df.columns else 0,
        "uniqueDrugs": int(df["drugId"].nunique()) if "drugId" in df.columns else 0,
        "uniqueProteins": int(df["protein"].nunique()) if "protein" in df.columns else 0,
        "uniqueDirections": int(df["direction"].nunique()) if "direction" in df.columns else 0,
        "missingInputReasonCounts": missing_counts(df),
    }


def write_jsonl(path: Path, df: pd.DataFrame, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            record = {field: json_safe(row.get(field)) for field in fields if field in df.columns}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def wrap_sequence(sequence: str, width: int = 80) -> list[str]:
    return [sequence[index : index + width] for index in range(0, len(sequence), width)]


def write_fasta(path: Path, df: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    proteins = (
        df[df["proteinSequence"].map(clean_str).ne("")]
        .sort_values(["protein", "externalQueueRank"] if "externalQueueRank" in df.columns else ["protein"])
        .drop_duplicates("protein", keep="first")
    )
    with path.open("w", encoding="utf-8") as handle:
        for _, row in proteins.iterrows():
            accession = safe_identifier(row.get("protein"))
            gene = safe_identifier(row.get("target") or row.get("libraryGeneName"))
            name = safe_identifier(short_text(row.get("proteinName") or row.get("libraryProteinName"), 60))
            handle.write(f">{accession}|{gene}|{name}\n")
            for line in wrap_sequence(clean_str(row.get("proteinSequence"))):
                handle.write(line + "\n")
    return int(len(proteins))


def write_smi(path: Path, df: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    drugs = (
        df[df["canonicalSmiles"].map(clean_str).ne("")]
        .sort_values(["drugId", "externalQueueRank"] if "externalQueueRank" in df.columns else ["drugId"])
        .drop_duplicates("drugId", keep="first")
    )
    with path.open("w", encoding="utf-8") as handle:
        for _, row in drugs.iterrows():
            identifier = safe_identifier(row.get("drugId"), row.get("drug"))
            handle.write(f"{clean_str(row.get('canonicalSmiles'))}\t{identifier}\n")
    return int(len(drugs))


def build_package(root: Path, candidate_path: Path, protein_path: Path, out_dir: Path, top_gnina: int, top_boltz: int, top_dti: int) -> dict[str, Any]:
    candidates = read_candidates(candidate_path, root)
    proteins = read_proteins(protein_path)
    candidates = attach_protein_sequences(candidates, proteins)
    candidates = add_pdbqt_paths(candidates, root)

    gnina_queue, gnina_rank = prepare_gnina_queue(candidates, top_gnina)
    boltz_queue, boltz_rank = prepare_boltz_chai_queue(candidates, top_boltz)
    dti_queue, dti_rank = prepare_dti_queue(candidates, top_dti)

    out_dir.mkdir(parents=True, exist_ok=True)
    gnina_path = out_dir / "gnina_top100_rescoring_queue.csv"
    boltz_path = out_dir / "boltz_chai_top50_complex_queue.csv"
    dti_path = out_dir / "independent_dti_top1000_queue.csv"
    union_path = out_dir / "external_sota_model_input_queue_union.csv"
    boltz_jsonl_path = out_dir / "boltz_chai_top50_complex_inputs.jsonl"
    dti_jsonl_path = out_dir / "independent_dti_top1000_inputs.jsonl"
    smi_path = out_dir / "top_candidate_ligands.smi"
    fasta_path = out_dir / "top_candidate_proteins.fasta"

    gnina_queue.to_csv(gnina_path, index=False)
    boltz_queue.to_csv(boltz_path, index=False)
    dti_queue.to_csv(dti_path, index=False)
    union = pd.concat([gnina_queue, boltz_queue, dti_queue], ignore_index=True, sort=False)
    union.to_csv(union_path, index=False)
    write_jsonl(
        boltz_jsonl_path,
        boltz_queue,
        [
            "externalQueueRank",
            "pairId",
            "direction",
            "drugId",
            "drug",
            "protein",
            "target",
            "proteinName",
            "canonicalSmiles",
            "proteinSequence",
            "proteinSequenceLength",
            "receptorPdbPath",
            "confidenceSdfPath",
            "selectionRankColumn",
            "selectionRankValue",
            "externalModelInputReady",
            "externalModelMissingInputReasons",
        ],
    )
    write_jsonl(
        dti_jsonl_path,
        dti_queue,
        [
            "externalQueueRank",
            "pairId",
            "direction",
            "drugId",
            "drug",
            "protein",
            "target",
            "proteinName",
            "canonicalSmiles",
            "proteinSequence",
            "proteinSequenceLength",
            "selectionRankColumn",
            "selectionRankValue",
            "externalModelInputReady",
            "externalModelMissingInputReasons",
        ],
    )
    ligand_count = write_smi(smi_path, union)
    protein_count = write_fasta(fasta_path, union)

    coverage = {
        "candidateRows": int(len(candidates)),
        "uniquePairs": int(candidates["pairId"].nunique()),
        "uniqueDrugs": int(candidates["drugId"].nunique()),
        "uniqueProteins": int(candidates["protein"].nunique()),
        "canonicalSmilesRows": int(candidates["smilesAvailable"].sum()),
        "sequenceMappedRows": int(candidates["sequenceAvailable"].sum()),
        "confidenceSdfExistingRows": int(candidates["confidenceSdfExists"].sum()),
        "receptorPdbExistingRows": int(candidates["receptorPdbExists"].sum()),
        "ligandPdbqtExistingRows": int(candidates["ligandPdbqtExists"].sum()),
        "receptorPdbqtExistingRows": int(candidates["receptorPdbqtExists"].sum()),
    }
    summary = {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "Executable input queues for external SOTA model extensions beyond the completed local pipeline.",
        "sourceCandidateMatrix": str(candidate_path),
        "sourceProteinLibrary": str(protein_path),
        "coverage": coverage,
        "selectionPolicy": {
            "gninaTop100": f"Top {top_gnina} unique drug-protein pairs by {gnina_rank}; requires DiffDock ligand SDF plus receptor PDB.",
            "boltzChaiTop50": f"Top {top_boltz} diverse drug-protein pairs by {boltz_rank}; caps early selection by drug, target, and disease direction; requires SMILES plus protein sequence.",
            "independentDtiTop1000": f"Top {top_dti} unique drug-protein pairs by {dti_rank}; requires SMILES plus protein sequence.",
        },
        "queues": {
            "gninaTop100": queue_summary(gnina_queue),
            "boltzChaiTop50": queue_summary(boltz_queue),
            "independentDtiTop1000": queue_summary(dti_queue),
        },
        "environment": {
            "binaries": {name: binary_status(name) for name in ["gnina", "smina", "vina", "obabel"]},
            "modules": {name: import_status(name) for name in ["boltz", "chai_lab", "dgl", "dgllife", "torch", "rdkit"]},
        },
        "sharedInputAssets": {
            "topCandidateLigandsSmi": str(smi_path),
            "topCandidateLigandCount": ligand_count,
            "topCandidateProteinsFasta": str(fasta_path),
            "topCandidateProteinCount": protein_count,
        },
        "artifacts": {
            "gninaQueueCsv": str(gnina_path),
            "boltzChaiQueueCsv": str(boltz_path),
            "independentDtiQueueCsv": str(dti_path),
            "queueUnionCsv": str(union_path),
            "boltzChaiJsonl": str(boltz_jsonl_path),
            "independentDtiJsonl": str(dti_jsonl_path),
            "ligandsSmi": str(smi_path),
            "proteinsFasta": str(fasta_path),
        },
        "interpretation": [
            "The queues are input-ready artifacts, not completed GNINA/Boltz/Chai/DrugBAN/DeepDTA scores.",
            "External model execution is now gated mainly by binaries, containers, dependencies, or validated checkpoints rather than by candidate data preparation.",
            "The independent DTI queue is de-duplicated by drug-protein pair because sequence/SMILES DTI models are direction-agnostic.",
        ],
    }
    write_json(out_dir / "external_sota_model_input_summary.json", summary)
    (out_dir / "EXTERNAL_SOTA_MODEL_INPUT_PACKAGE.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(summary: dict[str, Any]) -> str:
    queues = summary["queues"]
    coverage = summary["coverage"]
    env = summary["environment"]
    lines = [
        "# External SOTA Model Input Package",
        "",
        f"Generated: {summary['createdUtc']}",
        "",
        "This package converts the current BioMaster priority matrix into executable queues for external SOTA-adjacent model layers. It does not claim that those external model scores have already been computed.",
        "",
        "## Source Coverage",
        "",
        f"- Candidate rows: {coverage['candidateRows']}",
        f"- Unique drug-protein pairs: {coverage['uniquePairs']}",
        f"- Unique drugs: {coverage['uniqueDrugs']}",
        f"- Unique proteins: {coverage['uniqueProteins']}",
        f"- Rows with canonical SMILES: {coverage['canonicalSmilesRows']}",
        f"- Rows with mapped protein sequence: {coverage['sequenceMappedRows']}",
        f"- Rows with existing DiffDock ligand SDF: {coverage['confidenceSdfExistingRows']}",
        f"- Rows with existing receptor PDB: {coverage['receptorPdbExistingRows']}",
        f"- Rows with cached ligand/receptor PDBQT: {coverage['ligandPdbqtExistingRows']}/{coverage['receptorPdbqtExistingRows']}",
        "",
        "## Queues",
        "",
    ]
    for key, label in [
        ("gninaTop100", "GNINA CNN structural rescoring"),
        ("boltzChaiTop50", "Boltz/Chai complex spot-check"),
        ("independentDtiTop1000", "Independent DTI ensemble"),
    ]:
        item = queues[key]
        lines.extend(
            [
                f"### {label}",
                f"- Rows: {item['rows']}",
                f"- Input-ready rows: {item['inputReadyRows']}",
                f"- Unique drugs/proteins/directions: {item['uniqueDrugs']}/{item['uniqueProteins']}/{item['uniqueDirections']}",
                f"- Missing-input counts: {item['missingInputReasonCounts']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Environment Gates",
            "",
            f"- GNINA binary: {env['binaries']['gnina']}",
            f"- smina binary: {env['binaries']['smina']}",
            f"- Boltz module: {env['modules']['boltz']}",
            f"- Chai module: {env['modules']['chai_lab']}",
            f"- DGL/DGLLife modules: {env['modules']['dgl']} / {env['modules']['dgllife']}",
            "",
            "## Artifacts",
            "",
        ]
    )
    for label, path in summary["artifacts"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Next Execution Steps",
            "",
            "- Run GNINA or smina CNN rescoring on `gnina_top100_rescoring_queue.csv` after installing a binary or container.",
            "- Run Boltz/Chai/AF3-style complex spot-checks on `boltz_chai_top50_complex_inputs.jsonl` after selecting a model environment.",
            "- Run DrugBAN, DeepDTA, or another validated independent DTI ensemble on `independent_dti_top1000_inputs.jsonl` after installing dependencies and checkpoints.",
            "- Treat missing external scores as an engineering setup gap, not negative biological evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external SOTA model input queues from the final BioMaster priority matrix.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--candidate-matrix",
        default="outputs/sota_validation/final_prioritization/final_priority_vina_consensus_matrix.csv",
    )
    parser.add_argument(
        "--protein-library",
        default="outputs/druggable_proteome/protein_library_druggable_chembl.csv",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/external_sota_model_inputs")
    parser.add_argument("--top-gnina", type=int, default=100)
    parser.add_argument("--top-boltz", type=int, default=50)
    parser.add_argument("--top-dti", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidate_path = root / args.candidate_matrix
    protein_path = root / args.protein_library
    summary = build_package(
        root=root,
        candidate_path=candidate_path,
        protein_path=protein_path,
        out_dir=root / args.out_dir,
        top_gnina=args.top_gnina,
        top_boltz=args.top_boltz,
        top_dti=args.top_dti,
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
