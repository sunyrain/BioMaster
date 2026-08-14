#!/usr/bin/env python3
"""Extract frozen label-blind PIC-DTA drug and WT/modification features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/biomaster_pic_dta_label_blind_feature_freeze_20260814.json"
DEFAULT_OUTPUT = ROOT / "outputs/old_drug_target_sota_v1/pic_dta_label_blind_features_v1"
BASE_PROTEINS = ["abl1", "braf", "egfr", "fgfr3", "flt3", "kit", "lrrk2", "met", "pik3ca", "ret"]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def modified_mask(proteins: pd.Series) -> pd.Series:
    return (
        proteins.str.contains(r"_[a-z][0-9]", regex=True)
        | proteins.str.contains("_itd", regex=False)
        | proteins.str.contains("abl1_p", regex=False)
        | proteins.str.contains("s808g", regex=False)
    )


def base_protein(protein: str) -> str | None:
    for base in BASE_PROTEINS:
        if protein.startswith(f"{base}_"):
            return base
    return None


def alignment_blocks(
    wt: str, modified: str, aligner: PairwiseAligner
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    alignment = aligner.align(wt, modified)[0]
    indices = np.asarray(alignment.indices, dtype=int)
    if indices.shape[0] != 2:
        raise RuntimeError("Unexpected pairwise alignment index shape")
    unequal = []
    for column in range(indices.shape[1]):
        wt_index, modified_index = int(indices[0, column]), int(indices[1, column])
        unequal.append(
            wt_index < 0
            or modified_index < 0
            or wt[wt_index] != modified[modified_index]
        )
    blocks = []
    start = 0
    while start < len(unequal):
        if not unequal[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(unequal) and unequal[stop]:
            stop += 1
        wt_positions = [int(value) for value in indices[0, start:stop] if value >= 0]
        modified_positions = [int(value) for value in indices[1, start:stop] if value >= 0]
        left_wt = next(
            (int(indices[0, column]) for column in range(start - 1, -1, -1) if indices[0, column] >= 0),
            0,
        )
        left_modified = next(
            (int(indices[1, column]) for column in range(start - 1, -1, -1) if indices[1, column] >= 0),
            0,
        )
        wt_center = int(np.median(wt_positions)) if wt_positions else left_wt
        modified_center = (
            int(np.median(modified_positions)) if modified_positions else left_modified
        )
        wt_residues = "".join(wt[position] for position in wt_positions)
        modified_residues = "".join(modified[position] for position in modified_positions)
        substitutions = sum(
            1
            for wt_index, modified_index in zip(
                indices[0, start:stop], indices[1, start:stop]
            )
            if wt_index >= 0
            and modified_index >= 0
            and wt[int(wt_index)] != modified[int(modified_index)]
        )
        blocks.append(
            {
                "alignment_column_start_zero_based": start,
                "alignment_column_stop_exclusive": stop,
                "wt_center_zero_based": wt_center,
                "modified_center_zero_based": modified_center,
                "wt_changed_residues": wt_residues,
                "modified_changed_residues": modified_residues,
                "substitution_count": substitutions,
                "deleted_residue_count": max(0, len(wt_residues) - substitutions),
                "inserted_residue_count": max(0, len(modified_residues) - substitutions),
            }
        )
        start = stop
    counts = {
        "changed_block_count": len(blocks),
        "substitution_count": sum(block["substitution_count"] for block in blocks),
        "deleted_residue_count": sum(block["deleted_residue_count"] for block in blocks),
        "inserted_residue_count": sum(block["inserted_residue_count"] for block in blocks),
    }
    return blocks, counts


def request_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def build_protein_manifests(
    manifest: pd.DataFrame,
    sequences: dict[str, str],
    radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], np.ndarray]:
    selected = manifest[modified_mask(manifest["protein"])].copy().reset_index(drop=True)
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -1.0
    requests: dict[str, str] = {}
    construct_rows = []
    block_rows = []
    explicit_rows = []
    aa_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    for feature_index, row in enumerate(selected.itertuples(index=False)):
        protein = str(row.protein)
        base = base_protein(protein)
        paired = base is not None
        phosphorylation = str(row.protein_display_name).endswith("p")
        modified_sequence = sequences[protein]
        if paired:
            wt_sequence = sequences[base]
            blocks, counts = alignment_blocks(wt_sequence, modified_sequence, aligner)
        else:
            wt_sequence = ""
            blocks = []
            counts = {
                "changed_block_count": 0,
                "substitution_count": 0,
                "deleted_residue_count": 0,
                "inserted_residue_count": 0,
            }
        sequence_changing = paired and wt_sequence != modified_sequence
        if sequence_changing and not blocks:
            raise RuntimeError(f"No changed alignment block for {protein}")
        if paired and not sequence_changing and not phosphorylation:
            raise RuntimeError(f"Unexpected same-sequence non-phosphorylation construct: {protein}")
        source_counts = np.zeros(20, dtype=np.float32)
        destination_counts = np.zeros(20, dtype=np.float32)
        for block_number, block in enumerate(blocks, start=1):
            for amino_acid in block["wt_changed_residues"]:
                source_counts[aa_index[amino_acid]] += 1.0
            for amino_acid in block["modified_changed_residues"]:
                destination_counts[aa_index[amino_acid]] += 1.0
            wt_center = int(block["wt_center_zero_based"])
            modified_center = int(block["modified_center_zero_based"])
            wt_start = max(0, wt_center - radius)
            wt_stop = min(len(wt_sequence), wt_center + radius + 1)
            modified_start = max(0, modified_center - radius)
            modified_stop = min(len(modified_sequence), modified_center + radius + 1)
            wt_window = wt_sequence[wt_start:wt_stop]
            modified_window = modified_sequence[modified_start:modified_stop]
            wt_key = request_key(wt_window)
            modified_key = request_key(modified_window)
            requests.setdefault(wt_key, wt_window)
            requests.setdefault(modified_key, modified_window)
            block_rows.append(
                {
                    "construct_feature_index": feature_index,
                    "protein": protein,
                    "base_wildtype_protein": base,
                    "changed_block_number": block_number,
                    **block,
                    "wt_window_start_zero_based": wt_start,
                    "wt_window_stop_exclusive": wt_stop,
                    "modified_window_start_zero_based": modified_start,
                    "modified_window_stop_exclusive": modified_stop,
                    "wt_window_length": len(wt_window),
                    "modified_window_length": len(modified_window),
                    "wt_request_key": wt_key,
                    "modified_request_key": modified_key,
                    "wt_window_sequence": wt_window,
                    "modified_window_sequence": modified_window,
                }
            )
        scalars = np.array(
            [
                counts["changed_block_count"],
                counts["substitution_count"],
                counts["deleted_residue_count"],
                counts["inserted_residue_count"],
                len(modified_sequence) - len(wt_sequence) if paired else 0,
                int(phosphorylation),
                int(
                    counts["deleted_residue_count"] > 0
                    or counts["inserted_residue_count"] > 0
                ),
                int(
                    counts["changed_block_count"] > 1
                    or counts["substitution_count"]
                    + counts["deleted_residue_count"]
                    + counts["inserted_residue_count"]
                    > 1
                ),
            ],
            dtype=np.float32,
        )
        explicit_rows.append(np.concatenate([source_counts, destination_counts, scalars]))
        construct_rows.append(
            {
                "construct_feature_index": feature_index,
                "official_row_index": int(row.official_row_index),
                "gene_symbol_source": str(row.gene_symbol_source),
                "protein_display_name": str(row.protein_display_name),
                "protein": protein,
                "base_wildtype_protein": base,
                "paired_wildtype_available": paired,
                "is_phosphorylated_state": phosphorylation,
                "wt_domain_sequence_length": len(wt_sequence) if paired else 0,
                "modified_domain_sequence_length": len(modified_sequence),
                "sequence_changes_relative_to_wt": sequence_changing,
                **counts,
            }
        )
    return (
        pd.DataFrame(construct_rows),
        pd.DataFrame(block_rows),
        requests,
        np.stack(explicit_rows).astype(np.float32),
    )


def extract_esm2(
    requests: dict[str, str], checkpoint: Path, batch_size: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen ESM2 extraction")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["TORCH_HOME"] = str(checkpoint.parents[2])
    import esm  # noqa: PLC0415

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(torch.device("cuda")).eval()
    converter = alphabet.get_batch_converter()
    layer = int(model.num_layers)
    ordered = sorted(requests.items(), key=lambda item: (len(item[1]), item[0]))
    extracted: dict[str, np.ndarray] = {}
    batches = 0
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            selected = ordered[start : start + batch_size]
            _, _, tokens = converter([(key, sequence) for key, sequence in selected])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(
                    tokens.to("cuda", non_blocking=True),
                    repr_layers=[layer],
                    return_contacts=False,
                )
            representations = result["representations"][layer]
            for batch_index, (key, sequence) in enumerate(selected):
                mean = (
                    representations[batch_index, 1 : len(sequence) + 1]
                    .float()
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                extracted[key] = mean
            batches += 1
            if batches % 10 == 0 or start + len(selected) == len(ordered):
                print(
                    json.dumps(
                        {
                            "esm2_windows_completed": start + len(selected),
                            "esm2_windows_total": len(ordered),
                            "batches": batches,
                            "gpu_memory_allocated_mib": round(
                                torch.cuda.memory_allocated() / 1024**2, 1
                            ),
                        }
                    ),
                    flush=True,
                )
    del model
    torch.cuda.empty_cache()
    return extracted, {
        "requests": len(ordered),
        "batches": batches,
        "batch_size": batch_size,
        "layer": layer,
        "precision": "CUDA_AUTOCAST_FLOAT16_SAVE_FLOAT32",
    }


def aggregate_construct_embeddings(
    construct_manifest: pd.DataFrame,
    block_manifest: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    dimension: int,
) -> dict[str, np.ndarray]:
    count = len(construct_manifest)
    matrices = {
        "wt_local_mean": np.zeros((count, dimension), dtype=np.float32),
        "modified_local_mean": np.zeros((count, dimension), dtype=np.float32),
        "local_delta_mean": np.zeros((count, dimension), dtype=np.float32),
        "local_delta_signed_maxabs": np.zeros((count, dimension), dtype=np.float32),
    }
    for feature_index, blocks in block_manifest.groupby(
        "construct_feature_index", sort=False, observed=True
    ):
        wt = np.stack([embeddings[key] for key in blocks["wt_request_key"]]).astype(np.float32)
        modified = np.stack(
            [embeddings[key] for key in blocks["modified_request_key"]]
        ).astype(np.float32)
        delta = modified - wt
        feature_index = int(feature_index)
        matrices["wt_local_mean"][feature_index] = wt.mean(axis=0)
        matrices["modified_local_mean"][feature_index] = modified.mean(axis=0)
        matrices["local_delta_mean"][feature_index] = delta.mean(axis=0)
        max_index = np.abs(delta).argmax(axis=0)
        matrices["local_delta_signed_maxabs"][feature_index] = delta[
            max_index, np.arange(dimension)
        ]
    return matrices


def extract_drugs(ligand_pickle: Path) -> tuple[pd.DataFrame, np.ndarray]:
    with ligand_pickle.open("rb") as handle:
        dictionary = pickle.load(handle)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    rows = []
    fingerprints = []
    for drug_index, (name, values) in enumerate(dictionary.items()):
        cid, smiles = str(values[0]), str(values[1])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"Invalid official SMILES for {name}: {smiles}")
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints.append(array)
        rows.append(
            {
                "drug_feature_index": drug_index,
                "drug_name": str(name),
                "pubchem_cid": cid,
                "compound_iso_smiles": smiles,
                "canonical_smiles_rdkit": Chem.MolToSmiles(molecule, canonical=True),
                "morgan_on_bits": int(array.sum()),
            }
        )
    return pd.DataFrame(rows), np.stack(fingerprints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_LABEL_BLIND_FEATURE_EXTRACTION":
        raise RuntimeError("Unexpected feature-freeze status")
    paths = {}
    for name, spec in config["inputs"].items():
        path = Path(spec["path"])
        paths[name] = path if path.is_absolute() else ROOT / path
        if sha256(paths[name]) != spec["sha256"]:
            raise RuntimeError(f"Input hash mismatch for {name}")

    sequence_manifest = pd.read_csv(
        paths["official_construct_manifest"],
        usecols=config["inputs"]["official_construct_manifest"]["columns_allowed"],
    )
    sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse(paths["official_construct_full_fasta"], "fasta")
    }
    constructs, blocks, requests, explicit = build_protein_manifests(
        sequence_manifest,
        sequences,
        int(config["protein_representation"]["context"]["radius_residues_per_changed_block"]),
    )
    drugs, morgan = extract_drugs(paths["official_ligand_dictionary"])
    expected = config["frozen_counts_before_extraction"]
    phosphorylation_groups = json.loads(
        paths["sequence_reconstruction_summary"].read_text(encoding="utf-8")
    )["sequence_indistinguishability"]["phosphorylation_or_p_suffix_groups"]
    observed = {
        "official_constructs": len(sequence_manifest),
        "official_ligands": len(drugs),
        "official_modified_constructs": len(constructs),
        "paired_modified_constructs": int(constructs["paired_wildtype_available"].sum()),
        "unpaired_modified_constructs": int((~constructs["paired_wildtype_available"]).sum()),
        "paired_base_genes": int(
            constructs.loc[constructs["paired_wildtype_available"], "base_wildtype_protein"].nunique()
        ),
        "sequence_indistinguishable_phosphorylation_groups": len(phosphorylation_groups),
    }
    if observed != expected:
        raise RuntimeError(f"Frozen count drift: {observed} != {expected}")
    validation = {
        "config_sha256": sha256(config_path),
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "observed_counts": observed,
        "changed_blocks": len(blocks),
        "unique_esm2_window_requests": len(requests),
        "explicit_feature_shape": list(explicit.shape),
        "morgan_shape": list(morgan.shape),
        "forbidden_label_inputs_opened": [],
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID", **validation}, ensure_ascii=False, indent=2))
        return

    embeddings, extraction_audit = extract_esm2(
        requests,
        paths["esm2_checkpoint"],
        int(config["protein_representation"]["batch_size"]),
        int(config["protein_representation"]["deterministic_seed"]),
    )
    matrices = aggregate_construct_embeddings(
        constructs, blocks, embeddings, int(config["protein_representation"]["embedding_dimension"])
    )
    delta_norm = np.linalg.norm(matrices["local_delta_mean"], axis=1)
    signed_norm = np.linalg.norm(matrices["local_delta_signed_maxabs"], axis=1)
    sequence_changing = constructs["sequence_changes_relative_to_wt"].to_numpy(dtype=bool)
    paired = constructs["paired_wildtype_available"].to_numpy(dtype=bool)
    same_sequence_paired = paired & ~sequence_changing
    checks = {
        "all_input_hashes_match": True,
        "frozen_counts_match": observed == expected,
        "all_drugs_valid": len(drugs) == 72 and morgan.shape == (72, 2048),
        "all_embeddings_finite": all(np.isfinite(matrix).all() for matrix in matrices.values()),
        "all_sequence_changing_deltas_nonzero": bool(
            np.all(delta_norm[sequence_changing] > 0)
            and np.all(signed_norm[sequence_changing] > 0)
        ),
        "same_sequence_phosphorylation_deltas_exact_zero": bool(
            np.all(matrices["local_delta_mean"][same_sequence_paired] == 0)
            and np.all(matrices["local_delta_signed_maxabs"][same_sequence_paired] == 0)
        ),
        "unpaired_construct_embeddings_exact_zero": bool(
            all(np.all(matrix[~paired] == 0) for matrix in matrices.values())
        ),
        "no_forbidden_label_inputs_opened": True,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    construct_path = output / "PIC_DTA_MODIFIED_CONSTRUCT_FEATURE_MANIFEST_56_V1.csv"
    block_path = output / "PIC_DTA_CHANGED_BLOCK_MANIFEST_V1.csv.gz"
    drug_path = output / "PIC_DTA_DRUG_FEATURE_MANIFEST_72_V1.csv"
    explicit_path = output / "PIC_DTA_EXPLICIT_MODIFICATION_FEATURES_56X48_FLOAT32_V1.npy"
    morgan_path = output / "PIC_DTA_DRUG_MORGAN_72X2048_UINT8_V1.npy"
    constructs.assign(
        local_delta_mean_l2=delta_norm,
        local_delta_signed_maxabs_l2=signed_norm,
    ).to_csv(construct_path, index=False)
    blocks.to_csv(block_path, index=False, compression="gzip")
    drugs.to_csv(drug_path, index=False)
    np.save(explicit_path, explicit)
    np.save(morgan_path, morgan)
    matrix_paths = {}
    for name, matrix in matrices.items():
        path = output / f"PIC_DTA_ESM2_{name.upper()}_56X1280_FLOAT32_V1.npy"
        np.save(path, matrix)
        matrix_paths[name] = path
    outputs = [construct_path, block_path, drug_path, explicit_path, morgan_path, *matrix_paths.values()]
    summary = {
        "schema_version": "PIC_DTA_LABEL_BLIND_FEATURE_SUMMARY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config": relative(config_path),
        "validation": validation,
        "esm2_extraction": extraction_audit,
        "counts": {
            **observed,
            "changed_blocks": len(blocks),
            "sequence_changing_paired_constructs": int(sequence_changing.sum()),
            "same_sequence_phosphorylation_paired_constructs": int(same_sequence_paired.sum()),
            "explicit_feature_dimensions": int(explicit.shape[1]),
        },
        "checks": checks,
        "label_blind_boundary": "No affinity matrix, reconstructed label table, affinity value, pKd value, censoring indicator, or mutation-response outcome was opened.",
        "claim_boundary": "These are prior-art representations prepared for a frozen paired interval-censored mechanism test; they are not a validated model or algorithm innovation.",
        "inputs": {relative(path): sha256(path) for path in paths.values()},
        "outputs": {relative(path): sha256(path) for path in outputs},
    }
    summary_path = output / "PIC_DTA_LABEL_BLIND_FEATURE_SUMMARY_V1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "counts": summary["counts"],
        "checks": checks,
        "summary": relative(summary_path),
        "summary_sha256": sha256(summary_path),
    }, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
