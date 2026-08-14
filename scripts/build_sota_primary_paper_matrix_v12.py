#!/usr/bin/env python3
"""Audit AD-LSF's official release and append it to the SOTA matrix V12."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
REPO = ROOT / "third_party/sota_dti_2026/AD-LSF"
DATA = REPO / "data"
INPUT = BASE / "SOTA_PRIMARY_PAPER_TRAINING_DATA_RESULTS_MATRIX_V11.csv"
OUTPUT = BASE / "SOTA_PRIMARY_PAPER_TRAINING_DATA_RESULTS_MATRIX_V12.csv"
AUDIT_JSON = BASE / "AD_LSF_RELEASED_CODE_DATA_SPLIT_AUDIT_V1.json"
AUDIT_CSV = BASE / "AD_LSF_RELEASED_SPLIT_AUDIT_V1.csv"
AUDIT_MD = BASE / "AD_LSF_RELEASED_CODE_DATA_SPLIT_AUDIT_V1.md"
EXPECTED_COMMIT = "5bee039edaf6bf508eaae07f414f9074988a4ed8"


PAPER_TABLE = {
    "bindingdb": {"drugs": 14643, "proteins": 2623, "rows": 49199, "positive": 20674, "negative": 28525},
    "biosnap": {"drugs": 4510, "proteins": 2181, "rows": 27464, "positive": 13830, "negative": 13634},
    "human": {"drugs": 2726, "proteins": 2001, "rows": 5997, "positive": 2633, "negative": 3364},
    "celegans": {"drugs": 1767, "proteins": 1876, "rows": 7785, "positive": 3893, "negative": 3892},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["SMILES", "Protein", "Y"]
    if any(column not in frame.columns for column in required):
        raise RuntimeError(f"Missing columns: {required} versus {list(frame.columns)}")
    result = frame.copy()
    result["SMILES"] = result["SMILES"].astype(str)
    result["Protein"] = result["Protein"].astype(str)
    result["Y"] = pd.to_numeric(result["Y"], errors="raise").astype(int)
    if not set(result["Y"].unique()).issubset({0, 1}):
        raise RuntimeError("Nonbinary label")
    result["pair"] = list(zip(result["SMILES"], result["Protein"]))
    result["full_row"] = list(zip(result["SMILES"], result["Protein"], result["Y"]))
    return result


def label_map(frame: pd.DataFrame) -> dict[tuple[str, str], set[int]]:
    result: dict[tuple[str, str], set[int]] = defaultdict(set)
    for smiles, protein, label in frame[["SMILES", "Protein", "Y"]].itertuples(index=False, name=None):
        result[(str(smiles), str(protein))].add(int(label))
    return dict(result)


def overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    left_map = label_map(left)
    right_map = label_map(right)
    keys = set(left_map) & set(right_map)
    return {
        "pair_overlap": len(keys),
        "label_set_conflicts": sum(left_map[key] != right_map[key] for key in keys),
        "opposite_label_collisions": sum(len(left_map[key] | right_map[key]) > 1 for key in keys),
        "full_row_overlap": len(set(left["full_row"]) & set(right["full_row"])),
    }


def multiset(frame: pd.DataFrame) -> Counter[tuple[str, str, int]]:
    return Counter(frame["full_row"])


def sample_alignment(
    full: pd.DataFrame, role_frame: pd.DataFrame, sample_path: Path
) -> dict[str, Any]:
    samples = pd.read_csv(sample_path)
    samples = samples.loc[:, [column for column in samples.columns if not column.startswith("Unnamed")]]
    if list(samples.columns) != ["smiles", "sequence", "interactions"]:
        raise RuntimeError(f"Unexpected samples schema in {sample_path}: {list(samples.columns)}")
    smiles_order = list(dict.fromkeys(full["SMILES"]))
    protein_order = list(dict.fromkeys(full["Protein"]))
    smiles_index = {value: index for index, value in enumerate(smiles_order)}
    protein_index = {value: index for index, value in enumerate(protein_order)}
    expected_smiles = role_frame["SMILES"].map(smiles_index).to_numpy(dtype=int)
    expected_proteins = role_frame["Protein"].map(protein_index).to_numpy(dtype=int)
    expected_labels = role_frame["Y"].to_numpy(dtype=int)
    observed_smiles = samples["smiles"].to_numpy(dtype=int)
    observed_proteins = samples["sequence"].to_numpy(dtype=int)
    observed_labels = samples["interactions"].to_numpy(dtype=int)
    return {
        "rows": len(samples),
        "row_count_matches": len(samples) == len(role_frame),
        "smiles_indices_match": bool(np.array_equal(expected_smiles, observed_smiles)),
        "protein_indices_match": bool(np.array_equal(expected_proteins, observed_proteins)),
        "labels_match": bool(np.array_equal(expected_labels, observed_labels)),
    }


def seeded_random_replay(full: pd.DataFrame, dataset: str, seed: int) -> dict[str, Any]:
    indices = np.arange(len(full))
    np.random.seed(seed)
    np.random.shuffle(indices)
    n = len(indices)
    first, second = ((0.7, 0.8) if dataset in {"bindingdb", "biosnap"} else (0.8, 0.9))
    train_index, valid_index, test_index = np.split(indices, [int(first * n), int(second * n)])
    train = full.iloc[train_index]
    valid = full.iloc[valid_index]
    test = full.iloc[test_index]
    return {
        "seed": seed,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "test_unique_drugs_seen_in_train_fraction": len(set(test["SMILES"]) & set(train["SMILES"]))
        / max(1, test["SMILES"].nunique()),
        "test_unique_targets_seen_in_train_fraction": len(set(test["Protein"]) & set(train["Protein"]))
        / max(1, test["Protein"].nunique()),
        "train_test": overlap(train, test),
        "valid_test": overlap(valid, test),
        "test_original_row_indices": set(map(int, test_index)),
    }


def positional_encoding_audit(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PositionalEncoding"]
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one PositionalEncoding, found {len(nodes)}")
    isolated = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace: dict[str, Any] = {"torch": torch, "nn": nn, "math": math}
    exec(compile(isolated, "AD_LSF_PositionalEncoding", "exec"), namespace)
    cls = namespace["PositionalEncoding"]
    module = cls(d_model=8, max_len=20)
    zeros = torch.zeros(4, 6, 8)
    zero_output = module(zeros)
    within_sample_token_difference = float(
        (zero_output - zero_output[:, :1, :]).abs().max().item()
    )
    between_batch_slot_difference = float(
        (zero_output[0] - zero_output[1]).abs().max().item()
    )
    torch.manual_seed(20260814)
    x = torch.randn(4, 6, 8)
    permutation = torch.tensor([2, 0, 3, 1])
    permute_then_encode = module(x[permutation])
    encode_then_permute = module(x)[permutation]
    batch_permutation_equivariance_error = float(
        (permute_then_encode - encode_then_permute).abs().max().item()
    )
    return {
        "stored_pe_shape": list(module.pe.shape),
        "input_shape": list(zeros.shape),
        "within_sample_token_position_max_difference_for_zero_input": within_sample_token_difference,
        "between_batch_slot_max_difference_for_zero_input": between_batch_slot_difference,
        "batch_permutation_equivariance_max_error": batch_permutation_equivariance_error,
        "uses_batch_axis_instead_of_sequence_axis": within_sample_token_difference == 0.0
        and between_batch_slot_difference > 0.0
        and batch_permutation_equivariance_error > 0.0,
        "interpretation": (
            "pe is stored as [max_len,1,d] and sliced with x.size(0), so the code adds a different vector by batch "
            "slot and the same vector to every token within a sample. This is not sequence positional encoding and "
            "makes outputs non-equivariant to batch permutation."
        ),
    }


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    split_source = (REPO / "preprocess/split_dataset.py").read_text(encoding="utf-8")
    train_source = (REPO / "train.py").read_text(encoding="utf-8")
    test_source = (REPO / "test.py").read_text(encoding="utf-8")
    model_source = (REPO / "model/dti.py").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8").strip()
    positional = positional_encoding_audit(model_source)

    raw_summaries: dict[str, Any] = {}
    random_replays: dict[str, Any] = {}
    split_summaries: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    full_frames: dict[str, pd.DataFrame] = {}

    for dataset in PAPER_TABLE:
        full = normalize_frame(pd.read_csv(DATA / dataset / "fulldata.csv"))
        full_frames[dataset] = full
        unique_full = full.drop_duplicates(["SMILES", "Protein", "Y"])
        pair_labels = label_map(full)
        duplicate_excess_by_label = {
            str(label): int((full["Y"] == label).sum() - (unique_full["Y"] == label).sum())
            for label in [0, 1]
        }
        raw_summaries[dataset] = {
            "path": str((DATA / dataset / "fulldata.csv").relative_to(ROOT)),
            "rows": len(full),
            "positive": int(full["Y"].sum()),
            "negative": int((1 - full["Y"]).sum()),
            "unique_drugs": int(full["SMILES"].nunique()),
            "unique_targets": int(full["Protein"].nunique()),
            "unique_full_rows": len(unique_full),
            "duplicate_full_row_excess": len(full) - len(unique_full),
            "duplicate_excess_by_label": duplicate_excess_by_label,
            "unique_pairs": int(full["pair"].nunique()),
            "pair_label_conflicts": sum(len(labels) > 1 for labels in pair_labels.values()),
            "paper_table": PAPER_TABLE[dataset],
            "released_minus_paper_rows": len(full) - PAPER_TABLE[dataset]["rows"],
            "released_minus_paper_positive": int(full["Y"].sum()) - PAPER_TABLE[dataset]["positive"],
            "released_minus_paper_negative": int((1 - full["Y"]).sum()) - PAPER_TABLE[dataset]["negative"],
        }

        replay_42 = seeded_random_replay(full, dataset, 42)
        replay_43 = seeded_random_replay(full, dataset, 43)
        test_jaccard = len(
            replay_42["test_original_row_indices"] & replay_43["test_original_row_indices"]
        ) / len(replay_42["test_original_row_indices"] | replay_43["test_original_row_indices"])
        for replay in [replay_42, replay_43]:
            replay.pop("test_original_row_indices")
        random_replays[dataset] = {
            "official_published_split_identifiable": False,
            "reason": "the preprocessing script calls np.random.shuffle without a seed or saved random split files",
            "representative_seed_42": replay_42,
            "representative_seed_43": replay_43,
            "seed_42_vs_43_test_row_membership_jaccard": test_jaccard,
        }

        split_summaries[dataset] = {}
        for split_type in ["cold_drug", "cold_target"]:
            roles: dict[str, pd.DataFrame] = {}
            alignments: dict[str, Any] = {}
            for role in ["train", "valid", "test"]:
                raw_path = DATA / dataset / split_type / role / f"{role}.csv"
                sample_path = DATA / dataset / split_type / role / "samples.csv"
                role_frame = normalize_frame(pd.read_csv(raw_path))
                roles[role] = role_frame
                alignments[role] = sample_alignment(full, role_frame, sample_path)
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "split_type": split_type,
                        "role": role,
                        "rows": len(role_frame),
                        "positive": int(role_frame["Y"].sum()),
                        "negative": int((1 - role_frame["Y"]).sum()),
                        "unique_drugs": int(role_frame["SMILES"].nunique()),
                        "unique_targets": int(role_frame["Protein"].nunique()),
                        "unique_pairs": int(role_frame["pair"].nunique()),
                        "duplicate_full_row_excess": int(len(role_frame) - role_frame["full_row"].nunique()),
                        "sample_index_alignment": all(alignments[role].values()),
                    }
                )
            concatenated = pd.concat([roles[role] for role in ["train", "valid", "test"]], ignore_index=True)
            train_valid = overlap(roles["train"], roles["valid"])
            train_test = overlap(roles["train"], roles["test"])
            valid_test = overlap(roles["valid"], roles["test"])
            split_summaries[dataset][split_type] = {
                "role_rows": {role: len(frame) for role, frame in roles.items()},
                "all_full_rows_accounted_for_as_multiset": multiset(concatenated) == multiset(full),
                "sample_indices_align_with_reconstructed_first_occurrence_maps": all(
                    all(values.values()) for values in alignments.values()
                ),
                "train_valid": train_valid,
                "train_test": train_test,
                "valid_test": valid_test,
                "train_valid_drug_overlap": len(set(roles["train"]["SMILES"]) & set(roles["valid"]["SMILES"])),
                "train_test_drug_overlap": len(set(roles["train"]["SMILES"]) & set(roles["test"]["SMILES"])),
                "valid_test_drug_overlap": len(set(roles["valid"]["SMILES"]) & set(roles["test"]["SMILES"])),
                "train_valid_target_overlap": len(set(roles["train"]["Protein"]) & set(roles["valid"]["Protein"])),
                "train_test_target_overlap": len(set(roles["train"]["Protein"]) & set(roles["test"]["Protein"])),
                "valid_test_target_overlap": len(set(roles["valid"]["Protein"]) & set(roles["test"]["Protein"])),
                "cold_axis_train_disjoint_from_valid_and_test": (
                    split_type == "cold_drug"
                    and len(set(roles["train"]["SMILES"]) & (set(roles["valid"]["SMILES"]) | set(roles["test"]["SMILES"]))) == 0
                )
                or (
                    split_type == "cold_target"
                    and len(set(roles["train"]["Protein"]) & (set(roles["valid"]["Protein"]) | set(roles["test"]["Protein"]))) == 0
                ),
                "alignments": alignments,
            }

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    missing_runtime_artifacts = {
        "model_Fusion1_py": not (REPO / "model/Fusion1.py").exists(),
        "requirements_or_environment": not any(
            (REPO / name).exists()
            for name in ["requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml"]
        ),
        "smiles_index_files": not any(DATA.glob("*/smiles.csv")),
        "protein_index_files": not any(DATA.glob("*/protein.csv")),
        "precomputed_embeddings": not any(DATA.rglob("*_embeddings.npy")),
        "model_checkpoints": not any(
            path.suffix in {".ckpt", ".pth", ".pt"} for path in REPO.rglob("*") if path.is_file()
        ),
        "random_split_files": not any(DATA.glob("*/random/train/samples.csv")),
    }
    checks = {
        "official_commit_matches": commit == EXPECTED_COMMIT,
        "four_full_datasets_audited": set(full_frames) == set(PAPER_TABLE),
        "bindingdb_release_matches_paper_table": raw_summaries["bindingdb"]["released_minus_paper_rows"] == 0,
        "biosnap_paper_drug_count_differs_from_release": raw_summaries["biosnap"]["unique_drugs"] == 4505
        and PAPER_TABLE["biosnap"]["drugs"] == 4510,
        "human_release_duplicates_731_positive_rows": raw_summaries["human"]["duplicate_full_row_excess"] == 731
        and raw_summaries["human"]["duplicate_excess_by_label"]["1"] == 731,
        "human_unique_rows_match_paper_interactions": raw_summaries["human"]["unique_full_rows"] == 5997,
        "celegans_release_rows_one_above_paper": raw_summaries["celegans"]["released_minus_paper_rows"] == 1,
        "celegans_release_duplicates_1234_positive_rows": raw_summaries["celegans"]["duplicate_full_row_excess"] == 1234
        and raw_summaries["celegans"]["duplicate_excess_by_label"]["1"] == 1234,
        "all_released_cold_splits_cover_full_multisets": all(
            summary["all_full_rows_accounted_for_as_multiset"]
            for dataset in split_summaries.values()
            for summary in dataset.values()
        ),
        "all_released_cold_splits_are_train_disjoint_on_named_axis": all(
            summary["cold_axis_train_disjoint_from_valid_and_test"]
            for dataset in split_summaries.values()
            for summary in dataset.values()
        ),
        "all_sample_index_files_align": all(
            summary["sample_indices_align_with_reconstructed_first_occurrence_maps"]
            for dataset in split_summaries.values()
            for summary in dataset.values()
        ),
        "duplicate_pair_leak_exists_between_valid_and_test": any(
            summary["valid_test"]["pair_overlap"] > 0
            for dataset in split_summaries.values()
            for summary in dataset.values()
        ),
        "published_random_split_files_absent": missing_runtime_artifacts["random_split_files"],
        "random_generator_has_no_seed": "np.random.shuffle(dataset)" in split_source
        and "np.random.seed" not in split_source,
        "released_cold_types_not_generatable_by_split_script": 'choices=["random","cold","cluster"]' in split_source
        and "cold_drug" not in split_source
        and "cold_target" not in split_source,
        "model_import_references_missing_fusion1": "from model.Fusion1 import *" in model_source
        and missing_runtime_artifacts["model_Fusion1_py"],
        "positional_encoding_uses_batch_axis": positional["uses_batch_axis_instead_of_sequence_axis"],
        "paper_learning_rate_differs_from_release_configs": all(
            token not in (REPO / "config/config.yaml").read_text(encoding="utf-8")
            for token in ["5e-4", "0.0005"]
        ),
        "paper_five_repeat_orchestration_not_released": not any(
            "seed" in path.name.lower() or "repeat" in path.name.lower()
            for path in REPO.rglob("*.py")
        ),
        "detail_rows_cover_four_datasets_two_splits_three_roles": len(detail_rows) == 24,
    }
    if not all(checks.values()):
        raise RuntimeError(f"AD-LSF audit failed: {[key for key, value in checks.items() if not value]}")

    audit: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "verdict": {
            "executable_as_released": False,
            "published_random_split_exactly_reproducible": False,
            "released_cold_splits_auditable": True,
            "paper_headline_entity_cold": False,
            "same_data_directly_comparable_to_biomaster": False,
            "usable_as_algorithm_prior_art": True,
            "use_paper_headline_as_sota_evidence_for_biomaster": False,
            "reason": (
                "The paper headline uses warm random pair splits, while those exact splits, embeddings and weights "
                "are absent and the split generator has no seed. The model imports an unshipped Fusion1 module, "
                "the released small datasets contain duplicated positive rows that cross validation/test in cold "
                "files, and positional encoding uses batch position rather than token position."
            ),
        },
        "repository": {
            "official_code": "https://github.com/hatle/AD-LSF",
            "local_path": str(REPO.relative_to(ROOT)),
            "commit": commit,
            "readme": readme,
            "missing_runtime_artifacts": missing_runtime_artifacts,
            "train_type_choices_include_unreleased_generators": "random, cold, cluster, cold_drug, cold_target",
            "split_generator_type_choices": "random, cold, cluster",
            "standalone_test_type_choices": "random, cold, cluster",
        },
        "paper_report": {
            "primary_source": "https://doi.org/10.1021/acsomega.6c02700",
            "published_online": "2026-06-24",
            "dataset_table": PAPER_TABLE,
            "headline_protocol": "random pair split; 7:1:2 BindingDB/BioSNAP and 8:1:1 Human/C. elegans",
            "repetitions": "paper states five independent repeats and reports mean plus/minus standard deviation",
            "headline": {
                "BindingDB_AUROC": "0.967 +/- 0.001",
                "BindingDB_AUPRC": "0.959 +/- 0.001",
                "BindingDB_accuracy": "0.919 +/- 0.001",
                "scope": "paper random-split table, not a cold-start result",
            },
            "paper_learning_rate": 0.0005,
            "released_config_learning_rates": {
                "bindingdb": 0.0001,
                "biosnap": 0.0001,
                "human": 0.00005,
                "celegans": 0.00005,
            },
        },
        "released_full_data": raw_summaries,
        "released_random_split_replays_not_official": random_replays,
        "released_cold_splits": split_summaries,
        "implementation_semantics": {
            "missing_model_module": "model/dti.py imports model.Fusion1, but model/Fusion1.py is absent",
            "positional_encoding_runtime_audit": positional,
            "gpu_argument": "train.py accepts --gpus but hard-codes devices=[0]",
            "repeat_policy": "training fixes seed 42; no five-seed orchestration, logs, predictions or checkpoints are released",
        },
        "checks": checks,
        "detail_csv": str(AUDIT_CSV.relative_to(ROOT)),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(REPO.rglob("*"))
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        },
    }

    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        matrix_rows = list(reader)
    if any(row["model"] == "AD-LSF" for row in matrix_rows):
        raise RuntimeError("AD-LSF already exists in V11")
    matrix_rows.append(
        {
            "model": "AD-LSF",
            "year": "2026",
            "primary_source": "https://doi.org/10.1021/acsomega.6c02700",
            "official_code": "https://github.com/hatle/AD-LSF",
            "local_commit": commit,
            "input_modalities": (
                "drug graph GCN plus ChemBERTa-2; protein character CNN plus ESM-2; asymmetric dynamic gated "
                "modulation, latent-signal coordination and bidirectional cross-attention"
            ),
            "training_data": "BindingDB, BioSNAP, Human and C. elegans released fulldata; no published random split, embeddings or weights",
            "reported_scale": (
                "paper: 49,199/27,464/5,997/7,785 interactions; release: 49,199/27,464/6,728/7,786, with "
                "731 and 1,234 duplicated positive rows in Human and C. elegans"
            ),
            "split_protocol": (
                "headline: warm random pair split 7:1:2 or 8:1:1, five repeats; exact random files/seeds absent. "
                "Released cold-drug and cold-target files are train-disjoint on the named entity axis but their "
                "generator is absent and duplicate pairs cross validation/test"
            ),
            "headline_result": "random BindingDB AUROC/AUPRC/ACC .967/.959/.919 (mean over stated five repeats)",
            "main_strength": (
                "latest asymmetric local/global modality modulation and latent signal coordination reference; "
                "official raw and cold-split files permit detailed release audit"
            ),
            "critical_limit_for_our_claim": (
                "headline is warm random pair split; exact split and five repeats are unreleased; model imports missing "
                "Fusion1.py; Human/C. elegans releases oversample positives through exact duplicates; positional "
                "encoding depends on batch slot rather than token position; paper/config learning rates disagree"
            ),
            "implementation_role": (
                "recent architecture prior art and candidate corrected same-data control only after semantics are "
                "explicitly repaired; not an official executable baseline"
            ),
            "reported_result_status": "paper random-split headline; official code/data/split semantics audited",
            "same_data_directly_comparable": "NO",
            "reproduction_status": (
                "commit and 24 released cold-split roles audited; exact headline reproduction blocked by missing "
                "module, random split, embeddings, dependencies, weights and repeat orchestration"
            ),
            "algorithm_collision_with_eqir": (
                "HIGH for asymmetric gating, dynamic frequency decomposition, multisource disentanglement and "
                "bidirectional interactive alignment; LOW for label-independent evidence qualification, exact "
                "fallback and dual-query base-relative tail-regret constraints"
            ),
            "required_action": (
                "do not claim asymmetric gated modulation, latent signal disentanglement or bidirectional cross-"
                "fusion as BioMaster novelty; if used numerically, label a repaired port as AD-LSF-inspired, deduplicate "
                "before frozen S1-S5 splitting, fix positional semantics, use fold-local preprocessing and publish five seeds"
            ),
        }
    )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(matrix_rows)

    audit["matrix_update"] = {
        "input_v11": str(INPUT.relative_to(ROOT)),
        "input_v11_sha256": sha256(INPUT),
        "output_v12": str(OUTPUT.relative_to(ROOT)),
        "output_v12_sha256": sha256(OUTPUT),
        "v11_rows": len(matrix_rows) - 1,
        "v12_rows": len(matrix_rows),
        "only_added_model": "AD-LSF",
    }
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# AD-LSF 官方代码、数据与切分审计（2026-08-14）",
        "",
        "## 裁决",
        "",
        "AD-LSF 是目前更近的多模态架构先例，但论文 headline 来自 warm random pair split，不能与 BioMaster 的实体冷启动 S1–S5 直接比较。官方仓库附原始数据和 cold-drug/cold-target 文件，却缺失论文 random split、预计算嵌入、模型权重、依赖清单和一个被代码强制导入的 `model/Fusion1.py`。",
        "",
        "| 数据集 | 论文行数 | 释放行数 | 唯一完整行 | 重复正/负行 | 释放药物/靶点 | 与论文主要差异 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, summary in raw_summaries.items():
        paper = summary["paper_table"]
        duplicate = summary["duplicate_excess_by_label"]
        differences = []
        if summary["released_minus_paper_rows"]:
            differences.append(f"rows {summary['released_minus_paper_rows']:+d}")
        if summary["unique_drugs"] != paper["drugs"]:
            differences.append(f"drugs {summary['unique_drugs'] - paper['drugs']:+d}")
        if summary["released_minus_paper_positive"]:
            differences.append(f"positive {summary['released_minus_paper_positive']:+d}")
        if summary["released_minus_paper_negative"]:
            differences.append(f"negative {summary['released_minus_paper_negative']:+d}")
        lines.append(
            f"| {dataset} | {paper['rows']:,} | {summary['rows']:,} | {summary['unique_full_rows']:,} | "
            f"{duplicate['1']:,}/{duplicate['0']:,} | {summary['unique_drugs']:,}/{summary['unique_targets']:,} | "
            f"{'; '.join(differences) if differences else '一致'} |"
        )
    lines.extend(
        [
            "",
            "Human 的 6,728 个释放行恰好由论文的 5,997 个唯一 interaction 加 731 个重复阳性组成；C. elegans 释放文件也有 1,234 个重复阳性。训练时这些不是权重，而是被当作独立样本读取，并在部分 released cold split 的 validation/test 间形成相同 pair 重叠。",
            "",
            "## 代码语义",
            "",
            "- 论文 random split 文件没有释放，生成脚本 `np.random.shuffle` 前不设 seed；seed 42 与 43 的代表性重放产生不同测试成员，因此无法识别论文五次实验的确切切分。",
            "- 释放的 cold-drug/cold-target 文件覆盖四个原始文件的全部行，TRAIN 在命名冷轴上与 VALID/TEST 不相交；但生成脚本不支持这两个类型，且重复样本导致 VALID/TEST pair 重叠。",
            "- `model/dti.py` 强制导入不存在的 `model/Fusion1.py`，仓库也没有依赖清单、嵌入、checkpoint 或日志，因而不能按发布状态运行。",
            "- 运行时隔离验证显示 positional encoding 把位置向量加在 batch slot 而非 token 位置：同一样本所有 token 的位置差为 0，batch 置换等变误差为 "
            f"`{positional['batch_permutation_equivariance_max_error']:.6f}`。",
            "- 论文写学习率 5e-4；释放配置实际为 BindingDB/BioSNAP 1e-4、Human/C. elegans 5e-5。论文称五次独立重复，但仓库只固定 seed 42，没有重复调度、预测或日志。",
            "",
            "## 对 BioMaster 的约束",
            "",
            "非对称门控、动态频率分解、潜在信号解耦和双向交互对齐均已有直接先例，不能成为 EQIR 的模块首创。AD-LSF 的高 random-split 数值不证明严格冷启动能力；若后续纳入数值对照，只能构建明确标注的 corrected AD-LSF-inspired port，在冻结 S1–S5 前去重并修复位置语义。",
            "",
        ]
    )
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "commit": commit,
                "raw_summaries": raw_summaries,
                "positional_encoding": positional,
                "audit_json": str(AUDIT_JSON),
                "audit_json_sha256": sha256(AUDIT_JSON),
                "audit_csv": str(AUDIT_CSV),
                "audit_csv_sha256": sha256(AUDIT_CSV),
                "audit_markdown": str(AUDIT_MD),
                "audit_markdown_sha256": sha256(AUDIT_MD),
                "matrix_v12": str(OUTPUT),
                "matrix_v12_sha256": sha256(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
