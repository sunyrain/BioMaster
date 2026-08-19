from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/biomaster_recent_novel_target_external_freeze_20260819.json"
OUT = ROOT / "outputs/biomaster_recent_novel_target_external_v1"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def test_recent_target_panel_was_frozen_without_retraining_or_cherry_picking() -> None:
    payload = json.loads(FREEZE.read_text())
    assert payload["status"] == "FROZEN_BEFORE_MODEL_SCORING"
    assert payload["model_policy"]["retraining_or_finetuning"] is False
    assert payload["model_policy"]["report_all_frozen_controls"] is True
    assert payload["model_policy"]["ranking_direction"] == "within each new target over all 720 old drugs"
    assert [target["uniprot_accession"] for target in payload["targets"]] == ["Q99611", "A5D8V6"]
    for target in payload["targets"]:
        assert len(target["sequence"]) == target["sequence_length"]
        assert sequence_sha256(target["sequence"]) == target["sequence_sha256"]


def test_recent_target_features_reproduce_frozen_reference_pipelines() -> None:
    payload = json.loads((OUT / "RECENT_NOVEL_TARGET_FEATURE_AUDIT_V1.json").read_text())
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())
    assert payload["protbert"]["transformers_version"] == "4.46.3"
    assert payload["protbert"]["reference_max_abs_difference"] <= 1e-6
    assert payload["esm2"]["reference_max_abs_difference"] <= 1e-5


def test_recent_targets_and_pairs_are_database_disjoint_but_old_drugs_are_present() -> None:
    payload = json.loads((OUT / "RECENT_NOVEL_TARGET_DATABASE_AUDIT_V1.json").read_text())
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())
    for target in payload["targets"]:
        assert target["training_target_exact_sequence_rows"] == 0
        assert target["project_888_accession_rows"] == 0
        assert target["bindingdb_target_tokens_absent"] is True
        assert target["chembl37"]["target_absent_from_chembl37"] is True
        assert target["chembl37"]["experimental_pair_absent_from_chembl37"] is True
        assert target["old_drug_720_rows"] == 1


def test_recent_target_retrieval_reports_the_frozen_negative_result_exactly() -> None:
    summary = json.loads((OUT / "RECENT_NOVEL_TARGET_EXTERNAL_SUMMARY_V1.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["prospective_claim_allowed"] is False
    assert summary["claim_level"] == "PRIMARY_MISSED_PREDECLARED_TOP5_PERCENT"
    assert summary["ranking_space"] == {
        "targets": 2,
        "old_drugs_per_target": 720,
        "pairs": 1440,
        "direction": "old drugs ranked within each target",
    }
    ranks = {
        row["uniprot_accession"]: row["rank_within_target_720"]
        for row in summary["positive_control_results"]
    }
    assert ranks == {"Q99611": 395, "A5D8V6": 219}
    assert summary["predeclared_threshold_results"]["positive_controls_top5_percent"] == 0
    assert set(summary["structure_fallback"]["max_abs_final_minus_base_logit_by_seed"].values()) == {0.0}

    scores = pd.read_csv(OUT / "RECENT_NOVEL_TARGET_720_DRUG_SCORES_V1.csv.gz")
    assert len(scores) == 1440
    assert scores.groupby("uniprot_accession")["ligand_inchikey"].nunique().to_dict() == {
        "A5D8V6": 720,
        "Q99611": 720,
    }
    assert int(scores["is_literature_experimental_positive"].sum()) == 2
