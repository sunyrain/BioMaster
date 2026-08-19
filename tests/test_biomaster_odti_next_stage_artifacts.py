from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomaster.odti_w1_v3 import W1V3AdapterError, W1_V3_FIELDS, adapt_w1_v3_rows


ROOT = Path(__file__).resolve().parents[1]


def require_local_artifact(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"local generated artifact is not distributed: {path.relative_to(ROOT)}")
    return path


def test_e2_screen_protocol_is_frozen_without_promoting_a_new_weight() -> None:
    path = ROOT / "configs/biomaster_odti_e2_protein_residual_screen_20260817.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "READY_PENDING_FEATURES"
    assert payload["baseline"]["keep_target_base"] == "ProtBERT1024"
    assert payload["candidate"]["source_path"] is None
    assert payload["screen"]["protocols"] == [
        "S2_HOMOLOGY_COLD_TARGET",
        "S3_STRICT_DOUBLE_COLD",
        "S5_OLD_DRUG_ENTITY_COLD",
    ]
    assert payload["promotion_gates"]["unknown_pairs_are_not_negative"] is True


def test_davis_candidate_is_manifested_but_external_gate_is_closed() -> None:
    summary = ROOT / (
        "outputs/biomaster_odti_local_external_candidates_v1/"
        "LOCAL_EXTERNAL_ENTITY_COLD_CANDIDATES_V1.json"
    )
    require_local_artifact(summary)
    payload = json.loads(summary.read_text())
    davis = payload["sources"]["davis_complete_secondary_hf"]
    assert payload["status"] == "PASS"
    assert davis["status"] == "SCORED_POSITIVE_ONLY_ENTITY_COLD"
    assert davis["pre_score_status"] == "CANDIDATE_REQUIRES_NEW_FEATURES"
    assert davis["exact_frozen_pair_overlap_unique_pairs"] == 810
    assert davis["nonoverlap_unique_pairs"] == 24962
    assert davis["both_unseen_nonoverlap_unique_pairs"] == 8120
    assert payload["gate_decision"]["true_entity_cold_external_scored"] is False
    assert payload["gate_decision"]["davis_score_ready_in_current_store"] is True
    assert payload["gate_decision"]["davis_positive_only_scored"] is True
    manifest = ROOT / davis["frozen_manifest"]["path"]
    assert manifest.is_file()


def test_e2_feature_readiness_blocks_duplicate_local_weights() -> None:
    path = ROOT / (
        "outputs/biomaster_odti_e2_feature_readiness_v1/"
        "E2_FEATURE_SOURCE_READINESS_V1.json"
    )
    require_local_artifact(path)
    payload = json.loads(path.read_text())
    assert payload["state"] == "PROTOCOL_READY_BLOCKED_NO_DISTINCT_LOCAL_FEATURE_SOURCE"
    assert payload["eligible_distinct_local_weight_count"] == 0
    assert payload["decision"]["start_training_now"] is False


def test_thermoprot_is_not_misclassified_as_a_new_e2_plm() -> None:
    path = ROOT / (
        "outputs/biomaster_odti_e2_thermoprot_audit_v1/"
        "THERMOPROT_E2_ELIGIBILITY_AUDIT_V1.json"
    )
    require_local_artifact(path)
    payload = json.loads(path.read_text())
    assert payload["status"] == "PASS"
    assert payload["decision"]["eligible_as_e2_source"] is False
    assert payload["decision"]["start_e2_training"] is False
    assert payload["task_signature"]["uses_esm_input"] is True
    assert payload["task_signature"]["contains_contact_or_pair_task_heads"] is True
    assert payload["biomaster_e2_feature_contract"]["standalone_pooled_protein_embedding"] is False


def test_s5_calibration_audit_is_validation_only_and_not_promoted() -> None:
    path = ROOT / (
        "outputs/biomaster_odti_s5_calibration_audit_v1/"
        "ODTI_S5_CALIBRATION_AUDIT_V1.json"
    )
    require_local_artifact(path)
    payload = json.loads(path.read_text())
    assert payload["status"] == "PASS"
    assert payload["split"]["test_labels_untouched_for_fit"] is True
    assert payload["claim_status"].startswith("VALIDATION_ONLY")
    assert payload["test_metrics"]["beta"]["ece_15"] < payload["test_metrics"]["stored_temperature"]["ece_15"]


def test_w1_v3_training_preflight_separates_template_pass_from_training_ready() -> None:
    path = ROOT / (
        "outputs/biomaster_odti_w1_v3_training_preflight_v1/"
        "W1_V3_TRAINING_PREFLIGHT_AUDIT_V1.json"
    )
    require_local_artifact(path)
    payload = json.loads(path.read_text())
    assert payload["status"] == "PASS"
    assert payload["decision"]["w1_templates_structurally_ready"] is True
    assert payload["decision"]["identity_provenance_bridge_ready"] is True
    assert payload["decision"]["direct_v3_training_ready"] is False
    assert payload["decision"]["start_v3_training_now"] is False
    assert payload["counts"]["synthetic_pass_cells"] == 0
    assert "activity_class" in payload["semantic_adapter_required"]["fields"]
    assert "replicate_variance" in payload["semantic_adapter_required"]["fields"]


def test_w1_semantic_template_is_rejected_fail_closed() -> None:
    template = ROOT / (
        "outputs/biomaster_odti_w1_v3_semantic_input_v1/"
        "W1_V3_SEMANTIC_RESULT_INPUT_16_V1.csv"
    )
    require_local_artifact(template)
    import csv

    with template.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        adapt_w1_v3_rows(rows)
    except W1V3AdapterError as exc:
        codes = {item["code"] for item in exc.errors}
        assert "PENDING_OR_LOCKED_SEMANTICS" in codes
        assert "NOT_CONTROLLED_UNBLINDED" in codes
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("pending/locked W1 template must not enter V3")


def test_w1_semantic_adapter_accepts_explicit_unseen_pair_without_negative_imputation() -> None:
    base = {field: "" for field in W1_V3_FIELDS}

    def make(run: str) -> dict[str, str]:
        row = dict(base)
        row.update(
            {
                "w1_candidate_id": "W1_RANK_01",
                "prospective_pair_id": "W1_V17_R01_TARGET",
                "calibration_pair_id": "",
                "training_store_overlap_status": "UNSEEN_LIGAND_OR_PAIR",
                "drug_entity_key": "TEST-INCHIKEY",
                "target_entity_key": "CHEMBL_TEST",
                "assay_id": "ASSAY_TEST",
                "assay_lane": "ENZYME_BIOCHEMICAL",
                "plate_id": "PLATE_TEST",
                "blinded_sample_code": f"BLIND_{run}",
                "independent_run": run,
                "replicate_id": f"REP_{run}",
                "readout_value": "1.0",
                "readout_unit": "uM",
                "activity_class": "active",
                "assay_status": "PASS",
                "replicate_variance": "0.04",
                "assay_metadata_json": '{"construct":"test","qc":"pass"}',
                "raw_data_filename": f"raw_{run}.csv",
                "raw_data_file_sha256": "a" * 64,
                "unblinding_status": "CONTROLLED_UNBLINDED",
            }
        )
        return row

    normalized, summary = adapt_w1_v3_rows([make("1"), make("2")])
    assert len(normalized) == 2
    assert summary["unknown_pair_is_negative"] is False
    assert summary["prospective_pairs"] == 1


def test_readiness_exposes_fail_closed_semantic_adapter_state() -> None:
    path = ROOT / (
        "outputs/biomaster_odti_model_data_readiness_v1/"
        "BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.json"
    )
    require_local_artifact(path)
    payload = json.loads(path.read_text())
    adapter = payload["execution_readiness"]["w1_v3_semantic_adapter"]
    assert adapter["adapter_status"] == "REJECTED_NOT_TRAINING_READY"
    assert adapter["accepted"] is False
    assert adapter["output_csv_written"] is False
    assert adapter["training_ready"] is False
