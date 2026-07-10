from __future__ import annotations

import pandas as pd

from scripts.build_final_v4_delivery_package import (
    assign_review_queue,
    build_assay_matrix,
    build_nomination_plate_map,
    join_reviewed,
)


def test_review_queue_separates_novel_validated_and_deprioritized() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["novel", "validated", "contradictory"],
            "final384_rank": [1, 2, 3],
            "priority_score_v2": [80.0, 90.0, 95.0],
            "agent_feasibility_grade": ["A", "A", "B"],
            "agent_confidence": ["high", "high", "medium"],
            "agent_database_query_resolution": ["not_needed", "not_needed", "not_needed"],
            "chembl_activity_query_ok": [True, True, True],
            "lit_ok": [True, True, True],
            "agent_literature_class": [
                "no_exact_report_found",
                "exact_pair_validated",
                "contradictory",
            ],
            "chembl_exact_activity_status": [
                "no_exact_chembl_activity_record",
                "exact_binding_activity_pchembl_ge_5",
                "no_exact_chembl_activity_record",
            ],
        }
    )

    result = assign_review_queue(frame).set_index("pair_id")

    assert result.loc["novel", "review_queue"] == "A_novel_priority"
    assert result.loc["validated", "review_queue"] == "P_validated_control_or_rediscovery"
    assert result.loc["contradictory", "review_queue"] == "D_deprioritize"


def test_review_queue_fails_closed_when_database_queries_are_unresolved() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["unresolved", "manually_resolved"],
            "final384_rank": [1, 2],
            "priority_score_v2": [90.0, 89.0],
            "agent_feasibility_grade": ["A", "A"],
            "agent_confidence": ["high", "high"],
            "agent_database_query_resolution": ["unresolved", "resolved_manually"],
            "agent_literature_class": ["no_exact_report_found", "no_exact_report_found"],
            "chembl_exact_activity_status": ["no_exact_chembl_activity_record"] * 2,
            "chembl_activity_query_ok": [False, False],
            "lit_ok": [True, True],
        }
    )

    result = assign_review_queue(frame).set_index("pair_id")

    assert result.loc["unresolved", "review_queue"] == "Q_database_query_incomplete"
    assert result.loc["manually_resolved", "review_queue"] == "A_novel_priority"


def test_join_reviewed_requires_all_formal_pairs() -> None:
    formal = pd.DataFrame({"pair_id": ["p1", "p2"], "priority_score_v2": [1.0, 2.0]})
    reviewed = pd.DataFrame(
        {
            "pair_id": ["p1", "p2"],
            "agent_feasibility_grade": ["A", "B"],
            "agent_verdict": ["keep", "review"],
        }
    )

    result = join_reviewed(formal, reviewed)

    assert result["agent_feasibility_grade"].tolist() == ["A", "B"]


def test_assay_matrix_adds_known_positive_and_original_target_counterscreen() -> None:
    final = pd.DataFrame(
        {
            "pair_id": ["pair"],
            "sequence_key": ["SEQ1"],
            "drug_names": ["Candidate"],
            "primary_gene": ["GENE1"],
            "target_assay_family_v2": ["enzyme"],
            "default_assay_strategy": ["enzyme assay"],
            "agent_assay_plan": ["orthogonal binding"],
            "fda_target_names": ["ORIGINAL"],
        }
    )
    controls = pd.DataFrame(
        {
            "sequence_key": ["SEQ1"],
            "drug_chembl_id": ["CHEMBL1"],
            "known_action_type": ["INHIBITOR"],
            "known_source": ["known"],
        }
    )
    library = pd.DataFrame({"drug_id": ["CHEMBL1"], "drug_name": ["Positive"]})

    result = build_assay_matrix(final, controls, pd.DataFrame(), library)

    assert result.loc[0, "positive_control_names"] == "Positive"
    assert result.loc[0, "positive_control_review_required"] == False  # noqa: E712
    assert result.loc[0, "known_target_counterscreen"] == "ORIGINAL"


def test_nomination_map_is_exactly_four_blocks_of_96() -> None:
    final = pd.DataFrame(
        {
            "pair_id": [f"P{index}" for index in range(384)],
            "review_adjusted_rank": range(1, 385),
            "drug_names": [f"D{index}" for index in range(384)],
            "primary_gene": [f"G{index % 20}" for index in range(384)],
            "target_assay_family_v2": ["enzyme"] * 384,
            "review_queue": ["B_novel_testable"] * 384,
        }
    )

    result = build_nomination_plate_map(final)

    assert result.groupby("nomination_block").size().to_dict() == {
        "Block_1": 96,
        "Block_2": 96,
        "Block_3": 96,
        "Block_4": 96,
    }
    assert result["position"].nunique() == 96
