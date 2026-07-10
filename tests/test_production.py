from __future__ import annotations

import pandas as pd

from biomaster.production import (
    add_boltz_review_class_v3,
    add_priority_score_v2,
    annotate_candidate_risk,
    canonical_assay_family,
    diverse_select,
    select_formal_packages,
    select_reviewed_final384,
    shared_reference_percentile,
    specific_family_tokens,
)
from scripts.select_reviewed_final384_v4 import merge_review_evidence


def test_boltz_review_class_accepts_concordant_c_without_calling_it_ab() -> None:
    df = pd.DataFrame(
        {
            "boltz_completed_refined": [True, True, False],
            "boltz_support_tier_refined": [
                "C_boltz_partial_signal_review",
                "C_boltz_partial_signal_review",
                "U_boltz_not_completed",
            ],
            "boltz_affinity_probability_refined": [0.70, 0.20, 0.90],
            "boltz_ligand_iptm_refined": [0.60, 0.90, 0.90],
            "boltz_confidence_score_refined": [0.30, 0.90, 0.90],
            "boltz_complex_iplddt_refined": [0.30, 0.90, 0.90],
        }
    )

    reviewed = add_boltz_review_class_v3(df)

    assert reviewed["boltz_review_class_v3"].tolist() == [
        "C_concordant_multi_metric",
        "completed_low_or_single_metric",
        "low_or_incomplete",
    ]
    assert reviewed["boltz_substantive_signal_v3"].tolist() == [True, False, False]


def test_shared_reference_percentile_uses_one_scale() -> None:
    reference = pd.Series([0.1, 0.2, 0.3, 0.4])
    discovery = shared_reference_percentile(pd.Series([0.2, 0.4]), reference)
    controls = shared_reference_percentile(pd.Series([0.2, 0.4]), reference)

    assert discovery.tolist() == [0.5, 1.0]
    assert controls.tolist() == discovery.tolist()


def test_shared_reference_percentile_uses_midrank_for_ties() -> None:
    result = shared_reference_percentile(pd.Series([0.0]), pd.Series([0.0, 0.0, 0.0, 1.0]))

    assert result.tolist() == [0.5]


def test_priority_score_has_no_absolute_conplex_threshold_jump_or_pocket_double_count() -> None:
    df = pd.DataFrame(
        {
            "conplex_score": [0.0999, 0.1001],
            "rank_within_drug": [10, 10],
            "target_rank": [20, 20],
            "drug_pair_count_in_project_space": [463, 463],
            "target_pair_count_in_project_space": [750, 750],
            "structure_bin": ["B_strict_supported_overlap", "B_strict_supported_overlap"],
            "p2rank_puresnet_overlap_fraction": [0.0, 1.0],
            "top_pocket_probability": [0.0, 0.9],
            "anchor_availability_tier": ["A1_SM_approved_drug", "A1_SM_approved_drug"],
            "drug_feasibility_score": [10, 10],
            "experimental_feasibility_score": [15, 15],
        }
    )

    scored = add_priority_score_v2(df, pd.Series([0.0, 0.1, 0.2]))

    assert scored["target_pocket_prior_component_v2"].tolist() == [12.0, 12.0]
    assert abs(scored.loc[1, "pair_conplex_component_v2"] - scored.loc[0, "pair_conplex_component_v2"]) < 5.0


def test_canonical_assay_family_prefers_curated_ot_value() -> None:
    df = pd.DataFrame(
        {
            "target_assay_family": ["enzyme", "other_assayable", "transporter"],
            "anchor_project_assay_family": ["kinase", "ion_channel", "excluded_or_review"],
        }
    )

    assert canonical_assay_family(df).tolist() == ["kinase", "ion_channel", "transporter"]


def test_specific_family_tokens_do_not_treat_assay_class_as_family() -> None:
    assert specific_family_tokens("Enzyme; Kinase; Protein Kinase") == set()
    assert specific_family_tokens(
        "Enzyme; Kinase; Tyrosine protein kinase VEGFR family"
    ) == {"tyrosine protein kinase vegfr family"}
    assert specific_family_tokens("Transporter; SLC superfamily of solute carriers") == set()


def test_risk_annotation_separates_assay_similarity_from_family_extension() -> None:
    anchors = pd.DataFrame(
        {
            "gene": ["KNOWN", "CANDIDATE", "OTHER"],
            "project_assay_family": ["enzyme", "enzyme", "enzyme"],
            "target_class_labels": [
                "Enzyme; Named alpha family",
                "Enzyme; Named alpha family",
                "Enzyme; Named beta family",
            ],
        }
    )
    controls = pd.DataFrame(
        {"drug_chembl_id": ["CHEMBL1"], "gene_names": ["KNOWN"]}
    )
    candidates = pd.DataFrame(
        {
            "drug_chembl_id": ["CHEMBL1", "CHEMBL1"],
            "primary_gene": ["CANDIDATE", "OTHER"],
            "gene_names": ["CANDIDATE", "OTHER"],
            "target_assay_family": ["enzyme", "enzyme"],
            "anchor_project_assay_family": ["enzyme", "enzyme"],
            "anchor_target_class_labels": [
                "Enzyme; Named alpha family",
                "Enzyme; Named beta family",
            ],
            "fda_original_target_family": ["enzyme", "enzyme"],
            "is_known_fda_target_pair": [False, False],
        }
    )

    annotated = annotate_candidate_risk(candidates, controls, anchors)

    assert annotated["same_assay_family_only"].tolist() == [True, True]
    assert annotated["specific_target_family_extension_risk"].tolist() == [True, False]
    assert annotated["family_or_rediscovery_risk_v2"].tolist() == [True, False]


def test_known_kinase_context_catches_legacy_misclassification() -> None:
    anchors = pd.DataFrame(
        {
            "gene": ["CDK6", "CDK4"],
            "project_assay_family": ["kinase", "kinase"],
            "target_class_labels": ["Enzyme; Kinase", "Enzyme; Kinase"],
        }
    )
    controls = pd.DataFrame({"drug_chembl_id": ["CHEMBL1"], "gene_names": ["CDK6"]})
    candidate = pd.DataFrame(
        {
            "drug_chembl_id": ["CHEMBL1"],
            "primary_gene": ["CDK4"],
            "gene_names": ["CDK4"],
            "target_assay_family": ["enzyme"],
            "anchor_project_assay_family": ["kinase"],
            "anchor_target_class_labels": ["Enzyme; Kinase"],
            "fda_original_target_family": ["unknown"],
            "fda_target_names": ["CDK6/cyclin D1"],
            "fda_moa": [""],
            "is_known_fda_target_pair": [False],
        }
    )

    annotated = annotate_candidate_risk(candidate, controls, anchors)

    assert annotated.loc[0, "target_assay_family_v2"] == "kinase"
    assert bool(annotated.loc[0, "kinase_to_kinase_risk"])
    assert annotated.loc[0, "candidate_role_v2"] == "family_extension_or_rediscovery_control"


def test_known_context_is_shared_across_equivalent_active_moiety_ids() -> None:
    anchors = pd.DataFrame(
        {
            "gene": ["KNOWN"],
            "project_assay_family": ["enzyme"],
            "target_class_labels": ["Enzyme"],
        }
    )
    controls = pd.DataFrame(
        {
            "drug_chembl_id": ["SALT_A"],
            "knowledge_compound_key": ["ACTIVE"],
            "gene_names": ["KNOWN"],
        }
    )
    candidate = pd.DataFrame(
        {
            "drug_chembl_id": ["SALT_B"],
            "active_moiety_smiles": ["ACTIVE"],
            "primary_gene": ["KNOWN"],
            "gene_names": ["KNOWN"],
            "target_assay_family": ["enzyme"],
            "anchor_project_assay_family": ["enzyme"],
            "anchor_target_class_labels": ["Enzyme"],
            "fda_original_target_family": ["unknown"],
            "is_known_fda_target_pair": [False],
        }
    )

    annotated = annotate_candidate_risk(candidate, controls, anchors)

    assert bool(annotated.loc[0, "exact_known_target_v2"])


def test_diverse_select_is_deterministic_and_respects_initial_caps() -> None:
    rows = []
    for index in range(6):
        rows.append(
            {
                "pair_id": f"P{index}",
                "drug_chembl_id": f"D{index // 2}",
                "primary_gene": f"G{index}",
                "sequence_key": f"S{index}",
                "murcko_scaffold": f"M{index}",
                "target_assay_family_v2": "enzyme",
                "priority_score_v2": 100 - index,
                "pair_specific_evidence_score_v2": 50 - index,
                "conplex_score": 1 - index / 10,
            }
        )
    df = pd.DataFrame(rows)

    first = diverse_select(df, 3, drug_cap=1, target_cap=1, scaffold_cap=1)
    second = diverse_select(df, 3, drug_cap=1, target_cap=1, scaffold_cap=1)

    assert first["pair_id"].tolist() == second["pair_id"].tolist()
    assert first["drug_chembl_id"].nunique() == 3


def test_diverse_select_deduplicates_salt_forms_by_active_moiety_and_target() -> None:
    df = pd.DataFrame(
        {
            "pair_id": ["salt_G1", "parent_G1", "salt_G2"],
            "drug_chembl_id": ["SALT", "PARENT", "SALT"],
            "active_moiety_smiles": ["CCO", "CCO", "CCO"],
            "primary_gene": ["G1", "G1", "G2"],
            "sequence_key": ["S1", "S1", "S2"],
            "murcko_scaffold": ["M", "M", "M"],
            "target_assay_family_v2": ["enzyme", "enzyme", "enzyme"],
            "priority_score_v2": [100, 99, 98],
            "pair_specific_evidence_score_v2": [50, 49, 48],
            "conplex_score": [0.9, 0.8, 0.7],
        }
    )

    selected = diverse_select(df, 3, drug_cap=3, target_cap=3, scaffold_cap=3)

    assert selected["pair_id"].tolist() == ["salt_G1", "salt_G2"]


def test_diverse_select_does_not_silently_relax_hard_caps() -> None:
    df = pd.DataFrame(
        {
            "pair_id": ["P1", "P2"],
            "drug_chembl_id": ["D", "D"],
            "active_moiety_smiles": ["CC", "CC"],
            "primary_gene": ["G1", "G2"],
            "sequence_key": ["S1", "S2"],
            "murcko_scaffold": ["M", "M"],
            "target_assay_family_v2": ["enzyme", "enzyme"],
            "priority_score_v2": [2.0, 1.0],
        }
    )

    selected = diverse_select(df, 2, drug_cap=1, target_cap=2, scaffold_cap=2)

    assert selected["pair_id"].tolist() == ["P1"]


def test_formal_selection_does_not_apply_a_discontinuous_ab_bonus() -> None:
    rows = pd.DataFrame(
        {
            "pair_id": ["AB_LOW", "C_HIGH"],
            "drug_chembl_id": ["D1", "D2"],
            "active_moiety_smiles": ["CC", "CCC"],
            "primary_gene": ["G1", "G2"],
            "sequence_key": ["S1", "S2"],
            "murcko_scaffold": ["M1", "M2"],
            "target_assay_family_v2": ["enzyme", "enzyme"],
            "priority_score_v2": [50.0, 90.0],
            "pair_specific_evidence_score_v2": [25.0, 45.0],
            "conplex_score": [0.2, 0.4],
            "boltz_completed_refined": [True, True],
            "boltz_support_tier_refined": [
                "B_boltz_review_supported",
                "C_boltz_partial_signal_review",
            ],
            "boltz_affinity_probability_refined": [0.61, 0.75],
            "boltz_ligand_iptm_refined": [0.50, 0.70],
            "boltz_confidence_score_refined": [0.41, 0.38],
            "boltz_complex_iplddt_refined": [0.40, 0.38],
            "exact_known_target_v2": [False, False],
            "family_or_rediscovery_risk_v2": [False, False],
            "severe_compound_liability": [False, False],
            "anchor_project_standard_direct_sm": [True, True],
            "structure_bin": ["A_strict_overlapping_pocket", "A_strict_overlapping_pocket"],
            "ion_channel_feasibility_flag": [False, False],
            "pose_stability_completed": [True, True],
            "pose_stability_tier": ["A_stable_conditional_pose", "A_stable_conditional_pose"],
        }
    )
    config = {
        "final1000": {
            "size": 1,
            "drug_cap": 1,
            "target_cap": 1,
            "scaffold_cap": 1,
            "family_caps": {"enzyme": 1},
        },
        "final384": {
            "size": 1,
            "drug_cap": 1,
            "target_cap": 1,
            "scaffold_cap": 1,
            "family_caps": {"enzyme": 1},
        },
    }

    final, nomination = select_formal_packages(rows, config)

    assert final["pair_id"].tolist() == ["C_HIGH"]
    assert nomination["pair_id"].tolist() == ["C_HIGH"]


def test_nomination_fails_closed_without_pose_audit() -> None:
    row = pd.DataFrame(
        {
            "pair_id": ["P"],
            "drug_chembl_id": ["D"],
            "active_moiety_smiles": ["CC"],
            "primary_gene": ["G"],
            "sequence_key": ["S"],
            "murcko_scaffold": ["M"],
            "target_assay_family_v2": ["enzyme"],
            "priority_score_v2": [90.0],
            "pair_specific_evidence_score_v2": [50.0],
            "conplex_score": [0.5],
            "boltz_completed_refined": [True],
            "boltz_support_tier_refined": ["A_boltz_second_model_supported"],
            "boltz_affinity_probability_refined": [0.9],
            "boltz_ligand_iptm_refined": [0.9],
            "boltz_confidence_score_refined": [0.9],
            "boltz_complex_iplddt_refined": [0.9],
            "exact_known_target_v2": [False],
            "family_or_rediscovery_risk_v2": [False],
            "severe_compound_liability": [False],
            "anchor_project_standard_direct_sm": [True],
            "structure_bin": ["A_strict_overlapping_pocket"],
            "ion_channel_feasibility_flag": [False],
        }
    )
    config = {
        "final1000": {
            "size": 1,
            "drug_cap": 1,
            "target_cap": 1,
            "scaffold_cap": 1,
            "family_caps": {"enzyme": 1},
        },
        "final384": {
            "size": 1,
            "drug_cap": 1,
            "target_cap": 1,
            "scaffold_cap": 1,
            "family_caps": {"enzyme": 1},
        },
    }

    final, nomination = select_formal_packages(row, config)

    assert len(final) == 1
    assert nomination.empty


def test_post_review_selection_replaces_d_and_unresolved_rows() -> None:
    rows = pd.DataFrame(
        {
            "pair_id": ["D_HIGH", "QUERY_FAIL", "KEEP_A", "KEEP_B"],
            "drug_chembl_id": ["D1", "D2", "D3", "D4"],
            "active_moiety_smiles": ["CC", "CCC", "CCCC", "CCCCC"],
            "primary_gene": ["G1", "G2", "G3", "G4"],
            "sequence_key": ["S1", "S2", "S3", "S4"],
            "murcko_scaffold": ["M1", "M2", "M3", "M4"],
            "target_assay_family_v2": ["enzyme"] * 4,
            "priority_score_v2": [99.0, 98.0, 80.0, 79.0],
            "pair_specific_evidence_score_v2": [50.0, 49.0, 40.0, 39.0],
            "conplex_score": [0.9, 0.8, 0.7, 0.6],
            "agent_feasibility_grade": ["D", "A", "A", "B"],
            "agent_literature_class": ["no_exact_report_found"] * 4,
            "agent_confidence": ["high"] * 4,
            "agent_database_query_resolution": ["not_needed", "unresolved", "not_needed", "not_needed"],
            "chembl_activity_query_ok": [True, False, True, True],
            "lit_ok": [True] * 4,
            "chembl_exact_activity_status": ["no_exact_chembl_activity_record"] * 4,
        }
    )
    config = {
        "size": 2,
        "drug_cap": 1,
        "target_cap": 1,
        "scaffold_cap": 1,
        "family_caps": {"enzyme": 2},
    }

    selected = select_reviewed_final384(rows, config, validated_control_cap=1)

    assert selected["pair_id"].tolist() == ["KEEP_A", "KEEP_B"]


def test_prepare_merge_select_preserves_plain_lit_ok_fail_closed_status() -> None:
    pool = pd.DataFrame(
        {
            "pair_id": ["OK", "FAIL"],
            "drug_chembl_id": ["D1", "D2"],
            "active_moiety_smiles": ["CC", "CCC"],
            "primary_gene": ["G1", "G2"],
            "sequence_key": ["S1", "S2"],
            "murcko_scaffold": ["M1", "M2"],
            "target_assay_family_v2": ["enzyme", "enzyme"],
            "priority_score_v2": [80.0, 99.0],
            "pair_specific_evidence_score_v2": [40.0, 50.0],
            "conplex_score": [0.7, 0.9],
        }
    )
    reviewed = pd.DataFrame(
        {
            "pair_id": ["OK", "FAIL"],
            "agent_feasibility_grade": ["A", "A"],
            "agent_literature_class": ["no_exact_report_found", "no_exact_report_found"],
            "agent_confidence": ["high", "high"],
            "agent_database_query_resolution": ["not_needed", "unresolved"],
            "chembl_activity_query_ok": [True, True],
            "chembl_hierarchy_query_ok": [True, True],
            "lit_ok": [True, False],
            "pubmed_query_schema": ["v2", "v2"],
            "pubmed_query_sha256": ["hash-ok", "hash-fail"],
            "chembl_exact_activity_status": ["no_exact_chembl_activity_record"] * 2,
        }
    )
    merged = merge_review_evidence(pool, reviewed)
    config = {
        "size": 1,
        "drug_cap": 1,
        "target_cap": 1,
        "scaffold_cap": 1,
        "family_caps": {"enzyme": 1},
    }

    assert merged["lit_ok"].tolist() == [True, False]
    selected = select_reviewed_final384(merged, config, validated_control_cap=1)
    assert selected["pair_id"].tolist() == ["OK"]
