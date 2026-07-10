from __future__ import annotations

from scripts.audit_final_candidates_chembl_activity import summarize
from scripts.build_comprehensive_repurposing_literature_report import terms_for_drug


def test_chembl_summary_requires_binding_assay_and_numeric_pchembl() -> None:
    result = summarize(
        [
            {"assay_type": "F", "pchembl_value": "8.0", "activity_id": 1},
            {"assay_type": "B", "pchembl_value": "invalid", "activity_id": 2},
            {
                "assay_type": "B",
                "pchembl_value": "6.2",
                "activity_id": 3,
                "assay_chembl_id": "ASSAY3",
                "standard_relation": "=",
                "standard_flag": 1,
                "data_validity_comment": None,
                "assay_variant_mutation": None,
            },
        ],
        {"ASSAY3": {"ok": True, "confidence_score": 9}},
    )

    assert result["chembl_exact_activity_status"] == "exact_binding_activity_pchembl_ge_5"
    assert result["chembl_exact_binding_count"] == 2
    assert result["chembl_exact_binding_pchembl_ge_5_count"] == 1
    assert result["chembl_exact_max_binding_pchembl"] == 6.2


def test_pubmed_drug_terms_split_semicolon_aliases_and_strip_salts() -> None:
    terms = terms_for_drug("Rilpivirine Hydrochloride; Rilpivirine")

    assert "Rilpivirine Hydrochloride" in terms
    assert "Rilpivirine" in terms
    assert all(";" not in term for term in terms)


def test_chembl_strong_but_low_confidence_binding_requires_manual_review() -> None:
    result = summarize(
        [
            {
                "assay_type": "B",
                "pchembl_value": "7.0",
                "activity_id": 1,
                "assay_chembl_id": "LOW",
                "standard_relation": "=",
                "standard_flag": 1,
                "data_validity_comment": None,
                "assay_variant_mutation": None,
            }
        ],
        {"LOW": {"ok": True, "confidence_score": 4}},
    )

    assert result["chembl_exact_activity_status"] == "manual_exact_binding_review"
    assert result["chembl_exact_binding_pchembl_ge_5_count"] == 0
    assert result["chembl_exact_manual_binding_review_count"] == 1
