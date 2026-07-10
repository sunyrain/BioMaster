from __future__ import annotations

from scripts.annotate_compound_assay_liability import molecule_annotations


def test_brenk_only_alert_is_developability_not_assay_interference() -> None:
    result = molecule_annotations("Cc1cc(/C=C/C#N)cc(C)c1Nc1ccnc(Nc2ccc(C#N)cc2)n1")

    assert result["liability_brenk_alerts"]
    assert result["brenk_developability_review"] is True
    assert result["assay_interference_review"] is False
