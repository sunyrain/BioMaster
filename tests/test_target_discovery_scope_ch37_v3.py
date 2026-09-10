from __future__ import annotations

import pandas as pd

from scripts.build_target_discovery_scope_ch37_v3 import classify_discovery_scope


def _row(
    target_id: str,
    evidence: str,
    lane: str,
    structure: str = "none",
) -> dict[str, object]:
    return {
        "target_chembl_id": target_id,
        "gene_symbol": target_id,
        "uniprot_accession": target_id,
        "sequence": f"M{target_id}",
        "sequence_sha256": f"hash-{target_id}",
        "set_non_gpcr_all": True,
        "set_small_molecule_moa_non_gpcr": evidence == "chembl",
        "ot_project_standard_direct_sm": evidence == "ot",
        "structure_ready_permissive": structure in {"permissive", "strict"},
        "structure_ready_strict": structure == "strict",
        "af_exact_sequence_model": structure != "none",
        "calibration_8x8": evidence == "chembl",
        "assay_lane": lane,
        "evidence_class": evidence,
    }


def test_non_gpcr_registry_routes_instead_of_ranking_every_target_together() -> None:
    master = pd.DataFrame(
        [
            _row("BIOCHEM", "chembl", "ENZYME_BIOCHEMICAL", "strict"),
            _row("FUNCTIONAL", "ot", "ION_CHANNEL_FUNCTIONAL", "permissive"),
            _row("SPECIAL_DIRECT", "chembl", "EXTRACELLULAR_SPECIAL"),
            _row("EXPLORATORY", "none", "NUCLEAR_EPIGENETIC_DOMAIN"),
            _row("REGISTRY_ONLY", "none", "NONCANONICAL_REVIEW"),
            {
                **_row("GPCR", "chembl", "ENZYME_BIOCHEMICAL"),
                "set_non_gpcr_all": False,
            },
        ]
    )
    result = classify_discovery_scope(
        master,
        current_scored_target_ids={"BIOCHEM"},
        supervised_target_ids={"BIOCHEM", "FUNCTIONAL"},
        protbert_exact_sequences={"MBIOCHEM", "MFUNCTIONAL"},
    ).set_index("target_chembl_id")

    assert set(result.index) == {
        "BIOCHEM",
        "FUNCTIONAL",
        "SPECIAL_DIRECT",
        "EXPLORATORY",
        "REGISTRY_ONLY",
    }
    assert result["include_in_non_gpcr_registry_745"].all()
    assert result["include_in_primary_direct_sm_screen_450"].sum() == 2
    assert result.loc[
        "BIOCHEM", "include_in_primary_biochemical_screen_367"
    ]
    assert result.loc[
        "FUNCTIONAL", "include_in_primary_functional_screen_83"
    ]
    assert result.loc[
        "SPECIAL_DIRECT", "include_in_special_direct_sm_branch_42"
    ]
    assert result.loc[
        "EXPLORATORY", "include_in_exploratory_assayable_frontier_8"
    ]
    assert result.loc[
        "REGISTRY_ONLY", "registry_only_special_no_direct_sm_245"
    ]
    assert result.loc["REGISTRY_ONLY", "rank_contract"] == (
        "NO_CURRENT_PRODUCTION_RANK"
    )
    assert result["missing_relation_semantics"].eq("UNKNOWN_NOT_NEGATIVE").all()
