#!/usr/bin/env python3
"""Freeze the V18 full 30-candidate portfolio audit before bulk identity/risk/rank computation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V17 = RUN / "final_evidence_routing_v17"
OUT = RUN / "full_candidate_portfolio_v18"
PROTOCOL = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.json"
STAMP = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.sha256"
DEPENDENCIES = [
    V17 / "UNIFIED_V17_REPRODUCIBILITY_AUDIT.json",
    V17 / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz",
    V17 / "PROSPECTIVE_INTEGRATED_CASEBOOK_V10.csv",
    V17 / "MASTER_VALIDATION_QUEUE_63_ROWS_V17.csv",
    V17 / "W1_CANDIDATE_CHEMBL37_IDENTITY_8_V15.csv",
    V17 / "W1_BIDIRECTIONAL_MODEL_RANKS_64_V16.csv",
    V17 / "W1_PHYSICAL_MODEL_COVERAGE_AND_RANKS_8_V16.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in DEPENDENCIES if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "FROZEN_BEFORE_V18_BULK_PUBCHEM_FULL_30_RANK_AND_PORTFOLIO_RISK_COMPUTATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "population": {
            "candidate_pairs": 30,
            "unique_project_full_inchikey_entities": 28,
            "unique_targets": 16,
            "waves": {"W1_BLINDED_CANDIDATE_PILOT": 8, "W2_CONTINGENT_ONLY": 21, "VETO_NOT_AUTHORIZED": 1},
            "deployment": {"SEEDED_KNOWN_GRAPH_185": 19, "UNSEEDED_TARGET_DTA_199": 11},
            "selection_and_rank_immutable": True,
        },
        "authorization_boundaries": {
            "W1_BLINDED_CANDIDATE_PILOT": "May execute only after the frozen procurement and received-lot release gate passes.",
            "W2_CONTINGENT_ONLY": "Computational and pre-assay readiness audit allowed; procurement and assay execution remain unauthorized until a documented W1 trigger and new release decision.",
            "VETO_NOT_AUTHORIZED": "Computational audit only; no procurement, plate assignment or wet-lab execution is authorized.",
            "rank_rule": "No V18 result may promote, delete, demote or reorder a frozen candidate pair.",
        },
        "identity_audit": {
            "chembl37": [
                "Exact molecule_dictionary and compound_structures lookup using frozen ligand_species_chembl_ids_v7.",
                "Compare project full InChIKey to ChEMBL full InChIKey and separately compare source drug label to ChEMBL preferred entity name.",
                "Query prodrug, dosed ingredient, phase, approval year, molecule hierarchy and all known drug_mechanism rows.",
            ],
            "name_entity_classes": [
                "EXACT_NORMALIZED_NAME",
                "DISTINCT_NAMED_STEREOENTITY_LABEL_MISMATCH",
                "OTHER_NAME_ENTITY_MISMATCH_REQUIRES_MANUAL_ADJUDICATION",
            ],
            "predeclared_special_case": (
                "Frozen rank 21 source label ketoconazole uses CHEMBL295698/full InChIKey "
                "XMAYWYJOQHXEEK-ZEQKJWHPSA-N, whose ChEMBL preferred entity is LEVOKETOCONAZOLE. "
                "Keep rank and modeled entity frozen, but set ENTITY_NAME_ADJUDICATION_HOLD and never substitute "
                "racemic ketoconazole CHEMBL157101/full InChIKey XMAYWYJOQHXEEK-UHFFFAOYSA-N."
            ),
            "pubchem": (
                "Query current PubChem by exact project full InChIKey for 28 unique entities; retry missing keys "
                "individually; never relax primary identity to name or connectivity-only matching."
            ),
        },
        "chemistry_and_species_audit": {
            "unit": "28 unique full-InChIKey entities, then map without duplication to 30 frozen pairs",
            "descriptors": [
                "MW", "cLogP", "TPSA", "HBD", "HBA", "rotatable bonds", "formal charge",
                "fraction sp3", "specified chiral centers", "aromatic fraction",
            ],
            "catalog_alerts": ["PAINS", "Brenk"],
            "predeclared_substructures": [
                "boron", "nitro", "ester", "carboxylic acid/carboxylate", "nitrile",
                "alpha,beta-unsaturated carbonyl", "phenol", "amine", "disulfide",
            ],
            "prodrug_rule": (
                "All ChEMBL prodrug=1 entities require an explicit activation/metabolite/stability route. "
                "Known separate entity IDs remain separate and may never be merged."
            ),
            "risk_tier_rule": {
                "R3_HIGH_SPECIAL_HANDLING": (
                    "Predeclared prodrug/active-species ambiguity, project-ChEMBL full-key representation mismatch, "
                    "distinct named entity-label mismatch, boron covalent chemistry, or compound-specific known "
                    "reactive/covalent handling requirement."
                ),
                "R2_MODERATE_COUNTERASSAY": (
                    "Brenk/PAINS/property/readout alert or broad known-target family overlap without an R3 condition."
                ),
                "R1_STANDARD": "No R3 condition and no predeclared chemistry/property/family alert.",
                "interpretation": "Risk tier controls experiment design; it is never efficacy probability or reranking.",
            },
            "predeclared_compound_specific_r3": [
                "omaveloxolone", "nitazoxanide", "serdexmethylphenidate", "ixazomib",
                "romidepsin", "selinexor",
            ],
        },
        "full_portfolio_robustness": {
            "directions": ["TARGET_CENTERED_ACROSS_720_DRUGS", "DRUG_CENTERED_ACROSS_384_TARGETS_OR_FROZEN_BRANCH"],
            "views": ["CONPLEX", "DRUGCLIP", "DTA_CONSENSUS", "V10_BRANCH_PRIMARY"],
            "expected_rank_rows": 240,
            "target_model_pairwise_concordance_rows": 180,
            "rank_method": "Descending average-tie rank; empirical upper-tail percentile=(n-rank)/(n-1).",
            "top_thresholds": {"top10": "rank <= ceil(0.10*n)", "top20": "rank <= ceil(0.20*n)"},
            "robust_z": "(score-median)/(1.4826*MAD), undefined and explicitly retained when MAD=0 or finite n<50.",
            "physical_rule": (
                "Report Boltz and GNINA only when the exact frozen pair calculation exists; rank only within the "
                "target-specific selectively computed subset and expose missingness."
            ),
            "predeclared_missing_physical_case": (
                "Rank 21 modeled levoketoconazole-PARP2 has no completed main Boltz probability and no GNINA result; "
                "multiseed support metadata must not be used to impute the missing main result."
            ),
            "w1_regression_rule": "Ranks 1-8 must reproduce every matching V16 rank and physical field exactly.",
        },
        "target_class_assay_routes_for_w2": {
            "NUCLEAR_RECEPTOR": "coregulator/direct binding plus reporter or cellular engagement and receptor-selectivity controls",
            "KINASE": "biochemical ATP-near-Km plus NanoBRET/CETSA/direct binding, ATP shift and kinase selectivity panel",
            "METALLOPROTEASE_OR_PROTEASE": "substrate cleavage plus LC-MS/activity probe/direct binding and related-protease/readout controls",
            "METABOLIC_OR_SIGNAL_ENZYME": "substrate-product assay plus LC-MS/direct binding and substrate/redox/aggregation controls",
            "PARP": "DNA-dependent NAD+/PARylation assay plus direct engagement/cellular PAR and PARP1/selectivity controls",
        },
        "selection_bias_and_claim_boundaries": [
            "All 30 pairs were selected using overlapping internal evidence; ranks and agreement are descriptive sensitivity analysis, not independent validation or calibrated p-values.",
            "Target-lane retrospective support is calibration evidence and never exact-pair confirmation.",
            "Physical subsets are selectively computed and are not random negative-control panels.",
            "No computational score, identity match, vendor record, rank or agreement establishes binding, mechanism, efficacy or repurposing.",
            "No hard-gate target may be recovered; scope remains the frozen 384 targets including only the 46 recovered solely for lacking experimental pockets.",
            "W2 and veto audit artifacts do not authorize procurement, plate assignment or assay execution.",
        ],
        "frozen_dependencies": {str(path.relative_to(ROOT)): sha256(path) for path in DEPENDENCIES},
    }
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    value = sha256(PROTOCOL)
    STAMP.write_text(f"{value}  {PROTOCOL.name}\n")
    print(json.dumps({"status": "FROZEN", "protocol_sha256": value}, indent=2))


if __name__ == "__main__":
    main()
