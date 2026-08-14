#!/usr/bin/env python3
"""Independent verifier for V18 all-30 identity, chemistry, robustness and authorization artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V17 = RUN / "final_evidence_routing_v17"
OUT = RUN / "full_candidate_portfolio_v18"
PROTOCOL = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.json"
STAMP = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.sha256"
ROBUST_SUMMARY = OUT / "FULL30_COMPUTATIONAL_ROBUSTNESS_SUMMARY_V18.json"
IDENTITY_SUMMARY = OUT / "FULL30_IDENTITY_CHEMISTRY_READINESS_SUMMARY_V18.json"
AUDIT = OUT / "FULL30_PORTFOLIO_INDEPENDENT_AUDIT_V18.json"
PAIR = V17 / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz"
DB = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
RAW = OUT / "pubchem_full28_identity_raw_v18"
EXPECTED_PAIR_SHA256 = "8e1c73a0b16122bee6e4d3d7ca5dd9dbc816759c63a2976ec57799cdf97b7a9a"
HASH_CACHE: dict[Path, str] = {}


def sha256(path: Path) -> str:
    resolved = path.resolve()
    if resolved in HASH_CACHE:
        return HASH_CACHE[resolved]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest(); HASH_CACHE[resolved] = value
    return value


def vendor_boolean(payload: object) -> bool:
    if isinstance(payload, dict):
        if payload.get("TOCHeading") == "Chemical Vendors":
            return any(True in item.get("Value", {}).get("Boolean", []) for item in payload.get("Information", []))
        return any(vendor_boolean(value) for value in payload.values())
    if isinstance(payload, list):
        return any(vendor_boolean(value) for value in payload)
    return False


def recalc_chemistry(frame: pd.DataFrame) -> pd.DataFrame:
    pp = FilterCatalog.FilterCatalogParams(); pp.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    bp = FilterCatalog.FilterCatalogParams(); bp.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    pains = FilterCatalog.FilterCatalog(pp); brenk = FilterCatalog.FilterCatalog(bp)
    patterns = {
        "boron": "[#5]", "nitro": "[N+](=O)[O-]", "ester": "[CX3](=O)[OX2][#6]",
        "carboxylic_acid_or_carboxylate": "[CX3](=O)[OX1-,OX2H1]", "nitrile": "[CX2]#N",
        "alpha_beta_unsaturated_carbonyl": "[C,c]=[C,c][CX3](=O)", "phenol": "[c][OX2H]",
        "amine": "[NX3;!$(N-C=O);!$(N-S=O)]", "disulfide": "[SX2]-[SX2]",
    }
    queries = {name: Chem.MolFromSmarts(pattern) for name, pattern in patterns.items()}
    rows = []
    for row in frame.itertuples():
        mol = Chem.MolFromSmiles(row.project_smiles)
        pains_names = sorted({item.GetDescription() for item in pains.GetMatches(mol)})
        brenk_names = sorted({item.GetDescription() for item in brenk.GetMatches(mol)})
        counts = {name: len(mol.GetSubstructMatches(query)) for name, query in queries.items()}
        mw = Descriptors.MolWt(mol); clogp = Crippen.MolLogP(mol); tpsa = rdMolDescriptors.CalcTPSA(mol)
        charged_atoms = sum(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
        flags = []
        if mw > 600: flags.append("MW_GT_600")
        if clogp > 5: flags.append("CLOGP_GT_5")
        if tpsa > 140: flags.append("TPSA_GT_140")
        if clogp > 4 and tpsa < 50: flags.append("LOW_SOLUBILITY_HEURISTIC")
        if pains_names: flags.append("PAINS_ALERT")
        if brenk_names: flags.append("BRENK_ALERT")
        if counts["boron"]: flags.append("BORON_COVALENT_REACTIVITY_CONTROL")
        if counts["nitro"]: flags.append("NITRO_REDOX_READOUT_CONTROL")
        if counts["ester"]: flags.append("ESTER_HYDROLYSIS_STABILITY_CONTROL")
        if counts["alpha_beta_unsaturated_carbonyl"]: flags.append("ENONE_ELECTROPHILE_CONTROL")
        if counts["disulfide"]: flags.append("DISULFIDE_REDUCTION_STATE_CONTROL")
        if charged_atoms >= 2: flags.append("MULTI_CHARGED_OR_ZWITTERION_REPRESENTATION")
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
        rows.append({
            "project_full_inchikey": row.project_full_inchikey,
            "rdkit_inchikey": Chem.MolToInchiKey(mol), "rdkit_canonical_smiles": Chem.MolToSmiles(mol, True),
            "molecular_weight": mw, "clogp": clogp, "tpsa": tpsa,
            "hbd": Lipinski.NumHDonors(mol), "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()), "charged_atom_count": charged_atoms,
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "specified_chiral_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=False)),
            "aromatic_atom_fraction": aromatic_atoms / mol.GetNumHeavyAtoms(), **counts,
            "pains_alert_count": len(pains_names), "pains_alerts": "|".join(pains_names),
            "brenk_alert_count": len(brenk_names), "brenk_alerts": "|".join(brenk_names),
            "preassay_chemistry_flags": "|".join(flags) if flags else "NONE",
        })
    return pd.DataFrame(rows)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    robust_summary = json.loads(ROBUST_SUMMARY.read_text())
    identity_summary = json.loads(IDENTITY_SUMMARY.read_text())
    v17_audit = json.loads((V17 / "UNIFIED_V17_REPRODUCIBILITY_AUDIT.json").read_text())
    identities = pd.read_csv(OUT / "FULL28_CHEMBL37_PUBCHEM_IDENTITY_V18.csv", low_memory=False)
    mechanisms = pd.read_csv(OUT / "FULL28_CHEMBL37_KNOWN_MECHANISMS_V18.csv", low_memory=False)
    chemistry = pd.read_csv(OUT / "FULL28_CHEMISTRY_DESCRIPTORS_AND_ALERTS_V18.csv", low_memory=False)
    species = pd.read_csv(OUT / "FULL28_PRODRUG_ACTIVE_SPECIES_HANDLING_3_V18.csv", low_memory=False)
    portfolio = pd.read_csv(OUT / "FULL30_PAIR_IDENTITY_CHEMISTRY_RISK_V18.csv", low_memory=False)
    routes = pd.read_csv(OUT / "FULL30_ASSAY_ROUTE_AND_AUTHORIZATION_V18.csv", low_memory=False)
    ranks = pd.read_csv(OUT / "FULL30_BIDIRECTIONAL_MODEL_RANKS_240_V18.csv", low_memory=False)
    candidate_summary = pd.read_csv(OUT / "FULL30_BIDIRECTIONAL_ROBUSTNESS_SUMMARY_30_V18.csv", low_memory=False)
    concordance = pd.read_csv(OUT / "FULL30_TARGET_MODEL_CONCORDANCE_180_V18.csv", low_memory=False)
    physical = pd.read_csv(OUT / "FULL30_PHYSICAL_MODEL_COVERAGE_AND_RANKS_30_V18.csv", low_memory=False)
    independence = pd.read_csv(OUT / "FULL30_SELECTION_BIAS_AND_EVIDENCE_INDEPENDENCE_30_V18.csv", low_memory=False)

    protocol_deps = all((ROOT / rel).is_file() and sha256(ROOT / rel) == value for rel, value in protocol["frozen_dependencies"].items())
    summary_hashes = all(
        (ROOT / rel).is_file() and sha256(ROOT / rel) == value
        for summary in [robust_summary, identity_summary]
        for section in ["inputs", "outputs"]
        for rel, value in summary[section].items()
    )
    ids = identities["molecule_chembl_id"].tolist(); ph = ",".join("?" for _ in ids)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        direct_identity = pd.read_sql_query(f"""
            SELECT md.chembl_id AS molecule_chembl_id, md.pref_name, md.prodrug, md.max_phase,
                   md.first_approval, md.oral, md.parenteral, md.topical, md.dosed_ingredient,
                   md.structure_type, cs.standard_inchi_key AS chembl_standard_inchikey,
                   cs.canonical_smiles AS chembl_canonical_smiles,
                   pmd.chembl_id AS hierarchy_parent_chembl_id, pmd.pref_name AS hierarchy_parent_name
            FROM molecule_dictionary md LEFT JOIN compound_structures cs ON cs.molregno=md.molregno
            LEFT JOIN molecule_hierarchy mh ON mh.molregno=md.molregno
            LEFT JOIN molecule_dictionary pmd ON pmd.molregno=mh.parent_molregno
            WHERE md.chembl_id IN ({ph}) ORDER BY md.chembl_id""", con, params=ids)
        direct_mechanisms = pd.read_sql_query(f"""
            SELECT md.chembl_id AS molecule_chembl_id, md.pref_name, dm.mechanism_of_action,
                   dm.action_type, dm.direct_interaction, dm.molecular_mechanism,
                   td.chembl_id AS known_target_chembl_id, td.pref_name AS known_target_name,
                   td.target_type AS known_target_type, td.organism AS known_target_organism
            FROM molecule_dictionary md JOIN drug_mechanism dm ON dm.molregno=md.molregno
            LEFT JOIN target_dictionary td ON td.tid=dm.tid WHERE md.chembl_id IN ({ph})
            ORDER BY md.chembl_id,td.chembl_id,dm.mechanism_of_action""", con, params=ids)
    finally:
        con.close()
    identity_cols = list(direct_identity.columns)
    direct_identity_match = identities[identity_cols].sort_values("molecule_chembl_id").reset_index(drop=True).fillna("").astype(str).equals(
        direct_identity.reset_index(drop=True).fillna("").astype(str)
    )
    direct_mechanism_match = mechanisms.fillna("").astype(str).reset_index(drop=True).equals(
        direct_mechanisms.fillna("").astype(str).reset_index(drop=True)
    )
    recalculated = recalc_chemistry(chemistry)
    joined = chemistry.merge(recalculated, on="project_full_inchikey", suffixes=("_reported", "_recomputed"), validate="one_to_one")
    numeric_cols = [
        "molecular_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge",
        "charged_atom_count", "fraction_csp3", "specified_chiral_centers", "aromatic_atom_fraction",
        "boron", "nitro", "ester", "carboxylic_acid_or_carboxylate", "nitrile",
        "alpha_beta_unsaturated_carbonyl", "phenol", "amine", "disulfide", "pains_alert_count", "brenk_alert_count",
    ]
    text_cols = ["rdkit_inchikey", "rdkit_canonical_smiles", "pains_alerts", "brenk_alerts", "preassay_chemistry_flags"]
    chemistry_numeric_match = all(np.allclose(
        joined[f"{col}_reported"].astype(float), joined[f"{col}_recomputed"].astype(float), rtol=0, atol=1e-10
    ) for col in numeric_cols)
    chemistry_text_match = all(joined[f"{col}_reported"].fillna("").astype(str).equals(
        joined[f"{col}_recomputed"].fillna("").astype(str)
    ) for col in text_cols)

    raw_files = sorted(RAW.glob("*.json")); bulk_files = [p for p in raw_files if "BULK_IDENTITY" in p.name]
    vendor_files = [p for p in raw_files if "CHEMICAL_VENDORS" in p.name]
    bulk = json.loads(bulk_files[0].read_text())
    props = bulk.get("payload", {}).get("PropertyTable", {}).get("Properties", [])
    by_key = {str(item.get("InChIKey", "")): item for item in props}
    pubchem_bulk_match = len(by_key) == 28 and all(
        row.project_full_inchikey in by_key
        and int(by_key[row.project_full_inchikey]["CID"]) == int(row.pubchem_cid)
        and by_key[row.project_full_inchikey]["InChIKey"] == row.pubchem_returned_inchikey
        for row in identities.itertuples()
    )
    vendors = {}
    for path in vendor_files:
        vendors[int(path.name.split("_")[2])] = json.loads(path.read_text())
    vendor_match = len(vendors) == 28 and all(
        int(row.pubchem_cid) in vendors
        and bool(vendors[int(row.pubchem_cid)]["vendor_record_indicator"]) == bool(row.vendor_record_indicator)
        == vendor_boolean(vendors[int(row.pubchem_cid)].get("payload", {}))
        for row in identities.itertuples()
    )

    pair_cols = ["ligand_inchikey", "target_chembl_id", "conplex_score", "drugclip_cosine_mean", "dta_cross_target_consensus_score", "branch_primary_score_v10", "old_drug_target_deployment_branch_v10"]
    pair = pd.read_csv(PAIR, usecols=pair_cols, low_memory=False)
    views = {"CONPLEX":"conplex_score", "DRUGCLIP":"drugclip_cosine_mean", "DTA_CONSENSUS":"dta_cross_target_consensus_score", "V10_BRANCH_PRIMARY":"branch_primary_score_v10"}
    rank_checks = []
    for row in ranks.itertuples():
        if row.direction == "TARGET_CENTERED_ACROSS_DRUGS":
            group = pair[pair.target_chembl_id.eq(row.target_chembl_id)]
        else:
            group = pair[pair.ligand_inchikey.eq(row.ligand_inchikey)]
            if row.model_view == "V10_BRANCH_PRIMARY": group = group[group.old_drug_target_deployment_branch_v10.eq(row.deployment_branch)]
        col = views[row.model_view]; finite = group[pd.to_numeric(group[col],errors="coerce").notna()].copy()
        finite[col] = pd.to_numeric(finite[col],errors="coerce")
        mask = finite.ligand_inchikey.eq(row.ligand_inchikey) & finite.target_chembl_id.eq(row.target_chembl_id)
        avg = float(finite[col].rank(method="average",ascending=False).loc[mask].iloc[0])
        mn = float(finite[col].rank(method="min",ascending=False).loc[mask].iloc[0])
        mx = float(finite[col].rank(method="max",ascending=False).loc[mask].iloc[0])
        pct=(len(finite)-avg)/(len(finite)-1)
        rank_checks.append(
            len(finite)==row.finite_score_denominator and np.isclose(avg,row.recomputed_average_tie_rank)
            and np.isclose(mn,row.recomputed_min_tie_rank) and np.isclose(mx,row.recomputed_max_tie_rank)
            and np.isclose(pct,row.empirical_upper_tail_percentile)
        )
    concordance_checks=[]
    for row in concordance.itertuples():
        group=pair[pair.target_chembl_id.eq(row.target_chembl_id)]
        l,r=views[row.left_model_view],views[row.right_model_view]
        finite=group[[l,r]].apply(pd.to_numeric,errors="coerce").dropna()
        rho=float(finite[l].corr(finite[r],method="spearman"))
        concordance_checks.append(len(finite)==row.finite_pairwise_drug_count and np.isclose(rho,row.spearman_rho_across_drugs,rtol=0,atol=1e-12))

    high = set(candidate_summary.loc[candidate_summary.rank_discordance_flag.eq("HIGH_RANGE_GE_0.50"),"candidate_rank"])
    checks = {
        "protocol_stamp_and_dependencies_rehashed": STAMP.read_text().split()[0] == sha256(PROTOCOL) and protocol_deps,
        "both_builder_summaries_pass_20_of_20": (
            robust_summary["status"] == identity_summary["status"] == "PASS"
            and len(robust_summary["checks"]) == len(identity_summary["checks"]) == 20
            and all(robust_summary["checks"].values()) and all(identity_summary["checks"].values())
        ),
        "summary_inputs_and_outputs_all_rehashed": summary_hashes,
        "inherited_v17_audit_pass_40_of_40": v17_audit["checks_passed"] == v17_audit["checks_total"] == 40,
        "population_exact_30_pairs_28_entities_16_targets_8_21_1": (
            len(portfolio)==30 and len(identities)==28 and portfolio.target_chembl_id.nunique()==16
            and portfolio.execution_wave.value_counts().to_dict()=={"W2_CONTINGENT_ONLY":21,"W1_BLINDED_CANDIDATE_PILOT":8,"VETO_NOT_AUTHORIZED":1}
        ),
        "direct_chembl_identity_exact_28": direct_identity_match,
        "direct_chembl_mechanisms_exact_36": direct_mechanism_match and len(mechanisms)==36,
        "chembl_key_partition_27_exact_1_serdex": (
            identities.project_vs_chembl_full_inchikey_status.eq("EXACT_FULL_INCHIKEY_MATCH").sum()==27
            and identities.loc[identities.project_vs_chembl_full_inchikey_status.str.contains("MISMATCH"),"pref_name"].tolist()==["SERDEXMETHYLPHENIDATE"]
        ),
        "only_name_entity_mismatch_is_rank21_ketoconazole_label_levoketoconazole_entity": (
            portfolio.source_label_vs_chembl_preferred_name_class.ne("EXACT_NORMALIZED_NAME").sum()==1
            and portfolio.loc[portfolio.v10_integrated_case_rank.eq(21),"pref_name"].tolist()==["LEVOKETOCONAZOLE"]
            and portfolio.loc[portfolio.v10_integrated_case_rank.eq(21),"project_full_inchikey"].tolist()==["XMAYWYJOQHXEEK-ZEQKJWHPSA-N"]
        ),
        "prodrug_partition_and_species_routes_exact": (
            set(identities.loc[identities.prodrug.eq(1),"pref_name"])=={"NITAZOXANIDE","ROMIDEPSIN","SERDEXMETHYLPHENIDATE"}
            and len(species)==3 and species.entity_merge_policy.str.startswith("NEVER_MERGE").all()
        ),
        "rdkit_numeric_descriptors_and_substructures_recomputed": chemistry_numeric_match,
        "rdkit_keys_alert_names_and_flags_recomputed": chemistry_text_match,
        "pubchem_raw_exact_1_bulk_28_vendor_no_retry": len(raw_files)==29 and len(bulk_files)==1 and len(vendor_files)==28 and not any("RETRY" in p.name for p in raw_files),
        "pubchem_bulk_full_key_cid_exact_28": pubchem_bulk_match,
        "pubchem_vendor_indicators_recomputed_28": vendor_match,
        "risk_partition_exact_r3_8_r2_17_r1_5": portfolio.experimental_handling_risk_tier.value_counts().to_dict()=={"R2_MODERATE_COUNTERASSAY":17,"R3_HIGH_SPECIAL_HANDLING":8,"R1_STANDARD":5},
        "routes_exact_30_and_w2_veto_not_authorized": (
            len(routes)==30 and routes.loc[routes.execution_wave.eq("W2_CONTINGENT_ONLY"),"execution_authorization"].str.contains("NOT_AUTHORIZED").all()
            and routes.loc[routes.execution_wave.eq("VETO_NOT_AUTHORIZED"),"execution_authorization"].str.startswith("VETO_COMPUTATIONAL_AUDIT_ONLY_NO_PROCUREMENT_OR_ASSAY_AUTHORIZED").all()
        ),
        "all_240_rank_rows_independently_recomputed": len(ranks)==240 and all(rank_checks),
        "historical_min_tie_cases_exact_ranks12_19_23": set(ranks.loc[ranks.stored_rank_recalculation_status.str.contains("HISTORICAL_MIN_TIE"),"candidate_rank"])=={12,19,23},
        "all_180_spearman_correlations_recomputed": len(concordance)==180 and all(concordance_checks),
        "physical_coverage_exact_29_boltz_10_gnina_and_rank21_missing_not_imputed": (
            physical.boltz_primary_completed.sum()==29 and physical.gnina_primary_completed.sum()==10
            and physical.loc[physical.candidate_rank.eq(21),"physical_metadata_consistency_flag"].tolist()==["MULTISEED_SUPPORT_METADATA_WITHOUT_MAIN_RESULT_DO_NOT_IMPUTE"]
        ),
        "high_discordance_exact_16_and_no_rerank": len(high)==16 and candidate_summary.candidate_rank.tolist()==list(range(1,31)),
        "independence_exact_30_zero_external_exact_pair_validation": len(independence)==30 and ~independence.exact_pair_external_evidence_present.any(),
        "claim_boundaries_preserve_384_only46_and_forbid_binding_w2_authorization": (
            any("not independent validation" in item for item in protocol["selection_bias_and_claim_boundaries"])
            and any("No hard-gate target" in item for item in protocol["selection_bias_and_claim_boundaries"])
            and any("do not authorize procurement" in item for item in protocol["selection_bias_and_claim_boundaries"])
        ),
    }
    checks={k:bool(v) for k,v in checks.items()}
    report={
        "created_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if all(checks.values()) else "FAIL",
        "checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
        "key_hashes":{"protocol_sha256":sha256(PROTOCOL),"pair_core_sha256":sha256(PAIR),"robust_summary_sha256":sha256(ROBUST_SUMMARY),"identity_summary_sha256":sha256(IDENTITY_SUMMARY)},
        "independent_interpretation":(
            "All 28 entities, 30 pairs, RDKit descriptors, PubChem raw identities, 240 bidirectional ranks and 180 "
            "target-lane correlations were independently checked. Rank 21 remains the modeled levoketoconazole full-key "
            "entity under a ketoconazole-label adjudication hold; W2/veto are not execution-authorized."
        ),
    }
    AUDIT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks_passed":report["checks_passed"],"checks_total":report["checks_total"],"failed_checks":[k for k,v in checks.items() if not v],"audit_sha256":sha256(AUDIT)},ensure_ascii=False,indent=2))
    if report["status"]!="PASS": raise SystemExit(1)


if __name__ == "__main__": main()
