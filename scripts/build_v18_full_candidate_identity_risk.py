#!/usr/bin/env python3
"""Build exact identity, chemistry-risk, active-species and assay-route artifacts for all 30 candidates."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V17 = RUN / "final_evidence_routing_v17"
OUT = RUN / "full_candidate_portfolio_v18"
PROTOCOL = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.json"
STAMP = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.sha256"
DB = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
RAW = OUT / "pubchem_full28_identity_raw_v18"
USER_AGENT = "BioMaster-V18-full30-identity-risk/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def vendor_boolean(payload: object) -> bool:
    if isinstance(payload, dict):
        if payload.get("TOCHeading") == "Chemical Vendors":
            return any(True in item.get("Value", {}).get("Boolean", []) for item in payload.get("Information", []))
        return any(vendor_boolean(value) for value in payload.values())
    if isinstance(payload, list):
        return any(vendor_boolean(value) for value in payload)
    return False


def fetch_vendor(cid: int) -> dict[str, object]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    try:
        response = requests.get(
            url, params={"heading": "Chemical Vendors"}, headers={"User-Agent": USER_AGENT}, timeout=60
        )
        payload = response.json()
        return {
            "pubchem_cid": cid, "vendor_http_status": response.status_code,
            "vendor_record_indicator": response.status_code == 200 and vendor_boolean(payload),
            "vendor_network_error": "", "url": response.url, "payload": payload,
        }
    except requests.RequestException as error:
        return {
            "pubchem_cid": cid, "vendor_http_status": 0, "vendor_record_indicator": False,
            "vendor_network_error": f"{type(error).__name__}:{error}", "url": url, "payload": {},
        }


def query_chembl(entity_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    placeholders = ",".join("?" for _ in entity_ids)
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        identity = pd.read_sql_query(
            f"""
            SELECT md.chembl_id AS molecule_chembl_id, md.pref_name, md.prodrug, md.max_phase,
                   md.first_approval, md.oral, md.parenteral, md.topical, md.dosed_ingredient,
                   md.structure_type, cs.standard_inchi_key AS chembl_standard_inchikey,
                   cs.canonical_smiles AS chembl_canonical_smiles,
                   pmd.chembl_id AS hierarchy_parent_chembl_id, pmd.pref_name AS hierarchy_parent_name
            FROM molecule_dictionary md
            LEFT JOIN compound_structures cs ON cs.molregno=md.molregno
            LEFT JOIN molecule_hierarchy mh ON mh.molregno=md.molregno
            LEFT JOIN molecule_dictionary pmd ON pmd.molregno=mh.parent_molregno
            WHERE md.chembl_id IN ({placeholders})
            ORDER BY md.chembl_id
            """,
            connection, params=entity_ids,
        )
        mechanisms = pd.read_sql_query(
            f"""
            SELECT md.chembl_id AS molecule_chembl_id, md.pref_name, dm.mechanism_of_action,
                   dm.action_type, dm.direct_interaction, dm.molecular_mechanism,
                   td.chembl_id AS known_target_chembl_id, td.pref_name AS known_target_name,
                   td.target_type AS known_target_type, td.organism AS known_target_organism
            FROM molecule_dictionary md
            JOIN drug_mechanism dm ON dm.molregno=md.molregno
            LEFT JOIN target_dictionary td ON td.tid=dm.tid
            WHERE md.chembl_id IN ({placeholders})
            ORDER BY md.chembl_id, td.chembl_id, dm.mechanism_of_action
            """,
            connection, params=entity_ids,
        )
        relatives = pd.read_sql_query(
            """
            SELECT md.chembl_id AS related_molecule_chembl_id, md.pref_name AS related_molecule_name,
                   md.prodrug, md.max_phase, md.first_approval, md.dosed_ingredient,
                   cs.standard_inchi_key AS related_standard_inchikey,
                   cs.canonical_smiles AS related_canonical_smiles
            FROM molecule_dictionary md
            LEFT JOIN compound_structures cs ON cs.molregno=md.molregno
            WHERE md.pref_name IN (?, ?)
            ORDER BY md.pref_name
            """,
            connection, params=["TIZOXANIDE", "DEXMETHYLPHENIDATE"],
        )
    finally:
        connection.close()
    return identity, mechanisms, relatives


def pubchem_audit(entities: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    RAW.mkdir(parents=True, exist_ok=True)
    keys = entities["project_full_inchikey"].tolist()
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/" + quote(",".join(keys), safe=",")
        + "/property/Title,IUPACName,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    )
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    bulk_path = RAW / "PUBCHEM_FULL28_BULK_IDENTITY_RESPONSE_V18.json"
    bulk_path.write_text(json.dumps({
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "url": response.url,
        "http_status": response.status_code, "payload": payload,
    }, ensure_ascii=False, indent=2) + "\n")
    raw_paths = [bulk_path]
    by_key: dict[str, list[dict]] = {}
    for item in payload.get("PropertyTable", {}).get("Properties", []):
        by_key.setdefault(str(item.get("InChIKey", "")), []).append(item)
    retry_status: dict[str, int] = {}
    for key in [key for key in keys if key not in by_key]:
        retry_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/"
            "property/Title,IUPACName,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
        )
        retry = requests.get(retry_url, headers={"User-Agent": USER_AGENT}, timeout=60)
        retry_payload = retry.json()
        retry_status[key] = retry.status_code
        path = RAW / f"PUBCHEM_INCHIKEY_{key}_INDIVIDUAL_RETRY_V18.json"
        path.write_text(json.dumps({
            "retrieved_utc": datetime.now(timezone.utc).isoformat(), "url": retry.url,
            "http_status": retry.status_code, "payload": retry_payload,
        }, ensure_ascii=False, indent=2) + "\n")
        raw_paths.append(path)
        if retry.status_code == 200:
            for item in retry_payload.get("PropertyTable", {}).get("Properties", []):
                by_key.setdefault(str(item.get("InChIKey", "")), []).append(item)
    rows = []
    for entity in entities.itertuples():
        matches = by_key.get(entity.project_full_inchikey, [])
        item = matches[0] if len(matches) == 1 else {}
        rows.append({
            "project_full_inchikey": entity.project_full_inchikey,
            "exact_pubchem_match_count": len(matches), "pubchem_cid": item.get("CID", pd.NA),
            "pubchem_title": item.get("Title", ""), "pubchem_iupac_name": item.get("IUPACName", ""),
            "pubchem_smiles": item.get("SMILES", item.get("CanonicalSMILES", "")),
            "pubchem_returned_inchikey": item.get("InChIKey", ""),
            "individual_retry_http_status": retry_status.get(entity.project_full_inchikey, pd.NA),
        })
    audit = pd.DataFrame(rows)
    cids = sorted(audit["pubchem_cid"].dropna().astype(int).unique())
    vendor_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_vendor, cid): cid for cid in cids}
        for future in as_completed(futures):
            vendor_results.append(future.result())
    for result in sorted(vendor_results, key=lambda item: int(item["pubchem_cid"])):
        path = RAW / f"PUBCHEM_CID_{result['pubchem_cid']}_CHEMICAL_VENDORS_V18.json"
        path.write_text(json.dumps({
            "retrieved_utc": datetime.now(timezone.utc).isoformat(), "url": result["url"],
            "http_status": result["vendor_http_status"],
            "vendor_record_indicator": result["vendor_record_indicator"],
            "network_error": result["vendor_network_error"], "payload": result["payload"],
        }, ensure_ascii=False, indent=2) + "\n")
        raw_paths.append(path)
    vendor_frame = pd.DataFrame([
        {key: value for key, value in result.items() if key not in {"payload", "url"}}
        for result in vendor_results
    ])
    audit = audit.merge(vendor_frame, on="pubchem_cid", how="left", validate="one_to_one")
    audit["pubchem_identity_status"] = np.where(
        audit["exact_pubchem_match_count"].eq(1)
        & audit["project_full_inchikey"].eq(audit["pubchem_returned_inchikey"]),
        "EXACT_FULL_INCHIKEY_MATCH", "NO_EXACT_FULL_INCHIKEY_MATCH",
    )
    audit["pubchem_vendor_record_status"] = np.where(
        audit["vendor_record_indicator"].fillna(False),
        "VENDOR_RECORD_INDICATOR_PRESENT_NOT_STOCK_VERIFIED",
        "NO_VENDOR_RECORD_INDICATOR_OR_IDENTITY_UNRESOLVED",
    )
    return audit, sorted(raw_paths)


def chemistry_table(entities: pd.DataFrame) -> pd.DataFrame:
    pains_params = FilterCatalog.FilterCatalogParams(); pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_params = FilterCatalog.FilterCatalogParams(); brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    pains = FilterCatalog.FilterCatalog(pains_params); brenk = FilterCatalog.FilterCatalog(brenk_params)
    patterns = {
        "boron": "[#5]", "nitro": "[N+](=O)[O-]", "ester": "[CX3](=O)[OX2][#6]",
        "carboxylic_acid_or_carboxylate": "[CX3](=O)[OX1-,OX2H1]", "nitrile": "[CX2]#N",
        "alpha_beta_unsaturated_carbonyl": "[C,c]=[C,c][CX3](=O)", "phenol": "[c][OX2H]",
        "amine": "[NX3;!$(N-C=O);!$(N-S=O)]", "disulfide": "[SX2]-[SX2]",
    }
    queries = {name: Chem.MolFromSmarts(pattern) for name, pattern in patterns.items()}
    rows = []
    for entity in entities.itertuples():
        mol = Chem.MolFromSmiles(entity.project_smiles)
        if mol is None:
            raise ValueError(entity.project_full_inchikey)
        pains_names = sorted({item.GetDescription() for item in pains.GetMatches(mol)})
        brenk_names = sorted({item.GetDescription() for item in brenk.GetMatches(mol)})
        counts = {name: len(mol.GetSubstructMatches(query)) for name, query in queries.items()}
        mw = Descriptors.MolWt(mol); clogp = Crippen.MolLogP(mol); tpsa = rdMolDescriptors.CalcTPSA(mol)
        charged_atoms = sum(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
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
        rows.append({
            "project_full_inchikey": entity.project_full_inchikey, "project_smiles": entity.project_smiles,
            "rdkit_inchikey": Chem.MolToInchiKey(mol), "rdkit_canonical_smiles": Chem.MolToSmiles(mol, True),
            "molecular_weight": mw, "clogp": clogp, "tpsa": tpsa,
            "hbd": Lipinski.NumHDonors(mol), "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
            "charged_atom_count": charged_atoms, "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "specified_chiral_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=False)),
            "aromatic_atom_fraction": aromatic_atoms / mol.GetNumHeavyAtoms(), **counts,
            "pains_alert_count": len(pains_names), "pains_alerts": "|".join(pains_names),
            "brenk_alert_count": len(brenk_names), "brenk_alerts": "|".join(brenk_names),
            "preassay_chemistry_flags": "|".join(flags) if flags else "NONE",
        })
    return pd.DataFrame(rows)


def assay_plan(gene: str) -> dict[str, str]:
    plans = {
        "NR3C1": ("NUCLEAR_RECEPTOR", "NR3C1 coregulator-recruitment concentration response", "direct NR3C1 binding or cellular GR target engagement", "known receptor pharmacology, reporter interference, mock and viability"),
        "ALOX5": ("METABOLIC_OR_SIGNAL_ENZYME", "ALOX5 substrate-to-product assay with required cofactors", "LC-MS 5-HpETE/5-HETE or direct ALOX5 binding", "ALOX12/15, COX, iron/redox, aggregation and optical controls"),
        "COMT": ("METABOLIC_OR_SIGNAL_ENZYME", "COMT methyl-transfer assay with SAM/Mg2+", "LC-MS product plus orthogonal SAM turnover or direct binding", "substrate stability, redox/optical, active-species and no-enzyme controls"),
        "CSNK2A1": ("KINASE", "CSNK2A1 biochemical kinase assay at ATP near Km", "radiometric assay or NanoBRET/CETSA/direct binding", "CSNK2A2, ATP shift, kinase panel, solubility and aggregation"),
        "MMP3": ("METALLOPROTEASE_OR_PROTEASE", "MMP3 peptide-cleavage assay", "LC-MS cleavage/direct binding with second substrate", "MMP2/9, zinc/calcium, fluorescence, mock enzyme and active-species controls"),
        "TTK": ("KINASE", "TTK biochemical kinase assay at ATP near Km", "NanoBRET/CETSA/direct binding with phospho-substrate readout", "ALK/ROS1, ATP shift, kinase panel and cell-cycle toxicity"),
        "NR1H3": ("NUCLEAR_RECEPTOR", "NR1H3/LXR-alpha coregulator-recruitment concentration response", "direct LXR-alpha binding or cellular target-gene engagement", "ESR1/known target, LXR-beta, mock reporter, cytotoxicity and partitioning"),
        "CMA1": ("METALLOPROTEASE_OR_PROTEASE", "CMA1 cleavage assay with preincubation time series", "LC-MS/activity probe/intact-protein MS", "proteasome, related proteases, dilution/washout and no-enzyme stability"),
        "MMP7": ("METALLOPROTEASE_OR_PROTEASE", "MMP7 peptide-cleavage concentration response", "LC-MS cleavage or direct binding with second substrate", "MMP2/3/9, zinc/calcium, fluorescence quench, aggregation and mock enzyme"),
        "PDE4D": ("METABOLIC_OR_SIGNAL_ENZYME", "PDE4D cAMP-hydrolysis concentration response", "LC-MS/coupled orthogonal product assay or direct binding", "PDE4A/B/C, coupling-enzyme, fluorescence, aggregation and substrate controls"),
        "MAOB": ("METABOLIC_OR_SIGNAL_ENZYME", "MAOB amine-oxidation substrate-product assay", "LC-MS product or direct MAOB binding/thermal shift", "MAOA, peroxide/redox, optical, detergent aggregation and no-enzyme controls"),
        "WEE1": ("KINASE", "WEE1 biochemical kinase assay at ATP near Km", "NanoBRET/CETSA/direct binding with phospho-CDK readout", "known drug targets, ATP shift, kinase panel and cell-cycle toxicity"),
        "CASP1": ("METALLOPROTEASE_OR_PROTEASE", "CASP1 peptide-cleavage concentration response", "LC-MS cleavage or activity-based probe/direct binding", "other caspases, fluorescence, aggregation, inflammasome context and no-enzyme controls"),
        "DHODH": ("METABOLIC_OR_SIGNAL_ENZYME", "DHODH dihydroorotate-to-orotate assay", "LC-MS product or direct binding/thermal shift", "ubiquinone/redox coupling, mitochondrial ETC, optical, aggregation and no-enzyme controls"),
        "PARP2": ("PARP", "DNA-dependent PARP2 NAD+/PARylation assay", "direct binding/engagement or cellular PAR orthogonal readout", "PARP1, DNA dependence, NAD competition, optical and no-enzyme controls"),
        "AKT2": ("KINASE", "AKT2 biochemical kinase assay at ATP near Km", "NanoBRET/CETSA/direct binding with phospho-substrate readout", "AKT1/3, known drug targets, ATP shift, kinase panel and cytotoxicity"),
    }
    target_class, primary, orthogonal, counters = plans[gene]
    return {"target_assay_class": target_class, "primary_assay_route": primary, "orthogonal_confirmation_route": orthogonal, "required_counterassays": counters}


def main() -> None:
    if sha256(PROTOCOL) != STAMP.read_text().split()[0]:
        raise RuntimeError("V18 protocol stamp mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol["frozen_dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen dependency changed: {relative}")
    casebook = pd.read_csv(V17 / "PROSPECTIVE_INTEGRATED_CASEBOOK_V10.csv", low_memory=False)
    case = casebook[casebook["v10_integrated_case_rank"].between(1, 30)].sort_values("v10_integrated_case_rank")
    pair_columns = ["ligand_inchikey", "target_chembl_id", "ligand_smiles", "drug_names", "gene_symbol"]
    pair = pd.read_csv(V17 / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz", usecols=pair_columns, low_memory=False)
    selected = case[[
        "v10_integrated_case_rank", "ligand_inchikey", "target_chembl_id", "drug_names", "gene_symbol",
        "ligand_species_chembl_ids_v7", "old_drug_target_deployment_branch_v10",
        "v10_target_lane_evidence_status", "target_family_relation_v7", "known_target_component_count_v7",
    ]].merge(
        pair[["ligand_inchikey", "target_chembl_id", "ligand_smiles"]],
        on=["ligand_inchikey", "target_chembl_id"], validate="one_to_one",
    )
    master = pd.read_csv(V17 / "MASTER_VALIDATION_QUEUE_63_ROWS_V17.csv", low_memory=False)
    waves = master[master["validation_layer"].eq("L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS")][[
        "candidate_rank_context", "execution_wave"
    ]].rename(columns={"candidate_rank_context": "v10_integrated_case_rank"})
    selected = selected.merge(waves, on="v10_integrated_case_rank", validate="one_to_one")
    entity_rows = []
    for key, group in selected.groupby("ligand_inchikey", sort=True):
        entity_rows.append({
            "project_full_inchikey": key,
            "project_smiles": group["ligand_smiles"].iloc[0],
            "frozen_molecule_chembl_id": group["ligand_species_chembl_ids_v7"].iloc[0],
            "source_drug_names": "|".join(sorted(group["drug_names"].unique())),
            "candidate_ranks": ",".join(map(str, sorted(group["v10_integrated_case_rank"].astype(int)))),
            "candidate_pair_count": len(group),
            "target_chembl_ids": ",".join(sorted(group["target_chembl_id"].unique())),
        })
    entities = pd.DataFrame(entity_rows)
    chembl_identity, mechanisms, relatives = query_chembl(entities["frozen_molecule_chembl_id"].tolist())
    entities = entities.merge(
        chembl_identity, left_on="frozen_molecule_chembl_id", right_on="molecule_chembl_id",
        validate="one_to_one",
    )
    entities["project_vs_chembl_full_inchikey_status"] = np.where(
        entities["project_full_inchikey"].eq(entities["chembl_standard_inchikey"]),
        "EXACT_FULL_INCHIKEY_MATCH", "FULL_INCHIKEY_REPRESENTATION_MISMATCH",
    )
    entities["same_inchikey_connectivity_block"] = [
        left.split("-")[0] == right.split("-")[0]
        for left, right in zip(entities["project_full_inchikey"], entities["chembl_standard_inchikey"])
    ]
    name_classes = []
    entity_holds = []
    for row in entities.itertuples():
        names = row.source_drug_names.split("|")
        if all(normalize_name(name) == normalize_name(row.pref_name) for name in names):
            name_classes.append("EXACT_NORMALIZED_NAME")
            entity_holds.append("NO_NAME_ENTITY_HOLD")
        elif row.molecule_chembl_id == "CHEMBL295698" and names == ["ketoconazole"]:
            name_classes.append("DISTINCT_NAMED_STEREOENTITY_LABEL_MISMATCH")
            entity_holds.append("ENTITY_NAME_ADJUDICATION_HOLD_MODELED_ENTITY_IS_LEVOKETOCONAZOLE")
        else:
            name_classes.append("OTHER_NAME_ENTITY_MISMATCH_REQUIRES_MANUAL_ADJUDICATION")
            entity_holds.append("ENTITY_NAME_ADJUDICATION_HOLD")
    entities["source_label_vs_chembl_preferred_name_class"] = name_classes
    entities["entity_name_execution_hold"] = entity_holds

    pubchem, raw_paths = pubchem_audit(entities)
    chemistry = chemistry_table(entities)
    entities = entities.merge(pubchem, on="project_full_inchikey", validate="one_to_one").merge(
        chemistry, on=["project_full_inchikey", "project_smiles"], validate="one_to_one"
    )

    relation_by_name = relatives.set_index("related_molecule_name").to_dict("index")
    species_rows = []
    species_specs = [
        ("nitazoxanide", "CHEMBL1401", "TIZOXANIDE", "PARALLEL_ACTIVE_METABOLITE",
         "Run parent and tizoxanide separately and quantify hydrolysis/interconversion during assay incubation.",
         "ChEMBL37 exact entity record"),
        ("serdexmethylphenidate", "CHEMBL4301162", "DEXMETHYLPHENIDATE", "PARALLEL_RELEASED_ACTIVE_DRUG",
         "Run parent and dexmethylphenidate separately; track project zwitterion and ChEMBL neutral representation by full key.",
         "ChEMBL37 exact entity record"),
        ("romidepsin", "CHEMBL343448", "REDUCED_ROMIDEPSIN_DITHIOL", "REDUCTION_ACTIVATED_DITHIOL_STATE_NO_SEPARATE_CHEMBL37_ENTITY",
         "Measure disulfide reduction and reduced/oxidized state in assay matrix; include glutathione dependence, thiol-state and no-target controls.",
         "PMID:21587264; DOI:10.1038/ja.2011.35"),
    ]
    for parent_name, parent_id, related_name, relationship, rule, source in species_specs:
        relative = relation_by_name.get(related_name, {})
        parent = entities[entities["molecule_chembl_id"].eq(parent_id)].iloc[0]
        species_rows.append({
            "parent_drug_name": parent_name, "parent_molecule_chembl_id": parent_id,
            "parent_project_full_inchikey": parent["project_full_inchikey"],
            "related_species_name": related_name,
            "related_molecule_chembl_id": relative.get("related_molecule_chembl_id", "NO_SEPARATE_CHEMBL37_ENTITY"),
            "related_standard_inchikey": relative.get("related_standard_inchikey", "NO_SEPARATE_CHEMBL37_ENTITY"),
            "related_canonical_smiles": relative.get("related_canonical_smiles", "NO_SEPARATE_CHEMBL37_ENTITY"),
            "species_relationship": relationship, "execution_rule": rule, "evidence_source": source,
            "entity_merge_policy": "NEVER_MERGE_RESULTS_ACROSS_PARENT_RELATED_OR_REDUCTION_STATE_IDS",
        })
    species = pd.DataFrame(species_rows)

    entity_columns = [
        "project_full_inchikey", "molecule_chembl_id", "pref_name", "prodrug", "chembl_standard_inchikey",
        "project_vs_chembl_full_inchikey_status", "same_inchikey_connectivity_block",
        "source_label_vs_chembl_preferred_name_class", "entity_name_execution_hold",
        "pubchem_cid", "pubchem_identity_status", "pubchem_vendor_record_status", "vendor_record_indicator",
        "molecular_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge",
        "charged_atom_count", "fraction_csp3", "specified_chiral_centers", "aromatic_atom_fraction",
        "boron", "nitro", "ester", "carboxylic_acid_or_carboxylate", "nitrile",
        "alpha_beta_unsaturated_carbonyl", "phenol", "amine", "disulfide", "pains_alert_count",
        "pains_alerts", "brenk_alert_count", "brenk_alerts", "preassay_chemistry_flags",
    ]
    portfolio = selected.merge(
        entities[entity_columns], left_on="ligand_inchikey", right_on="project_full_inchikey", validate="many_to_one"
    )
    r3_compounds = set(protocol["chemistry_and_species_audit"]["predeclared_compound_specific_r3"])
    risk_tiers = []
    risk_reasons = []
    for row in portfolio.itertuples():
        name = str(row.drug_names).lower()
        reasons = []
        if name in r3_compounds: reasons.append("PREDECLARED_COMPOUND_SPECIFIC_R3")
        if int(row.prodrug) == 1: reasons.append("CHEMBL_PRODRUG_FLAG")
        if row.project_vs_chembl_full_inchikey_status != "EXACT_FULL_INCHIKEY_MATCH": reasons.append("PROJECT_CHEMBL_FULL_KEY_REPRESENTATION_MISMATCH")
        if row.source_label_vs_chembl_preferred_name_class != "EXACT_NORMALIZED_NAME": reasons.append("SOURCE_LABEL_ENTITY_NAME_MISMATCH")
        if int(row.boron) > 0: reasons.append("BORON_COVALENT_CHEMISTRY")
        if reasons:
            tier = "R3_HIGH_SPECIAL_HANDLING"
        elif row.preassay_chemistry_flags != "NONE" or row.target_family_relation_v7 == "F3_BROAD_PROTEIN_CLASS_OVERLAP":
            tier = "R2_MODERATE_COUNTERASSAY"
            if row.preassay_chemistry_flags != "NONE": reasons.append("CHEMISTRY_OR_PROPERTY_ALERT")
            if row.target_family_relation_v7 == "F3_BROAD_PROTEIN_CLASS_OVERLAP": reasons.append("KNOWN_TARGET_BROAD_FAMILY_OVERLAP")
        else:
            tier = "R1_STANDARD"
            reasons.append("NO_PREDECLARED_R3_OR_R2_FLAG")
        risk_tiers.append(tier); risk_reasons.append("|".join(reasons))
    portfolio["experimental_handling_risk_tier"] = risk_tiers
    portfolio["risk_tier_reasons"] = risk_reasons
    portfolio["identity_execution_status"] = np.where(
        portfolio["entity_name_execution_hold"].eq("NO_NAME_ENTITY_HOLD"),
        "IDENTITY_PRECHECK_PASS_NOT_PROCUREMENT_RELEASE",
        portfolio["entity_name_execution_hold"],
    )

    mechanism_text = mechanisms.groupby("molecule_chembl_id").agg(
        chembl_known_mechanism_count=("mechanism_of_action", "size"),
        chembl_known_mechanisms=("mechanism_of_action", lambda values: " | ".join(values.dropna().astype(str))),
        chembl_known_target_ids=("known_target_chembl_id", lambda values: "|".join(values.dropna().astype(str))),
    ).reset_index()
    portfolio = portfolio.merge(mechanism_text, on="molecule_chembl_id", how="left", validate="many_to_one")
    portfolio["chembl_known_mechanism_count"] = portfolio["chembl_known_mechanism_count"].fillna(0).astype(int)
    portfolio[["chembl_known_mechanisms", "chembl_known_target_ids"]] = portfolio[[
        "chembl_known_mechanisms", "chembl_known_target_ids"
    ]].fillna("NO_CHEMBL37_DRUG_MECHANISM_ROW")

    route_rows = []
    for row in portfolio.sort_values("v10_integrated_case_rank").itertuples():
        plan = assay_plan(row.gene_symbol)
        if row.execution_wave == "W1_BLINDED_CANDIDATE_PILOT":
            authorization = "W1_PENDING_FROZEN_V17_PROCUREMENT_AND_RECEIVED_LOT_QC_NOT_RELEASED"
        elif row.execution_wave == "W2_CONTINGENT_ONLY":
            authorization = "W2_CONTINGENT_COMPUTATIONAL_READINESS_ONLY_PROCUREMENT_AND_ASSAY_NOT_AUTHORIZED"
        else:
            authorization = "VETO_COMPUTATIONAL_AUDIT_ONLY_NO_PROCUREMENT_OR_ASSAY_AUTHORIZED"
        if row.entity_name_execution_hold != "NO_NAME_ENTITY_HOLD":
            authorization += "|ENTITY_NAME_ADJUDICATION_HOLD"
        route_rows.append({
            "candidate_rank": int(row.v10_integrated_case_rank), "execution_wave": row.execution_wave,
            "drug_name": row.drug_names, "modeled_chembl_preferred_entity": row.pref_name,
            "project_full_inchikey": row.project_full_inchikey, "target_chembl_id": row.target_chembl_id,
            "gene_symbol": row.gene_symbol, **plan,
            "experimental_handling_risk_tier": row.experimental_handling_risk_tier,
            "risk_tier_reasons": row.risk_tier_reasons,
            "active_species_requirement": (
                species.loc[species["parent_molecule_chembl_id"].eq(row.molecule_chembl_id), "execution_rule"].iloc[0]
                if row.molecule_chembl_id in set(species["parent_molecule_chembl_id"])
                else "TEST_EXACT_FROZEN_ENTITY; NO_PREDECLARED_SEPARATE_ACTIVE_SPECIES"
            ),
            "execution_authorization": authorization,
            "success_gate": "Two QC-valid independent curves <=10 uM plus nonidentical/direct orthogonal confirmation and required species/counterassay gates.",
            "claim_boundary": "Readiness route only; no procurement/assay authorization beyond frozen wave and no binding claim.",
        })
    routes = pd.DataFrame(route_rows)

    paths = {
        "identity": OUT / "FULL28_CHEMBL37_PUBCHEM_IDENTITY_V18.csv",
        "mechanisms": OUT / "FULL28_CHEMBL37_KNOWN_MECHANISMS_V18.csv",
        "chemistry": OUT / "FULL28_CHEMISTRY_DESCRIPTORS_AND_ALERTS_V18.csv",
        "species": OUT / "FULL28_PRODRUG_ACTIVE_SPECIES_HANDLING_3_V18.csv",
        "portfolio": OUT / "FULL30_PAIR_IDENTITY_CHEMISTRY_RISK_V18.csv",
        "routes": OUT / "FULL30_ASSAY_ROUTE_AND_AUTHORIZATION_V18.csv",
    }
    entities.to_csv(paths["identity"], index=False)
    mechanisms.to_csv(paths["mechanisms"], index=False)
    chemistry.to_csv(paths["chemistry"], index=False)
    species.to_csv(paths["species"], index=False)
    portfolio.sort_values("v10_integrated_case_rank").to_csv(paths["portfolio"], index=False)
    routes.to_csv(paths["routes"], index=False)

    w1_identity = pd.read_csv(V17 / "W1_CANDIDATE_CHEMBL37_IDENTITY_8_V15.csv", low_memory=False)
    w1_readiness = pd.read_csv(V17 / "W1_CANDIDATE_ORTHOGONAL_READINESS_8_V15.csv", low_memory=False)
    w1 = portfolio[portfolio["v10_integrated_case_rank"].between(1, 8)].sort_values("v10_integrated_case_rank")
    w1_identity_fields_match = (
        w1["ligand_inchikey"].tolist() == w1_identity["molecule_inchikey"].tolist()
        and w1["molecule_chembl_id"].tolist() == w1_identity["molecule_chembl_id"].tolist()
        and w1["chembl_standard_inchikey"].tolist() == w1_identity["chembl_standard_inchikey"].tolist()
        and w1["prodrug"].astype(int).tolist() == w1_identity["prodrug"].astype(int).tolist()
        and w1["project_vs_chembl_full_inchikey_status"].tolist()
        == w1_identity["project_vs_chembl_full_inchikey_status"].tolist()
    )
    numeric_fields = [
        "molecular_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge",
        "charged_atom_count", "fraction_csp3", "specified_chiral_centers", "aromatic_atom_fraction",
        "pains_alert_count", "brenk_alert_count",
    ]
    w1_chemistry_match = all(np.allclose(
        w1[column].astype(float), w1_identity[column].astype(float), rtol=0, atol=1e-10
    ) for column in numeric_fields)
    w1_risk_match = w1["experimental_handling_risk_tier"].tolist() == w1_readiness[
        "experimental_handling_risk_tier"
    ].tolist()
    name_mismatches = entities[entities["source_label_vs_chembl_preferred_name_class"].ne("EXACT_NORMALIZED_NAME")]
    checks = {
        "protocol_stamp_and_dependencies_verified": True,
        "exact_30_pairs_28_entities_16_targets": (
            len(portfolio) == 30 and entities["project_full_inchikey"].nunique() == len(entities) == 28
            and portfolio["target_chembl_id"].nunique() == 16
        ),
        "all_28_frozen_chembl_ids_resolved_one_to_one": len(chembl_identity) == 28 and entities["molecule_chembl_id"].notna().all(),
        "project_chembl_key_partition_27_exact_1_serdex_representation": (
            entities["project_vs_chembl_full_inchikey_status"].eq("EXACT_FULL_INCHIKEY_MATCH").sum() == 27
            and entities["project_vs_chembl_full_inchikey_status"].eq("FULL_INCHIKEY_REPRESENTATION_MISMATCH").sum() == 1
            and entities.loc[
                entities["project_vs_chembl_full_inchikey_status"].str.contains("MISMATCH"), "pref_name"
            ].tolist() == ["SERDEXMETHYLPHENIDATE"]
        ),
        "source_name_partition_27_exact_1_ketoconazole_levoketoconazole_hold": (
            len(name_mismatches) == 1 and name_mismatches.iloc[0]["molecule_chembl_id"] == "CHEMBL295698"
            and name_mismatches.iloc[0]["pref_name"] == "LEVOKETOCONAZOLE"
            and name_mismatches.iloc[0]["source_drug_names"] == "ketoconazole"
        ),
        "rank21_modeled_entity_full_key_preserved_and_not_substituted_racemate": (
            portfolio.loc[portfolio["v10_integrated_case_rank"].eq(21), "project_full_inchikey"].tolist()
            == ["XMAYWYJOQHXEEK-ZEQKJWHPSA-N"]
            and portfolio.loc[portfolio["v10_integrated_case_rank"].eq(21), "identity_execution_status"].str.contains("ADJUDICATION_HOLD").all()
        ),
        "chembl_prodrug_partition_exact_nitazoxanide_romidepsin_serdex": set(
            entities.loc[entities["prodrug"].eq(1), "pref_name"]
        ) == {"NITAZOXANIDE", "ROMIDEPSIN", "SERDEXMETHYLPHENIDATE"},
        "active_species_handling_exact_3_and_never_merge": (
            len(species) == 3 and set(species["parent_molecule_chembl_id"])
            == {"CHEMBL1401", "CHEMBL343448", "CHEMBL4301162"}
            and species["entity_merge_policy"].str.startswith("NEVER_MERGE").all()
        ),
        "romidepsin_reduced_dithiol_not_fabricated_as_chembl_entity": (
            species.loc[species["parent_molecule_chembl_id"].eq("CHEMBL343448"), "related_molecule_chembl_id"].tolist()
            == ["NO_SEPARATE_CHEMBL37_ENTITY"]
            and species.loc[species["parent_molecule_chembl_id"].eq("CHEMBL343448"), "evidence_source"].str.contains("21587264").all()
        ),
        "all_28_project_smiles_regenerate_exact_project_full_key": chemistry["rdkit_inchikey"].eq(
            chemistry["project_full_inchikey"]
        ).all(),
        "pubchem_all_network_resolved_and_exact_full_key_28": (
            len(pubchem) == 28 and pubchem["vendor_network_error"].fillna("").eq("").all()
            and pubchem["pubchem_identity_status"].eq("EXACT_FULL_INCHIKEY_MATCH").all()
        ),
        "pubchem_raw_exact_1_bulk_plus_28_vendor_no_retry": (
            len(raw_paths) == 29 and sum("BULK_IDENTITY" in path.name for path in raw_paths) == 1
            and sum("CHEMICAL_VENDORS" in path.name for path in raw_paths) == 28
            and not any("RETRY" in path.name for path in raw_paths)
        ),
        "full30_risk_tiers_complete_and_never_change_rank": (
            len(portfolio) == 30 and portfolio["experimental_handling_risk_tier"].notna().all()
            and portfolio["v10_integrated_case_rank"].astype(int).tolist() == list(range(1, 31))
        ),
        "w1_identity_fields_reproduce_v15": w1_identity_fields_match,
        "w1_chemistry_numeric_fields_reproduce_v15": w1_chemistry_match,
        "w1_risk_tiers_reproduce_v15": w1_risk_match,
        "assay_routes_exact_30_all_nonempty": (
            len(routes) == 30 and routes[[
                "primary_assay_route", "orthogonal_confirmation_route", "required_counterassays",
                "active_species_requirement", "execution_authorization",
            ]].fillna("").ne("").all().all()
        ),
        "wave_authorization_exact_8_w1_21_w2_1_veto": (
            routes["execution_authorization"].str.startswith("W1_").sum() == 8
            and routes["execution_authorization"].str.startswith("W2_").sum() == 21
            and routes["execution_authorization"].str.startswith("VETO_").sum() == 1
        ),
        "w2_and_veto_never_assay_authorized": (
            routes.loc[routes["execution_wave"].eq("W2_CONTINGENT_ONLY"), "execution_authorization"]
            .str.startswith("W2_CONTINGENT_COMPUTATIONAL_READINESS_ONLY_PROCUREMENT_AND_ASSAY_NOT_AUTHORIZED").all()
            and routes.loc[routes["execution_wave"].eq("VETO_NOT_AUTHORIZED"), "execution_authorization"]
            .str.startswith("VETO_COMPUTATIONAL_AUDIT_ONLY_NO_PROCUREMENT_OR_ASSAY_AUTHORIZED").all()
        ),
        "vendor_record_claim_never_stock_verified": entities["pubchem_vendor_record_status"].str.contains(
            "NOT_STOCK_VERIFIED|NO_VENDOR_RECORD"
        ).all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "counts": {
            "candidate_pairs": 30, "unique_entities": 28, "unique_targets": 16,
            "chembl_full_key_exact": int(entities["project_vs_chembl_full_inchikey_status"].eq("EXACT_FULL_INCHIKEY_MATCH").sum()),
            "chembl_representation_mismatch": int(entities["project_vs_chembl_full_inchikey_status"].str.contains("MISMATCH").sum()),
            "name_entity_mismatch_holds": len(name_mismatches), "chembl_prodrugs": int(entities["prodrug"].eq(1).sum()),
            "active_species_routes": len(species), "pubchem_exact_full_key": int(pubchem["pubchem_identity_status"].eq("EXACT_FULL_INCHIKEY_MATCH").sum()),
            "pubchem_vendor_record_indicators": int(pubchem["vendor_record_indicator"].fillna(False).sum()),
            "risk_tier_counts_30_pairs": {str(k): int(v) for k, v in portfolio["experimental_handling_risk_tier"].value_counts().items()},
            "chembl_known_mechanism_rows": len(mechanisms), "entities_with_chembl_known_mechanism": mechanisms["molecule_chembl_id"].nunique(),
        },
        "claim_boundaries": protocol["selection_bias_and_claim_boundaries"],
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [PROTOCOL, DB, V17 / "PROSPECTIVE_INTEGRATED_CASEBOOK_V10.csv"]},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in [*paths.values(), *raw_paths]},
    }
    summary_path = OUT / "FULL30_IDENTITY_CHEMISTRY_READINESS_SUMMARY_V18.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": summary["status"], "checks_passed": sum(checks.values()), "checks_total": len(checks),
        **summary["counts"], "summary_sha256": sha256(summary_path),
    }, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
