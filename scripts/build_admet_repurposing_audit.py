from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}

DIRECTION_TERMS = {
    "oncology": ["oncology", "neoplasm", "carcinoma", "cancer", "tumor", "tumour", "leukemia", "lymphoma", "myeloma"],
    "infectious_disease": ["infectious", "infection", "hiv", "viral", "bacterial", "tuberculosis", "hepatitis", "antiviral", "antibacterial"],
    "cardiovascular": ["cardiovascular", "hypertension", "heart", "coronary", "thrombosis", "arrhythmia", "vascular", "stroke"],
    "neurology_psychiatry": [
        "neurology",
        "psychiatry",
        "psychiatric",
        "nervous",
        "parkinson",
        "alzheimer",
        "depression",
        "schizophrenia",
        "migraine",
        "epilepsy",
        "pain",
    ],
    "immunology_inflammation": [
        "immunology",
        "inflammation",
        "inflammatory",
        "immune",
        "arthritis",
        "asthma",
        "dermatitis",
        "psoriasis",
        "lupus",
        "colitis",
        "crohn",
    ],
}

SYSTEMIC_ROUTES = {
    "oral",
    "intravenous",
    "subcutaneous",
    "intramuscular",
    "injection",
    "infusion",
    "transdermal",
    "sublingual",
    "buccal",
    "rectal",
    "inhalation",
}

LOCAL_ROUTE_TERMS = {"topical", "ophthalmic", "otic", "nasal", "vaginal", "dental"}
CYP_DDI_TERMS = ["cytochrome p450", "cyp", "p-glycoprotein", "p glycoprotein", "p-gp", "p gp", "transporter"]
HIGH_RISK_INDICATION_TERMS = ["contrast", "diagnostic", "imaging", "anesthesia", "anaesthesia", "sedation", "radiopharmaceutical"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def bool_int(value: bool) -> int:
    return 1 if value else 0


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def has_any(text: str, terms: list[str] | set[str]) -> bool:
    lower = norm(text)
    return any(term in lower for term in terms)


def split_terms(text: str | None) -> list[str]:
    return [item.strip() for item in re.split(r"[;/,|]+", text or "") if item.strip()]


def best_catalog_match(catalog: FilterCatalog, mol: Chem.Mol) -> tuple[int, str]:
    matches = catalog.GetMatches(mol)
    if not matches:
        return 0, ""
    descriptions = sorted({match.GetDescription() for match in matches})
    return len(descriptions), "; ".join(descriptions[:8])


def build_filter_catalogs() -> tuple[FilterCatalog, FilterCatalog]:
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(pains_params), FilterCatalog(brenk_params)


def route_class(route: str) -> str:
    lower = norm(route)
    if has_any(lower, SYSTEMIC_ROUTES):
        return "systemic"
    if has_any(lower, LOCAL_ROUTE_TERMS):
        return "local"
    return "other_or_unclear"


def direction_fit(direction: str, therapeutic_area: str, indication: str) -> tuple[int, str]:
    text = " ".join([therapeutic_area or "", indication or ""])
    terms = DIRECTION_TERMS.get(direction, [])
    matched = [term for term in terms if term in norm(text)]
    return bool_int(bool(matched)), "; ".join(matched[:8])


def descriptor_row(row: dict[str, str], pains: FilterCatalog, brenk: FilterCatalog) -> dict[str, Any]:
    smiles = row.get("canonical_smiles") or row.get("isomeric_smiles") or row.get("SMILES") or ""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {
            "drugId": row.get("drug_id", ""),
            "drug": row.get("drug_name", ""),
            "chemblId": row.get("chembl_id", ""),
            "smilesValid": 0,
            "admetTier": "D",
            "admetScore": 0,
            "admetFlags": "invalid_smiles",
            "repurposingRouteClass": route_class(row.get("route", "")),
        }

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    qed = QED.qed(mol)
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber_violations = sum([rotb > 10, tpsa > 140])
    egan_violations = sum([logp > 5.88, tpsa > 131.6])
    pains_count, pains_notes = best_catalog_match(pains, mol)
    brenk_count, brenk_notes = best_catalog_match(brenk, mol)

    route = row.get("route", "")
    therapeutic_area = row.get("therapeutic_area", "")
    indication = row.get("indication", "")
    mechanism = row.get("mechanism_of_action", "")
    target_name = row.get("target_name", "")
    text = " ".join([therapeutic_area, indication, mechanism, target_name, route])
    systemic = route_class(route) == "systemic"
    local = route_class(route) == "local"
    cyp_ddi = has_any(text, CYP_DDI_TERMS)
    diagnostic_like = has_any(text, HIGH_RISK_INDICATION_TERMS)

    flags: list[str] = []
    if ro5_violations:
        flags.append(f"ro5_violations={ro5_violations}")
    if veber_violations:
        flags.append(f"veber_violations={veber_violations}")
    if egan_violations:
        flags.append(f"egan_violations={egan_violations}")
    if qed < 0.25:
        flags.append("low_qed")
    if pains_count:
        flags.append("pains_alert")
    if brenk_count >= 3:
        flags.append("multiple_brenk_alerts")
    if abs(formal_charge) >= 2:
        flags.append("high_formal_charge")
    if local:
        flags.append("local_route")
    if cyp_ddi:
        flags.append("cyp_or_transporter_ddi_text")
    if diagnostic_like:
        flags.append("diagnostic_or_procedure_like")

    score = 100
    score -= 12 * ro5_violations
    score -= 8 * veber_violations
    score -= 6 * egan_violations
    score -= 25 if qed < 0.25 else 0
    score -= 18 if pains_count else 0
    score -= min(20, 4 * brenk_count)
    score -= 10 if abs(formal_charge) >= 2 else 0
    score -= 8 if local else 0
    score -= 8 if cyp_ddi else 0
    score -= 10 if diagnostic_like else 0
    score += 6 if systemic and ro5_violations <= 1 and qed >= 0.35 else 0
    score = max(0, min(100, score))

    if score >= 80:
        tier = "A"
    elif score >= 65:
        tier = "B"
    elif score >= 45:
        tier = "C"
    else:
        tier = "D"

    return {
        "drugId": row.get("drug_id", ""),
        "drug": row.get("drug_name", ""),
        "brandName": row.get("brand_name", ""),
        "chemblId": row.get("chembl_id", ""),
        "approvalYear": row.get("approval_year", ""),
        "route": route,
        "routeClass": route_class(route),
        "therapeuticArea": therapeutic_area,
        "indication": indication,
        "mechanismOfAction": mechanism,
        "targetName": target_name,
        "smilesValid": 1,
        "molWt": round(mw, 3),
        "logP": round(logp, 3),
        "hbd": int(hbd),
        "hba": int(hba),
        "tpsa": round(tpsa, 3),
        "rotatableBonds": int(rotb),
        "heavyAtoms": int(heavy),
        "qed": round(qed, 4),
        "aromaticRings": int(aromatic),
        "formalCharge": int(formal_charge),
        "fractionCsp3": round(fsp3, 4),
        "ro5Violations": int(ro5_violations),
        "veberViolations": int(veber_violations),
        "eganViolations": int(egan_violations),
        "painsAlerts": int(pains_count),
        "painsNotes": pains_notes,
        "brenkAlerts": int(brenk_count),
        "brenkNotes": brenk_notes,
        "cypDdiTextFlag": bool_int(cyp_ddi),
        "diagnosticProcedureTextFlag": bool_int(diagnostic_like),
        "admetScore": int(round(score)),
        "admetTier": tier,
        "admetFlags": "; ".join(flags) if flags else "none",
    }


def build_drug_audit(drugs_csv: Path) -> list[dict[str, Any]]:
    pains, brenk = build_filter_catalogs()
    return [descriptor_row(row, pains, brenk) for row in read_csv(drugs_csv)]


def enrich_candidates(candidates_csv: Path, drug_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    drug_lookup = {row["drugId"]: row for row in drug_rows}
    rows: list[dict[str, Any]] = []
    for row in read_csv(candidates_csv):
        rank = int(float(row.get("rank") or 999999))
        if rank > top_n:
            continue
        drug = drug_lookup.get(row.get("drugId", ""), {})
        fit, matched_terms = direction_fit(
            row.get("direction", ""),
            drug.get("therapeuticArea") or row.get("therapeuticArea", ""),
            drug.get("indication") or row.get("indication", ""),
        )
        admet_score = number(drug.get("admetScore")) or 0
        credibility = number(row.get("credibilityScore")) or 0
        direction_score = number(row.get("directionScore")) or 0
        affinity = number(row.get("affinityScore")) or 0
        structure_bonus = 8 if row.get("status") == "completed" else -8
        route_bonus = 5 if drug.get("routeClass") == "systemic" else 0
        fit_bonus = 8 if fit else 0
        raw_translational_score = (
            0.36 * direction_score * 100
            + 0.24 * affinity * 100
            + 0.20 * credibility
            + 0.20 * admet_score
            + structure_bonus
            + route_bonus
            + fit_bonus
        )
        translational_score = round(max(0.0, min(100.0, raw_translational_score)), 3)
        if translational_score >= 85:
            posture = "A_high_priority_review"
        elif translational_score >= 75:
            posture = "B_mechanism_review"
        elif translational_score >= 60:
            posture = "C_secondary_review"
        else:
            posture = "D_low_priority_or_requires_rescue"
        rows.append(
            {
                "direction": row.get("direction", ""),
                "labelZh": DIRECTION_LABELS.get(row.get("direction", ""), row.get("direction", "")),
                "rank": rank,
                "pairId": row.get("pairId", ""),
                "drugId": row.get("drugId", ""),
                "drug": row.get("drug", ""),
                "target": row.get("target", ""),
                "protein": row.get("protein", ""),
                "directionScore": row.get("directionScore", ""),
                "affinityScore": row.get("affinityScore", ""),
                "diffdock": row.get("diffdock", ""),
                "status": row.get("status", ""),
                "credibilityScore": row.get("credibilityScore", ""),
                "admetScore": drug.get("admetScore", ""),
                "admetTier": drug.get("admetTier", ""),
                "admetFlags": drug.get("admetFlags", ""),
                "routeClass": drug.get("routeClass", ""),
                "route": drug.get("route", ""),
                "directionLabelFit": fit,
                "directionFitTerms": matched_terms,
                "qed": drug.get("qed", ""),
                "ro5Violations": drug.get("ro5Violations", ""),
                "veberViolations": drug.get("veberViolations", ""),
                "painsAlerts": drug.get("painsAlerts", ""),
                "brenkAlerts": drug.get("brenkAlerts", ""),
                "cypDdiTextFlag": drug.get("cypDdiTextFlag", ""),
                "diagnosticProcedureTextFlag": drug.get("diagnosticProcedureTextFlag", ""),
                "translationalScore": translational_score,
                "translationalPosture": posture,
            }
        )
    return sorted(rows, key=lambda item: (item["direction"], -item["translationalScore"], item["rank"]))


def summarize(drug_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts = Counter(row.get("admetTier", "NA") for row in drug_rows)
    route_counts = Counter(row.get("routeClass", "NA") for row in drug_rows)
    flag_counts: Counter[str] = Counter()
    for row in drug_rows:
        for flag in split_terms(row.get("admetFlags", "")):
            if flag and flag != "none":
                flag_counts[flag] += 1
    by_direction: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[row["direction"]].append(row)
    for direction, rows in sorted(grouped.items()):
        by_direction[direction] = {
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "rows": len(rows),
            "medianTranslationalScore": float(pd.Series([row["translationalScore"] for row in rows]).median()) if rows else None,
            "admetTierCounts": dict(Counter(row.get("admetTier", "NA") for row in rows)),
            "routeClassCounts": dict(Counter(row.get("routeClass", "NA") for row in rows)),
            "directionLabelFitRows": int(sum(1 for row in rows if int(row.get("directionLabelFit") or 0))),
            "highPriorityRows": int(sum(1 for row in rows if str(row.get("translationalPosture", "")).startswith("A_"))),
        }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "drugRows": len(drug_rows),
        "candidateRows": len(candidate_rows),
        "drugAdmetTierCounts": dict(tier_counts),
        "drugRouteClassCounts": dict(route_counts),
        "drugFlagCounts": dict(flag_counts.most_common()),
        "candidateByDirection": by_direction,
        "note": "Rule-based ADMET/repurposability audit using transparent RDKit descriptors and text flags; not a substitute for experimental ADMET or clinical safety review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build transparent ADMET and repurposability audit tables.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--candidates", default="outputs/disease_directions/disease_direction_integrated_candidates.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/admet_repurposing")
    parser.add_argument("--top-n-per-direction", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    drug_rows = build_drug_audit(root / args.drugs)
    candidate_rows = enrich_candidates(root / args.candidates, drug_rows, args.top_n_per_direction)
    summary = summarize(drug_rows, candidate_rows)
    summary["inputs"] = {
        "drugs": args.drugs,
        "candidates": args.candidates,
        "top_n_per_direction": args.top_n_per_direction,
    }
    summary["outputs"] = {
        "drugAudit": str((out_dir / "drug_admet_repurposing_audit.csv").resolve()),
        "candidateAudit": str((out_dir / "candidate_admet_repurposing_audit_topn.csv").resolve()),
        "summary": str((out_dir / "admet_repurposing_summary.json").resolve()),
    }

    drug_fields = [
        "drugId",
        "drug",
        "brandName",
        "chemblId",
        "approvalYear",
        "route",
        "routeClass",
        "therapeuticArea",
        "indication",
        "mechanismOfAction",
        "targetName",
        "smilesValid",
        "molWt",
        "logP",
        "hbd",
        "hba",
        "tpsa",
        "rotatableBonds",
        "heavyAtoms",
        "qed",
        "aromaticRings",
        "formalCharge",
        "fractionCsp3",
        "ro5Violations",
        "veberViolations",
        "eganViolations",
        "painsAlerts",
        "painsNotes",
        "brenkAlerts",
        "brenkNotes",
        "cypDdiTextFlag",
        "diagnosticProcedureTextFlag",
        "admetScore",
        "admetTier",
        "admetFlags",
    ]
    candidate_fields = [
        "direction",
        "labelZh",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "directionScore",
        "affinityScore",
        "diffdock",
        "status",
        "credibilityScore",
        "admetScore",
        "admetTier",
        "admetFlags",
        "routeClass",
        "route",
        "directionLabelFit",
        "directionFitTerms",
        "qed",
        "ro5Violations",
        "veberViolations",
        "painsAlerts",
        "brenkAlerts",
        "cypDdiTextFlag",
        "diagnosticProcedureTextFlag",
        "translationalScore",
        "translationalPosture",
    ]
    write_csv(out_dir / "drug_admet_repurposing_audit.csv", drug_fields, drug_rows)
    write_csv(out_dir / "candidate_admet_repurposing_audit_topn.csv", candidate_fields, candidate_rows)
    write_json(out_dir / "admet_repurposing_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
