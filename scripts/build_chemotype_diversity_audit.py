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

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


RDLogger.DisableLog("rdApp.*")

CUTOFFS = [20, 50, 100, 200, 500, 1000]
DIRECTION_TOP_CUTOFFS = [20, 50, 100, 200]


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def hhi(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return sum((count / total) ** 2 for count in counts.values())


def chembl_base(drug_id: Any) -> str:
    text = norm(drug_id)
    return text.split("__", 1)[0] if "__" in text else text


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = norm(value)
        if text:
            return text
    return ""


def aggregate_terms(values: pd.Series, limit: int = 8) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = norm(value)
        if not text or text in seen:
            continue
        terms.append(text)
        seen.add(text)
        if len(terms) >= limit:
            break
    return "; ".join(terms)


def read_fda_structures(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="FDA Small Molecules 2005-2026").fillna("")
    raw["ChEMBL ID"] = raw["ChEMBL ID"].astype(str).str.strip()
    raw["SMILES"] = raw["SMILES"].astype(str).str.strip()
    rows: list[dict[str, Any]] = []
    for chembl_id, group in raw.groupby("ChEMBL ID", dropna=False):
        valid = group[group["SMILES"].astype(str).str.strip().ne("")]
        source = valid.iloc[0] if not valid.empty else group.iloc[0]
        rows.append(
            {
                "chemblId": chembl_id,
                "smiles": norm(source.get("SMILES", "")),
                "genericNames": aggregate_terms(group.get("Generic Name (INN)", pd.Series(dtype=str))),
                "brandNames": aggregate_terms(group.get("Brand Name", pd.Series(dtype=str))),
                "approvalYears": aggregate_terms(group.get("Approval Year", pd.Series(dtype=str))),
                "therapeuticAreas": aggregate_terms(group.get("Therapeutic Area", pd.Series(dtype=str))),
                "routes": aggregate_terms(group.get("Route", pd.Series(dtype=str))),
                "indications": aggregate_terms(group.get("Indication", pd.Series(dtype=str)), limit=4),
                "fdaRecordCount": int(len(group)),
                "sourceSmilesRecordIndex": int(source.name) if source.name is not None else "",
            }
        )
    return pd.DataFrame(rows)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def molecular_row(row: pd.Series) -> dict[str, Any]:
    smiles = norm(row.get("smiles", ""))
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "chemblId": row.get("chemblId", ""),
            "smiles": smiles,
            "smilesValid": 0,
            "structureError": "invalid_or_missing_smiles",
        }
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        scaffold = f"acyclic:{Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol), isomericSmiles=False)}"
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return {
        "chemblId": row.get("chemblId", ""),
        "smiles": smiles,
        "canonicalSmiles": canonical,
        "smilesValid": 1,
        "structureError": "",
        "murckoScaffold": scaffold,
        "molecularFormula": formula,
        "molecularWeight": round(float(Descriptors.MolWt(mol)), 4),
        "logP": round(float(Crippen.MolLogP(mol)), 4),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        "qed": round(float(QED.qed(mol)), 4),
        "heavyAtoms": int(mol.GetNumHeavyAtoms()),
        "rotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
    }


def fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def assign_clusters(drug_rows: list[dict[str, Any]], similarity_threshold: float) -> None:
    valid_indices: list[int] = []
    fps: list[DataStructs.ExplicitBitVect] = []
    for idx, row in enumerate(drug_rows):
        mol = mol_from_smiles(str(row.get("canonicalSmiles") or row.get("smiles") or ""))
        if mol is None:
            row["chemotypeClusterId"] = ""
            row["nearestNeighborSimilarity"] = ""
            row["nearestNeighborChemblId"] = ""
            continue
        valid_indices.append(idx)
        fps.append(fingerprint(mol))

    if not fps:
        return

    distances: list[float] = []
    nearest: list[tuple[float, str]] = [(0.0, "") for _ in fps]
    for i in range(1, len(fps)):
        sims = list(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
        distances.extend([1.0 - sim for sim in sims])
        if sims:
            best_j = int(np.argmax(sims))
            if sims[best_j] > nearest[i][0]:
                nearest[i] = (float(sims[best_j]), str(drug_rows[valid_indices[best_j]].get("chemblId", "")))
            for j, sim in enumerate(sims):
                if sim > nearest[j][0]:
                    nearest[j] = (float(sim), str(drug_rows[valid_indices[i]].get("chemblId", "")))

    clusters = Butina.ClusterData(
        distances,
        len(fps),
        distThresh=1.0 - similarity_threshold,
        isDistData=True,
        reordering=True,
    )
    cluster_order = sorted(clusters, key=lambda cluster: (-len(cluster), min(cluster)))
    for cluster_id, cluster in enumerate(cluster_order, start=1):
        for local_idx in cluster:
            row = drug_rows[valid_indices[local_idx]]
            row["chemotypeClusterId"] = f"C{cluster_id:03d}"
            row["chemotypeClusterSizeUniqueDrugs"] = int(len(cluster))
            row["nearestNeighborSimilarity"] = round(nearest[local_idx][0], 4)
            row["nearestNeighborChemblId"] = nearest[local_idx][1]


def build_drug_audit(final: pd.DataFrame, fda: pd.DataFrame) -> pd.DataFrame:
    fda_by_id = {row["chemblId"]: row for _, row in fda.iterrows()}
    rows: list[dict[str, Any]] = []
    for drug_id, group in final.groupby("drugId", dropna=False):
        drug_id_text = norm(drug_id)
        base_id = chembl_base(drug_id_text)
        match_id = ""
        match_status = "missing_fda_structure"
        if drug_id_text in fda_by_id:
            match_id = drug_id_text
            match_status = "exact_chembl_match"
        elif base_id in fda_by_id:
            match_id = base_id
            match_status = "base_chembl_match_from_compound_suffix"
        fda_row = fda_by_id.get(match_id, pd.Series(dtype=object))
        mol_fields = molecular_row(fda_row) if match_id else {"smilesValid": 0, "structureError": "no_fda_match"}
        row: dict[str, Any] = {
            "drugId": drug_id_text,
            "drugIdBase": base_id,
            "matchedChemblId": match_id,
            "structureMatchStatus": match_status,
            "drug": first_nonempty(group["drug"]),
            "candidateRows": int(len(group)),
            "uniqueTargets": int(group["protein"].nunique()),
            "directions": ";".join(sorted(group["direction"].astype(str).unique())),
            "directionCount": int(group["direction"].nunique()),
            "bestFinalRankGlobal": int(pd.to_numeric(group["finalRankGlobal"], errors="coerce").min()),
            "bestFinalPriorityScore": round(float(pd.to_numeric(group["finalPriorityScore"], errors="coerce").max()), 4),
            "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
            "reviewTrackCounts": dict(Counter(group["reviewTrack"].astype(str))),
            "noveltyClassCounts": dict(Counter(group["noveltyClass"].astype(str))),
            "genericNames": norm(fda_row.get("genericNames", "")) if match_id else "",
            "therapeuticAreas": norm(fda_row.get("therapeuticAreas", "")) if match_id else "",
            "approvalYears": norm(fda_row.get("approvalYears", "")) if match_id else "",
            "fdaRecordCount": int(fda_row.get("fdaRecordCount", 0)) if match_id else 0,
        }
        row.update(mol_fields)
        rows.append(row)
    assign_clusters(rows, similarity_threshold=0.70)
    return pd.DataFrame(rows)


def merge_candidate_audit(final: pd.DataFrame, drug_audit: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "drugId",
        "drugIdBase",
        "matchedChemblId",
        "structureMatchStatus",
        "smilesValid",
        "murckoScaffold",
        "chemotypeClusterId",
        "chemotypeClusterSizeUniqueDrugs",
        "nearestNeighborSimilarity",
        "nearestNeighborChemblId",
        "molecularWeight",
        "logP",
        "tpsa",
        "qed",
        "heavyAtoms",
        "rotatableBonds",
        "aromaticRings",
        "therapeuticAreas",
        "approvalYears",
    ]
    return final.merge(drug_audit[[col for col in keep if col in drug_audit.columns]], on="drugId", how="left")


def topk_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_ranked = df.sort_values("finalRankGlobal")
    for cutoff in CUTOFFS:
        top = global_ranked.head(cutoff)
        if top.empty:
            continue
        rows.append(concentration_row(top, "global", "all", cutoff))
    for direction, group in df.groupby("direction"):
        ranked = group.sort_values("finalRankWithinDirection")
        for cutoff in DIRECTION_TOP_CUTOFFS:
            top = ranked.head(cutoff)
            if top.empty:
                continue
            rows.append(concentration_row(top, "direction", direction, cutoff))
    return rows


def concentration_row(top: pd.DataFrame, group_type: str, group_value: str, cutoff: int) -> dict[str, Any]:
    scaffold_values = top["murckoScaffold"].fillna("").replace("", "unmapped").astype(str).tolist()
    cluster_values = top["chemotypeClusterId"].fillna("").replace("", "unmapped").astype(str).tolist()
    drug_values = top["drugId"].astype(str).tolist()
    nn_values = pd.to_numeric(top["nearestNeighborSimilarity"], errors="coerce").dropna()
    scaffold_counts = Counter(scaffold_values)
    cluster_counts = Counter(cluster_values)
    drug_counts = Counter(drug_values)
    known_hits = int(pd.to_numeric(top.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum())
    return {
        "groupType": group_type,
        "groupValue": group_value,
        "cutoff": cutoff,
        "rows": int(len(top)),
        "uniqueDrugs": int(top["drugId"].nunique()),
        "uniqueScaffolds": int(len(set(scaffold_values) - {"unmapped"})),
        "uniqueChemotypeClusters": int(len(set(cluster_values) - {"unmapped"})),
        "unmappedRows": int(sum(1 for value in scaffold_values if value == "unmapped")),
        "drugHHI": round(hhi(drug_values), 4),
        "scaffoldHHI": round(hhi(scaffold_values), 4),
        "chemotypeClusterHHI": round(hhi(cluster_values), 4),
        "topDrug": drug_counts.most_common(1)[0][0],
        "topDrugPct": round(pct(drug_counts.most_common(1)[0][1], len(top)), 4),
        "topScaffold": scaffold_counts.most_common(1)[0][0],
        "topScaffoldPct": round(pct(scaffold_counts.most_common(1)[0][1], len(top)), 4),
        "topChemotypeCluster": cluster_counts.most_common(1)[0][0],
        "topChemotypeClusterPct": round(pct(cluster_counts.most_common(1)[0][1], len(top)), 4),
        "medianNearestNeighborSimilarity": round(float(nn_values.median()), 4) if not nn_values.empty else "",
        "knownDrugTargetRows": known_hits,
    }


def scaffold_cluster_tables(candidate: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scaffold_rows: list[dict[str, Any]] = []
    for scaffold, group in candidate.groupby("murckoScaffold", dropna=False):
        scaffold_text = norm(scaffold) or "unmapped"
        scaffold_rows.append(
            {
                "murckoScaffold": scaffold_text,
                "candidateRows": int(len(group)),
                "uniqueDrugs": int(group["drugId"].nunique()),
                "uniqueTargets": int(group["protein"].nunique()),
                "directions": ";".join(sorted(group["direction"].astype(str).unique())),
                "bestFinalRankGlobal": int(pd.to_numeric(group["finalRankGlobal"], errors="coerce").min()),
                "bestDrug": group.sort_values("finalRankGlobal").iloc[0].get("drug", ""),
                "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
            }
        )
    cluster_rows: list[dict[str, Any]] = []
    for cluster, group in candidate.groupby("chemotypeClusterId", dropna=False):
        cluster_text = norm(cluster) or "unmapped"
        cluster_rows.append(
            {
                "chemotypeClusterId": cluster_text,
                "candidateRows": int(len(group)),
                "uniqueDrugs": int(group["drugId"].nunique()),
                "uniqueScaffolds": int(group["murckoScaffold"].nunique()),
                "uniqueTargets": int(group["protein"].nunique()),
                "directions": ";".join(sorted(group["direction"].astype(str).unique())),
                "bestFinalRankGlobal": int(pd.to_numeric(group["finalRankGlobal"], errors="coerce").min()),
                "bestDrug": group.sort_values("finalRankGlobal").iloc[0].get("drug", ""),
                "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
            }
        )
    return (
        sorted(scaffold_rows, key=lambda row: (-row["candidateRows"], row["bestFinalRankGlobal"])),
        sorted(cluster_rows, key=lambda row: (-row["candidateRows"], row["bestFinalRankGlobal"])),
    )


def select_diverse_shortlist(candidate: pd.DataFrame, per_direction: int = 40) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for direction, group in candidate.groupby("direction"):
        ranked = group.sort_values(["finalPriorityScore", "finalRankWithinDirection"], ascending=[False, True])
        scaffold_counts: Counter[str] = Counter()
        cluster_counts: Counter[str] = Counter()
        drug_counts: Counter[str] = Counter()
        direction_rows: list[pd.Series] = []
        for _, row in ranked.iterrows():
            if len(direction_rows) >= per_direction:
                break
            scaffold = norm(row.get("murckoScaffold", "")) or "unmapped"
            cluster = norm(row.get("chemotypeClusterId", "")) or "unmapped"
            drug = norm(row.get("drugId", ""))
            if scaffold_counts[scaffold] >= 4:
                continue
            if cluster_counts[cluster] >= 6:
                continue
            if drug_counts[drug] >= 2:
                continue
            direction_rows.append(row)
            scaffold_counts[scaffold] += 1
            cluster_counts[cluster] += 1
            drug_counts[drug] += 1
        for _, row in ranked.iterrows():
            if len(direction_rows) >= per_direction:
                break
            if any(existing["pairId"] == row["pairId"] for existing in direction_rows):
                continue
            drug = norm(row.get("drugId", ""))
            if drug_counts[drug] >= 3:
                continue
            direction_rows.append(row)
            scaffold_counts[norm(row.get("murckoScaffold", "")) or "unmapped"] += 1
            cluster_counts[norm(row.get("chemotypeClusterId", "")) or "unmapped"] += 1
            drug_counts[drug] += 1
        selected.extend(direction_rows)
    if not selected:
        return pd.DataFrame(columns=candidate.columns)
    result = pd.DataFrame(selected).sort_values(["direction", "finalPriorityScore"], ascending=[True, False]).copy()
    result["chemotypeDiverseRankWithinDirection"] = result.groupby("direction").cumcount() + 1
    return result


def build_summary(candidate: pd.DataFrame, drug: pd.DataFrame, topk: list[dict[str, Any]], diverse: pd.DataFrame) -> dict[str, Any]:
    top_by_key = {(row["groupType"], row["groupValue"], row["cutoff"]): row for row in topk}
    exact_or_base = int(drug["structureMatchStatus"].isin(["exact_chembl_match", "base_chembl_match_from_compound_suffix"]).sum())
    valid_drugs = drug[drug["smilesValid"].eq(1)]
    valid_candidate_rows = int(candidate["smilesValid"].fillna(0).astype(int).sum())
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(candidate)),
        "uniqueDrugs": int(drug["drugId"].nunique()),
        "structureMappedUniqueDrugs": exact_or_base,
        "structureMappedUniqueDrugPct": round(pct(exact_or_base, len(drug)), 4),
        "validStructureUniqueDrugs": int(valid_drugs["drugId"].nunique()),
        "validStructureCandidateRows": valid_candidate_rows,
        "validStructureCandidatePct": round(pct(valid_candidate_rows, len(candidate)), 4),
        "exactChemblMatchedUniqueDrugs": int(drug["structureMatchStatus"].eq("exact_chembl_match").sum()),
        "baseChemblMatchedUniqueDrugs": int(drug["structureMatchStatus"].eq("base_chembl_match_from_compound_suffix").sum()),
        "unmatchedUniqueDrugs": int(drug["structureMatchStatus"].eq("missing_fda_structure").sum()),
        "uniqueMurckoScaffolds": int(valid_drugs["murckoScaffold"].nunique()),
        "uniqueChemotypeClusters": int(valid_drugs["chemotypeClusterId"].nunique()),
        "top100UniqueDrugs": top_by_key.get(("global", "all", 100), {}).get("uniqueDrugs"),
        "top100UniqueScaffolds": top_by_key.get(("global", "all", 100), {}).get("uniqueScaffolds"),
        "top100UniqueChemotypeClusters": top_by_key.get(("global", "all", 100), {}).get("uniqueChemotypeClusters"),
        "top100TopScaffoldPct": top_by_key.get(("global", "all", 100), {}).get("topScaffoldPct"),
        "top100TopChemotypeClusterPct": top_by_key.get(("global", "all", 100), {}).get("topChemotypeClusterPct"),
        "top100ScaffoldHHI": top_by_key.get(("global", "all", 100), {}).get("scaffoldHHI"),
        "top100ChemotypeClusterHHI": top_by_key.get(("global", "all", 100), {}).get("chemotypeClusterHHI"),
        "top500UniqueDrugs": top_by_key.get(("global", "all", 500), {}).get("uniqueDrugs"),
        "top500UniqueScaffolds": top_by_key.get(("global", "all", 500), {}).get("uniqueScaffolds"),
        "top500UniqueChemotypeClusters": top_by_key.get(("global", "all", 500), {}).get("uniqueChemotypeClusters"),
        "diverseShortlistRows": int(len(diverse)),
        "diverseShortlistUniqueDrugs": int(diverse["drugId"].nunique()) if not diverse.empty else 0,
        "diverseShortlistUniqueScaffolds": int(diverse["murckoScaffold"].nunique()) if not diverse.empty else 0,
        "diverseShortlistUniqueClusters": int(diverse["chemotypeClusterId"].nunique()) if not diverse.empty else 0,
        "structureMatchStatusCounts": dict(Counter(drug["structureMatchStatus"].astype(str))),
        "methodNote": "Drug chemotypes are computed from FDA small-molecule SMILES using Murcko scaffolds and Morgan-fingerprint Butina clusters at Tanimoto similarity >= 0.70. Final ranking is preserved; the shortlist only caps repeated drugs/scaffolds/clusters for expert review.",
    }


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final Priority Chemotype Diversity Audit",
            "",
            f"Generated: {summary.get('created_utc')}",
            "",
            "## Method",
            "",
            "FDA small-molecule SMILES were matched to candidate drugs by ChEMBL ID. If a candidate drug ID carried a compound suffix, the base ChEMBL ID was used as a conservative fallback and recorded in the match status. Valid structures were summarized by Murcko scaffold and Morgan-fingerprint Butina cluster.",
            "",
            "## Summary",
            "",
            f"- Candidate rows: {summary.get('candidateRows')}",
            f"- Unique drugs: {summary.get('uniqueDrugs')}",
            f"- Structure-mapped unique drugs: {summary.get('structureMappedUniqueDrugs')} ({summary.get('structureMappedUniqueDrugPct')}%)",
            f"- Valid-structure candidate rows: {summary.get('validStructureCandidateRows')} ({summary.get('validStructureCandidatePct')}%)",
            f"- Unique Murcko scaffolds / chemotype clusters: {summary.get('uniqueMurckoScaffolds')} / {summary.get('uniqueChemotypeClusters')}",
            f"- Top100 unique drugs/scaffolds/clusters: {summary.get('top100UniqueDrugs')} / {summary.get('top100UniqueScaffolds')} / {summary.get('top100UniqueChemotypeClusters')}",
            f"- Top100 top-scaffold pct / top-cluster pct: {summary.get('top100TopScaffoldPct')}% / {summary.get('top100TopChemotypeClusterPct')}%",
            f"- Diverse shortlist rows: {summary.get('diverseShortlistRows')}",
            "",
            "## Outputs",
            "",
            "- Candidate audit: `outputs/sota_validation/final_prioritization/final_priority_chemotype_diversity_audit.csv`",
            "- Drug audit: `outputs/sota_validation/chemotype_diversity/chemotype_diversity_drug_audit.csv`",
            "- TopK concentration: `outputs/sota_validation/chemotype_diversity/chemotype_diversity_topk_concentration.csv`",
            "- Diverse shortlist: `outputs/sota_validation/final_prioritization/final_priority_chemotype_diverse_shortlist.csv`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit final candidate chemotype and scaffold diversity.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--fda-structures", default="FDA_approved_small_molecules_2005_2026_with_structures.xlsx")
    parser.add_argument("--out-dir", default="outputs/sota_validation/chemotype_diversity")
    parser.add_argument("--final-out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    final_out_dir = root / args.final_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_out_dir.mkdir(parents=True, exist_ok=True)

    final = pd.read_csv(root / args.final_table).fillna("")
    fda = read_fda_structures(root / args.fda_structures)
    drug_audit = build_drug_audit(final, fda)
    candidate = merge_candidate_audit(final, drug_audit)
    topk = topk_rows(candidate)
    scaffold_rows, cluster_rows = scaffold_cluster_tables(candidate)
    diverse = select_diverse_shortlist(candidate, per_direction=40)
    summary = build_summary(candidate, drug_audit, topk, diverse)

    drug_audit.to_csv(out_dir / "chemotype_diversity_drug_audit.csv", index=False)
    candidate.to_csv(final_out_dir / "final_priority_chemotype_diversity_audit.csv", index=False)
    write_csv(out_dir / "chemotype_diversity_topk_concentration.csv", topk)
    write_csv(out_dir / "chemotype_diversity_scaffold_summary.csv", scaffold_rows)
    write_csv(out_dir / "chemotype_diversity_cluster_summary.csv", cluster_rows)
    diverse.to_csv(final_out_dir / "final_priority_chemotype_diverse_shortlist.csv", index=False)
    write_json(out_dir / "chemotype_diversity_summary.json", summary)
    write_json(final_out_dir / "final_priority_chemotype_diversity_summary.json", summary)
    (final_out_dir / "FINAL_PRIORITY_CHEMOTYPE_DIVERSITY_AUDIT.md").write_text(markdown(summary), encoding="utf-8")

    print(json.dumps({"summary": summary, "out_dir": args.out_dir, "final_out_dir": args.final_out_dir}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
