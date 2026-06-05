from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATA_PATTERNS = [
    "data/**/*LINCS*",
    "data/**/*lincs*",
    "data/**/*CMap*",
    "data/**/*cmap*",
    "data/**/*CLUE*",
    "data/**/*clue*",
    "data/**/*L1000*",
    "data/**/*l1000*",
]

DISEASE_SIGNATURE_PATTERNS = [
    "data/**/*DEG*",
    "data/**/*deg*",
    "data/**/*differential*expression*",
    "data/**/*signature*",
    "data/**/*Signature*",
]


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\b(hydrochloride|dihydrochloride|monohydrate|tosylate|mesylate|bromide|citrate|sodium|potassium|calcium|phosphate|sulfate|acetate)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_chembl(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"CHEMBL\d+", text)
    return match.group(0) if match else ""


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def find_files(root: Path, patterns: list[str], max_files: int = 200) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                matches.append(path)
            if len(matches) >= max_files:
                return sorted(set(matches))
    return sorted(set(matches))


def file_rows(root: Path, perturbation_files: list[Path], disease_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for category, files in [("perturbation_signature_candidate", perturbation_files), ("disease_signature_candidate", disease_files)]:
        for path in files:
            if path in seen:
                continue
            seen.add(path)
            stat = path.stat()
            rows.append(
                {
                    "category": category,
                    "path": rel(path, root),
                    "suffix": path.suffix.lower(),
                    "sizeBytes": stat.st_size,
                    "mtimeUtc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
    return pd.DataFrame(rows)


def read_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"direction", "drug", "drugId", "pairId", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate matrix is missing required columns: {sorted(missing)}")
    df["drugNameNorm"] = df["drug"].map(normalize_name)
    df["drugChemblNorm"] = df["drugId"].map(normalize_chembl)
    return df


def read_fda_structures(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["fdaDrugNameNorm", "fdaChemblNorm", "SMILES", "Generic Name (INN)", "Brand Name"])
    df = pd.read_excel(path)
    for col in ["Generic Name (INN)", "Brand Name", "SMILES", "ChEMBL ID"]:
        if col not in df.columns:
            df[col] = ""
    df["fdaDrugNameNorm"] = df["Generic Name (INN)"].map(normalize_name)
    df["fdaBrandNameNorm"] = df["Brand Name"].map(normalize_name)
    df["fdaChemblNorm"] = df["ChEMBL ID"].map(normalize_chembl)
    df["hasSmiles"] = df["SMILES"].astype(str).str.strip().ne("")
    return df


def build_drug_scope(candidates: pd.DataFrame, fda: pd.DataFrame) -> pd.DataFrame:
    drug_scope = (
        candidates.groupby(["drugId", "drug", "drugChemblNorm", "drugNameNorm"], dropna=False)
        .agg(
            candidateRows=("pairId", "count"),
            uniqueDirections=("direction", "nunique"),
            directions=("direction", lambda x: ";".join(sorted(set(map(str, x))))),
            uniqueTargets=("target", "nunique"),
            bestSotaContextRank=(
                "sotaContextRankGlobal" if "sotaContextRankGlobal" in candidates.columns else "finalRankGlobal",
                "min",
            ),
            bestSotaContextScore=("sotaContextScore" if "sotaContextScore" in candidates.columns else "finalPriorityScore", "max"),
        )
        .reset_index()
    )
    if fda.empty:
        drug_scope["fdaStructureMatchType"] = "missing_fda_structure_table"
        drug_scope["fdaSmilesAvailable"] = False
        drug_scope["fdaChemblId"] = ""
        drug_scope["fdaGenericName"] = ""
        drug_scope["fdaBrandName"] = ""
        drug_scope["smiles"] = ""
        return drug_scope

    by_chembl = (
        fda[fda["fdaChemblNorm"].astype(str).ne("")]
        .sort_values(["hasSmiles"], ascending=False)
        .drop_duplicates("fdaChemblNorm")
        .set_index("fdaChemblNorm")
    )
    by_name = (
        fda[fda["fdaDrugNameNorm"].astype(str).ne("")]
        .sort_values(["hasSmiles"], ascending=False)
        .drop_duplicates("fdaDrugNameNorm")
        .set_index("fdaDrugNameNorm")
    )
    rows: list[dict[str, Any]] = []
    for row in drug_scope.itertuples(index=False):
        match_type = "unmatched"
        match = None
        if row.drugChemblNorm and row.drugChemblNorm in by_chembl.index:
            match_type = "chembl_exact"
            match = by_chembl.loc[row.drugChemblNorm]
        elif row.drugNameNorm and row.drugNameNorm in by_name.index:
            match_type = "generic_name_exact"
            match = by_name.loc[row.drugNameNorm]
        item = row._asdict()
        if match is not None:
            item.update(
                {
                    "fdaStructureMatchType": match_type,
                    "fdaSmilesAvailable": bool(match.get("hasSmiles", False)),
                    "fdaChemblId": match.get("ChEMBL ID", ""),
                    "fdaGenericName": match.get("Generic Name (INN)", ""),
                    "fdaBrandName": match.get("Brand Name", ""),
                    "smiles": match.get("SMILES", ""),
                }
            )
        else:
            item.update(
                {
                    "fdaStructureMatchType": match_type,
                    "fdaSmilesAvailable": False,
                    "fdaChemblId": "",
                    "fdaGenericName": "",
                    "fdaBrandName": "",
                    "smiles": "",
                }
            )
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["candidateRows", "bestSotaContextScore"], ascending=[False, False])


def build_direction_scope(candidates: pd.DataFrame, drug_scope: pd.DataFrame) -> pd.DataFrame:
    drug_map = drug_scope[["drugId", "fdaSmilesAvailable", "fdaStructureMatchType"]].drop_duplicates("drugId")
    merged = candidates.merge(drug_map, on="drugId", how="left")
    rows = (
        merged.groupby("direction", dropna=False)
        .agg(
            candidateRows=("pairId", "count"),
            uniqueDrugs=("drugId", "nunique"),
            uniqueTargets=("target", "nunique"),
            structureMappedRows=("fdaSmilesAvailable", lambda x: int(x.fillna(False).astype(bool).sum())),
            structureMappedUniqueDrugs=("drugId", lambda x: int(drug_scope[drug_scope["drugId"].isin(set(x))]["fdaSmilesAvailable"].fillna(False).astype(bool).sum())),
        )
        .reset_index()
    )
    rows["structureMappedRowsPct"] = (rows["structureMappedRows"] / rows["candidateRows"] * 100.0).round(4)
    rows["structureMappedUniqueDrugsPct"] = (rows["structureMappedUniqueDrugs"] / rows["uniqueDrugs"] * 100.0).round(4)
    rows["requiresDiseaseDegSignature"] = True
    rows["requiresCmapPerturbationSignature"] = True
    return rows.sort_values("candidateRows", ascending=False)


def build_summary(
    source_path: Path,
    fda_path: Path,
    candidates: pd.DataFrame,
    drug_scope: pd.DataFrame,
    direction_scope: pd.DataFrame,
    files: pd.DataFrame,
) -> dict[str, Any]:
    perturbation_file_count = int((files["category"] == "perturbation_signature_candidate").sum()) if not files.empty else 0
    disease_file_count = int((files["category"] == "disease_signature_candidate").sum()) if not files.empty else 0
    structure_mapped_drugs = int(drug_scope["fdaSmilesAvailable"].fillna(False).astype(bool).sum()) if not drug_scope.empty else 0
    unique_drugs = int(drug_scope["drugId"].nunique()) if not drug_scope.empty else 0
    ready = perturbation_file_count > 0 and disease_file_count > 0
    return {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "LINCS/CMap expression-reversal readiness audit for final SOTA candidate drugs and disease directions.",
        "source": str(source_path),
        "sourceRows": int(len(candidates)),
        "candidateDirections": sorted(map(str, candidates["direction"].dropna().unique())),
        "uniqueCandidateDrugs": unique_drugs,
        "uniqueCandidateTargets": int(candidates["target"].nunique()),
        "fdaStructureTable": str(fda_path),
        "fdaStructureTablePresent": bool(fda_path.exists()),
        "structureMappedUniqueDrugs": structure_mapped_drugs,
        "structureMappedUniqueDrugsPct": round(structure_mapped_drugs / unique_drugs * 100.0, 4) if unique_drugs else 0.0,
        "perturbationSignatureCandidateFiles": perturbation_file_count,
        "diseaseSignatureCandidateFiles": disease_file_count,
        "calculationReadiness": "ready_to_score" if ready else "not_ready_external_signature_data_missing",
        "canComputeExpressionReversalNow": bool(ready),
        "requiredInputs": [
            "LINCS/CMap L1000 drug perturbation signatures with compound identifiers or structure/name mappings.",
            "Disease-direction differential-expression signatures with up-regulated and down-regulated gene sets.",
            "A mapping table from FDA/ChEMBL candidate drugs to CMap perturbagen identifiers.",
        ],
        "directionRows": int(len(direction_scope)),
        "structureMatchTypeCounts": dict(Counter(drug_scope["fdaStructureMatchType"].fillna("unmatched"))) if not drug_scope.empty else {},
        "methodNote": (
            "Expression reversal is an orthogonal drug-disease evidence layer. It does not predict binding or docking; "
            "it tests whether a drug perturbation signature is directionally opposite to a disease signature."
        ),
    }


def write_markdown(out_path: Path, summary: dict[str, Any], direction_scope: pd.DataFrame, files: pd.DataFrame) -> None:
    lines = [
        "# LINCS/CMap Expression-Reversal Readiness Audit",
        "",
        f"Generated: {summary['createdUtc']}",
        "",
        "## Purpose",
        "",
        "This audit prepares the transcriptomic expression-reversal validation layer. The layer asks whether a candidate drug produces a perturbation signature that opposes a disease signature.",
        "",
        "## Candidate Scope",
        "",
        f"- Candidate matrix: `{summary['source']}`.",
        f"- Candidate rows: {summary['sourceRows']}.",
        f"- Disease directions: {', '.join(summary['candidateDirections'])}.",
        f"- Unique candidate drugs: {summary['uniqueCandidateDrugs']}.",
        f"- Unique candidate targets: {summary['uniqueCandidateTargets']}.",
        f"- FDA/structure-mapped candidate drugs: {summary['structureMappedUniqueDrugs']}/{summary['uniqueCandidateDrugs']} ({summary['structureMappedUniqueDrugsPct']:.2f}%).",
        "",
        "## Data Readiness",
        "",
        f"- Perturbation-signature candidate files found: {summary['perturbationSignatureCandidateFiles']}.",
        f"- Disease-signature candidate files found: {summary['diseaseSignatureCandidateFiles']}.",
        f"- Calculation readiness: {summary['calculationReadiness']}.",
        "",
        "## Direction Scope",
        "",
        "| Direction | Rows | Drugs | Targets | Structure-Mapped Drugs | Structure-Mapped Drug % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in direction_scope.itertuples(index=False):
        lines.append(
            f"| {row.direction} | {int(row.candidateRows)} | {int(row.uniqueDrugs)} | {int(row.uniqueTargets)} | "
            f"{int(row.structureMappedUniqueDrugs)} | {float(row.structureMappedUniqueDrugsPct):.2f}% |"
        )
    lines.extend(["", "## Local Signature File Candidates", ""])
    if files.empty:
        lines.append("No local LINCS/CMap perturbation signatures or disease DEG/signature files were found under `data/` or `outputs/`.")
    else:
        lines.extend(["| Category | Path | Size bytes |", "|---|---|---:|"])
        for row in files.head(30).itertuples(index=False):
            lines.append(f"| {row.category} | `{row.path}` | {int(row.sizeBytes)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a readiness and scope audit, not an expression-reversal score.",
            "- Missing LINCS/CMap or disease DEG files are data-access gaps, not negative drug-disease evidence.",
            "- Once perturbation and disease signatures are available, candidate drugs can be scored by reversal of disease up/down genes and merged back into the SOTA prioritization matrix.",
            "",
            "## Output Files",
            "",
            "- `lincs_cmap_readiness_summary.json`: machine-readable scope and readiness summary.",
            "- `lincs_cmap_drug_scope.csv`: candidate drug structure/mapping scope.",
            "- `lincs_cmap_direction_scope.csv`: disease-direction scope for expression-reversal scoring.",
            "- `lincs_cmap_data_file_audit.csv`: local signature-file discovery audit.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    source_path = root / args.source
    out_dir = root / args.out_dir
    fda_path = root / args.fda_structures
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(source_path)
    fda = read_fda_structures(fda_path)
    perturbation_files = find_files(root, DATA_PATTERNS)
    disease_files = find_files(root, DISEASE_SIGNATURE_PATTERNS)
    files = file_rows(root, perturbation_files, disease_files)
    drug_scope = build_drug_scope(candidates, fda)
    direction_scope = build_direction_scope(candidates, drug_scope)
    summary = build_summary(source_path, fda_path, candidates, drug_scope, direction_scope, files)

    drug_scope.to_csv(out_dir / "lincs_cmap_drug_scope.csv", index=False)
    direction_scope.to_csv(out_dir / "lincs_cmap_direction_scope.csv", index=False)
    files.to_csv(out_dir / "lincs_cmap_data_file_audit.csv", index=False)
    write_json(out_dir / "lincs_cmap_readiness_summary.json", summary)
    write_markdown(out_dir / "LINCS_CMAP_READINESS_AUDIT.md", summary, direction_scope, files)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit readiness for a LINCS/CMap expression-reversal validation layer.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source",
        default="outputs/sota_validation/final_prioritization/final_priority_sota_context_matrix.csv",
        help="Candidate matrix to scope for expression-reversal scoring.",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/lincs_cmap_readiness")
    parser.add_argument(
        "--fda-structures",
        default="FDA_approved_small_molecules_2005_2026_with_structures.xlsx",
        help="FDA structure table used to audit candidate drug identifiers and SMILES availability.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
