from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DIRECTIONS = [
    "oncology",
    "cardiovascular",
    "infectious_disease",
    "neurology_psychiatry",
    "immunology_inflammation",
]

SALT_WORDS = [
    "hydrochloride",
    "hydrochrloride",
    "dihydrochloride",
    "hydrobromide",
    "monohydrate",
    "hydrate",
    "tosylate",
    "mesylate",
    "besylate",
    "bromide",
    "chloride",
    "iodide",
    "citrate",
    "sodium",
    "potassium",
    "calcium",
    "phosphate",
    "sulfate",
    "sulphate",
    "acetate",
    "maleate",
    "dimaleate",
    "fumarate",
    "succinate",
    "tartrate",
    "lactate",
    "nitrate",
    "esylate",
    "camsylate",
    "bitartrate",
    "diolamine",
    "ditosylate",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(" + "|".join(map(re.escape, SALT_WORDS)) + r")\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def inchikey_first(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.split("-")[0] if text else ""


def bounded(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def read_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"pairId", "drugId", "drug", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate table missing required columns: {sorted(missing)}")
    return df


def read_drug_library(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["drug_id", "chembl_id", "drug_name", "inchikey", "canonical_smiles"]:
        if col not in df.columns:
            df[col] = ""
    df["drugIdKey"] = df["drug_id"].astype(str)
    df["chemblIdKey"] = df["chembl_id"].astype(str)
    df["drugNameNorm"] = df["drug_name"].map(normalize_name)
    df["inchiKeyFull"] = df["inchikey"].fillna("").astype(str).str.upper()
    df["inchiKeyFirst"] = df["inchiKeyFull"].map(inchikey_first)
    return df


def read_compounds(path: Path) -> pd.DataFrame:
    comp = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ["pert_id", "cmap_name", "inchi_key", "compound_aliases"]:
        if col not in comp.columns:
            comp[col] = ""
    comp["cmapNameNorm"] = comp["cmap_name"].map(normalize_name)
    comp["inchiKeyFull"] = comp["inchi_key"].fillna("").astype(str).str.upper()
    comp["inchiKeyFirst"] = comp["inchiKeyFull"].map(inchikey_first)
    return comp


def split_aliases(value: Any) -> list[str]:
    text = str(value or "")
    aliases: list[str] = []
    for part in re.split(r"[|;,]", text):
        norm = normalize_name(part)
        if norm:
            aliases.append(norm)
    return aliases


def build_candidate_drug_scope(candidates: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    drugs = candidates[["drugId", "drug"]].drop_duplicates().copy()
    lib_by_drug_id = (
        library.sort_values(["inchiKeyFull"], ascending=False).drop_duplicates("drugIdKey").set_index("drugIdKey")
    )
    lib_by_chembl = (
        library[library["chemblIdKey"].astype(str).ne("")]
        .sort_values(["inchiKeyFull"], ascending=False)
        .drop_duplicates("chemblIdKey")
        .set_index("chemblIdKey")
    )
    rows: list[dict[str, Any]] = []
    for row in drugs.itertuples(index=False):
        item = {"drugId": row.drugId, "drug": row.drug}
        lib_row = None
        if str(row.drugId) in lib_by_drug_id.index:
            lib_row = lib_by_drug_id.loc[str(row.drugId)]
            item["libraryMatchType"] = "drug_id_exact"
        elif str(row.drugId) in lib_by_chembl.index:
            lib_row = lib_by_chembl.loc[str(row.drugId)]
            item["libraryMatchType"] = "chembl_id_exact"
        else:
            item["libraryMatchType"] = "missing_library_row"
        if lib_row is not None:
            item.update(
                {
                    "libraryDrugName": lib_row.get("drug_name", ""),
                    "libraryChemblId": lib_row.get("chembl_id", ""),
                    "libraryInchiKey": lib_row.get("inchiKeyFull", ""),
                    "libraryInchiKeyFirst": lib_row.get("inchiKeyFirst", ""),
                    "libraryDrugNameNorm": lib_row.get("drugNameNorm", ""),
                }
            )
        else:
            item.update(
                {
                    "libraryDrugName": "",
                    "libraryChemblId": "",
                    "libraryInchiKey": "",
                    "libraryInchiKeyFirst": "",
                    "libraryDrugNameNorm": normalize_name(row.drug),
                }
            )
        item["candidateDrugNameNorm"] = normalize_name(row.drug)
        rows.append(item)
    return pd.DataFrame(rows)


def read_siginfo(path: Path, valid_pert_ids: set[str] | None = None) -> pd.DataFrame:
    usecols = [
        "sig_id",
        "pert_id",
        "pert_type",
        "cmap_name",
        "cell_iname",
        "pert_time",
        "pert_dose",
        "pert_dose_unit",
        "tas",
        "qc_pass",
        "is_hiq",
        "is_exemplar_sig",
        "is_ncs_sig",
        "is_null_sig",
    ]
    sig = pd.read_csv(path, sep="\t", usecols=lambda col: col in usecols, low_memory=False)
    sig = sig[sig["pert_type"].eq("trt_cp")].copy()
    sig["qc_pass"] = pd.to_numeric(sig.get("qc_pass"), errors="coerce").fillna(0).astype(int)
    sig = sig[sig["qc_pass"].eq(1)].copy()
    sig["pert_id"] = sig["pert_id"].astype(str)
    if valid_pert_ids is not None:
        sig = sig[sig["pert_id"].isin(valid_pert_ids)].copy()
    for col in ["tas", "pert_time", "pert_dose", "is_hiq", "is_exemplar_sig", "is_ncs_sig", "is_null_sig"]:
        if col in sig.columns:
            sig[col] = pd.to_numeric(sig[col], errors="coerce")
    return sig


def build_drug_cmap_mapping(drug_scope: pd.DataFrame, compounds: pd.DataFrame, siginfo: pd.DataFrame) -> pd.DataFrame:
    valid_pert_ids = set(siginfo["pert_id"].dropna().astype(str))
    compounds = compounds[compounds["pert_id"].astype(str).isin(valid_pert_ids)].copy()

    by_full_ik: dict[str, list[int]] = defaultdict(list)
    by_first_ik: dict[str, list[int]] = defaultdict(list)
    by_name: dict[str, list[int]] = defaultdict(list)
    by_alias: dict[str, list[int]] = defaultdict(list)

    for idx, row in compounds.reset_index(drop=True).iterrows():
        full = str(row.get("inchiKeyFull", ""))
        first = str(row.get("inchiKeyFirst", ""))
        name = str(row.get("cmapNameNorm", ""))
        if full:
            by_full_ik[full].append(idx)
        if first:
            by_first_ik[first].append(idx)
        if name:
            by_name[name].append(idx)
        for alias in split_aliases(row.get("compound_aliases", "")):
            by_alias[alias].append(idx)

    rows: list[dict[str, Any]] = []
    comp_reset = compounds.reset_index(drop=True)
    sig_counts = siginfo.groupby("pert_id")["sig_id"].nunique().to_dict()
    for drug in drug_scope.itertuples(index=False):
        match_indices: list[tuple[int, str]] = []
        full = str(getattr(drug, "libraryInchiKey", "") or "")
        first = str(getattr(drug, "libraryInchiKeyFirst", "") or "")
        lib_name = str(getattr(drug, "libraryDrugNameNorm", "") or "")
        cand_name = str(getattr(drug, "candidateDrugNameNorm", "") or "")
        if full and full in by_full_ik:
            match_indices.extend((idx, "inchikey_full") for idx in by_full_ik[full])
        elif first and first in by_first_ik:
            match_indices.extend((idx, "inchikey_first_block") for idx in by_first_ik[first])
        elif lib_name and lib_name in by_name:
            match_indices.extend((idx, "name_exact") for idx in by_name[lib_name])
        elif cand_name and cand_name in by_name:
            match_indices.extend((idx, "candidate_name_exact") for idx in by_name[cand_name])
        else:
            names = [name for name in [lib_name, cand_name] if name]
            for name in names:
                if name in by_alias:
                    match_indices.extend((idx, "alias_exact") for idx in by_alias[name])
                    break

        seen: set[str] = set()
        for idx, match_type in match_indices:
            comp = comp_reset.iloc[idx]
            pert_id = str(comp["pert_id"])
            if pert_id in seen:
                continue
            seen.add(pert_id)
            rows.append(
                {
                    "drugId": drug.drugId,
                    "drug": drug.drug,
                    "pertId": pert_id,
                    "cmapName": comp.get("cmap_name", ""),
                    "cmapInchiKey": comp.get("inchi_key", ""),
                    "matchType": match_type,
                    "cmapSignatureCountQcPass": int(sig_counts.get(pert_id, 0)),
                }
            )
        if not seen:
            rows.append(
                {
                    "drugId": drug.drugId,
                    "drug": drug.drug,
                    "pertId": "",
                    "cmapName": "",
                    "cmapInchiKey": "",
                    "matchType": "not_mapped_to_cmap",
                    "cmapSignatureCountQcPass": 0,
                }
            )
    return pd.DataFrame(rows)


def read_gene_info(path: Path) -> pd.DataFrame:
    gene = pd.read_csv(path, sep="\t", low_memory=False)
    gene["geneSymbolNorm"] = gene["gene_symbol"].fillna("").astype(str).str.upper()
    gene["geneIdStr"] = gene["gene_id"].astype(str)
    return gene


def load_direction_gene_sets(root: Path, gene_info: pd.DataFrame, top_n: int) -> dict[str, dict[str, Any]]:
    symbol_to_gene_ids: dict[str, list[str]] = defaultdict(list)
    for row in gene_info[["geneSymbolNorm", "geneIdStr"]].dropna().itertuples(index=False):
        if row.geneSymbolNorm and row.geneIdStr:
            symbol_to_gene_ids[row.geneSymbolNorm].append(row.geneIdStr)

    direction_sets: dict[str, dict[str, Any]] = {}
    for direction in DIRECTIONS:
        path = root / direction / "creeds_gene_counts.csv"
        if not path.exists():
            direction_sets[direction] = {
                "upSymbols": [],
                "downSymbols": [],
                "upGeneIds": [],
                "downGeneIds": [],
                "status": "missing_creeds_gene_counts",
            }
            continue
        df = pd.read_csv(path)
        for col in ["upSignatureCount", "downSignatureCount", "totalSignatureCount"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["geneNorm"] = df["gene"].fillna("").astype(str).str.upper()
        up_ranked = df[df["upSignatureCount"].gt(0)].sort_values(
            ["upSignatureCount", "totalSignatureCount", "geneNorm"],
            ascending=[False, False, True],
        )
        down_ranked = df[df["downSignatureCount"].gt(0)].sort_values(
            ["downSignatureCount", "totalSignatureCount", "geneNorm"],
            ascending=[False, False, True],
        )
        up_symbols = list(up_ranked["geneNorm"].drop_duplicates().head(top_n))
        down_symbols = list(down_ranked["geneNorm"].drop_duplicates().head(top_n))

        # Resolve overlaps by retaining the stronger disease-direction side.
        up_count = dict(zip(df["geneNorm"], df["upSignatureCount"]))
        down_count = dict(zip(df["geneNorm"], df["downSignatureCount"]))
        overlap = set(up_symbols) & set(down_symbols)
        if overlap:
            up_symbols = [
                gene for gene in up_symbols if gene not in overlap or up_count.get(gene, 0) > down_count.get(gene, 0)
            ]
            down_symbols = [
                gene
                for gene in down_symbols
                if gene not in overlap or down_count.get(gene, 0) > up_count.get(gene, 0)
            ]

        def map_symbols(symbols: list[str]) -> list[str]:
            ids: list[str] = []
            for symbol in symbols:
                ids.extend(symbol_to_gene_ids.get(symbol, []))
            return sorted(set(ids), key=lambda x: int(x) if x.isdigit() else x)

        direction_sets[direction] = {
            "upSymbols": up_symbols,
            "downSymbols": down_symbols,
            "upGeneIds": map_symbols(up_symbols),
            "downGeneIds": map_symbols(down_symbols),
            "status": "ok",
        }
    return direction_sets


def read_gctx_ids(path: Path) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    with h5py.File(path, "r") as handle:
        sig_ids = [item.decode("utf-8") for item in handle["0/META/COL/id"][:]]
        gene_ids = [item.decode("utf-8") for item in handle["0/META/ROW/id"][:]]
    return sig_ids, gene_ids, {sid: idx for idx, sid in enumerate(sig_ids)}, {gid: idx for idx, gid in enumerate(gene_ids)}


def direction_column_sets(direction_sets: dict[str, dict[str, Any]], gene_id_to_col: dict[str, int]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for direction, values in direction_sets.items():
        up_cols = [gene_id_to_col[gid] for gid in values["upGeneIds"] if gid in gene_id_to_col]
        down_cols = [gene_id_to_col[gid] for gid in values["downGeneIds"] if gid in gene_id_to_col]
        result[direction] = {
            **values,
            "upCols": np.array(sorted(set(up_cols)), dtype=np.int64),
            "downCols": np.array(sorted(set(down_cols)), dtype=np.int64),
        }
    return result


def score_signature_matrix(
    gctx_path: Path,
    selected_siginfo: pd.DataFrame,
    direction_sets: dict[str, dict[str, Any]],
    sig_id_to_row: dict[str, int],
    batch_size: int,
) -> pd.DataFrame:
    selected = selected_siginfo.copy()
    selected["gctxRowIndex"] = selected["sig_id"].map(sig_id_to_row)
    selected = selected[selected["gctxRowIndex"].notna()].copy()
    selected["gctxRowIndex"] = selected["gctxRowIndex"].astype(int)
    selected = selected.sort_values("gctxRowIndex").reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame()

    direction_col_ix = {
        direction: {
            "up": values["upCols"],
            "down": values["downCols"],
        }
        for direction, values in direction_sets.items()
    }
    rows: list[dict[str, Any]] = []
    with h5py.File(gctx_path, "r") as handle:
        matrix = handle["0/DATA/0/matrix"]
        n = len(selected)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = selected.iloc[start:end].copy()
            row_indices = batch["gctxRowIndex"].to_numpy(dtype=np.int64)
            values = matrix[row_indices, :]
            for direction, cols in direction_col_ix.items():
                up_cols = cols["up"]
                down_cols = cols["down"]
                if len(up_cols) == 0 or len(down_cols) == 0:
                    up_mean = np.full(len(batch), np.nan, dtype=np.float32)
                    down_mean = np.full(len(batch), np.nan, dtype=np.float32)
                    raw = np.full(len(batch), np.nan, dtype=np.float32)
                else:
                    up_mean = np.nanmean(values[:, up_cols], axis=1)
                    down_mean = np.nanmean(values[:, down_cols], axis=1)
                    raw = down_mean - up_mean
                for idx, base in enumerate(batch.itertuples(index=False)):
                    rows.append(
                        {
                            "direction": direction,
                            "sigId": base.sig_id,
                            "pertId": base.pert_id,
                            "cmapName": base.cmap_name,
                            "cell": base.cell_iname,
                            "pertTime": getattr(base, "pert_time", np.nan),
                            "pertDose": getattr(base, "pert_dose", np.nan),
                            "pertDoseUnit": getattr(base, "pert_dose_unit", ""),
                            "tas": getattr(base, "tas", np.nan),
                            "isHiq": getattr(base, "is_hiq", np.nan),
                            "isExemplar": getattr(base, "is_exemplar_sig", np.nan),
                            "diseaseUpGeneCountMapped": int(len(up_cols)),
                            "diseaseDownGeneCountMapped": int(len(down_cols)),
                            "drugMeanDiseaseUpGenes": float(up_mean[idx]) if math.isfinite(float(up_mean[idx])) else np.nan,
                            "drugMeanDiseaseDownGenes": float(down_mean[idx]) if math.isfinite(float(down_mean[idx])) else np.nan,
                            "rawReversalScore": float(raw[idx]) if math.isfinite(float(raw[idx])) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_drug_direction_scores(signature_scores: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    valid_mapping = mapping[mapping["pertId"].astype(str).ne("")].copy()
    sig = signature_scores.merge(valid_mapping[["drugId", "drug", "pertId", "matchType"]], on="pertId", how="inner")
    if sig.empty:
        return pd.DataFrame()
    sig["rawReversalScore"] = pd.to_numeric(sig["rawReversalScore"], errors="coerce")
    sig["tas"] = pd.to_numeric(sig.get("tas"), errors="coerce")
    grouped_rows: list[dict[str, Any]] = []
    for (drug_id, drug, direction), group in sig.groupby(["drugId", "drug", "direction"], dropna=False):
        group = group[group["rawReversalScore"].notna()].copy()
        if group.empty:
            continue
        group = group.sort_values(["rawReversalScore", "tas"], ascending=[False, False])
        best = group.iloc[0]
        top_n = max(1, min(5, len(group)))
        top_raw = group.head(top_n)["rawReversalScore"]
        grouped_rows.append(
            {
                "drugId": drug_id,
                "drug": drug,
                "direction": direction,
                "cmapMapped": True,
                "cmapPerturbagenCount": int(group["pertId"].nunique()),
                "cmapSignatureCount": int(group["sigId"].nunique()),
                "cmapBestRawReversal": round(float(best["rawReversalScore"]), 6),
                "cmapMedianRawReversal": round(float(group["rawReversalScore"].median()), 6),
                "cmapTop5MeanRawReversal": round(float(top_raw.mean()), 6),
                "cmapPositiveSignaturePct": round(float(group["rawReversalScore"].gt(0).mean() * 100.0), 4),
                "cmapBestSigId": best["sigId"],
                "cmapBestPertId": best["pertId"],
                "cmapBestCmapName": best.get("cmapName", ""),
                "cmapBestCell": best.get("cell", ""),
                "cmapBestDose": best.get("pertDose", ""),
                "cmapBestDoseUnit": best.get("pertDoseUnit", ""),
                "cmapBestTime": best.get("pertTime", ""),
                "cmapBestTas": round(float(best["tas"]), 6) if pd.notna(best.get("tas")) else np.nan,
                "cmapMatchTypes": ";".join(sorted(set(map(str, group["matchType"].dropna())))),
            }
        )
    out = pd.DataFrame(grouped_rows)
    if out.empty:
        return out
    out["cmapReversalPercentileWithinDirection"] = 0.0
    for direction, idx in out.groupby("direction").groups.items():
        values = out.loc[idx, "cmapBestRawReversal"].rank(method="average", pct=True)
        out.loc[idx, "cmapReversalPercentileWithinDirection"] = (values * 100.0).round(4)
    out["cmapReversalScore"] = out.apply(
        lambda row: round(
            bounded(
                (50.0 + 0.5 * row["cmapReversalPercentileWithinDirection"])
                if row["cmapBestRawReversal"] > 0
                else (0.5 * row["cmapReversalPercentileWithinDirection"])
            ),
            4,
        ),
        axis=1,
    )
    out["cmapReversalTier"] = out.apply(cmap_tier, axis=1)
    out["cmapReversalInterpretation"] = out.apply(cmap_interpretation, axis=1)
    return out.sort_values(["direction", "cmapReversalScore", "cmapBestRawReversal"], ascending=[True, False, False])


def cmap_tier(row: pd.Series) -> str:
    raw = float(row.get("cmapBestRawReversal", 0.0))
    score = float(row.get("cmapReversalScore", 0.0))
    if raw >= 1.0 and score >= 85:
        return "A_strong_reversal"
    if raw >= 0.5 and score >= 70:
        return "B_moderate_reversal"
    if raw > 0:
        return "C_weak_or_context_dependent_reversal"
    return "D_no_reversal_signal"


def cmap_interpretation(row: pd.Series) -> str:
    tier = row.get("cmapReversalTier", "")
    best = row.get("cmapBestRawReversal", "")
    cell = row.get("cmapBestCell", "")
    sig_count = row.get("cmapSignatureCount", "")
    if str(tier).startswith("A_"):
        return f"Strong LINCS reversal signal across {sig_count} mapped drug signatures; best raw reversal {best} in {cell}."
    if str(tier).startswith("B_"):
        return f"Moderate LINCS reversal signal; best raw reversal {best} in {cell}."
    if str(tier).startswith("C_"):
        return f"Weak or context-dependent LINCS reversal signal; use as supporting evidence only."
    return "No positive LINCS reversal signal among mapped CMap signatures."


def build_candidate_audit(candidates: pd.DataFrame, drug_direction: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapped_drugs = set(mapping[mapping["pertId"].astype(str).ne("")]["drugId"].astype(str))
    out = candidates.merge(drug_direction, on=["drugId", "drug", "direction"], how="left")
    out["cmapMapped"] = out["drugId"].astype(str).isin(mapped_drugs)
    out["cmapReversalTier"] = out["cmapReversalTier"].fillna(
        out["cmapMapped"].map(lambda x: "D_no_direction_score" if x else "U_not_mapped_to_cmap")
    )
    out["cmapReversalInterpretation"] = out["cmapReversalInterpretation"].fillna(
        out["cmapMapped"].map(
            lambda x: "Drug mapped to CMap but no usable direction score was computed."
            if x
            else "Drug was not mapped to a CMap compound perturbagen."
        )
    )
    numeric_defaults = {
        "cmapPerturbagenCount": 0,
        "cmapSignatureCount": 0,
        "cmapBestRawReversal": np.nan,
        "cmapMedianRawReversal": np.nan,
        "cmapTop5MeanRawReversal": np.nan,
        "cmapPositiveSignaturePct": np.nan,
        "cmapReversalPercentileWithinDirection": np.nan,
        "cmapReversalScore": 0.0,
    }
    for col, value in numeric_defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)
    selected_cols = [
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "finalPriorityScore",
        "sotaContextScore",
        "sotaReadyScore",
        "noveltyClass",
        "reviewTrack",
        "admetTier",
        "structureConfidenceTier",
        "poseAuditStatus",
        "cmapMapped",
        "cmapPerturbagenCount",
        "cmapSignatureCount",
        "cmapBestRawReversal",
        "cmapMedianRawReversal",
        "cmapTop5MeanRawReversal",
        "cmapPositiveSignaturePct",
        "cmapReversalPercentileWithinDirection",
        "cmapReversalScore",
        "cmapReversalTier",
        "cmapBestSigId",
        "cmapBestPertId",
        "cmapBestCmapName",
        "cmapBestCell",
        "cmapBestDose",
        "cmapBestDoseUnit",
        "cmapBestTime",
        "cmapBestTas",
        "cmapMatchTypes",
        "cmapReversalInterpretation",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "validationGatesZh",
    ]
    available = [col for col in selected_cols if col in out.columns]
    return out.sort_values(["cmapReversalScore", "finalPriorityScore"], ascending=[False, False])[available]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LINCS/CMap Disease-Signature Reversal Audit",
        "",
        f"Created UTC: {summary['createdUtc']}",
        "",
        "## Scope",
        "",
        "This audit scores whether CMap compound perturbation signatures are directionally opposite to CREEDS disease-direction signatures.",
        "The score is drug-direction evidence and is not target-specific binding evidence.",
        "",
        "## Coverage",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Candidate drugs: {summary['candidateDrugs']}",
        f"- CMap-mapped candidate drugs: {summary['mappedCandidateDrugs']}",
        f"- Candidate rows with mapped drug: {summary['mappedCandidateRows']}",
        f"- Selected CMap signatures: {summary['selectedSignatureCount']}",
        "",
        "## Candidate Signal",
        "",
        f"- Candidate rows with positive reversal score: {summary['candidatePositiveReversalRows']}",
        f"- Top100 rows with positive reversal score: {summary['top100PositiveReversalRows']}",
        "",
        "## Tier Counts",
        "",
    ]
    for tier, count in summary["candidateTierCounts"].items():
        lines.append(f"- {tier}: {count}")
    lines.extend(
        [
            "",
            "## Direction Coverage",
            "",
            "| Direction | Up genes | Down genes | Mapped up genes | Mapped down genes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["directionGeneSetSummary"]:
        lines.append(
            f"| {row['direction']} | {row['upSymbols']} | {row['downSymbols']} | {row['mappedUpGeneIds']} | {row['mappedDownGeneIds']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Positive reversal means disease-up genes tend to be lower in the drug perturbation signature and disease-down genes tend to be higher.",
            "Because CMap signatures are cell-line, dose, and time dependent, this layer is used as orthogonal support for expert review rather than definitive efficacy evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LINCS/CMap drug-disease signature reversal over BioMaster candidates.")
    parser.add_argument("--candidates", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_gtex_context_matrix.csv"))
    parser.add_argument("--drug-library", type=Path, default=Path("data/processed/drug_library_pubchem_chembl_mapped.csv"))
    parser.add_argument("--lincs-dir", type=Path, default=Path("data/external/lincs_cmap"))
    parser.add_argument("--disease-signature-root", type=Path, default=Path("data/external/disease_signatures"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/sota_validation/lincs_cmap_reversal"))
    parser.add_argument("--top-genes", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-signatures-per-pert", type=int, default=80)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    final_out = Path("outputs/sota_validation/final_prioritization")
    final_out.mkdir(parents=True, exist_ok=True)
    gctx_path = args.lincs_dir / "level5_beta_trt_cp_n720216x12328.gctx"
    siginfo_path = args.lincs_dir / "siginfo_beta.txt"
    geneinfo_path = args.lincs_dir / "geneinfo_beta.txt"
    compound_path = args.lincs_dir / "compoundinfo_beta.txt"

    candidates = read_candidates(args.candidates)
    drug_library = read_drug_library(args.drug_library)
    drug_scope = build_candidate_drug_scope(candidates, drug_library)
    siginfo_all_qc = read_siginfo(siginfo_path)
    compounds = read_compounds(compound_path)
    mapping = build_drug_cmap_mapping(drug_scope, compounds, siginfo_all_qc)
    mapped_pert_ids = set(mapping[mapping["pertId"].astype(str).ne("")]["pertId"].astype(str))

    selected_siginfo = siginfo_all_qc[siginfo_all_qc["pert_id"].isin(mapped_pert_ids)].copy()
    selected_siginfo["tasFill"] = pd.to_numeric(selected_siginfo.get("tas"), errors="coerce").fillna(-1)
    selected_siginfo = (
        selected_siginfo.sort_values(["pert_id", "is_hiq", "is_exemplar_sig", "tasFill"], ascending=[True, False, False, False])
        .groupby("pert_id", group_keys=False)
        .head(args.max_signatures_per_pert)
        .drop(columns=["tasFill"], errors="ignore")
        .reset_index(drop=True)
    )

    gene_info = read_gene_info(geneinfo_path)
    sig_ids, gene_ids, sig_id_to_row, gene_id_to_col = read_gctx_ids(gctx_path)
    direction_sets_raw = load_direction_gene_sets(args.disease_signature_root, gene_info, args.top_genes)
    direction_sets = direction_column_sets(direction_sets_raw, gene_id_to_col)
    selected_siginfo = selected_siginfo[selected_siginfo["sig_id"].isin(sig_id_to_row)].copy()

    signature_scores = score_signature_matrix(
        gctx_path=gctx_path,
        selected_siginfo=selected_siginfo,
        direction_sets=direction_sets,
        sig_id_to_row=sig_id_to_row,
        batch_size=args.batch_size,
    )
    drug_direction = aggregate_drug_direction_scores(signature_scores, mapping)
    candidate_audit = build_candidate_audit(candidates, drug_direction, mapping)

    mapping.to_csv(args.outdir / "lincs_cmap_drug_mapping.csv", index=False)
    selected_siginfo.to_csv(args.outdir / "lincs_cmap_selected_signatures.csv", index=False)
    signature_scores.to_csv(args.outdir / "lincs_cmap_signature_reversal_scores.csv", index=False)
    drug_direction.to_csv(args.outdir / "lincs_cmap_drug_direction_reversal_scores.csv", index=False)
    candidate_audit.to_csv(args.outdir / "candidate_lincs_cmap_reversal_audit.csv", index=False)
    candidate_audit.to_csv(final_out / "final_priority_lincs_cmap_augmented_table.csv", index=False)
    candidate_audit.head(300).to_csv(final_out / "final_priority_lincs_cmap_top300_expert_shortlist.csv", index=False)

    direction_gene_summary = []
    for direction, values in direction_sets.items():
        direction_gene_summary.append(
            {
                "direction": direction,
                "upSymbols": len(values["upSymbols"]),
                "downSymbols": len(values["downSymbols"]),
                "mappedUpGeneIds": int(len(values["upCols"])),
                "mappedDownGeneIds": int(len(values["downCols"])),
            }
        )

    mapped_drugs = mapping[mapping["pertId"].astype(str).ne("")]["drugId"].nunique()
    mapped_rows = int(candidate_audit["cmapMapped"].astype(bool).sum())
    positive_rows = int(pd.to_numeric(candidate_audit["cmapBestRawReversal"], errors="coerce").fillna(-999).gt(0).sum())
    top100_positive = int(
        pd.to_numeric(candidate_audit.sort_values("finalPriorityScore", ascending=False).head(100)["cmapBestRawReversal"], errors="coerce")
        .fillna(-999)
        .gt(0)
        .sum()
    )
    summary = {
        "createdUtc": utc_now(),
        "inputs": {
            "candidates": str(args.candidates),
            "gctx": str(gctx_path),
            "siginfo": str(siginfo_path),
            "geneinfo": str(geneinfo_path),
            "compoundinfo": str(compound_path),
            "diseaseSignatureRoot": str(args.disease_signature_root),
        },
        "candidateRows": int(len(candidates)),
        "candidateDrugs": int(candidates["drugId"].nunique()),
        "mappedCandidateDrugs": int(mapped_drugs),
        "mappedCandidateRows": mapped_rows,
        "allQcTrtCpSignatureRows": int(len(siginfo_all_qc)),
        "selectedSignatureCount": int(len(selected_siginfo)),
        "signatureScoreRows": int(len(signature_scores)),
        "drugDirectionScoreRows": int(len(drug_direction)),
        "candidatePositiveReversalRows": positive_rows,
        "top100PositiveReversalRows": top100_positive,
        "candidateTierCounts": candidate_audit["cmapReversalTier"].value_counts().to_dict(),
        "mappingMatchTypeCounts": mapping["matchType"].value_counts().to_dict(),
        "directionGeneSetSummary": direction_gene_summary,
        "outputs": {
            "drugMapping": str(args.outdir / "lincs_cmap_drug_mapping.csv"),
            "selectedSignatures": str(args.outdir / "lincs_cmap_selected_signatures.csv"),
            "signatureScores": str(args.outdir / "lincs_cmap_signature_reversal_scores.csv"),
            "drugDirectionScores": str(args.outdir / "lincs_cmap_drug_direction_reversal_scores.csv"),
            "candidateAudit": str(args.outdir / "candidate_lincs_cmap_reversal_audit.csv"),
            "finalAugmentedTable": str(final_out / "final_priority_lincs_cmap_augmented_table.csv"),
        },
        "methodNote": "Drug-direction expression reversal is computed as mean(drug signature over disease-down genes) minus mean(drug signature over disease-up genes). Higher values indicate stronger disease-signature reversal.",
    }
    (args.outdir / "lincs_cmap_reversal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.outdir / "LINCS_CMAP_REVERSAL_AUDIT.md", summary)
    write_markdown(final_out / "FINAL_PRIORITY_LINCS_CMAP_REVERSAL_AUDIT.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
