#!/usr/bin/env python3
"""Build a reproducible physics-first funnel from 106k discovery pairs to 1000.

The script intentionally starts from the recovered 106,561 direct-action
discovery pairs, not from any previous strict/top-ready table. Known FDA target
pairs are scored separately as calibration controls and never enter the final
discovery candidate list.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    MurckoScaffold = None
    rdMolStandardize = None
    RDKIT_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "final_1000_funnel_v1"

DISCOVERY_PAIRS = ROOT / "outputs" / "recovered_recall_sources" / "direct_action_discovery_pairs.csv"
ALL_DIRECT_PAIRS = ROOT / "outputs" / "recovered_recall_sources" / "direct_action_target_engagement_pairs.csv"
KNOWN_CONTROLS = ROOT / "outputs" / "recovered_recall_sources" / "direct_action_known_target_controls.csv"
TARGET_STRUCTURE = ROOT / "outputs" / "puresnet_gpu_pocket_audit_v1" / "puresnet_gpu_target_summary_merged.csv"
FDA_XLSX = ROOT / "FDA_approved_small_molecules_2005_2026_with_structures.xlsx"
BOLTZ_TOP50 = ROOT / "outputs" / "boltz2_structure_affinity_v1" / "boltz2_top50_known_vs_discovery_detailed.csv"


NUMERIC_DEFAULTS: dict[str, float] = {
    "conplex_score": 0.0,
    "rank_within_drug": 999.0,
    "target_rank": 9999.0,
    "top_pocket_probability": 0.0,
    "top_pocket_score": 0.0,
    "max_probability": 0.0,
    "mean_probability": 0.0,
    "p2rank_puresnet_overlap_fraction": 0.0,
    "p2rank_puresnet_jaccard": 0.0,
    "p2rank_top_residue_count": 0.0,
    "puresnet_best_cluster_residue_count": 0.0,
    "receptor_residue_count": 9999.0,
    "fda_mw": math.nan,
    "fda_qed": math.nan,
    "fda_ro5_violations": math.nan,
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def join_unique(values: pd.Series, limit: int = 2000) -> str:
    out: list[str] = []
    for value in values:
        for part in re.split(r"[;|]", clean(value)):
            item = part.strip()
            if item and item not in out:
                out.append(item)
    return ";".join(out)[:limit]


def as_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def tier_starts(series: pd.Series, letters: tuple[str, ...]) -> pd.Series:
    prefixes = tuple(f"{letter}_" for letter in letters)
    return series.fillna("").astype(str).str.startswith(prefixes)


def safe_round(value: Any, digits: int = 3) -> Any:
    try:
        if pd.isna(value):
            return ""
        return round(float(value), digits)
    except Exception:
        return value


def read_fda_drugs() -> pd.DataFrame:
    fda = pd.read_excel(FDA_XLSX, sheet_name="FDA Small Molecules 2005-2026").fillna("")
    fda = fda.rename(
        columns={
            "ChEMBL ID": "drug_chembl_id",
            "Generic Name (INN)": "fda_generic_name",
            "Brand Name": "fda_brand_name",
            "Therapeutic Area": "fda_therapeutic_area",
            "Indication": "fda_indication",
            "Mechanism of Action": "fda_moa",
            "Action Type": "fda_action_type",
            "Route": "fda_route",
            "Target Name": "fda_target_names",
            "SMILES": "canonical_smiles",
            "Molecular Weight (Da)": "fda_mw",
            "QED": "fda_qed",
            "Ro5 Violations": "fda_ro5_violations",
            "Approval Year": "fda_approval_year",
            "LogP (Crippen)": "fda_logp",
            "TPSA (Å²)": "fda_tpsa",
        }
    )
    fda["drug_chembl_id"] = fda["drug_chembl_id"].map(clean)
    fda = fda[fda["drug_chembl_id"].str.startswith("CHEMBL")].copy()
    for col in [
        "fda_mw",
        "fda_qed",
        "fda_ro5_violations",
        "fda_approval_year",
        "fda_logp",
        "fda_tpsa",
    ]:
        if col in fda.columns:
            fda[col] = pd.to_numeric(fda[col], errors="coerce")

    grouped = fda.groupby("drug_chembl_id", dropna=False).agg(
        fda_generic_name=("fda_generic_name", join_unique),
        fda_brand_name=("fda_brand_name", join_unique),
        canonical_smiles=("canonical_smiles", lambda x: next((clean(v) for v in x if clean(v)), "")),
        fda_therapeutic_area=("fda_therapeutic_area", join_unique),
        fda_indication=("fda_indication", join_unique),
        fda_moa=("fda_moa", join_unique),
        fda_action_type=("fda_action_type", join_unique),
        fda_route=("fda_route", join_unique),
        fda_target_names=("fda_target_names", join_unique),
        fda_mw=("fda_mw", "median"),
        fda_qed=("fda_qed", "median"),
        fda_ro5_violations=("fda_ro5_violations", "median"),
        fda_approval_year=("fda_approval_year", "min"),
        fda_logp=("fda_logp", "median"),
        fda_tpsa=("fda_tpsa", "median"),
    )
    grouped = grouped.reset_index()

    scaffolds = grouped["canonical_smiles"].map(standardize_smiles)
    grouped["rdkit_parse_ok"] = [item["rdkit_parse_ok"] for item in scaffolds]
    grouped["active_moiety_smiles"] = [item["active_moiety_smiles"] for item in scaffolds]
    grouped["murcko_scaffold"] = [item["murcko_scaffold"] for item in scaffolds]
    grouped["canonical_smiles_rdkit"] = [item["canonical_smiles_rdkit"] for item in scaffolds]
    return grouped


def standardize_smiles(smiles: Any) -> dict[str, Any]:
    text = clean(smiles)
    if not text or not RDKIT_AVAILABLE:
        return {
            "rdkit_parse_ok": False,
            "canonical_smiles_rdkit": text,
            "active_moiety_smiles": text,
            "murcko_scaffold": "",
        }
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError("rdkit_parse_failed")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if fragments:
            parent = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        else:
            parent = mol
        try:
            parent = rdMolStandardize.Uncharger().uncharge(parent)
        except Exception:
            pass
        active = Chem.MolToSmiles(parent, isomericSmiles=True)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
        except Exception:
            scaffold = ""
        if not scaffold:
            scaffold = active
        return {
            "rdkit_parse_ok": True,
            "canonical_smiles_rdkit": canonical,
            "active_moiety_smiles": active,
            "murcko_scaffold": scaffold,
        }
    except Exception:
        return {
            "rdkit_parse_ok": False,
            "canonical_smiles_rdkit": text,
            "active_moiety_smiles": text,
            "murcko_scaffold": "",
        }


def pdb_residue_count(path_value: Any) -> int:
    path_text = clean(path_value)
    if not path_text:
        return 0
    path = Path(path_text)
    if not path.exists():
        return 0
    residues: set[tuple[str, str, str]] = set()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                resn = line[17:20].strip()
                if not resn:
                    continue
                chain = line[21].strip() or "A"
                resi = line[22:26].strip()
                icode = line[26].strip()
                if resi:
                    residues.add((chain, resi, icode))
    except Exception:
        return 0
    return len(residues)


def target_assay_family(row: pd.Series) -> str:
    text = " ".join(
        [
            clean(row.get("gene_names")),
            clean(row.get("protein_names")),
            clean(row.get("target_classes")),
            clean(row.get("target_feasibility_reason")),
        ]
    ).lower()
    gene = clean(row.get("gene_names"))
    if re.search(r"\b(gpcr|g protein-coupled receptor|7tm)\b", text):
        return "unexpected_gpcr"
    if re.search(r"\b(BCL2|BCL2L1|BCL2L2|MCL1|BCL2A1|BIRC)\b", gene, flags=re.I):
        return "other_assayable"
    if "kinase" in text or re.search(
        r"\b(ABL|AKT|ALK|BTK|CDK|EGFR|ERBB|FGFR|FLT|JAK|KIT|LCK|LYN|MAPK|MET|MTOR|NTRK|RET|ROS1|SRC|SYK|TYK|YES)\b",
        gene,
        flags=re.I,
    ):
        return "kinase"
    if "ion channel" in text:
        return "ion_channel"
    if "transporter" in text:
        return "transporter"
    if any(term in text for term in ["nuclear receptor", "transcription factor", "epigenetic", "chromatin"]):
        return "nuclear_epigenetic"
    if "enzyme" in text:
        return "enzyme"
    return "other_assayable"


def original_target_family(row: pd.Series) -> str:
    text = " ".join(
        [
            clean(row.get("fda_target_names")),
            clean(row.get("fda_moa")),
            clean(row.get("fda_action_type")),
        ]
    ).lower()
    if re.search(
        r"kinase|cyclin-dependent|bcr/abl|\b(cdk|abl|egfr|erbb|fgfr|jak|btk|alk|ret|met|trk|ntrk|flt|kit|src)\b",
        text,
    ):
        return "kinase"
    if re.search(
        r"ion channel|sodium channel|calcium channel|potassium channel|chloride channel|channel blocker",
        text,
    ):
        return "ion_channel"
    if re.search(
        r"transporter|reuptake|pump|slc|p-glycoprotein|sodium-dependent|vesicular monoamine",
        text,
    ):
        return "transporter"
    if re.search(
        r"nuclear receptor|estrogen receptor|androgen receptor|progesterone receptor|glucocorticoid receptor|"
        r"mineralocorticoid receptor|bile acid receptor|farnesoid x receptor|thyroid hormone receptor|"
        r"retinoic acid receptor|retinoid x receptor|peroxisome proliferator|ppar|liver x receptor|"
        r"pregnane x receptor|vitamin d receptor|transcription|histone|hdac|methyltransferase|epigenetic",
        text,
    ):
        return "nuclear_epigenetic"
    if re.search(
        r"enzyme|cyclooxygenase|factor x|thrombin|protease|polymerase|topoisomerase|reductase|"
        r"synthase|hydrolase|transferase|phosphatase|dehydrogenase|aromatase|monoamine oxidase|"
        r"acetylcholinesterase|carbonic anhydrase",
        text,
    ):
        return "enzyme"
    if re.search(r"receptor|agonist|antagonist|modulator", text):
        return "receptor_other"
    return "unknown"


def read_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGET_STRUCTURE, low_memory=False).fillna("")
    targets = targets.drop_duplicates("sequence_key", keep="first").copy()
    targets["receptor_residue_count"] = targets["pdb_path"].map(pdb_residue_count)
    targets["target_assay_family"] = targets.apply(target_assay_family, axis=1)
    return targets


def add_target_rank(pairs: pd.DataFrame, all_pairs: pd.DataFrame) -> pd.DataFrame:
    ranked = all_pairs[["drug_chembl_id", "sequence_key", "conplex_score"]].copy()
    ranked["conplex_score"] = pd.to_numeric(ranked["conplex_score"], errors="coerce").fillna(0.0)
    ranked["target_rank"] = ranked.groupby("sequence_key")["conplex_score"].rank(
        method="first",
        ascending=False,
    )
    return pairs.merge(
        ranked[["drug_chembl_id", "sequence_key", "target_rank"]],
        on=["drug_chembl_id", "sequence_key"],
        how="left",
    )


def load_boltz_top50() -> pd.DataFrame:
    if not BOLTZ_TOP50.exists():
        return pd.DataFrame(columns=["drug_chembl_id", "sequence_key"])
    boltz = pd.read_csv(BOLTZ_TOP50, low_memory=False).fillna("")

    def parse_pair_id(pair_id: Any) -> tuple[str, str]:
        text = clean(pair_id)
        drug_match = re.search(r"(CHEMBL[0-9A-Z]+)", text)
        seq_match = re.search(r"(SEQ\d+)", text)
        return (
            drug_match.group(1) if drug_match else "",
            seq_match.group(1) if seq_match else "",
        )

    parsed = boltz["pairId"].map(parse_pair_id)
    boltz["drug_chembl_id"] = [item[0] for item in parsed]
    boltz["sequence_key"] = [item[1] for item in parsed]
    boltz = boltz.rename(
        columns={
            "validationSet": "boltz_validation_set",
            "boltzCompleted": "boltz_completed",
            "boltzSupportTier": "boltz_support_tier",
            "boltzCompositeScore": "boltz_composite_score",
            "boltzConfidenceScore": "boltz_confidence_score",
            "boltzLigandIptm": "boltz_ligand_iptm",
            "boltzComplexIplddt": "boltz_complex_iplddt",
            "boltzAffinityProbabilityBinary": "boltz_affinity_probability",
            "boltzAffinityPredValue": "boltz_affinity_pred_value",
        }
    )
    keep = [
        "drug_chembl_id",
        "sequence_key",
        "boltz_validation_set",
        "boltz_completed",
        "boltz_support_tier",
        "boltz_composite_score",
        "boltz_confidence_score",
        "boltz_ligand_iptm",
        "boltz_complex_iplddt",
        "boltz_affinity_probability",
        "boltz_affinity_pred_value",
    ]
    return boltz[[col for col in keep if col in boltz.columns]].drop_duplicates(
        ["drug_chembl_id", "sequence_key"],
        keep="first",
    )


def score_dataframe(pairs: pd.DataFrame, is_control: bool = False) -> pd.DataFrame:
    df = pairs.copy()
    for col, default in NUMERIC_DEFAULTS.items():
        df[col] = as_numeric(df, col, default)

    conplex = df["conplex_score"].clip(lower=0.0)
    rank = df["rank_within_drug"]
    target_rank = df["target_rank"]
    pct = conplex.rank(pct=True, method="average")

    abs_score = pd.Series(0.0, index=df.index)
    abs_score += (conplex >= 0.05).astype(float) * 1.5
    abs_score += (conplex >= 0.10).astype(float) * 2.0
    abs_score += (conplex >= 0.20).astype(float) * 2.5
    abs_score += (conplex >= 0.30).astype(float) * 2.0
    abs_score += (conplex >= 0.50).astype(float) * 1.0

    drug_rank_score = pd.Series(0.0, index=df.index)
    drug_rank_score += (rank <= 300).astype(float) * 1.0
    drug_rank_score += (rank <= 100).astype(float) * 2.0
    drug_rank_score += (rank <= 50).astype(float) * 2.0
    drug_rank_score += (rank <= 20).astype(float) * 2.0
    drug_rank_score += (rank <= 10).astype(float) * 1.0

    target_rank_score = pd.Series(0.0, index=df.index)
    target_rank_score += (target_rank <= 200).astype(float) * 1.0
    target_rank_score += (target_rank <= 100).astype(float) * 1.0
    target_rank_score += (target_rank <= 50).astype(float) * 1.5
    target_rank_score += (target_rank <= 20).astype(float) * 1.5
    target_rank_score += (target_rank <= 10).astype(float) * 1.0

    df["conplex_global_percentile"] = pct
    df["target_recall_evidence_score"] = (
        abs_score + drug_rank_score + target_rank_score + 4.0 * pct
    ).clip(upper=25.0)

    strict = df["strict_structure_tier"].fillna("").astype(str)
    consensus = df["structure_consensus_tier"].fillna("").astype(str)
    p2rank = df["p2rank_pocketability_tier"].fillna("").astype(str)
    puresnet = df["puresnet_tier"].fillna("").astype(str)

    structure = pd.Series(0.0, index=df.index)
    structure += strict.str.startswith("A_").astype(float) * 14.0
    structure += strict.str.startswith("B_").astype(float) * 12.0
    structure += strict.str.startswith("C_").astype(float) * 6.0
    structure += consensus.str.startswith("A_").astype(float) * 4.0
    structure += consensus.str.startswith("B_").astype(float) * 3.0
    structure += consensus.str.startswith("C_").astype(float) * 2.0
    structure += p2rank.str.startswith("A_").astype(float) * 3.0
    structure += p2rank.str.startswith("B_").astype(float) * 2.0
    structure += p2rank.str.startswith("C_").astype(float) * 1.0
    structure += puresnet.str.startswith("A_").astype(float) * 3.0
    structure += puresnet.str.startswith("B_").astype(float) * 2.0
    structure += puresnet.str.startswith("C_").astype(float) * 1.0
    structure += df["p2rank_puresnet_overlap_fraction"].clip(0, 1) * 3.0
    structure += (df["top_pocket_probability"].clip(0, 0.9) / 0.9) * 2.0
    residue_count = df["p2rank_top_residue_count"]
    structure += ((residue_count >= 8) & (residue_count <= 80)).astype(float) * 1.0
    df["structure_pocket_evidence_score"] = structure.clip(upper=30.0)

    family = df["target_assay_family"].fillna("other_assayable").astype(str)
    assay = pd.Series(0.0, index=df.index)
    assay += family.map(
        {
            "kinase": 6.0,
            "enzyme": 5.5,
            "ion_channel": 5.0,
            "transporter": 5.0,
            "nuclear_epigenetic": 5.0,
            "other_assayable": 3.0,
            "unexpected_gpcr": 0.0,
        }
    ).fillna(3.0)
    modality = df.get("druggable_modalities", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    assay += modality.str.contains("small molecule", regex=False).astype(float) * 3.0
    receptor_len = df["receptor_residue_count"]
    assay += (receptor_len.between(80, 900)).astype(float) * 4.0
    assay += ((receptor_len > 900) & (receptor_len <= 1200)).astype(float) * 2.0
    assay += (df["p2rank_top_residue_count"] >= 8).astype(float) * 2.0
    df["experimental_feasibility_score"] = assay.clip(upper=15.0)

    mw = df["fda_mw"]
    qed = df["fda_qed"]
    ro5 = df["fda_ro5_violations"]
    route = df.get("fda_route", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
    drug_score = pd.Series(0.0, index=df.index)
    drug_score += mw.between(120, 650).astype(float) * 2.5
    drug_score += mw.between(650.0001, 900).astype(float) * 1.0
    drug_score += (ro5 <= 1).astype(float) * 2.0
    drug_score += (qed >= 0.20).astype(float) * 1.5
    drug_score += (qed >= 0.35).astype(float) * 1.0
    drug_score += df.get("canonical_smiles", pd.Series([""] * len(df), index=df.index)).astype(str).str.len().gt(0).astype(float)
    systemic = route.str.contains("oral|intravenous|subcutaneous|intramuscular|inhalation|infusion|injection", regex=True)
    drug_score += systemic.astype(float) * 2.0
    df["drug_feasibility_score"] = drug_score.clip(upper=10.0)

    gene = df.get("gene_names", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    target_names = df.get("fda_target_names", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    moa = df.get("fda_moa", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    overlap = []
    for gene_text, target_text, moa_text in zip(gene, target_names, moa):
        haystack = f"{target_text} {moa_text}".lower()
        gene_tokens = [token.strip() for token in re.split(r"[,;/\s]+", gene_text) if token.strip()]
        overlap.append(any(len(token) >= 3 and re.search(rf"\b{re.escape(token.lower())}\b", haystack) for token in gene_tokens))
    df["fda_label_target_text_overlap"] = overlap
    df["fda_original_target_family"] = df.apply(original_target_family, axis=1)
    df["same_family_or_label_risk"] = (
        pd.Series(overlap, index=df.index)
        | (
            df["fda_original_target_family"].isin(
                ["kinase", "enzyme", "ion_channel", "transporter", "nuclear_epigenetic"]
            )
            & df["fda_original_target_family"].eq(family)
        )
    )
    exact_known = df.get("is_known_fda_target_pair", pd.Series([False] * len(df), index=df.index)).astype(str).str.lower().isin(
        {"true", "1", "1.0", "yes"}
    )
    novelty = pd.Series(0.0, index=df.index)
    novelty += (~exact_known).astype(float) * 3.0
    novelty += (~pd.Series(overlap, index=df.index)).astype(float) * 2.0
    novelty += (~df["same_family_or_label_risk"]).astype(float) * 2.5
    novelty += (family != "kinase").astype(float) * 1.0
    novelty += (df["drug_pair_count_in_106k"] <= 150).astype(float) * 1.0
    novelty += (df["target_pair_count_in_106k"] <= 300).astype(float) * 0.5
    if is_control:
        novelty = novelty.where(~exact_known, 0.0)
    df["novelty_leakage_control_score"] = novelty.clip(upper=10.0)

    diversity = pd.Series(0.0, index=df.index)
    diversity += family.map(
        {
            "kinase": 3.0,
            "enzyme": 4.0,
            "ion_channel": 5.0,
            "transporter": 5.0,
            "nuclear_epigenetic": 5.0,
            "other_assayable": 4.0,
            "unexpected_gpcr": 0.0,
        }
    ).fillna(4.0)
    diversity += (df["drug_pair_count_in_106k"] <= 150).astype(float) * 2.0
    diversity += (df["target_pair_count_in_106k"] <= 300).astype(float) * 2.0
    diversity += (df["scaffold_pair_count_in_106k"] <= 800).astype(float) * 1.0
    df["diversity_reserve_score"] = diversity.clip(upper=10.0)

    df["physics_first_pass_score"] = (
        df["target_recall_evidence_score"]
        + df["structure_pocket_evidence_score"]
        + df["experimental_feasibility_score"]
        + df["drug_feasibility_score"]
        + df["novelty_leakage_control_score"]
        + df["diversity_reserve_score"]
    ).clip(upper=100.0)

    df["conplex_bin"] = pd.cut(
        df["conplex_score"],
        bins=[-0.001, 0.05, 0.10, 0.20, 0.30, 1.1],
        labels=["very_low_lt005", "low_005_010", "mid_010_020", "good_020_030", "high_ge030"],
    ).astype(str)
    df["structure_bin"] = "D_no_primary_structure_support"
    df.loc[tier_starts(strict, ("A",)), "structure_bin"] = "A_strict_overlapping_pocket"
    df.loc[tier_starts(strict, ("B",)), "structure_bin"] = "B_strict_supported_overlap"
    df.loc[tier_starts(strict, ("C",)), "structure_bin"] = "C_manual_review_structure"

    df["core_physics_eligible"] = (
        df["has_alphafold_pdb"].astype(str).str.lower().isin({"true", "1", "1.0"})
        & df["structure_bin"].isin(
            [
                "A_strict_overlapping_pocket",
                "B_strict_supported_overlap",
                "C_manual_review_structure",
            ]
        )
        & (
            (df["conplex_score"] >= 0.05)
            | (df["rank_within_drug"] <= 100)
            | (df["target_rank"] <= 100)
        )
        & (family != "unexpected_gpcr")
        & df.get("canonical_smiles", pd.Series([""] * len(df), index=df.index)).astype(str).str.len().gt(0)
    )

    tier = pd.Series("D_background_or_deprioritized", index=df.index)
    tier.loc[
        (df["physics_first_pass_score"] >= 75)
        & df["structure_bin"].isin(["A_strict_overlapping_pocket", "B_strict_supported_overlap"])
        & ((df["conplex_score"] >= 0.20) | (df["rank_within_drug"] <= 50) | (df["target_rank"] <= 50))
    ] = "A_high_physics_priority"
    tier.loc[
        (tier == "D_background_or_deprioritized")
        & (df["physics_first_pass_score"] >= 65)
        & df["structure_bin"].isin(
            [
                "A_strict_overlapping_pocket",
                "B_strict_supported_overlap",
                "C_manual_review_structure",
            ]
        )
        & ((df["conplex_score"] >= 0.10) | (df["rank_within_drug"] <= 100) | (df["target_rank"] <= 100))
    ] = "B_good_physics_review"
    tier.loc[
        (tier == "D_background_or_deprioritized")
        & (df["physics_first_pass_score"] >= 55)
        & df["core_physics_eligible"]
    ] = "C_diversity_or_rescue_review"
    df["physics_priority_tier"] = tier
    df["boltz_stage2_priority"] = df["physics_priority_tier"].isin(
        ["A_high_physics_priority", "B_good_physics_review"]
    )
    df["discovery_queue_class"] = "background_or_deprioritized"
    df.loc[exact_known, "discovery_queue_class"] = "known_control_calibration"
    df.loc[
        (~exact_known) & df["same_family_or_label_risk"],
        "discovery_queue_class",
    ] = "positive_control_or_family_extension"
    df.loc[
        (~exact_known)
        & (~df["same_family_or_label_risk"])
        & df["physics_priority_tier"].isin(["A_high_physics_priority", "B_good_physics_review"])
        & ((df["conplex_score"] >= 0.20) | (df["rank_within_drug"] <= 50) | (df["target_rank"] <= 50)),
        "discovery_queue_class",
    ] = "novel_high_physics"
    df.loc[
        (~exact_known)
        & (~df["same_family_or_label_risk"])
        & df["core_physics_eligible"]
        & (df["discovery_queue_class"] == "background_or_deprioritized"),
        "discovery_queue_class",
    ] = "novel_review_or_rescue"
    df["selection_score"] = (
        df["physics_first_pass_score"]
        - df["same_family_or_label_risk"].astype(float) * 8.0
        - (family == "kinase").astype(float) * 1.5
        + (df["discovery_queue_class"].eq("novel_review_or_rescue")).astype(float) * 1.5
    )
    return df


def add_pool_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["drug_pair_count_in_106k"] = out.groupby("drug_chembl_id")["sequence_key"].transform("size")
    out["target_pair_count_in_106k"] = out.groupby("sequence_key")["drug_chembl_id"].transform("size")
    scaffold = out.get("murcko_scaffold", pd.Series([""] * len(out), index=out.index)).replace("", "NO_SCAFFOLD")
    out["scaffold_pair_count_in_106k"] = scaffold.map(scaffold.value_counts()).fillna(0).astype(int)
    return out


def greedy_select(
    df: pd.DataFrame,
    n: int,
    caps: dict[str, int],
    family_minimums: dict[str, int] | None = None,
    score_column: str = "selection_score",
) -> pd.DataFrame:
    selected: list[int] = []
    counts: dict[str, Counter[str]] = {
        "drug": Counter(),
        "target": Counter(),
        "scaffold": Counter(),
        "family": Counter(),
        "queue_class": Counter(),
    }
    if score_column not in df.columns:
        score_column = "physics_first_pass_score"
    rows = list(df.sort_values([score_column, "physics_first_pass_score"], ascending=False).itertuples())

    def row_ok(row: Any, active_caps: dict[str, int]) -> bool:
        drug = clean(getattr(row, "drug_chembl_id"))
        target = clean(getattr(row, "sequence_key"))
        scaffold = clean(getattr(row, "murcko_scaffold")) or "NO_SCAFFOLD"
        family = clean(getattr(row, "target_assay_family")) or "other_assayable"
        queue_class = clean(getattr(row, "discovery_queue_class")) or "background_or_deprioritized"
        return (
            counts["drug"][drug] < active_caps.get("drug", 10_000)
            and counts["target"][target] < active_caps.get("target", 10_000)
            and counts["scaffold"][scaffold] < active_caps.get("scaffold", 10_000)
            and counts["family"][family] < active_caps.get(f"family:{family}", active_caps.get("family", 10_000))
            and counts["queue_class"][queue_class] < active_caps.get(
                f"queue_class:{queue_class}",
                active_caps.get("queue_class", 10_000),
            )
        )

    def add_row(row: Any) -> None:
        selected.append(row.Index)
        counts["drug"][clean(getattr(row, "drug_chembl_id"))] += 1
        counts["target"][clean(getattr(row, "sequence_key"))] += 1
        counts["scaffold"][clean(getattr(row, "murcko_scaffold")) or "NO_SCAFFOLD"] += 1
        counts["family"][clean(getattr(row, "target_assay_family")) or "other_assayable"] += 1
        counts["queue_class"][clean(getattr(row, "discovery_queue_class")) or "background_or_deprioritized"] += 1

    if family_minimums:
        for family, minimum in family_minimums.items():
            for row in rows:
                if len(selected) >= n or counts["family"][family] >= minimum:
                    break
                if row.Index in selected:
                    continue
                if clean(getattr(row, "target_assay_family")) != family:
                    continue
                if row_ok(row, caps):
                    add_row(row)

    for row in rows:
        if len(selected) >= n:
            break
        if row.Index in selected:
            continue
        if row_ok(row, caps):
            add_row(row)

    if len(selected) < n:
        relaxed = dict(caps)
        relaxed["drug"] = max(caps.get("drug", 0) * 2, caps.get("drug", 0) + 3)
        relaxed["target"] = max(caps.get("target", 0) * 2, caps.get("target", 0) + 5)
        relaxed["scaffold"] = max(caps.get("scaffold", 0) * 2, caps.get("scaffold", 0) + 10)
        relaxed["family"] = max(caps.get("family", 0) * 2, caps.get("family", 0) + 100)
        for row in rows:
            if len(selected) >= n:
                break
            if row.Index in selected:
                continue
            if row_ok(row, relaxed):
                add_row(row)

    if len(selected) < n:
        for row in rows:
            if len(selected) >= n:
                break
            if row.Index not in selected:
                add_row(row)

    selected_df = df.loc[selected].copy()
    selected_df = selected_df.sort_values([score_column, "physics_first_pass_score"], ascending=False)
    selected_df["selection_rank"] = range(1, len(selected_df) + 1)
    return selected_df


def make_pair_id(df: pd.DataFrame) -> pd.Series:
    gene = df.get("gene_names", pd.Series(["TARGET"] * len(df), index=df.index)).fillna("TARGET").astype(str)
    gene_token = gene.str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_").replace("", "TARGET")
    return df["drug_chembl_id"].astype(str) + "_" + gene_token + "_" + df["sequence_key"].astype(str)


def build_teacher_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "selection_rank": "推荐顺序",
        "wetlab_priority_band": "湿实验优先批次",
        "physics_priority_tier": "证据分层",
        "discovery_queue_class": "队列类型",
        "selection_score": "平衡后选择分",
        "drug_names": "药物名称",
        "drug_chembl_id": "药物ChEMBL",
        "gene_names": "候选靶点基因",
        "protein_names": "候选靶点蛋白",
        "sequence_key": "唯一蛋白序列ID",
        "target_assay_family": "靶点实验类型",
        "target_classes": "靶点类别",
        "conplex_score": "ConPLEx分数",
        "rank_within_drug": "药物内ConPLEx排名",
        "target_rank": "靶点内ConPLEx排名",
        "structure_bin": "结构口袋分层",
        "p2rank_pocketability_tier": "P2Rank口袋等级",
        "puresnet_tier": "PUResNet口袋等级",
        "p2rank_puresnet_overlap_fraction": "双模型口袋重叠比例",
        "physics_first_pass_score": "物理优先总分",
        "target_recall_evidence_score": "DTI/召回证据分",
        "structure_pocket_evidence_score": "结构口袋证据分",
        "experimental_feasibility_score": "实验可行性分",
        "drug_feasibility_score": "药物可行性分",
        "novelty_leakage_control_score": "新靶点/泄露控制分",
        "diversity_reserve_score": "多样性保留分",
        "fda_therapeutic_area": "原FDA治疗领域",
        "fda_indication": "原FDA适应症",
        "fda_action_type": "FDA动作类型",
        "fda_moa": "FDA原MoA",
        "fda_target_names": "FDA原记录靶点",
        "fda_original_target_family": "原标签靶点家族",
        "same_family_or_label_risk": "同家族或标签重叠风险",
        "fda_label_target_text_overlap": "是否疑似原标签靶点重叠",
        "canonical_smiles": "SMILES",
        "murcko_scaffold": "Murcko骨架",
        "boltz_stage2_priority": "是否优先进入Boltz二阶段",
        "boltz_support_tier": "已跑Boltz支持等级",
        "boltz_affinity_probability": "已跑Boltz结合概率",
        "go_no_go_note": "第一轮建议",
    }
    out = df.copy()
    out["go_no_go_note"] = out.apply(make_go_no_go_note, axis=1)
    keep = [col for col in columns if col in out.columns]
    out = out[keep].rename(columns={col: columns[col] for col in keep})
    for col in ["ConPLEx分数", "双模型口袋重叠比例", "物理优先总分", "平衡后选择分", "DTI/召回证据分", "结构口袋证据分", "实验可行性分", "药物可行性分", "新靶点/泄露控制分", "多样性保留分", "已跑Boltz结合概率"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: safe_round(x, 3))
    return out


def make_go_no_go_note(row: pd.Series) -> str:
    tier = clean(row.get("physics_priority_tier"))
    family = clean(row.get("target_assay_family"))
    structure = clean(row.get("structure_bin"))
    conplex = float(row.get("conplex_score", 0) or 0)
    if tier.startswith("A_"):
        return f"优先湿实验；{family}；{structure}；ConPLEx={conplex:.3f}。"
    if tier.startswith("B_"):
        return f"可进入第一轮；建议先做target engagement/功能readout；ConPLEx={conplex:.3f}。"
    return f"保留为多样性或救援候选；需先做Boltz/对接或人工审计；ConPLEx={conplex:.3f}。"


def summarize_counts(pool: pd.DataFrame, known: pd.DataFrame, pre_boltz: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    def vc(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}

    filters = {
        "baseline_known_controls": pd.Series([True] * len(known), index=known.index),
        "known_core_physics_eligible": known["core_physics_eligible"],
        "known_strict_A_or_B": known["structure_bin"].isin(["A_strict_overlapping_pocket", "B_strict_supported_overlap"]),
        "known_conplex_ge_0.20_or_drug_top50": (known["conplex_score"] >= 0.20) | (known["rank_within_drug"] <= 50),
        "known_A_or_B_physics_tier": known["physics_priority_tier"].isin(["A_high_physics_priority", "B_good_physics_review"]),
    }
    known_recall = []
    total_known = max(len(known), 1)
    for name, mask in filters.items():
        n = int(mask.sum())
        known_recall.append(
            {
                "filter": name,
                "known_retained": n,
                "known_total": int(len(known)),
                "known_recall": n / total_known,
            }
        )

    summary = {
        "discovery_input_rows": int(len(pool)),
        "discovery_unique_drugs": int(pool["drug_chembl_id"].nunique()),
        "discovery_unique_targets": int(pool["sequence_key"].nunique()),
        "known_control_rows": int(len(known)),
        "pool_core_physics_eligible_rows": int(pool["core_physics_eligible"].sum()),
        "pool_tier_counts": vc(pool["physics_priority_tier"]),
        "pool_structure_bin_counts": vc(pool["structure_bin"]),
        "pool_conplex_bin_counts": vc(pool["conplex_bin"]),
        "pool_target_assay_family_counts": vc(pool["target_assay_family"]),
        "pre_boltz_rows": int(len(pre_boltz)),
        "pre_boltz_tier_counts": vc(pre_boltz["physics_priority_tier"]),
        "pre_boltz_family_counts": vc(pre_boltz["target_assay_family"]),
        "final_rows": int(len(final)),
        "final_unique_drugs": int(final["drug_chembl_id"].nunique()),
        "final_unique_targets": int(final["sequence_key"].nunique()),
        "final_unique_scaffolds": int(final["murcko_scaffold"].replace("", "NO_SCAFFOLD").nunique()),
        "final_tier_counts": vc(final["physics_priority_tier"]),
        "final_wetlab_priority_band_counts": vc(final["wetlab_priority_band"]),
        "final_queue_class_counts": vc(final["discovery_queue_class"]),
        "final_same_family_or_label_risk_rows": int(final["same_family_or_label_risk"].sum()),
        "final_structure_bin_counts": vc(final["structure_bin"]),
        "final_conplex_bin_counts": vc(final["conplex_bin"]),
        "final_target_assay_family_counts": vc(final["target_assay_family"]),
        "final_score_summary": {
            "min": float(final["physics_first_pass_score"].min()),
            "p25": float(final["physics_first_pass_score"].quantile(0.25)),
            "median": float(final["physics_first_pass_score"].median()),
            "p75": float(final["physics_first_pass_score"].quantile(0.75)),
            "max": float(final["physics_first_pass_score"].max()),
        },
        "known_recall_by_filter": known_recall,
        "rdkit_available": RDKIT_AVAILABLE,
    }
    return summary


def boltz_comparison(pool: pd.DataFrame, known: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    boltz = load_boltz_top50()
    if boltz.empty:
        return boltz, {"available": False}
    scored = pd.concat([pool.assign(scored_set="discovery"), known.assign(scored_set="known_control")], ignore_index=True)
    merged = boltz.merge(
        scored,
        on=["drug_chembl_id", "sequence_key"],
        how="left",
        suffixes=("", "_scored"),
    )
    for col in ["boltz_affinity_probability", "boltz_composite_score", "conplex_score", "physics_first_pass_score"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    disc = merged[merged["boltz_validation_set"].astype(str).str.contains("discovery", case=False, na=False)].copy()
    known_b = merged[merged["boltz_validation_set"].astype(str).str.contains("known", case=False, na=False)].copy()
    summary = {
        "available": True,
        "rows": int(len(merged)),
        "completed_rows": int(merged.get("boltz_completed", pd.Series(dtype=object)).astype(str).str.lower().isin({"true", "1", "1.0"}).sum()),
        "known_rows": int(len(known_b)),
        "discovery_rows": int(len(disc)),
        "known_affinity_probability_median": float(known_b["boltz_affinity_probability"].median()) if "boltz_affinity_probability" in known_b else None,
        "discovery_affinity_probability_median": float(disc["boltz_affinity_probability"].median()) if "boltz_affinity_probability" in disc else None,
        "conplex_boltz_spearman": float(
            merged[["conplex_score", "boltz_affinity_probability"]].corr(method="spearman").iloc[0, 1]
        )
        if {"conplex_score", "boltz_affinity_probability"}.issubset(merged.columns)
        and merged[["conplex_score", "boltz_affinity_probability"]].dropna().shape[0] >= 5
        else None,
    }
    return merged, summary


def write_report(summary: dict[str, Any], boltz_summary: dict[str, Any]) -> None:
    known_recall_rows = "\n".join(
        f"| {item['filter']} | {item['known_retained']} / {item['known_total']} | {item['known_recall']:.2%} |"
        for item in summary["known_recall_by_filter"]
    )

    def dict_rows(data: dict[str, int]) -> str:
        return "\n".join(f"| {key} | {value} |" for key, value in data.items())

    text = f"""# 106k 到 1000 候选的物理优先筛选漏斗 v1

## 输入与边界

- 起点：`direct_action_discovery_pairs.csv`，{summary['discovery_input_rows']:,} 条发现候选 pair。
- 发现池覆盖：{summary['discovery_unique_drugs']:,} 个 FDA 药物，{summary['discovery_unique_targets']:,} 个唯一蛋白序列靶点。
- 阳性校准：`direct_action_known_target_controls.csv`，{summary['known_control_rows']:,} 条已知 FDA direct-action 药物-靶点 pair，单独评估，不进入发现候选。
- 结构证据：AlphaFold receptor + P2Rank pocket + PUResNet pocket + 双模型重叠。
- DTI 证据：ConPLEx 绝对分数、药物内排名、靶点内排名、全局百分位。
- Boltz-2：已有 Top50 known/control 与 Top50 discovery 校准结果接入表格；全 1000 的 Boltz 属于下一步二阶段 GPU 任务。

## 筛选原则

1. 不把已知 FDA 靶点当作发现结果；已知 pair 只做流程校准。
2. 不再用 Top300 本身作为推荐理由；Top300 只是历史设计空间的一部分，当前推荐必须同时有结构口袋、ConPLEx/排名和实验可行性。
3. ConPLEx 不做唯一硬阈值；低分会被降权，高分也必须通过结构口袋和多样性控制。
4. 最终 1000 是第一轮 wet-lab discovery package，不是最终疗效结论。推荐先做 target engagement、酶活/受体功能/通路 readout 与 counterscreen。

## 漏斗数量

- 全量发现池：{summary['discovery_input_rows']:,}
- core physics eligible：{summary['pool_core_physics_eligible_rows']:,}
- Boltz 二阶段优先队列：{summary['pre_boltz_rows']:,}
- 最终第一轮候选：{summary['final_rows']:,}
- 最终覆盖：{summary['final_unique_drugs']:,} 个药物，{summary['final_unique_targets']:,} 个靶点，{summary['final_unique_scaffolds']:,} 个 Murcko/active-moiety 骨架。

## 最终 1000 质量分层

| 分层 | 数量 |
| --- | ---: |
{dict_rows(summary['final_tier_counts'])}

## 最终 1000 湿实验优先批次

| 批次 | 数量 |
| --- | ---: |
{dict_rows(summary['final_wetlab_priority_band_counts'])}

## 最终 1000 队列类型

| 队列类型 | 数量 |
| --- | ---: |
{dict_rows(summary['final_queue_class_counts'])}

同家族或标签重叠风险候选：{summary['final_same_family_or_label_risk_rows']:,} / {summary['final_rows']:,}。这部分主要作为同家族扩展、流程阳性和实验体系校准，不作为主发现叙事。

## 最终 1000 靶点实验类型分布

| 类型 | 数量 |
| --- | ---: |
{dict_rows(summary['final_target_assay_family_counts'])}

## 最终 1000 结构分布

| 结构分层 | 数量 |
| --- | ---: |
{dict_rows(summary['final_structure_bin_counts'])}

## 最终 1000 ConPLEx 分数分布

| ConPLEx 分箱 | 数量 |
| --- | ---: |
{dict_rows(summary['final_conplex_bin_counts'])}

## 已知 direct-action 阳性校准

| 过滤条件 | 已知命中保留 | 召回比例 |
| --- | ---: | ---: |
{known_recall_rows}

解释：这个召回不是“凭空发现能力”，而是阳性校准。它回答的是：当我们用同一套物理优先规则筛选时，已知 FDA direct-action 互作会被保留多少。发现候选仍然需要二阶段结构/文献/实验验证。

## Boltz-2 校准状态

- 是否已有校准结果：{boltz_summary.get('available')}
- 校准行数：{boltz_summary.get('rows')}
- known/control 行数：{boltz_summary.get('known_rows')}
- discovery 行数：{boltz_summary.get('discovery_rows')}
- known/control Boltz affinity probability 中位数：{boltz_summary.get('known_affinity_probability_median')}
- discovery Boltz affinity probability 中位数：{boltz_summary.get('discovery_affinity_probability_median')}
- ConPLEx 与 Boltz affinity probability 的 Spearman 相关：{boltz_summary.get('conplex_boltz_spearman')}

当前结论：Boltz-2 可以作为二阶段物理校验，但不能替代第一阶段漏斗。全量 10w pair 直接跑 Boltz 成本过高，且已有 Top50 校准显示它与 ConPLEx 不是简单同一个信号，适合作为互补证据层。

## 输出文件

- `candidate_pool_106k_scored.csv`：106,561 条全量发现候选打分表。
- `known_control_calibration_scored.csv`：242 条已知阳性校准表。
- `pre_boltz_shortlist_3000.csv`：建议优先跑 Boltz-2 的 3000 条二阶段队列。
- `final_1000_candidates.csv`：最终 1000 条机器可读候选表。
- `final_1000_teacher_readable_zh.csv`：最终 1000 条中文简洁汇报表。
- `final_first_wave_top300.csv` / `final_first_wave_top300_teacher_readable_zh.csv`：第一波 300 条候选。
- `boltz_top300_input_package/`：Top300 Boltz-2 结构+affinity 输入包。
- `boltz_top1000_input_package/`：Top1000 Boltz-2 结构+affinity 输入包。
- `conplex_vs_boltz_top50_comparison.csv`：已有 Boltz Top50 校准与 ConPLEx 对照。
- `funnel_summary.json`：全部计数与分布。
"""
    (OUTDIR / "FINAL_1000_FUNNEL_REPORT_ZH.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    discovery = pd.read_csv(DISCOVERY_PAIRS, low_memory=False).fillna("")
    all_pairs = pd.read_csv(ALL_DIRECT_PAIRS, low_memory=False).fillna("")
    known = pd.read_csv(KNOWN_CONTROLS, low_memory=False).fillna("")
    targets = read_targets()
    drugs = read_fda_drugs()
    boltz = load_boltz_top50()

    def prepare(pairs: pd.DataFrame) -> pd.DataFrame:
        df = add_target_rank(pairs, all_pairs)
        target_cols = [
            "sequence_key",
            "representative_protein_id",
            "gene_names",
            "protein_names",
            "target_classes",
            "druggable_modalities",
            "max_clinical_phase",
            "has_alphafold_pdb",
            "pdb_path",
            "p2rank_pocketability_tier",
            "top_pocket_rank",
            "top_pocket_score",
            "top_pocket_probability",
            "top_pocket_sas_points",
            "top_pocket_center_x",
            "top_pocket_center_y",
            "top_pocket_center_z",
            "top_pocket_residue_ids",
            "p2rank26_top_pocket_volume",
            "p2rank26_top_pocket_num_residues",
            "puresnet_tier",
            "structure_consensus_tier",
            "strict_structure_tier",
            "predicted_atoms",
            "clusters",
            "max_cluster_atoms",
            "max_probability",
            "mean_probability",
            "p2rank_top_residue_count",
            "puresnet_best_cluster_residue_count",
            "p2rank_puresnet_overlap_fraction",
            "p2rank_puresnet_jaccard",
            "receptor_residue_count",
            "target_assay_family",
        ]
        target_cols = [col for col in target_cols if col in targets.columns]
        before = set(df.columns)
        df = df.merge(targets[target_cols], on="sequence_key", how="left", suffixes=("", "_target"))
        for col in ["gene_names", "protein_names", "target_classes"]:
            target_col = f"{col}_target"
            if col in before and target_col in df.columns:
                df[col] = df[col].where(df[col].astype(str).str.len() > 0, df[target_col])
                df = df.drop(columns=[target_col])
        df = df.merge(drugs, on="drug_chembl_id", how="left")
        if not boltz.empty:
            df = df.merge(boltz, on=["drug_chembl_id", "sequence_key"], how="left")
        return df

    pool = prepare(discovery)
    known_scored_input = prepare(known)

    pool = add_pool_counts(pool)
    known_scored_input = known_scored_input.merge(
        pool[["drug_chembl_id", "sequence_key", "drug_pair_count_in_106k", "target_pair_count_in_106k", "scaffold_pair_count_in_106k"]],
        on=["drug_chembl_id", "sequence_key"],
        how="left",
    )
    for col in ["drug_pair_count_in_106k", "target_pair_count_in_106k", "scaffold_pair_count_in_106k"]:
        known_scored_input[col] = pd.to_numeric(known_scored_input[col], errors="coerce").fillna(0)

    pool = score_dataframe(pool, is_control=False)
    known_scored = score_dataframe(known_scored_input, is_control=True)
    pool["pair_id"] = make_pair_id(pool)
    known_scored["pair_id"] = make_pair_id(known_scored)

    core = pool[pool["core_physics_eligible"]].copy()
    tiered_core = core[core["physics_priority_tier"].isin(
        ["A_high_physics_priority", "B_good_physics_review", "C_diversity_or_rescue_review"]
    )].copy()
    pre_boltz = greedy_select(
        tiered_core,
        n=min(3000, len(tiered_core)),
        caps={
            "drug": 12,
            "target": 40,
            "scaffold": 70,
            "family": 1400,
            "queue_class:positive_control_or_family_extension": 600,
        },
        family_minimums={
            "kinase": 250,
            "enzyme": 700,
            "ion_channel": 120,
            "transporter": 250,
            "nuclear_epigenetic": 120,
            "other_assayable": 120,
        },
    )
    final = greedy_select(
        tiered_core,
        n=min(1000, len(tiered_core)),
        caps={
            "drug": 6,
            "target": 18,
            "scaffold": 30,
            "family": 520,
            "queue_class:positive_control_or_family_extension": 120,
        },
        family_minimums={
            "kinase": 90,
            "enzyme": 300,
            "ion_channel": 60,
            "transporter": 140,
            "nuclear_epigenetic": 60,
            "other_assayable": 60,
        },
    )
    final["wetlab_priority_band"] = "P3_rank701_1000_reserve_review"
    final.loc[final["selection_rank"] <= 700, "wetlab_priority_band"] = "P2_rank301_700_balanced_physics"
    final.loc[final["selection_rank"] <= 300, "wetlab_priority_band"] = "P1_rank1_300_first_wave"
    final["final_package"] = True
    pre_boltz["boltz_package"] = True

    boltz_cmp, boltz_summary = boltz_comparison(pool, known_scored)
    summary = summarize_counts(pool, known_scored, pre_boltz, final)
    summary["boltz_summary"] = boltz_summary

    sort_cols = ["physics_first_pass_score", "conplex_score"]
    pool.sort_values(sort_cols, ascending=False).to_csv(OUTDIR / "candidate_pool_106k_scored.csv", index=False)
    known_scored.sort_values(sort_cols, ascending=False).to_csv(OUTDIR / "known_control_calibration_scored.csv", index=False)
    pre_boltz.sort_values("selection_rank").to_csv(OUTDIR / "pre_boltz_shortlist_3000.csv", index=False)
    final.sort_values("selection_rank").to_csv(OUTDIR / "final_1000_candidates.csv", index=False)
    final.loc[final["selection_rank"] <= 300].sort_values("selection_rank").to_csv(
        OUTDIR / "final_first_wave_top300.csv",
        index=False,
    )
    final.loc[final["selection_rank"] <= 700].sort_values("selection_rank").to_csv(
        OUTDIR / "final_top700_balanced_physics.csv",
        index=False,
    )
    build_teacher_table(final.sort_values("selection_rank")).to_csv(
        OUTDIR / "final_1000_teacher_readable_zh.csv",
        index=False,
    )
    build_teacher_table(final.loc[final["selection_rank"] <= 300].sort_values("selection_rank")).to_csv(
        OUTDIR / "final_first_wave_top300_teacher_readable_zh.csv",
        index=False,
    )
    if not boltz_cmp.empty:
        boltz_cmp.to_csv(OUTDIR / "conplex_vs_boltz_top50_comparison.csv", index=False)

    pd.DataFrame(summary["known_recall_by_filter"]).to_csv(OUTDIR / "known_control_recall_by_filter.csv", index=False)
    with (OUTDIR / "funnel_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_report(summary, boltz_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
