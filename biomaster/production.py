"""Reusable contracts and selection helpers for the production screening funnel.

The original project grew through one-off analysis scripts.  This module keeps
the rules that affect formal candidate packages in one importable, testable
place.  Scores produced here are prioritisation scores, not calibrated binding
probabilities.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CANONICAL_ASSAY_FAMILIES = {
    "kinase",
    "enzyme",
    "ion_channel",
    "transporter",
    "nuclear_epigenetic",
    "other_assayable",
}

OT_ASSAY_FAMILY_MAP = {
    "kinase": "kinase",
    "enzyme": "enzyme",
    "ion_channel": "ion_channel",
    "transporter": "transporter",
    "nuclear_epigenetic_transcription": "nuclear_epigenetic",
}

TRUE_VALUES = {"1", "1.0", "true", "yes", "y"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def shared_reference_percentile(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Return empirical mid-rank percentiles on one shared reference set."""

    ref = pd.to_numeric(reference, errors="coerce").dropna().sort_values().to_numpy()
    numeric = pd.to_numeric(values, errors="coerce")
    if len(ref) == 0:
        return pd.Series(0.0, index=values.index, dtype=float)
    # Mid-ranks prevent a large tie block (notably ConPLEx zero scores) from
    # inheriting the percentile of the block's upper edge.
    import numpy as np

    query = numeric.fillna(-math.inf).to_numpy()
    left = np.searchsorted(ref, query, side="left")
    right = np.searchsorted(ref, query, side="right")
    result = (left + right + 1.0) / (2.0 * len(ref))
    return pd.Series(result, index=values.index, dtype=float)


def canonical_assay_family(df: pd.DataFrame) -> pd.Series:
    """Prefer curated Open Targets classification over the legacy regex label."""

    legacy = df.get("target_assay_family", pd.Series("other_assayable", index=df.index))
    legacy = legacy.fillna("").astype(str).str.strip()
    legacy = legacy.where(legacy.isin(CANONICAL_ASSAY_FAMILIES), "other_assayable")
    ot = df.get("anchor_project_assay_family", pd.Series("", index=df.index))
    mapped = ot.fillna("").astype(str).str.strip().map(OT_ASSAY_FAMILY_MAP)
    return mapped.where(mapped.notna(), legacy)


def specific_family_tokens(labels: Any) -> set[str]:
    """Extract conservative, named protein-family labels from OT target classes.

    Generic assay classes are intentionally excluded.  For example, two
    proteins being enzymes is not evidence that they are in the same family.
    """

    tokens: set[str] = set()
    for raw in clean(labels).split(";"):
        token = re.sub(r"\s+", " ", raw.strip().lower())
        if not token or ("family" not in token and "subfamily" not in token):
            continue
        if token in {"slc superfamily of solute carriers", "atp-binding cassette"}:
            continue
        if "small molecule receptor" in token or "peptide receptor" in token:
            continue
        tokens.add(token)
    return tokens


def build_known_target_context(
    known_controls: pd.DataFrame,
    anchor_table: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    labels_by_gene = (
        anchor_table.drop_duplicates("gene").set_index("gene").get("target_class_labels", pd.Series(dtype=str)).to_dict()
    )
    genes_by_drug: dict[str, set[str]] = {}
    families_by_drug: dict[str, set[str]] = {}
    assay_by_gene = anchor_table.drop_duplicates("gene").set_index("gene").get(
        "project_assay_family", pd.Series(dtype=str)
    ).to_dict()
    assays_by_drug: dict[str, set[str]] = {}
    grouping_column = (
        "knowledge_compound_key" if "knowledge_compound_key" in known_controls.columns else "drug_chembl_id"
    )
    for drug, group in known_controls.groupby(grouping_column):
        genes: set[str] = set()
        families: set[str] = set()
        assays: set[str] = set()
        for value in group.get("gene_names", pd.Series(dtype=str)):
            for gene in re.split(r"[;,|\s]+", clean(value)):
                if not gene:
                    continue
                genes.add(gene)
                families.update(specific_family_tokens(labels_by_gene.get(gene, "")))
                assay = OT_ASSAY_FAMILY_MAP.get(clean(assay_by_gene.get(gene, "")))
                if assay:
                    assays.add(assay)
        genes_by_drug[clean(drug)] = genes
        families_by_drug[clean(drug)] = families
        assays_by_drug[clean(drug)] = assays
    return genes_by_drug, families_by_drug, assays_by_drug


def annotate_candidate_risk(
    df: pd.DataFrame,
    known_controls: pd.DataFrame,
    anchor_table: pd.DataFrame,
) -> pd.DataFrame:
    """Split exact-known, family-extension, assay and feasibility risks.

    The legacy implementation treated every enzyme-to-enzyme or
    transporter-to-transporter hypothesis as a same-family pair.  This function
    keeps broad assay similarity as an annotation and uses conservative class
    labels for the actual family-extension flag.
    """

    out = df.copy()
    out["target_assay_family_legacy"] = out.get("target_assay_family", "")
    out["target_assay_family_v2"] = canonical_assay_family(out)
    out["assay_family_corrected"] = (
        out["target_assay_family_legacy"].fillna("").astype(str) != out["target_assay_family_v2"]
    )

    known_genes, known_families, known_assays = build_known_target_context(known_controls, anchor_table)
    exact_from_source = bool_series(out, "is_known_fda_target_pair")
    exact_from_map = pd.Series(False, index=out.index)
    specific_family = pd.Series(False, index=out.index)
    for idx, row in out.iterrows():
        drug = clean(row.get("active_moiety_smiles")) or clean(row.get("drug_chembl_id"))
        gene = clean(row.get("primary_gene")) or clean(row.get("gene_names")).split(";")[0]
        exact_from_map.loc[idx] = gene in known_genes.get(drug, set())
        candidate_families = specific_family_tokens(row.get("anchor_target_class_labels", ""))
        specific_family.loc[idx] = bool(candidate_families & known_families.get(drug, set()))

    out["exact_known_target_v2"] = exact_from_source | exact_from_map
    out["specific_target_family_extension_risk"] = specific_family
    original_family = out.get("fda_original_target_family", pd.Series("unknown", index=out.index)).astype(str)
    compound_keys = out.get("active_moiety_smiles", pd.Series("", index=out.index)).map(clean)
    compound_keys = compound_keys.where(
        compound_keys.ne(""), out.get("drug_chembl_id", pd.Series("", index=out.index)).map(clean)
    )
    out["known_target_assay_families_v2"] = compound_keys.map(
        lambda drug: ";".join(sorted(known_assays.get(clean(drug), set())))
    )
    out["same_known_assay_family_only"] = [
        family in known_assays.get(clean(drug), set())
        for drug, family in zip(compound_keys, out["target_assay_family_v2"])
    ]
    out["same_assay_family_only"] = original_family.eq(out["target_assay_family_v2"])
    label_text = (
        out.get("fda_target_names", pd.Series("", index=out.index)).astype(str)
        + " "
        + out.get("fda_moa", pd.Series("", index=out.index)).astype(str)
    )
    labelled_kinase = label_text.str.contains(
        r"kinase|\b(?:CDK|FGFR|EGFR|ERBB|VEGFR|PDGFR|JAK|BTK|ALK|RET|MET|NTRK|FLT|KIT|SRC)\d*\b",
        case=False,
        regex=True,
        na=False,
    )
    out["kinase_to_kinase_risk"] = out["target_assay_family_v2"].eq("kinase") & (
        original_family.eq("kinase") | out["known_target_assay_families_v2"].str.contains(r"\bkinase\b") | labelled_kinase
    )
    labels = out.get("anchor_target_class_labels", pd.Series("", index=out.index)).astype(str)
    out["nuclear_receptor_extension_risk"] = original_family.eq("nuclear_epigenetic") & labels.str.contains(
        r"\bNuclear receptor\b", case=False, regex=True, na=False
    )
    gene = out.get("primary_gene", out.get("gene_names", pd.Series("", index=out.index))).astype(str)
    out["carbonic_anhydrase_rediscovery_risk"] = gene.str.fullmatch(r"CA(?:[1-9]|1[0-4])", case=False, na=False)
    out["ion_channel_feasibility_flag"] = out["target_assay_family_v2"].eq("ion_channel")
    out["family_or_rediscovery_risk_v2"] = out[
        [
            "specific_target_family_extension_risk",
            "kinase_to_kinase_risk",
            "nuclear_receptor_extension_risk",
            "carbonic_anhydrase_rediscovery_risk",
        ]
    ].any(axis=1)

    reasons: list[str] = []
    for _, row in out.iterrows():
        row_reasons = []
        for column, label in [
            ("exact_known_target_v2", "exact_known_target"),
            ("specific_target_family_extension_risk", "specific_target_family_extension"),
            ("kinase_to_kinase_risk", "kinase_to_kinase_extension"),
            ("nuclear_receptor_extension_risk", "nuclear_receptor_extension"),
            ("carbonic_anhydrase_rediscovery_risk", "carbonic_anhydrase_rediscovery"),
            ("ion_channel_feasibility_flag", "ion_channel_assay_feasibility"),
        ]:
            if bool(row[column]):
                row_reasons.append(label)
        reasons.append(";".join(row_reasons))
    out["risk_notes_v2"] = reasons
    out["candidate_role_v2"] = "novel_discovery_hypothesis"
    out.loc[out["ion_channel_feasibility_flag"], "candidate_role_v2"] = "novel_feasibility_review"
    out.loc[out["family_or_rediscovery_risk_v2"], "candidate_role_v2"] = "family_extension_or_rediscovery_control"
    out.loc[out["exact_known_target_v2"], "candidate_role_v2"] = "known_positive_control"
    return out


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def add_priority_score_v2(df: pd.DataFrame, conplex_reference: pd.Series) -> pd.DataFrame:
    """Add transparent evidence components and a 0-100 prioritisation score."""

    out = df.copy()
    conplex = _numeric(out, "conplex_score")
    drug_rank = _numeric(out, "rank_within_drug", 9999)
    target_rank = _numeric(out, "target_rank", 9999)
    out["conplex_reference_percentile_v2"] = shared_reference_percentile(conplex, conplex_reference)

    target_count = _numeric(out, "drug_pair_count_in_project_space", 463).clip(lower=2)
    drug_count = _numeric(out, "target_pair_count_in_project_space", 750).clip(lower=2)
    drug_rank_percentile = (1.0 - (drug_rank - 1.0) / (target_count - 1.0)).clip(0, 1)
    target_rank_percentile = (1.0 - (target_rank - 1.0) / (drug_count - 1.0)).clip(0, 1)
    conplex_composite_percentile = (
        0.50 * out["conplex_reference_percentile_v2"]
        + 0.30 * drug_rank_percentile
        + 0.20 * target_rank_percentile
    )
    out["pair_conplex_component_v2"] = (25.0 * conplex_composite_percentile).clip(0, 25)

    affinity = _numeric(out, "boltz_affinity_probability_refined").clip(0, 1)
    pose_tier = out.get("pose_stability_tier", pd.Series("", index=out.index)).astype(str)
    pose_component = pose_tier.map(
        {"A_stable_conditional_pose": 6.0, "B_moderate_conditional_pose": 4.0}
    ).fillna(0.0)
    out["conditional_pose_stability_component_v3"] = pose_component
    # Complex generation is conditioned on a P2Rank pocket contact.  Interface
    # and structure confidence are therefore conditional pose-quality signals,
    # not independent evidence that the ligand chose that pocket.  Keep the
    # affinity head dominant and cap the conditional structural contribution.
    out["pair_boltz_component_v2"] = (24.0 * affinity + pose_component).clip(0, 30)

    structure_bin = out.get("structure_bin", pd.Series("", index=out.index)).astype(str)
    structure = structure_bin.map(
        {
            "A_strict_overlapping_pocket": 15.0,
            "B_strict_supported_overlap": 12.0,
            "C_manual_review_structure": 6.0,
        }
    ).fillna(0.0)
    out["target_pocket_prior_component_v2"] = structure.clip(0, 15)

    tractability = out.get("anchor_availability_tier", pd.Series("", index=out.index)).astype(str).map(
        {
            "A1_SM_approved_drug": 10.0,
            "A2_SM_clinical": 9.0,
            "B_high_quality_ligand_or_pocket": 8.0,
            "C_structure_or_medium_pocket": 6.0,
            "D_family_only_or_low_direct_evidence": 3.0,
            "E_no_OT_SM_tractability": 0.0,
        }
    ).fillna(0.0)
    out["target_tractability_component_v2"] = tractability.clip(0, 10)

    drug_feasibility = _numeric(out, "drug_feasibility_score").clip(0, 10)
    drug_feasibility -= bool_series(out, "assay_interference_review").astype(float) * 2.0
    drug_feasibility -= bool_series(out, "multi_product_label_review").astype(float) * 1.0
    drug_feasibility -= bool_series(out, "severe_compound_liability").astype(float) * 8.0
    out["drug_feasibility_component_v2"] = drug_feasibility.clip(0, 10)
    experiment = _numeric(out, "experimental_feasibility_score").clip(0, 15) * (5.0 / 15.0)
    experiment -= bool_series(out, "ion_channel_feasibility_flag").astype(float) * 1.0
    out["experimental_feasibility_component_v2"] = experiment.clip(0, 5)
    novelty = pd.Series(5.0, index=out.index)
    novelty -= bool_series(out, "family_or_rediscovery_risk_v2").astype(float) * 4.0
    novelty -= bool_series(out, "exact_known_target_v2").astype(float) * 5.0
    out["novelty_component_v2"] = novelty.clip(0, 5)

    components = [
        "pair_conplex_component_v2",
        "pair_boltz_component_v2",
        "target_pocket_prior_component_v2",
        "target_tractability_component_v2",
        "drug_feasibility_component_v2",
        "experimental_feasibility_component_v2",
        "novelty_component_v2",
    ]
    out["priority_score_v2"] = out[components].sum(axis=1).clip(0, 100)
    out["pair_specific_evidence_score_v2"] = out[
        ["pair_conplex_component_v2", "pair_boltz_component_v2"]
    ].sum(axis=1)
    return out


def add_evidence_tier_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = add_boltz_review_class_v3(df)
    boltz_tier = out.get("boltz_support_tier_refined", pd.Series("", index=out.index)).astype(str)
    conplex = _numeric(out, "conplex_score")
    drug_rank = _numeric(out, "rank_within_drug", 9999)
    target_rank = _numeric(out, "target_rank", 9999)
    structure_ok = out.get("structure_bin", pd.Series("", index=out.index)).isin(
        ["A_strict_overlapping_pocket", "B_strict_supported_overlap"]
    )
    a = boltz_tier.str.startswith("A_", na=False)
    completed = bool_series(out, "boltz_completed_refined")
    substantive = bool_series(out, "boltz_substantive_signal_v3")
    if "pose_stability_tier" in out.columns:
        pose_supported = bool_series(out, "pose_stability_completed") & out["pose_stability_tier"].isin(
            ["A_stable_conditional_pose", "B_moderate_conditional_pose"]
        )
    else:
        pose_supported = pd.Series(False, index=out.index)
    out["evidence_tier_v2"] = "E_low_or_uncompleted"
    out.loc[completed, "evidence_tier_v2"] = "D_completed_low_or_single_metric"
    out.loc[substantive & structure_ok, "evidence_tier_v2"] = "C_concordant_physics_review"
    out.loc[
        substantive
        & pose_supported
        & (conplex >= 0.20)
        & (drug_rank <= 50)
        & (target_rank <= 100)
        & structure_ok,
        "evidence_tier_v2",
    ] = "B_orthogonal_physics_supported"
    out.loc[
        a
        & pose_supported
        & (conplex >= 0.30)
        & (drug_rank <= 50)
        & (target_rank <= 50)
        & structure_ok,
        "evidence_tier_v2",
    ] = "A_high_physics_priority"
    return out


def add_boltz_review_class_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Classify Boltz evidence without treating an heuristic tier as truth.

    The A/B thresholds were defined for review convenience and were not
    calibrated as binding-probability cutoffs.  A candidate can therefore
    carry useful, concordant C-tier evidence when at least two continuous
    Boltz outputs agree.  The resulting class is an audit annotation and a
    nomination gate; the continuous measurements remain in the score.
    """

    out = df.copy()
    tier = out.get("boltz_support_tier_refined", pd.Series("", index=out.index)).astype(str)
    completed = bool_series(out, "boltz_completed_refined")
    affinity = _numeric(out, "boltz_affinity_probability_refined").clip(0, 1)
    ligand_iptm = _numeric(out, "boltz_ligand_iptm_refined").clip(0, 1)
    confidence = _numeric(out, "boltz_confidence_score_refined").clip(0, 1)
    complex_iplddt = _numeric(out, "boltz_complex_iplddt_refined").clip(0, 1)
    ab = tier.str.startswith(("A_", "B_"), na=False)
    concordant_c = tier.str.startswith("C_", na=False) & (affinity >= 0.50) & (
        (ligand_iptm >= 0.40)
        | (confidence >= 0.35)
        | (complex_iplddt >= 0.35)
    )
    out["boltz_review_class_v3"] = "low_or_incomplete"
    out.loc[completed, "boltz_review_class_v3"] = "completed_low_or_single_metric"
    out.loc[concordant_c, "boltz_review_class_v3"] = "C_concordant_multi_metric"
    out.loc[ab, "boltz_review_class_v3"] = "AB_heuristic_supported"
    out["boltz_substantive_signal_v3"] = completed & (ab | concordant_c)
    return out


def diverse_select(
    df: pd.DataFrame,
    n: int,
    *,
    score_column: str = "priority_score_v2",
    drug_cap: int = 4,
    target_cap: int = 8,
    scaffold_cap: int = 10,
    family_caps: dict[str, int] | None = None,
    category_column: str | None = None,
    category_caps: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Deterministically select high-ranked rows under diversity caps."""

    if n <= 0 or df.empty:
        return df.head(0).copy()
    family_caps = family_caps or {}
    category_caps = category_caps or {}
    order_cols = [score_column]
    ascending = [False]
    for tie_breaker in ["pair_specific_evidence_score_v2", "conplex_score", "pair_id"]:
        if tie_breaker in df.columns:
            order_cols.append(tie_breaker)
            ascending.append(tie_breaker == "pair_id")
    ordered = df.sort_values(order_cols, ascending=ascending, kind="mergesort")
    selected: list[int] = []
    selected_compound_target: set[tuple[str, str]] = set()
    counts = {
        "drug": Counter(),
        "target": Counter(),
        "scaffold": Counter(),
        "family": Counter(),
        "category": Counter(),
    }

    def keys(row: pd.Series) -> tuple[str, str, str, str]:
        compound = (
            clean(row.get("active_moiety_smiles"))
            or clean(row.get("canonical_smiles_rdkit"))
            or clean(row.get("drug_chembl_id"))
            or clean(row.get("drug_names"))
        )
        return (
            compound,
            clean(row.get("primary_gene")) or clean(row.get("sequence_key")),
            clean(row.get("murcko_scaffold")) or compound or "NO_SCAFFOLD",
            clean(row.get("target_assay_family_v2")) or "other_assayable",
        )

    def can_add(row: pd.Series) -> bool:
        drug, target, scaffold, family = keys(row)
        category = clean(row.get(category_column, "")) if category_column else ""
        return (
            (drug, target) not in selected_compound_target
            and counts["drug"][drug] < drug_cap
            and counts["target"][target] < target_cap
            and counts["scaffold"][scaffold] < scaffold_cap
            and counts["family"][family] < family_caps.get(family, n)
            and (
                not category_column
                or counts["category"][category] < category_caps.get(category, n)
            )
        )

    for idx, row in ordered.iterrows():
        if len(selected) >= n:
            break
        if not can_add(row):
            continue
        selected.append(idx)
        drug, target, scaffold, family = keys(row)
        counts["drug"][drug] += 1
        counts["target"][target] += 1
        counts["scaffold"][scaffold] += 1
        counts["family"][family] += 1
        if category_column:
            counts["category"][clean(row.get(category_column, ""))] += 1
        selected_compound_target.add((drug, target))
    result = df.loc[selected].copy()
    result = result.sort_values(order_cols, ascending=ascending, kind="mergesort")
    result["selection_rank_v2"] = range(1, len(result) + 1)
    return result


def nomination_eligible_rows(final1000: pd.DataFrame) -> pd.DataFrame:
    """Return rows eligible for direct wet-lab nomination before manual review."""

    completed = bool_series(final1000, "boltz_completed_refined")
    direct_sm = bool_series(final1000, "anchor_project_standard_direct_sm")
    structure_ok = final1000.get(
        "structure_bin", pd.Series("", index=final1000.index)
    ).isin(["A_strict_overlapping_pocket", "B_strict_supported_overlap"])
    eligible = final1000[
        completed
        & direct_sm
        & structure_ok
        & ~bool_series(final1000, "exact_known_target_v2")
        & ~bool_series(final1000, "family_or_rediscovery_risk_v2")
        & ~bool_series(final1000, "severe_compound_liability")
        & ~bool_series(final1000, "ion_channel_feasibility_flag")
        & ~bool_series(final1000, "structure_sequence_mismatch_v4")
    ].copy()
    if "pose_stability_tier" not in eligible.columns:
        return eligible.head(0).copy()
    return eligible[
        bool_series(eligible, "pose_stability_completed")
        & eligible["pose_stability_tier"].isin(
            ["A_stable_conditional_pose", "B_moderate_conditional_pose"]
        )
    ].copy()


def build_agent_review_pool(
    final1000: pd.DataFrame,
    review_pool_config: dict[str, Any],
) -> pd.DataFrame:
    """Select a larger pre-review pool so rejected nominations can be replaced."""

    eligible = nomination_eligible_rows(final1000)
    pool = diverse_select(
        eligible,
        int(review_pool_config["size"]),
        drug_cap=int(review_pool_config["drug_cap"]),
        target_cap=int(review_pool_config["target_cap"]),
        scaffold_cap=int(review_pool_config["scaffold_cap"]),
        family_caps={
            key: int(value) for key, value in review_pool_config["family_caps"].items()
        },
    )
    pool["review_pool_rank"] = pool["selection_rank_v2"]
    return pool


def select_reviewed_final384(
    reviewed_pool: pd.DataFrame,
    final_config: dict[str, Any],
    *,
    validated_control_cap: int = 96,
) -> pd.DataFrame:
    """Select the active 384 only after literature and feasibility review.

    Grade-D and contradictory rows are fail-closed. Exact-pair literature or
    ChEMBL hits remain useful rediscovery controls, but a hard cap prevents the
    active package from becoming a literature-confirmed benchmark set.
    """

    require_columns(
        reviewed_pool,
        [
            "pair_id",
            "agent_feasibility_grade",
            "agent_literature_class",
            "agent_confidence",
            "agent_database_query_resolution",
            "agent_active_species_status",
            "priority_score_v2",
        ],
        "reviewed pool",
    )
    data = reviewed_pool.copy()
    grade = data["agent_feasibility_grade"].astype(str).str.strip()
    literature = data["agent_literature_class"].astype(str).str.strip()
    confidence = data["agent_confidence"].astype(str).str.strip()
    query_resolution = data["agent_database_query_resolution"].astype(str).str.strip()
    active_species = data["agent_active_species_status"].astype(str).str.strip()
    invalid_grade = sorted(set(grade) - {"A", "B", "C", "D"})
    if invalid_grade:
        raise ValueError(f"Invalid agent grades in reviewed pool: {invalid_grade}")
    invalid_active_species = sorted(
        set(active_species)
        - {
            "parent_drug_relevant",
            "salt_normalization_adequate",
            "active_species_uncertain",
            "prodrug_active_metabolite_requires_rerun",
        }
    )
    if invalid_active_species:
        raise ValueError(f"Invalid active-species status in reviewed pool: {invalid_active_species}")

    exact_chembl = data.get(
        "chembl_exact_activity_status", pd.Series("", index=data.index)
    ).astype(str).eq("exact_binding_activity_pchembl_ge_5")
    exact_literature = literature.eq("exact_pair_validated")
    contradictory = literature.eq("contradictory")
    chembl_query_ok = bool_series(data, "chembl_activity_query_ok")
    pubmed_query_ok = bool_series(data, "lit_ok")
    automated_query_failed = ~chembl_query_ok | ~pubmed_query_ok
    unresolved_query = automated_query_failed & query_resolution.ne("resolved_manually")
    active_species_rerun = active_species.eq("prodrug_active_metabolite_requires_rerun")
    data["review_candidate_class_v4"] = "novel_hypothesis"
    data.loc[exact_chembl | exact_literature, "review_candidate_class_v4"] = (
        "validated_control_or_rediscovery"
    )
    data["review_exclusion_reason_v4"] = ""
    data.loc[grade.eq("D"), "review_exclusion_reason_v4"] = "agent_grade_D"
    data.loc[contradictory, "review_exclusion_reason_v4"] = (
        data.loc[contradictory, "review_exclusion_reason_v4"]
        .replace("", "contradictory_evidence")
        .replace("agent_grade_D", "agent_grade_D;contradictory_evidence")
    )
    data.loc[unresolved_query, "review_exclusion_reason_v4"] = "unresolved_database_query_failure"
    data.loc[active_species_rerun, "review_exclusion_reason_v4"] = (
        "prodrug_active_metabolite_requires_rerun"
    )
    eligible = data[
        grade.ne("D") & ~contradictory & ~unresolved_query & ~active_species_rerun
    ].copy()
    grade_bonus = grade.map({"A": 4.0, "B": 2.0, "C": 0.0, "D": -100.0})
    confidence_bonus = confidence.map({"high": 1.0, "medium": 0.5, "low": 0.0}).fillna(0.0)
    eligible["review_selection_score_v4"] = (
        _numeric(eligible, "priority_score_v2")
        + grade_bonus.loc[eligible.index].fillna(0.0)
        + confidence_bonus.loc[eligible.index]
    )
    selected = diverse_select(
        eligible,
        int(final_config["size"]),
        score_column="review_selection_score_v4",
        drug_cap=int(final_config["drug_cap"]),
        target_cap=int(final_config["target_cap"]),
        scaffold_cap=int(final_config["scaffold_cap"]),
        family_caps={key: int(value) for key, value in final_config["family_caps"].items()},
        category_column="review_candidate_class_v4",
        category_caps={"validated_control_or_rediscovery": int(validated_control_cap)},
    )
    expected = int(final_config["size"])
    if len(selected) != expected:
        raise RuntimeError(
            "Reviewed final package could not satisfy hard contracts: "
            f"selected={len(selected)}, expected={expected}, "
            f"eligible_after_review_fail_closed={len(eligible)}"
        )
    selected["final384_rank"] = selected["selection_rank_v2"]
    selected["final384_selection_stage"] = "post_agent_literature_feasibility_review"
    return selected


def select_formal_packages(
    top3000: pd.DataFrame,
    selection_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a nested final1000 -> final384 package from one scored Top3000."""

    data = top3000.copy()
    completed = bool_series(data, "boltz_completed_refined")
    data = add_boltz_review_class_v3(data)
    # Boltz is already represented by four continuous outputs in
    # pair_boltz_component_v2.  Do not add a discontinuous +100 tier bonus,
    # which would override every other evidence source at a threshold edge.
    data["formal_selection_score_v2"] = _numeric(data, "priority_score_v2")
    discovery_base = data[
        completed
        & ~bool_series(data, "exact_known_target_v2")
        & ~bool_series(data, "family_or_rediscovery_risk_v2")
        & ~bool_series(data, "severe_compound_liability")
        & ~bool_series(data, "structure_sequence_mismatch_v4")
    ].copy()
    f1000_cfg = selection_config["final1000"]
    final1000 = diverse_select(
        discovery_base,
        int(f1000_cfg["size"]),
        score_column="formal_selection_score_v2",
        drug_cap=int(f1000_cfg["drug_cap"]),
        target_cap=int(f1000_cfg["target_cap"]),
        scaffold_cap=int(f1000_cfg["scaffold_cap"]),
        family_caps={key: int(value) for key, value in f1000_cfg["family_caps"].items()},
    )
    final1000["final1000_rank"] = final1000["selection_rank_v2"]

    nomination_base = nomination_eligible_rows(final1000)
    f384_cfg = selection_config["final384"]
    final384 = diverse_select(
        nomination_base,
        int(f384_cfg["size"]),
        drug_cap=int(f384_cfg["drug_cap"]),
        target_cap=int(f384_cfg["target_cap"]),
        scaffold_cap=int(f384_cfg["scaffold_cap"]),
        family_caps={key: int(value) for key, value in f384_cfg["family_caps"].items()},
    )
    final384["final384_rank"] = final384["selection_rank_v2"]
    return final1000, final384


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_unique_pairs(df: pd.DataFrame, label: str) -> None:
    require_columns(df, ["drug_chembl_id", "sequence_key"], label)
    duplicates = int(df.duplicated(["drug_chembl_id", "sequence_key"]).sum())
    if duplicates:
        raise ValueError(f"{label} contains {duplicates} duplicate drug-target sequence pairs")
