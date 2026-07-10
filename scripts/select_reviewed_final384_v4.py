#!/usr/bin/env python3
"""Select the active final384 after manual literature and feasibility review."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.production import (  # noqa: E402
    assert_unique_pairs,
    bool_series,
    file_sha256,
    select_reviewed_final384,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def merge_review_evidence(pool: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """Join reviewed evidence without dropping plain query-status fields."""

    review_columns = [
        column
        for column in reviewed.columns
        if column.startswith("agent_")
        or column.startswith("ot_full_")
        or column.startswith("chembl_exact_")
        or column.startswith("chembl_activity_")
        or column.startswith("pair_pubmed_")
        or column.startswith("post_approval_pair_")
        or column.startswith("representative_pair_")
        or column.startswith("representative_post_")
        or column
        in {
            "literature_class",
            "lit_ok",
            "pubmed_query_schema",
            "pubmed_query_sha256",
            "chembl_hierarchy_query_ok",
            "chembl_molecule_ids_queried",
            "post_approval_pubmed_query_error",
        }
    ]
    overlapping = [column for column in review_columns if column in pool.columns]
    base = pool.drop(columns=overlapping, errors="ignore")
    evidence = reviewed[["pair_id", *review_columns]].copy()
    return base.merge(evidence, on="pair_id", how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/current_pipeline_v4.yaml"))
    parser.add_argument(
        "--review-pool",
        default=str(
            ROOT
            / "outputs/current_production_package_v2/formal_full_universe_v4"
            / "agent_review_pool_v4_complete.csv"
        ),
    )
    parser.add_argument("--reviewed", required=True)
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "outputs/current_production_package_v2/formal_full_universe_v4"
            / "final384_reviewed_selected_v4_complete.csv"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    pool_path = Path(args.review_pool).resolve()
    reviewed_path = Path(args.reviewed).resolve()
    output_path = Path(args.output).resolve()
    for path in [config_path, pool_path, reviewed_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pool = pd.read_csv(pool_path, low_memory=False).fillna("")
    reviewed = pd.read_csv(reviewed_path, low_memory=False).fillna("")
    assert_unique_pairs(pool, "formal agent review pool")
    if reviewed["pair_id"].duplicated().any():
        raise ValueError("Reviewed table contains duplicate pair_id values")
    expected = set(pool["pair_id"].astype(str))
    observed = set(reviewed["pair_id"].astype(str))
    if expected != observed:
        raise ValueError(
            "Review coverage differs from the formal pool: "
            f"missing={sorted(expected - observed)[:10]}, "
            f"unexpected={sorted(observed - expected)[:10]}"
        )

    merged = merge_review_evidence(pool, reviewed)

    final_config = config["selection"]["final384"]
    control_cap = int(config["selection"].get("post_review", {}).get("validated_control_cap", 96))
    selected = select_reviewed_final384(
        merged,
        final_config,
        validated_control_cap=control_cap,
    )
    assert_unique_pairs(selected, "post-review final384")
    for column in [
        "exact_known_target_v2",
        "family_or_rediscovery_risk_v2",
        "severe_compound_liability",
        "ion_channel_feasibility_flag",
        "structure_sequence_mismatch_v4",
    ]:
        if bool_series(selected, column).any():
            raise RuntimeError(f"Post-review final384 contains forbidden rows: {column}")
    if not selected["pose_stability_tier"].isin(
        ["A_stable_conditional_pose", "B_moderate_conditional_pose"]
    ).all():
        raise RuntimeError("Post-review final384 contains an unsupported conditional pose")

    selected_ids = set(selected["pair_id"].astype(str))
    audit = merged.copy()
    audit["selected_final384_v4"] = audit["pair_id"].astype(str).isin(selected_ids)
    audit["post_review_disposition_v4"] = "not_selected_rank_or_diversity_cap"
    audit.loc[
        audit["agent_feasibility_grade"].astype(str).eq("D"),
        "post_review_disposition_v4",
    ] = "excluded_agent_grade_D"
    audit.loc[
        audit["agent_literature_class"].astype(str).eq("contradictory"),
        "post_review_disposition_v4",
    ] = "excluded_contradictory_evidence"
    query_failed = (
        ~bool_series(audit, "chembl_activity_query_ok")
        | ~bool_series(audit, "lit_ok")
    ) & audit["agent_database_query_resolution"].astype(str).ne("resolved_manually")
    audit.loc[query_failed, "post_review_disposition_v4"] = "excluded_unresolved_database_query"
    active_species_rerun = audit["agent_active_species_status"].astype(str).eq(
        "prodrug_active_metabolite_requires_rerun"
    )
    audit.loc[active_species_rerun, "post_review_disposition_v4"] = (
        "excluded_active_metabolite_rerun_required"
    )
    audit.loc[audit["selected_final384_v4"], "post_review_disposition_v4"] = "selected_final384"
    audit = audit.sort_values(
        ["selected_final384_v4", "review_pool_rank"],
        ascending=[False, True],
        kind="mergesort",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    audit_path = output_path.with_name("agent_review_pool_post_review_disposition_v4_complete.csv")
    audit.to_csv(audit_path, index=False)
    exact = selected["review_candidate_class_v4"].eq("validated_control_or_rediscovery")
    summary = {
        "created_utc": now_utc(),
        "review_pool_rows": int(len(pool)),
        "reviewed_rows": int(len(reviewed)),
        "selected_rows": int(len(selected)),
        "selected_unique_drugs": int(selected["drug_chembl_id"].nunique()),
        "selected_unique_targets": int(selected["primary_gene"].nunique()),
        "selected_agent_grades": selected["agent_feasibility_grade"].value_counts().to_dict(),
        "selected_validated_control_or_rediscovery_rows": int(exact.sum()),
        "validated_control_cap": control_cap,
        "excluded_grade_D_rows": int(
            merged["agent_feasibility_grade"].astype(str).eq("D").sum()
        ),
        "excluded_contradictory_rows": int(
            merged["agent_literature_class"].astype(str).eq("contradictory").sum()
        ),
        "excluded_unresolved_database_query_rows": int(query_failed.sum()),
        "excluded_active_metabolite_rerun_rows": int(active_species_rerun.sum()),
        "source_sha256": {
            "config": file_sha256(config_path),
            "review_pool": file_sha256(pool_path),
            "reviewed": file_sha256(reviewed_path),
        },
        "output_sha256": {
            "selected_final384": file_sha256(output_path),
            "review_disposition": file_sha256(audit_path),
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
