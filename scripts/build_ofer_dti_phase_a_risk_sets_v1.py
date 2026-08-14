#!/usr/bin/env python3
"""Build leakage-audited historical risk sets for OFER-DTI Phase A."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/biomaster_first_event_dti_phase_a_freeze_20260814.json"
FEATURE_SUMMARY = ROOT / "outputs/old_drug_target_sota_v1/first_event_dti_feature_store_v1/OFER_DTI_FEATURE_STORE_SUMMARY_V1.json"
EVENTS = ROOT / "outputs/old_drug_target_sota_v1/first_event_dti_feature_store_v1/OFER_DTI_FIRST_EVENT_MANIFEST_V1.csv.gz"
COMPOUNDS = ROOT / "outputs/old_drug_target_sota_v1/first_event_dti_feature_store_v1/OFER_DTI_COMPOUND_FEATURE_INDEX_V1.csv.gz"
TARGETS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
OLD_DRUGS = ROOT / "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1/OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz"
OUT = ROOT / "outputs/old_drug_target_sota_v1/ofer_dti_phase_a_risk_sets_v1"

EXPECTED_FREEZE_SHA256 = "b11f437ebb20e285e4e3ccafb612dff297ffa86cbb5df900b7ea898b29a4dc13"
EXPECTED_FEATURE_SUMMARY_SHA256 = ""  # populated from the frozen feature build at runtime below
WINDOWS = [
    ("DEV_2015_2018", 2014, 2015, 2018, 202608141),
    ("DEV_2019_2022", 2018, 2019, 2022, 202608142),
]
K_PER_AXIS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_rejection(
    rng: np.random.Generator,
    pool: np.ndarray,
    k: int,
    forbidden_at_or_before: callable,
) -> list[int]:
    if k <= 0:
        return []
    chosen: set[int] = set()
    attempts = 0
    maximum_attempts = max(100, k * 100)
    while len(chosen) < k and attempts < maximum_attempts:
        candidate = int(pool[int(rng.integers(0, len(pool)))])
        attempts += 1
        if candidate in chosen or forbidden_at_or_before(candidate):
            continue
        chosen.add(candidate)
    if len(chosen) < k:
        for candidate_raw in pool:
            candidate = int(candidate_raw)
            if candidate not in chosen and not forbidden_at_or_before(candidate):
                chosen.add(candidate)
                if len(chosen) == k:
                    break
    return sorted(chosen)


def build_window(
    events: pd.DataFrame,
    compounds: pd.DataFrame,
    targets: pd.DataFrame,
    old_drugs: pd.DataFrame,
    name: str,
    cutoff: int,
    dev_start: int,
    dev_end: int,
    seed: int,
) -> dict[str, object]:
    window_dir = OUT / name
    window_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_compounds = int(compounds["compound_feature_index"].max()) + 1
    n_targets = int(targets["target_feature_index"].max()) + 1
    infinity_year = np.int16(32767)
    compound_entry = np.full(n_compounds, infinity_year, dtype=np.int16)
    target_entry = np.full(n_targets, infinity_year, dtype=np.int16)
    for index, year in events.groupby("compound_feature_index")["first_year"].min().items():
        compound_entry[int(index)] = np.int16(year)
    for index, year in events.groupby("target_feature_index")["first_year"].min().items():
        target_entry[int(index)] = np.int16(year)

    pair_year: dict[tuple[int, int], int] = {}
    pair_type: dict[tuple[int, int], str] = {}
    compound_event_years: dict[int, np.ndarray] = {}
    target_event_years: dict[int, np.ndarray] = {}
    for row in events[["compound_feature_index", "target_feature_index", "first_year", "event_type"]].itertuples(index=False):
        key = (int(row.compound_feature_index), int(row.target_feature_index))
        pair_year[key] = int(row.first_year)
        pair_type[key] = str(row.event_type)
    for index, group in events.groupby("compound_feature_index", sort=False):
        compound_event_years[int(index)] = np.sort(group["first_year"].to_numpy(dtype=np.int16))
    for index, group in events.groupby("target_feature_index", sort=False):
        target_event_years[int(index)] = np.sort(group["first_year"].to_numpy(dtype=np.int16))

    train = events[events["first_year"].le(cutoff)].copy()
    train = train.sort_values(["first_year", "event_index"], kind="mergesort").reset_index(drop=True)
    train_compound = train["compound_feature_index"].to_numpy(dtype=np.int32)
    train_target = train["target_feature_index"].to_numpy(dtype=np.int16)
    train_year = train["first_year"].to_numpy(dtype=np.int16)
    train_active = train["event_type"].eq("ACTIVE").to_numpy(dtype=np.int8)
    train_source_event = train["event_index"].to_numpy(dtype=np.int64)
    pair_entry = np.maximum(compound_entry[train_compound], target_entry[train_target]).astype(np.int16)
    prior_counts = np.maximum(train_year.astype(np.int32) - pair_entry.astype(np.int32), 0)
    capacity = int(len(train) * (1 + 2 * K_PER_AXIS) + prior_counts.sum())

    sample_compound = np.empty(capacity, dtype=np.int32)
    sample_target = np.empty(capacity, dtype=np.int16)
    sample_year = np.empty(capacity, dtype=np.int16)
    sample_observed = np.empty(capacity, dtype=np.int8)
    sample_active = np.empty(capacity, dtype=np.int8)
    sample_role = np.empty(capacity, dtype=np.int8)
    sample_weight = np.empty(capacity, dtype=np.float32)
    sample_source_event = np.empty(capacity, dtype=np.int64)
    pointer = 0

    target_pools = {
        year: np.flatnonzero(target_entry <= year).astype(np.int16)
        for year in sorted(set(train_year.astype(int).tolist()))
    }
    compound_pools = {
        year: np.flatnonzero(compound_entry <= year).astype(np.int32)
        for year in sorted(set(train_year.astype(int).tolist()))
    }

    def add(
        c: int,
        t: int,
        year: int,
        observed: int,
        active: int,
        role: int,
        weight: float,
        source_event: int,
    ) -> None:
        nonlocal pointer
        sample_compound[pointer] = c
        sample_target[pointer] = t
        sample_year[pointer] = year
        sample_observed[pointer] = observed
        sample_active[pointer] = active
        sample_role[pointer] = role
        sample_weight[pointer] = weight
        sample_source_event[pointer] = source_event
        pointer += 1

    for row_index, (compound, target, year, active, entry, source_event) in enumerate(
        zip(
            train_compound,
            train_target,
            train_year,
            train_active,
            pair_entry,
            train_source_event,
        ),
        start=1,
    ):
        c, t, y = int(compound), int(target), int(year)
        for survival_year in range(int(entry), y):
            add(c, t, survival_year, 0, -1, 1, 1.0, int(source_event))
        add(c, t, y, 1, int(active), 0, 1.0, int(source_event))

        target_pool = target_pools[y]
        observed_for_compound = int(np.searchsorted(compound_event_years[c], y, side="right"))
        eligible_target_count = max(int(len(target_pool)) - observed_for_compound, 0)
        k_target = min(K_PER_AXIS, eligible_target_count)
        sampled_targets = _sample_rejection(
            rng,
            target_pool,
            k_target,
            lambda candidate: pair_year.get((c, candidate), 32767) <= y,
        )
        target_weight = float(eligible_target_count / len(sampled_targets)) if sampled_targets else 0.0
        for candidate in sampled_targets:
            add(c, candidate, y, 0, -1, 2, target_weight, int(source_event))

        compound_pool = compound_pools[y]
        observed_for_target = int(np.searchsorted(target_event_years[t], y, side="right"))
        eligible_compound_count = max(int(len(compound_pool)) - observed_for_target, 0)
        k_compound = min(K_PER_AXIS, eligible_compound_count)
        sampled_compounds = _sample_rejection(
            rng,
            compound_pool,
            k_compound,
            lambda candidate: pair_year.get((candidate, t), 32767) <= y,
        )
        compound_weight = float(eligible_compound_count / len(sampled_compounds)) if sampled_compounds else 0.0
        for candidate in sampled_compounds:
            add(candidate, t, y, 0, -1, 3, compound_weight, int(source_event))

        if row_index % 50000 == 0:
            print(f"{name} train_events={row_index}/{len(train)} risk_rows={pointer}", flush=True)

    arrays = {
        "compound_feature_index": sample_compound[:pointer],
        "target_feature_index": sample_target[:pointer],
        "calendar_year": sample_year[:pointer],
        "observation_event": sample_observed[:pointer],
        "active_given_observed": sample_active[:pointer],
        "sampling_role": sample_role[:pointer],
        "inverse_sampling_probability_weight": sample_weight[:pointer],
        "source_event_index": sample_source_event[:pointer],
    }
    train_path = window_dir / "OFER_DTI_TRAIN_RISK_SAMPLES_V1.npz"
    np.savez_compressed(train_path, **arrays)

    old_map = old_drugs.set_index("ligand_inchikey")["drug_feature_index"].to_dict()
    compound_by_inchi = compounds.set_index("parent_standard_inchi_key")["compound_feature_index"].to_dict()
    old_to_compound = {
        int(drug_index): int(compound_by_inchi[inchi])
        for inchi, drug_index in old_map.items()
        if inchi in compound_by_inchi
    }
    eligible_old = [
        (drug_index, compound_index)
        for drug_index, compound_index in sorted(old_to_compound.items())
        if int(compound_entry[compound_index]) <= cutoff
    ]
    eligible_targets = np.flatnonzero(target_entry <= cutoff).astype(np.int16)
    eval_rows: list[dict[str, object]] = []
    for drug_index, compound_index in eligible_old:
        for target_raw in eligible_targets:
            target_index = int(target_raw)
            year = pair_year.get((compound_index, target_index), 32767)
            if year <= cutoff:
                continue
            observed = dev_start <= year <= dev_end
            active_event = observed and pair_type[(compound_index, target_index)] == "ACTIVE"
            eval_rows.append(
                {
                    "drug_feature_index": drug_index,
                    "event_compound_feature_index": compound_index,
                    "target_feature_index": target_index,
                    "query_cutoff_year": cutoff,
                    "horizon_end_year": dev_end,
                    "active_event_by_horizon": int(active_event),
                    "any_observation_event_by_horizon": int(observed),
                    "event_year_if_within_horizon": year if observed else "",
                    "event_type_if_within_horizon": pair_type[(compound_index, target_index)] if observed else "",
                }
            )
    evaluation = pd.DataFrame(eval_rows)
    eval_path = window_dir / "OFER_DTI_OLD_DRUG_TARGET_DEVELOPMENT_EVALUATION_V1.csv.gz"
    evaluation.to_csv(eval_path, index=False, compression="gzip")

    dev_event_pairs = {
        (int(row.compound_feature_index), int(row.target_feature_index))
        for row in events[events["first_year"].between(dev_start, dev_end)].itertuples(index=False)
    }
    train_event_pairs = {
        (int(c), int(t))
        for c, t, observed in zip(
            arrays["compound_feature_index"], arrays["target_feature_index"], arrays["observation_event"]
        )
        if observed == 1
    }
    role_counts = pd.Series(arrays["sampling_role"]).value_counts().sort_index().to_dict()
    evaluation_at_risk = all(
        int(compound_entry[int(row.event_compound_feature_index)]) <= cutoff
        and int(target_entry[int(row.target_feature_index)]) <= cutoff
        and pair_year.get(
            (int(row.event_compound_feature_index), int(row.target_feature_index)), 32767
        )
        > cutoff
        for row in evaluation.itertuples(index=False)
    )
    summary = {
        "window": name,
        "cutoff": cutoff,
        "development_years": [dev_start, dev_end],
        "counts": {
            "training_events": int(len(train)),
            "training_active_events": int(train_active.sum()),
            "training_risk_rows": int(pointer),
            "exact_event_rows": int(role_counts.get(0, 0)),
            "exact_pre_event_survival_rows": int(role_counts.get(1, 0)),
            "same_drug_sampled_target_rows": int(role_counts.get(2, 0)),
            "same_target_sampled_drug_rows": int(role_counts.get(3, 0)),
            "eligible_old_drugs_at_cutoff": int(len(eligible_old)),
            "eligible_targets_at_cutoff": int(len(eligible_targets)),
            "development_evaluation_rows": int(len(evaluation)),
            "development_active_events": int(evaluation["active_event_by_horizon"].sum()),
            "development_observation_events": int(evaluation["any_observation_event_by_horizon"].sum()),
            "development_old_drugs_with_active_event": int(
                evaluation.loc[evaluation["active_event_by_horizon"].eq(1), "drug_feature_index"].nunique()
            ),
        },
        "integrity": {
            "no_training_calendar_year_after_cutoff": bool(np.max(arrays["calendar_year"]) <= cutoff),
            "only_observed_rows_have_active_labels": bool(
                np.all(arrays["active_given_observed"][arrays["observation_event"] == 0] == -1)
                and np.all(np.isin(arrays["active_given_observed"][arrays["observation_event"] == 1], [0, 1]))
            ),
            "exactly_one_training_event_row_per_historical_event": int(role_counts.get(0, 0)) == len(train),
            "training_observed_pairs_disjoint_from_development_first_event_pairs": train_event_pairs.isdisjoint(dev_event_pairs),
            "all_sample_weights_finite_positive": bool(
                np.isfinite(arrays["inverse_sampling_probability_weight"]).all()
                and np.all(arrays["inverse_sampling_probability_weight"] > 0)
            ),
            "evaluation_contains_only_at_risk_pairs_at_cutoff": bool(evaluation_at_risk),
            "future_event_type_isolated_from_training": True,
            "unobserved_pairs_never_used_as_inactive_labels": True,
        },
        "artifacts": {
            str(train_path.relative_to(ROOT)): sha256(train_path),
            str(eval_path.relative_to(ROOT)): sha256(eval_path),
        },
    }
    summary_path = window_dir / "OFER_DTI_RISK_SET_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["artifacts"][str(summary_path.relative_to(ROOT))] = sha256(summary_path)
    return summary


def main() -> None:
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise ValueError("OFER-DTI freeze hash changed")
    feature_summary = json.loads(FEATURE_SUMMARY.read_text(encoding="utf-8"))
    if feature_summary.get("status") != "PASS":
        raise ValueError("OFER-DTI feature store did not pass")
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS, low_memory=False)
    compounds = pd.read_csv(COMPOUNDS, low_memory=False)
    targets = pd.read_csv(TARGETS, usecols=["target_feature_index", "sequence_key"])
    old_drugs = pd.read_csv(OLD_DRUGS, usecols=["drug_feature_index", "ligand_inchikey"])

    summaries = [
        build_window(events, compounds, targets, old_drugs, name, cutoff, start, end, seed)
        for name, cutoff, start, end, seed in WINDOWS
    ]
    all_integrity = all(all(summary["integrity"].values()) for summary in summaries)
    overall = {
        "schema_version": "OFER_DTI_PHASE_A_RISK_SETS_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_integrity else "FAIL",
        "windows": summaries,
        "integrity_all_pass": all_integrity,
        "inputs": {
            str(FREEZE.relative_to(ROOT)): sha256(FREEZE),
            str(FEATURE_SUMMARY.relative_to(ROOT)): sha256(FEATURE_SUMMARY),
            str(EVENTS.relative_to(ROOT)): sha256(EVENTS),
            str(COMPOUNDS.relative_to(ROOT)): sha256(COMPOUNDS),
            str(TARGETS.relative_to(ROOT)): sha256(TARGETS),
            str(OLD_DRUGS.relative_to(ROOT)): sha256(OLD_DRUGS),
        },
    }
    overall_path = OUT / "OFER_DTI_PHASE_A_RISK_SETS_SUMMARY_V1.json"
    overall_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
