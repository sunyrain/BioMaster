#!/usr/bin/env python3
"""Audit frozen prospective selections without changing selection or inventing outcomes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/retargetmap_spr512_ours_frozen_20260904'
OUT = ROOT / 'outputs/biomaster_training_preparation_20260908/candidate_audit'


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest(), size_bytes=path.stat().st_size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    master_path = next(SOURCE.glob('*INTERNAL_MASTER_V2.csv'))
    roster_path = next(SOURCE.glob('*TARGET_ROSTER_V2.csv'))
    summary_path = next(SOURCE.glob('*SUMMARY_V2.json'))
    master, roster = pd.read_csv(master_path), pd.read_csv(roster_path)
    candidates = master[master.lab_row_type.eq('BLINDED_CANDIDATE')].copy()
    controls = master[master.lab_row_type.eq('TARGET_POSITIVE_CONTROL')].copy()
    drugs = candidates.drop_duplicates('ligand_inchikey')
    core = candidates.experiment_arm.eq('FROZEN_384_CORE_DISCOVERY')
    high = candidates.selection_role.eq('OUR_FROZEN_MODEL_HIGH')
    intermediate = candidates.selection_role.eq('OUR_FROZEN_MODEL_INTERMEDIATE')
    low = candidates.selection_role.eq('OUR_FROZEN_MODEL_LOW_BACKGROUND')
    candidates['selection_rank'] = candidates.retargetmap_rank_384.where(core, candidates.routed_rank_within_drug)
    candidates['selection_denominator'] = core.map({True: 384, False: 367})
    candidates['evidence_tier'] = core.map({True: 'FROZEN_S5_MODEL_PROSPECTIVE_HYPOTHESIS', False: 'EXTENDED_ROUTE_PROSPECTIVE_HYPOTHESIS'})
    candidates['measured_outcome_available_in_frozen_package'] = False
    # This is a descriptive cross-route check, never a replacement selection.
    candidates['other_route_rank_descriptive_only'] = candidates.routed_rank_within_drug.where(core)
    degree = candidates.groupby('ligand_inchikey').size()
    ranks_ok = ((~high | (candidates.selection_rank <= core.map({True: 20, False: 30}))) &
                (~intermediate | candidates.selection_rank.between(80, 180)) &
                (~low | candidates.selection_rank.gt(250)))
    checks = dict(rows_512=len(master) == 512, candidates_480=len(candidates) == 480,
                  controls_32=len(controls) == 32, unique_pairs=not master.pair_id.duplicated().any(),
                  targets_32=candidates.target_chembl_id.nunique() == 32,
                  candidates_per_target_15=candidates.groupby('target_chembl_id').size().eq(15).all(),
                  per_target_roles_correct=candidates.groupby(['target_chembl_id', 'selection_role']).size().unstack().eq(
                      pd.Series({'OUR_FROZEN_MODEL_HIGH': 10, 'OUR_FROZEN_MODEL_INTERMEDIATE': 3,
                                 'OUR_FROZEN_MODEL_LOW_BACKGROUND': 2})).all().all(),
                  declared_rank_intervals=bool(ranks_ok.all()), candidate_drugs_120=len(drugs) == 120,
                  drug_degrees_2_to_6=degree.between(2, 6).all(),
                  no_duplicate_connectivity=not drugs.connectivity_key.duplicated().any())
    summary = json.loads(summary_path.read_text())
    # Recompute the exact intersection against the historical training input.
    train_path = ROOT / 'outputs/retrain_20260901/comprehensive_training_v1/COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz'
    train = pd.read_csv(train_path, usecols=['parent_standard_inchi_key', 'target_chembl_id']).rename(
        columns={'parent_standard_inchi_key': 'ligand_inchikey'})
    overlap = candidates.merge(train.drop_duplicates(), on=['ligand_inchikey', 'target_chembl_id'])
    checks['no_exact_candidate_in_historical_fullfit'] = len(overlap) == 0
    candidates.to_csv(OUT / 'PAIR_AUDIT.csv', index=False)
    risk_drugs = drugs[drugs.chemical_risk_points.gt(0)][['drug_names', 'ligand_inchikey', 'chemical_risk_points', 'rdkit_clogp', 'rdkit_mw']]
    risk_drugs.to_csv(OUT / 'RULE_FLAGGED_DRUGS.csv', index=False)
    per_target = candidates.groupby(['gene_symbol', 'experiment_arm']).agg(
        candidates=('pair_id', 'size'), rule_flagged=('chemical_risk_points', lambda v: int(v.gt(0).sum())),
        local_model_available=('structure_mask', 'sum')).reset_index()
    per_target = per_target.merge(roster[['gene_symbol', 'positive_control_drug', 'control_standard_types',
                                        'structure_ready_strict', 'positive_compounds', 'negative_compounds']], on='gene_symbol')
    per_target.to_csv(OUT / 'TARGET_AUDIT.csv', index=False)
    kd = roster.control_standard_types.fillna('').str.split(',').apply(lambda v: 'Kd' in v)
    ki = roster.control_standard_types.fillna('').str.split(',').apply(lambda v: 'Ki' in v)
    result = dict(status='AUDITED_PROSPECTIVE_CANDIDATES_NOT_MEASURED', utc=datetime.now(timezone.utc).isoformat(),
                  checks={k: bool(v) for k, v in checks.items()}, all_design_checks_pass=bool(all(checks.values())),
                  candidate_counts=candidates.selection_role.value_counts().to_dict(),
                  candidate_drugs=len(drugs), targets=len(roster), controls=len(controls),
                  core_high_pairs=int((core & high).sum()), extension_high_pairs=int((~core & high).sum()),
                  rule_flagged_drugs=len(risk_drugs), rule_flagged_pairs=int(candidates.chemical_risk_points.gt(0).sum()),
                  control_evidence=dict(includes_Kd=int(kd.sum()), Ki_without_Kd=int((ki & ~kd).sum()),
                                        IC50_only=int((~ki & ~kd).sum())),
                  ic50_only_controls=roster[~ki & ~kd][['gene_symbol', 'positive_control_drug']].to_dict('records'),
                  historical_training_exact_overlap=len(overlap),
                  inherited_external_snapshot_exclusions=summary['external_exclusions'],
                  external_novelty_researched_again=False,
                  structure_mask_counts=candidates.structure_mask.value_counts().to_dict(),
                  risk_scale='Original heuristic only; zero does not establish solubility, specificity or SPR validity.',
                  known_measured_outcomes_in_frozen_package=False,
                  empirical_hit_rate=None, selection_changed=False,
                  evidence_limit='S5 observed-pair AP is not the prospective SPR hit rate. 384 and 367 rankings are separate protocols.',
                  inputs=[identity(p) for p in [master_path, roster_path, summary_path, train_path, Path(__file__)]])
    (OUT / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
