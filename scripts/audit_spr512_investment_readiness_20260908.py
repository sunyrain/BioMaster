"""Read-only review of frozen selection; writes separate descriptive audit assets.

Flags indicate evidence gaps, not predicted biological inactivity or hit probability.
"""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/retargetmap_spr512_ours_frozen_20260904'
OUT = ROOT / 'outputs/retargetmap_spr512_quality_review_20260908'


def read(fragment):
    path = next(SOURCE.glob('*' + fragment + '*.csv'))
    with path.open(newline='') as handle:
        return path, list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    master, rows = read('INTERNAL_MASTER')
    roster, targets = read('TARGET_ROSTER')
    candidates = [r for r in rows if r['selection_role'] != 'POSITIVE_CONTROL']
    high = [r for r in candidates if r['selection_role'] == 'OUR_FROZEN_MODEL_HIGH']
    core_high = [r for r in high if r['experiment_arm'] == 'FROZEN_384_CORE_DISCOVERY']
    tmap = {r['gene_symbol']: r for r in targets}
    target_rows = []
    for target in targets:
        selected = [r for r in candidates if r['gene_symbol'] == target['gene_symbol']]
        high_rows = [r for r in selected if r['selection_role'] == 'OUR_FROZEN_MODEL_HIGH']
        evidence = set(target['control_standard_types'].split(','))
        target_rows.append(dict(
            gene=target['gene_symbol'], arm=target['experiment_arm'],
            candidates=len(selected), high_candidates=len(high_rows),
            high_drug_cold=sum(r['drug_seen_in_full_fit_training'] == 'False' for r in high_rows),
            positive_reference_compounds=int(target['positive_compounds']),
            negative_reference_compounds=int(target['negative_compounds']),
            reference_counts_are_model_performance=False,
            strict_pocket_ready=target['structure_ready_strict'],
            control=target['positive_control_drug'], control_endpoints=target['control_standard_types'],
            control_evidence_category='includes_Kd' if 'Kd' in evidence else 'Ki_without_Kd' if 'Ki' in evidence else 'IC50_only',
            construct_status_in_frozen_file=target['construct_boundary_status'],
            extra_review_reason=';'.join(
                (['strict_pocket_not_ready'] if target['structure_ready_strict'] != 'True' else [])
                + (['control_only_IC50'] if not evidence.intersection({'Kd', 'Ki'}) else [])),
            disposition='CONSTRUCT_AND_CONTROL_CONFIRMATION_PENDING_NOT_A_BINDING_VERDICT'))
    pair_rows = []
    for r in candidates:
        target = tmap[r['gene_symbol']]
        flags = ['construct_confirmation_pending_in_frozen_file', 'pair_literature_patent_review_pending']
        if r['experiment_arm'] == 'EXTENDED_367_TARGET_DISCOVERY':
            flags.append('extension_score_no_inherited_core_S5_performance')
        if r['drug_seen_in_full_fit_training'] == 'False':
            flags.append('drug_cold_in_historical_fullfit')
        if float(r['structure_mask']) == 0:
            flags.append('no_local_structure_mask_not_proof_of_no_pocket')
        if target['structure_ready_strict'] != 'True':
            flags.append('strict_target_pocket_not_ready')
        if not set(target['control_standard_types'].split(',')).intersection({'Kd', 'Ki'}):
            flags.append('control_only_IC50')
        if float(r['chemical_risk_points']) > 0:
            flags.append('original_chemical_rule_flag')
        if float(r['rdkit_clogp']) > 5:
            flags.append('clogp_above_5_solubility_measurement_needed')
        if r['drug_names'] in {'enalapril', 'serdexmethylphenidate'}:
            flags.append('label_verified_prodrug_parent_metabolite_question_unresolved')
        pair_rows.append(dict(
            blind_experiment_id=r['blind_experiment_id'], pair_id=r['pair_id'],
            drug=r['drug_names'], gene=r['gene_symbol'], role=r['selection_role'],
            arm=r['experiment_arm'], selecting_rank=r['retargetmap_rank_384']
            if r['experiment_arm'] == 'FROZEN_384_CORE_DISCOVERY' else r['routed_rank_within_drug'],
            posthoc_dtiam_rank_384=r['posthoc_only_dtiam_rank_384'],
            review_flags=';'.join(flags),
            binding_probability='', disposition='REVIEW_REQUIRED_NOT_AUTOMATIC_REJECTION'))
    checks = dict(rows_512=len(rows) == 512, candidates_480=len(candidates) == 480,
                  controls_32=len(rows)-len(candidates) == 32,
                  unique_pairs=len({(r['ligand_inchikey'], r['uniprot_accession']) for r in rows}) == len(rows),
                  targets_32=len(targets) == 32,
                  roles_per_target=all(Counter(r['selection_role'] for r in candidates
                      if r['gene_symbol'] == t['gene_symbol']) == {
                          'OUR_FROZEN_MODEL_HIGH': 10, 'OUR_FROZEN_MODEL_INTERMEDIATE': 3,
                          'OUR_FROZEN_MODEL_LOW_BACKGROUND': 2} for t in targets),
                  valid_selecting_ranks=all(
                      (float(r['selecting_rank']) <= (20 if r['arm'] == 'FROZEN_384_CORE_DISCOVERY' else 30))
                      if r['role'] == 'OUR_FROZEN_MODEL_HIGH' else
                      (80 <= float(r['selecting_rank']) <= 180)
                      if r['role'] == 'OUR_FROZEN_MODEL_INTERMEDIATE' else float(r['selecting_rank']) > 250
                      for r in pair_rows))
    summary = dict(
        scope='Frozen 2026-09-04 SPR512 V2; descriptive evidence and readiness review',
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [master, roster]},
        checks=checks, all_checks_pass=all(checks.values()),
        candidate_roles=dict(Counter(r['selection_role'] for r in candidates)),
        high_drug_cold=sum(r['drug_seen_in_full_fit_training'] == 'False' for r in high),
        core_high_dtiam_top20=sum(float(r['posthoc_only_dtiam_rank_384']) <= 20 for r in core_high),
        core_high_dtiam_median_rank=statistics.median(float(r['posthoc_only_dtiam_rank_384']) for r in core_high),
        core_high_denominator=len(core_high),
        control_evidence=dict(Counter(r['control_evidence_category'] for r in target_rows)),
        extra_target_review=[r['gene'] for r in target_rows if r['extra_review_reason']],
        role_chemistry_medians={role: {field: statistics.median(float(r[field]) for r in candidates if r['selection_role'] == role)
            for field in ['rdkit_mw', 'rdkit_clogp', 'rdkit_tpsa']}
            for role in sorted({r['selection_role'] for r in candidates})},
        limits=['No new experimental outcomes', 'No full 480-pair literature/patent search',
                'No per-candidate hit-probability calibration', 'No chemical matching causal analysis',
                'Only enalapril and serdexmethylphenidate label-verified here; not an exhaustive prodrug inventory',
                'DTIAM is posthoc context, not a rejection rule or an independent truth label',
                'Frozen input files and selection remain unchanged'])
    write_csv('TARGET_REVIEW.csv', target_rows)
    write_csv('PAIR_REVIEW.csv', pair_rows)
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit('Frozen design checks failed; inspect before interpreting review.')


if __name__ == '__main__':
    main()
