#!/usr/bin/env python3
"""Identify the actual SPR384 rank source and audit model substitutions of its rank gate."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.portable_ranker_v2 import digest
from score_ab_catalog_retention_20260911 import OUT, FINAL, REVIEW, select_fixed_slots


def main():
    p = pd.read_parquet(OUT / 'CORE276480_SELECTION_AUDIT.parquet')
    final = pd.read_csv(FINAL)
    finalids = set(final.pair_id)
    source = ROOT / 'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz'
    legacy = pd.read_csv(source, usecols=['pairId', 'independent_validation_rank_score']).rename(columns={'pairId':'pair_id'})
    a = p[['pair_id', 'independent_validation_rank_score']].merge(legacy, on='pair_id', validate='one_to_one', suffixes=['_audit', '_source'])
    assert (a.independent_validation_rank_score_audit == a.independent_validation_rank_score_source).all()
    actual = p.groupby('ligand_inchikey', sort=False).independent_validation_rank_score.rank(method='first', ascending=False)
    assert actual.eq(p.binding_rank_384).all()
    assert p.loc[p.in_frozen384, 'binding_rank_384'].le(20).all()
    selected, solver = select_fixed_slots(p[p.counterfactual_eligible], p.set_index('pair_id').independent_validation_rank_score,
                                          final.groupby('新靶点ChEMBL编号').size().to_dict())
    assert selected is not None
    selected.to_csv(OUT / 'LEGACY_SCORE_FIXED112_SELECTION.csv', index=False)
    reviewed = pd.read_csv(REVIEW / 'FULL574_COMPREHENSIVE_AUDIT.csv')
    approved = set(reviewed.loc[reviewed.eligible_after_comprehensive_review, 'pair_id'])
    gate = []
    names = [('actual_legacy_ranker', 'binding_rank_384')] + [(c.removesuffix('_drug_rank384'), c) for c in p if c.endswith('_drug_rank384')]
    for name, col in names:
        passes = p.counterfactual_eligible & p[col].le(20)
        remaining = set(p.loc[passes, 'pair_id']) & approved
        gate.append(dict(model=name, full_core_gate_pass=int(passes.sum()), original384_still_top20=int((passes & p.in_frozen384).sum()),
                         previously_reviewed_449_still_eligible=len(remaining),
                         can_fill384_without_new_pair_reviews=len(remaining)>=384))
    pd.DataFrame(gate).to_csv(OUT / 'ORIGINAL_TOP20_GATE_REPLAY.csv', index=False)
    selectedids = set(selected.pair_id)
    source_summary = source.with_name('BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json')
    result = dict(actual_rank_source=str(source.relative_to(ROOT)), source_sha256=digest(source),
                  exact_score_and_rank_reproduction=True, frozen384_all_original_rank_top20=True,
                  actual_rank_features=json.loads(source_summary.read_text())['selection']['independent_validation_rank'],
                  actual_legacy_score_with_fixed112_model_only_retained=len(selectedids & finalids),
                  solver=solver, legacy_ranker_is_not_selected_neural_checkpoint=True,
                  interpretation='Low overlap mixes changed scorer and changed selection objective. Original384 also used positive-neighbor, disease and review utility; counterfactual model-only selection does not replicate that full policy.')
    (OUT / 'SELECTION_LINEAGE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(pd.DataFrame(gate).to_string(index=False))


if __name__ == '__main__':
    main()
