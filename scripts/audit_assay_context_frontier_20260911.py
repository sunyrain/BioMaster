#!/usr/bin/env python3
"""Read-only assay recoverability audit; never changes training members or labels."""
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
RAW = ROOT / 'data/external/assay_context_research_20260911'
OUT = ROOT / 'outputs/biomaster_frontier_research_20260911'
ARMS = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def zipped(name):
    with zipfile.ZipFile(RAW / f'BindingDB_{name}_202609_tsv.zip') as z:
        return pd.read_csv(z.open(z.namelist()[0]), sep='\t', dtype=str, keep_default_na=False)


def main():
    OUT.mkdir(exist_ok=True)
    inputs = [DATA / name for name in ['OBSERVATIONS_WITH_QC.parquet', 'TRAIN_REGRESSION.parquet',
              'ALL_PAIR_TASK_LABELS.parquet']]
    inputs += [ARMS / f'{arm}_TRAIN.parquet' for arm in ['kdki_inactive', 'all_inactive']]
    inputs += list(RAW.glob('*.zip'))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    obs = pd.read_parquet(inputs[0], columns=['measurement_id', 'source', 'source_record_id',
        'molecule_id', 'target_id', 'endpoint', 'relation', 'value_nM', 'assay_id', 'record_qc'])
    obs = obs[obs.record_qc.eq('ACCEPTED')].copy()
    obs['pair_id'] = obs.molecule_id + '__' + obs.target_id
    obs['resolved_assay_id'] = ''
    cm = obs.source.eq('ChEMBL37') & obs.assay_id.fillna('').str.fullmatch(r'CHEMBL\d+')
    obs.loc[cm, 'resolved_assay_id'] = 'ChEMBL:' + obs.loc[cm, 'assay_id']
    mapping = zipped('rsid_eaids').drop_duplicates()
    ambiguous = mapping.groupby('REACTANT_SET_ID').ENTRYID_ASSAYID.nunique()
    bad = set(ambiguous[ambiguous.gt(1)].index)
    mapping = mapping[~mapping.REACTANT_SET_ID.isin(bad)].drop_duplicates('REACTANT_SET_ID')
    lookup = mapping.set_index('REACTANT_SET_ID').ENTRYID_ASSAYID
    assay = zipped('Assays')
    assay['key'] = assay.ENTRYID + '_' + assay.ASSAYID
    descriptions = set(assay.loc[assay.DESCRIPTION.str.strip().ne(''), 'key'])
    bm = obs.source.eq('BindingDB_202609')
    resolved = obs.loc[bm, 'source_record_id'].str.split(':').str[0].map(lookup)
    good = resolved.notna() & resolved.isin(descriptions)
    obs.loc[resolved[good].index, 'resolved_assay_id'] = 'BindingDB:' + resolved[good]
    source = []
    for name, g in obs.groupby('source'):
        source.append(dict(source=name, accepted_observations=len(g),
            legacy_assay_field_present=int(g.assay_id.fillna('').ne('').sum()),
            resolved_with_source_assay=int(g.resolved_assay_id.ne('').sum()),
            distinct_resolved_assays=g.loc[g.resolved_assay_id.ne(''), 'resolved_assay_id'].nunique()))
    pd.DataFrame(source).to_csv(OUT / 'ASSAY_SOURCE_RECOVERY.csv', index=False)
    splits = pd.read_parquet(inputs[2], columns=['pair_id', 'split']).drop_duplicates()
    assert splits.pair_id.is_unique
    splitmap = splits.set_index('pair_id').split
    obs['split'] = obs.pair_id.map(splitmap)
    keys = ['resolved_assay_id', 'target_id', 'endpoint']
    # Source namespaces stay separate: no unverified ChEMBL/BindingDB assay merge.
    context = obs[obs.resolved_assay_id.ne('')].copy()
    held = pd.MultiIndex.from_frame(context.loc[context.split.isin(['validation', 'test']), keys].drop_duplicates())
    regs = pd.read_parquet(inputs[1], columns=['pair_id', 'endpoint'])
    assert not regs.duplicated(['pair_id', 'endpoint']).any()
    exact = context[context.relation.eq('=') & context.value_nM.gt(0) & context.split.eq('train')]
    exact = exact.merge(regs, on=['pair_id', 'endpoint'], validate='many_to_one')
    exact['p_activity'] = 9 - np.log10(exact.value_nM)
    exact = exact.drop_duplicates(keys + ['pair_id', 'value_nM'])
    grouped = exact.groupby(keys + ['pair_id'], as_index=False).agg(
        p_activity=('p_activity', 'median'), minimum=('p_activity', 'min'), maximum=('p_activity', 'max'))
    grouped = grouped[(grouped.maximum - grouped.minimum).le(1.)].copy()
    grouped['shares_validation_or_test_assay'] = pd.MultiIndex.from_frame(grouped[keys]).isin(held)
    stats = []
    for arm, endpoints in [('kdki_inactive', ['Kd', 'Ki']), ('all_inactive', ['Kd', 'Ki', 'IC50', 'EC50'])]:
        members = pd.read_parquet(ARMS / f'{arm}_TRAIN.parquet', columns=['pair_id'])
        candidate = grouped[grouped.pair_id.isin(members.pair_id) & grouped.endpoint.isin(endpoints)]
        eligible = regs[regs.pair_id.isin(members.pair_id) & regs.endpoint.isin(endpoints)]
        for policy in ['current_train_only', 'exclude_validation_test_assays']:
            frame = candidate if policy == 'current_train_only' else candidate[~candidate.shares_validation_or_test_assay]
            for threshold in [1, 2, 5, 10]:
                count = frame.groupby(keys).pair_id.transform('nunique')
                use = frame[count.ge(threshold)]
                stats.append(dict(arm=arm, policy=policy, min_compounds_per_assay=threshold,
                    training_pairs=len(members), eligible_regression_pair_endpoints=len(eligible),
                    assay_target_endpoint_groups=len(use[keys].drop_duplicates()),
                    assay_pair_endpoint_rows=len(use), unique_pairs=use.pair_id.nunique(),
                    unique_pair_endpoints=len(use[['pair_id', 'endpoint']].drop_duplicates()),
                    covered_targets=use.target_id.nunique(),
                    pair_fraction_of_binary_members=use.pair_id.nunique()/len(members),
                    fraction_of_eligible_regression_pair_endpoints=len(use[['pair_id', 'endpoint']].drop_duplicates())/len(eligible)))
    pd.DataFrame(stats).to_csv(OUT / 'ASSAY_TRAINING_FEASIBILITY.csv', index=False)
    db = ROOT / 'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as c:
        columns = [r[1] for r in c.execute('pragma table_info(assays)')]
    summary = dict(status='DESCRIPTIVE_AUDIT_ONLY_NO_TRAINING', input_sha256=hashes,
        inputs_unchanged=all(sha(ROOT/p) == h for p, h in hashes.items()),
        bindingdb_mapping_rows=len(mapping), bindingdb_ambiguous_reactant_ids=len(bad),
        bindingdb_description_rows=len(assay), chembl_assay_columns=columns,
        verified_checks=dict(training_pair_ids_unique=splits.pair_id.is_unique,
            regression_pair_endpoint_unique=not regs.duplicated(['pair_id', 'endpoint']).any()),
        limitations=['Source assay ID recovery is not confirmation of identical constructs, conditions or mechanism.',
            'Recovered coverage is descriptive, not a model performance gain.',
            'ChEMBL/BindingDB assay namespaces are not merged; cross-source duplicate evidence can remain.',
            'Coverage uses existing exact-regression eligibility and original A/B members; all raw evidence is not newly admitted.',
            'Excluding shared assays from this auxiliary view alone would not purge them from current BCE training.',
            'Censored observations and generic inactive labels are excluded from exact within-assay regression coverage.',
            'Assay descriptions can vary within a source group; metadata and dynamic-range checks remain before training.'])
    assert summary['inputs_unchanged']
    summary['producer_sha256'] = sha(Path(__file__))
    (OUT / 'ASSAY_AUDIT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'DOWNLOAD_MANIFEST.json').write_text((RAW / 'DOWNLOAD_MANIFEST.json').read_text())
    print(pd.DataFrame(source).to_string(index=False))
    print(pd.DataFrame(stats).query('min_compounds_per_assay == 5').to_string(index=False))


if __name__ == '__main__':
    main()
