#!/usr/bin/env python3
"""Build versioned auxiliary assay views without changing the frozen A/B universe."""
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from audit_assay_context_frontier_20260911 import zipped
from prepare_endpoint_multitask_20260911 import OUT as PARENT, SOURCE, DATA, FEATURES, write_json
from biomaster.portable_ranker_v2 import digest
from biomaster.training_dataset_v1 import MUTANT

OUT = ROOT/'outputs/biomaster_assay_aware_20260911'
VARIANTS = {'balanced': dict(sampler='soft', regression=False, assay=False),
            'rotate150': dict(sampler='rotate150', regression=False, assay=False),
            'regression': dict(sampler='soft', regression=True, assay=False),
            'assay': dict(sampler='soft', regression=True, assay=True)}


def protocol():
    p = json.loads((PARENT/'PROTOCOL.json').read_text())
    p.update(status='FROZEN_BEFORE_OPTIMIZED_TRAINING', variants=VARIANTS, total_fits=30,
        primary_fits=24, adaptation_fits=6,
        auxiliary_weights=dict(regression=.1, assay_difference=.2),
        assay_pairs_per_endpoint=32, assay_minimum_compounds=5, assay_minimum_iqr=.3,
        assay_minimum_pair_difference=.3, target_count_smoothing=20., maximum_class_relative_gain=5.,
        rotating_cap=150, minimum_unique_binary_fraction=1.,
        binary_objective='Original measured A/B labels and two binary heads; change only weights or rotating exposure.',
        regression_objective='Same existing exact endpoint labels, train-only scaling and endpoint-balanced Huber.',
        assay_objective='Huber on intra-assay exact p-activity differences; separate endpoints, equal endpoint and assay weights.',
        source_duplicate_policy='Prefer ChEMBL for pair-endpoint auxiliary evidence when available; otherwise BindingDB. No merging of unverified assay namespaces.',
        auxiliary_context='Source assay descriptions required; annotated mutation/fusion contexts excluded; remaining context is source-reported, not fully experimentally certified.',
        partition_policy='Retain original scaffold split for matched comparisons; audit source-assay and document-disjoint subsets separately.',
        adaptation=dict(parents=['kdki_inactive', 'all_inactive'], parent_variant='assay',
            target_arm='kdki_inactive', learning_rate=1e-4,
            purpose='Compare B-to-A with A-to-A continued training using identical downstream data and loss; upstream compute still differs.'),
        test_policy='Score no optimized checkpoints on test until all 30 fits are frozen. Existing test is a diagnostic panel, not pristine.',
        limitations=['Operational validation convergence, not a guarantee of global optimality.',
            'The classifier remains pooled; endpoint regression heads are separate. This round tests weighting, intra-assay supervision and adaptation.',
            '150 means a rotating target/class budget without the historical scaffold-selection heuristic.',
            'Current scaffold splits can share source assays; only explicitly flagged subsets claim source-assay separation.',
            'No real SPR outcome inferred from model agreement or calibration.',
            'All measurements and production assets retained; research models are not automatically deployed.'])
    for key in ['target_rank_weight', 'drug_rank_weight', 'rank_objective', 'rank_queries_per_task_direction',
                'rank_candidates_per_class', 'regression_weight', 'weight_policy']:
        p.pop(key, None)
    return p


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'DATA_MANIFEST.json').exists():
        m = json.loads((OUT/'DATA_MANIFEST.json').read_text())
        for path, h in {**m['inputs'], **m['prepared']}.items():
            assert digest(ROOT/path) == h, path
        assert digest(OUT/'PROTOCOL.json') == m['protocol_sha256']
        return m
    p = protocol()
    if (OUT/'PROTOCOL.json').exists():
        assert json.loads((OUT/'PROTOCOL.json').read_text()) == p, 'protocol changed during preparation'
    else:
        write_json(OUT/'PROTOCOL.json', p)
    inputs = [DATA/'OBSERVATIONS_WITH_QC.parquet', DATA/'TRAIN_REGRESSION.parquet', DATA/'ALL_PAIR_TASK_LABELS.parquet',
        PARENT/'PROTOCOL.json', PARENT/'DATA_MANIFEST.json', PARENT/'MODEL_CONFIG.json', FEATURES/'MANIFEST.json']
    inputs += [SOURCE/f'{arm}_TRAIN.parquet' for arm in ['kdki_inactive', 'all_inactive']]
    inputs += [SOURCE/'COMMON_VALIDATION.parquet', SOURCE/'COMMON_TEST.parquet']
    raw = ROOT/'data/external/assay_context_research_20260911'
    downloads = json.loads((raw/'DOWNLOAD_MANIFEST.json').read_text())
    for item in downloads:
        path = ROOT/item['file']; assert digest(path) == item['sha256']; inputs.append(path)
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in inputs}
    obs = pd.read_parquet(DATA/'OBSERVATIONS_WITH_QC.parquet')
    obs = obs[obs.record_qc.eq('ACCEPTED')].copy()
    obs['pair_id'] = obs.molecule_id + '__' + obs.target_id
    obs['resolved_assay_id'] = ''; obs['description'] = ''
    cm = obs.source.eq('ChEMBL37') & obs.assay_id.fillna('').str.fullmatch(r'CHEMBL\d+')
    db = ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
    columns = 'chembl_id,description,assay_type,bao_format,assay_cell_type,assay_tissue,variant_id'
    metadata = []
    ids = obs.loc[cm, 'assay_id'].unique().tolist()
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as c:
        for start in range(0, len(ids), 900):
            chunk = ids[start:start+900]
            metadata.append(pd.read_sql_query(f"select {columns} from assays where chembl_id in ({','.join('?' for _ in chunk)})", c, params=chunk))
    metadata = pd.concat(metadata, ignore_index=True)
    assert metadata.chembl_id.is_unique
    metadata.to_parquet(OUT/'CHEMBL_ASSAY_METADATA.parquet', index=False)
    desc = metadata.set_index('chembl_id').description.fillna('')
    obs.loc[cm, 'resolved_assay_id'] = 'ChEMBL:' + obs.loc[cm, 'assay_id']
    obs.loc[cm, 'description'] = obs.loc[cm, 'assay_id'].map(desc).fillna('')
    mapping = zipped('rsid_eaids').drop_duplicates()
    ambiguous = mapping.groupby('REACTANT_SET_ID').ENTRYID_ASSAYID.nunique()
    assert not ambiguous.gt(1).any(), 'ambiguous BindingDB record to assay mapping'
    lookup = mapping.drop_duplicates('REACTANT_SET_ID').set_index('REACTANT_SET_ID').ENTRYID_ASSAYID
    descriptions = zipped('Assays'); descriptions['key'] = descriptions.ENTRYID + '_' + descriptions.ASSAYID
    different = descriptions.groupby('key').DESCRIPTION.nunique()
    bad_descriptions = set(different[different.gt(1)].index)
    descriptions = descriptions[~descriptions.key.isin(bad_descriptions)].drop_duplicates('key')
    bm = obs.source.eq('BindingDB_202609')
    resolved = obs.loc[bm, 'source_record_id'].str.split(':').str[0].map(lookup)
    valid = resolved.notna()
    obs.loc[resolved[valid].index, 'resolved_assay_id'] = 'BindingDB:' + resolved[valid]
    obs.loc[bm, 'description'] = resolved.map(descriptions.set_index('key').DESCRIPTION).fillna('')
    obs['assay_group'] = obs.resolved_assay_id + '|' + obs.target_id + '|' + obs.endpoint
    # Normalize text for conservative mutation/fusion exclusion, never for extracting labels.
    mutation = obs.description.str.contains(MUTANT, na=False)
    known_context = obs.resolved_assay_id.ne('') & obs.description.str.strip().ne('')
    labels = pd.read_parquet(DATA/'ALL_PAIR_TASK_LABELS.parquet', columns=['pair_id', 'split']).drop_duplicates()
    assert labels.pair_id.is_unique
    obs['split'] = obs.pair_id.map(labels.set_index('pair_id').split)
    regs = pd.read_parquet(DATA/'TRAIN_REGRESSION.parquet')
    use = obs[known_context & ~mutation & obs.split.eq('train') & obs.relation.eq('=') & obs.value_nM.gt(0)].copy()
    use = use.merge(regs[['pair_id', 'endpoint', 'drug_feature_index', 'target_feature_index']],
                    on=['pair_id', 'endpoint'], validate='many_to_one')
    has_cm = pd.MultiIndex.from_frame(use.loc[use.source.eq('ChEMBL37'), ['pair_id', 'endpoint']].drop_duplicates())
    duplicate = use.source.eq('BindingDB_202609') & pd.MultiIndex.from_frame(use[['pair_id', 'endpoint']]).isin(has_cm)
    removed_duplicate = int(duplicate.sum()); use = use[~duplicate].copy()
    use['p_activity'] = 9 - np.log10(use.value_nM)
    use = use.drop_duplicates(['assay_group', 'pair_id', 'value_nM'])
    keys = ['assay_group', 'resolved_assay_id', 'source', 'target_id', 'endpoint', 'pair_id', 'drug_feature_index', 'target_feature_index']
    view = use.groupby(keys, as_index=False).agg(p_activity=('p_activity','median'),
        minimum=('p_activity','min'), maximum=('p_activity','max'))
    view = view[(view.maximum-view.minimum).le(1.)].copy()
    stats = []; prepared = [OUT/'CHEMBL_ASSAY_METADATA.parquet']
    held = set(obs.loc[obs.split.isin(['validation', 'test']) & known_context, 'assay_group'])
    for arm, endpoints in [('kdki_inactive', ['Kd', 'Ki']), ('all_inactive', ['Kd', 'Ki', 'IC50', 'EC50'])]:
        members = pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet')
        f = view[view.pair_id.isin(members.pair_id) & view.endpoint.isin(endpoints)].copy()
        summary = f.groupby('assay_group').p_activity.agg(['size', lambda x:x.quantile(.75)-x.quantile(.25)])
        summary.columns = ['molecules', 'iqr']
        good = summary.index[summary.molecules.ge(p['assay_minimum_compounds']) & summary.iqr.ge(p['assay_minimum_iqr'])]
        f = f[f.assay_group.isin(good)].copy(); f['shares_validation_test_assay'] = f.assay_group.isin(held)
        assert not f.duplicated(['assay_group','pair_id']).any()
        assert f.pair_id.isin(members.pair_id).all() and np.isfinite(f.p_activity).all()
        for endpoint in endpoints:
            part = f[f.endpoint.eq(endpoint)]
            assert not part.empty, (arm, endpoint, 'no eligible assay supervision')
            stats.append(dict(arm=arm,endpoint=endpoint,rows=len(part),groups=part.assay_group.nunique(),
                unique_pairs=part.pair_id.nunique(),targets=part.target_id.nunique(),
                shared_validation_test_assay_row_fraction=part.shares_validation_test_assay.mean()))
        path = OUT/f'{arm}_ASSAY_TRAIN.parquet'; f.to_parquet(path,index=False); prepared.append(path)
        # Audit query context separately; do not alter the fixed panel membership or labels.
        train_obs = obs[obs.pair_id.isin(members.pair_id)]
        train_groups = set(train_obs.loc[train_obs.resolved_assay_id.ne(''),'assay_group'])
        observed = obs[obs.resolved_assay_id.ne('')].copy()
        observed['shared'] = observed.assay_group.isin(train_groups)
        shared_pairs = set(observed.loc[observed.shared, 'pair_id'])
        missing_pairs = set(obs.loc[~known_context, 'pair_id'])
        available_pairs = set(observed.pair_id)
        for split in ['VALIDATION','TEST']:
            frame = pd.read_parquet(SOURCE/f'COMMON_{split}.parquet')
            frame['source_assay_disjoint'] = frame.pair_id.isin(available_pairs) & ~frame.pair_id.isin(shared_pairs | missing_pairs)
            frame['source_assay_and_document_disjoint'] = frame.source_assay_disjoint & frame.document_disjoint
            path = OUT/f'{arm}_{split}_CONTEXT.parquet'; frame.to_parquet(path,index=False); prepared.append(path)
    pd.DataFrame(stats).to_csv(OUT/'ASSAY_VIEW_COUNTS.csv',index=False)
    write_json(OUT/'PREPARATION_AUDIT.json', dict(accepted_observations=len(obs), known_description=int(known_context.sum()),
        mutation_or_fusion_description=int(mutation.sum()), bindingdb_ambiguous_descriptions=len(bad_descriptions),
        secondary_bindingdb_observations_removed=removed_duplicate, context_certification='source-reported; not laboratory-verified',
        original_scaffold_split_preserved=True, test_scores_used=False,
        cross_source_policy=p['source_duplicate_policy']))
    (OUT/'MODEL_CONFIG.json').write_text((PARENT/'MODEL_CONFIG.json').read_text()); prepared.append(OUT/'MODEL_CONFIG.json')
    parent = json.loads((PARENT/'DATA_MANIFEST.json').read_text())
    for path,h in {**hashes, **parent['frozen_inputs']}.items():
        assert digest(ROOT/path) == h, path
    manifest = dict(inputs=hashes, prepared={str(x.relative_to(ROOT)):digest(x) for x in prepared},
        protocol_sha256=digest(OUT/'PROTOCOL.json'), frozen_inputs=parent['frozen_inputs'],
        producer_sha256=digest(Path(__file__)))
    write_json(OUT/'DATA_MANIFEST.json',manifest)
    return manifest


if __name__ == '__main__':
    prepare()
