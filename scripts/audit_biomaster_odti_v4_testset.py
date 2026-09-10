#!/usr/bin/env python3
"""Audit actual R1 test provenance, observation coverage and support overlap.

Read-only with respect to the frozen experiment. Writes a separate audit;
does not change predictions, labels, candidates, checkpoints or old metrics.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_biomaster_odti_v4_features import sha256, write_json


def main():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    run = ROOT / 'outputs/biomaster_odti_v4_20260905'
    out = run / 'testset_audit'
    out.mkdir(exist_ok=True)
    protocol = json.loads((run / 'protocol/PROTOCOL_V4.json').read_text())
    data_path = Path(protocol['data']['prepared_relations_path'])
    raw_path = Path(protocol['data']['identity']['relations_path'])
    assert sha256(data_path) == protocol['data']['prepared_relations_sha256']
    assert sha256(raw_path) == protocol['data']['identity']['relations_sha256']
    data, raw = pd.read_csv(data_path), pd.read_csv(raw_path, low_memory=False)
    old_path = ROOT / 'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1/DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz'
    old = pd.read_csv(old_path)
    assets = pd.read_csv(ROOT / 'outputs/biomaster_odti_v4_plan_20260905/CURRENT_MOLECULE_ASSETS_V4.csv.gz').set_index('drug_feature_index')
    panel_path = run / 'protocol/test_panel.npz'
    assert sha256(panel_path) == protocol['panels']['test']['panel_sha256']
    with np.load(panel_path) as f:
        queries, targets, positive = f['drug_indices'], f['target_indices'], f['known_positive']
    train, test = data.loc[data.split.eq('train')], data.loc[data.split.eq('test')]
    part = test.loc[test.drug_feature_index.isin(queries)]
    source = raw.iloc[part.source_row].copy()
    source['drug_feature_index'] = part.drug_feature_index.to_numpy()
    counts = part.groupby('drug_feature_index').binary_label.agg(['size', 'sum', 'nunique'])
    old_keys, old_smiles = set(old.ligand_inchikey), set(old.ligand_smiles)
    key_match = set(source.loc[source.parent_standard_inchi_key.isin(old_keys), 'drug_feature_index'])
    smiles_match = set(assets.index[assets.model_ligand_smiles.isin(old_smiles)]) & set(queries)
    evidence = {n: np.load(p, mmap_mode='r') for n,p in protocol['panels']['test']['support'].items()}
    similarity = np.where(evidence['mask'][:,1], evidence['similarities'][:,1], 0).max(-1).reshape(positive.shape)
    nearest = similarity.max(-1)
    prediction = np.load(run / 'evaluation/positive_nearest_PREDICTIONS_V4.npz')
    assert np.array_equal(prediction['panel_scores'], similarity)
    same_scaffold = set(train.murcko_scaffold.dropna()) - {''}
    query_scaffolds = part.drop_duplicates('drug_feature_index').murcko_scaffold
    lookup = {(int(d),int(t)):int(s) for d,t,s in zip(data.drug_feature_index,data.target_feature_index,data.source_row)}
    shared_doc, shared_assay, examples = 0, 0, []
    def identifiers(value):
        return set(str(value).split(',')) - {'nan', 'None', ''}
    for qi, ti in np.argwhere(positive):
        row = qi * len(targets) + ti
        ref = int(evidence['indices'][row,1,0])
        d, t = int(queries[qi]), int(targets[ti])
        query_record = raw.iloc[lookup[d,t]]
        if ref < 0:
            continue
        ref_record = raw.iloc[lookup[ref,t]]
        doc = identifiers(query_record.doc_ids) & identifiers(ref_record.doc_ids)
        assay = identifiers(query_record.assay_ids) & identifiers(ref_record.assay_ids)
        shared_doc += bool(doc)
        shared_assay += bool(assay)
        examples.append({'query_drug_index':d,'target_index':t,'reference_drug_index':ref,
            'query_chembl_id':query_record.parent_molecule_chembl_id,
            'reference_chembl_id':ref_record.parent_molecule_chembl_id,
            'similarity':float(evidence['similarities'][row,1,0]),
            'shared_doc_ids':','.join(sorted(doc)), 'shared_assay_ids':','.join(sorted(assay))})
    identical_fingerprint = nearest >= .999999
    best_target = similarity.argmax(-1)
    rows = np.arange(len(queries))*len(targets)+best_target
    slots = evidence['similarities'][rows,1].argmax(-1)
    refs = evidence['indices'][rows,1,slots]
    same_input, same_without_stereo = 0, 0
    for d, ref in zip(queries[identical_fingerprint], refs[identical_fingerprint]):
        a,b = assets.loc[d,'model_ligand_smiles'], assets.loc[ref,'model_ligand_smiles']
        same_input += a == b
        ma,mb = Chem.MolFromSmiles(a),Chem.MolFromSmiles(b)
        Chem.RemoveStereochemistry(ma)
        Chem.RemoveStereochemistry(mb)
        same_without_stereo += Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb)
    # Independently check every valid full-panel support against train triples.
    width = len(assets)
    valid_codes = np.sort(((train.target_feature_index.to_numpy()*width + train.drug_feature_index.to_numpy())*2 + train.binary_label.to_numpy()).astype(np.int64))
    index_path = ROOT / 'outputs/biomaster_v3_20260905/data/support_store/support_index.npz'
    with np.load(index_path) as f:
        entity_codes = f['entity_codes']
    violations, self_support, valid_count = 0,0,0
    ds,ts = np.repeat(queries,len(targets)), np.tile(targets,len(queries))
    for start in range(0,len(ds),4096):
        stop = min(start+4096,len(ds))
        mask = evidence['mask'][start:stop]
        ids = evidence['indices'][start:stop]
        codes = ((ts[start:stop,None,None]*width + ids)*2 + np.arange(2)[None,:,None])[mask]
        positions = np.searchsorted(valid_codes,codes)
        violations += int(((positions == len(valid_codes)) | (valid_codes[np.minimum(positions,len(valid_codes)-1)] != codes)).sum())
        self_support += int(((entity_codes[np.maximum(ids,0)] == entity_codes[ds[start:stop,None,None]]) & mask).sum())
        valid_count += int(mask.sum())
    assert violations == self_support == 0
    audit = {
        'scope':'R1 historical general-compound known-positive retrieval; unsuitable as primary old-drug novel-target benchmark',
        'data_files':{str(p.relative_to(ROOT)):sha256(p) for p in [data_path,raw_path,panel_path,old_path]},
        'test_observed':{'rows':len(test),'compounds':test.drug_feature_index.nunique(),
            'positive':int(test.binary_label.sum()),'negative':int(test.binary_label.eq(0).sum()),
            'sources':raw.iloc[test.source_row].source_kind.value_counts().to_dict()},
        'scored_panel':{'compounds':len(queries),'targets':len(targets),'scored_pairs':int(positive.size),
            'observed_pairs':len(part),'known_positive':int(positive.sum()),'known_negative':int(part.binary_label.eq(0).sum()),
            'unobserved':int(positive.size-len(part)),'unobserved_fraction':float(1-len(part)/positive.size),
            'compounds_one_observed_pair':int(counts['size'].eq(1).sum()),
            'compounds_one_known_positive':int(counts['sum'].eq(1).sum()),
            'compounds_with_positive_and_negative':int(counts['nunique'].eq(2).sum()),
            'observed_sources':source.source_kind.value_counts().to_dict(),
            'matched_project720_inchikey':len(key_match),'matched_project720_model_smiles':len(smiles_match),
            'old_drug_matching_limit':'exact current identifiers; no assertion about all approved drugs or untested alias normalization'},
        'overlap':{'queries_with_train_scaffold':int(query_scaffolds.isin(same_scaffold).sum()),
            'queries_similarity_ge_0_7':int((nearest>=.7).sum()),'queries_similarity_lt_0_4':int((nearest<.4).sum()),
            'queries_identical_morgan_to_train_positive':int(identical_fingerprint.sum()),
            'among_identical_morgan_exact_smiles':int(same_input),
            'among_identical_morgan_same_structure_without_stereo':int(same_without_stereo),
            'known_positive_pairs_shared_doc_with_top_same_target_positive_reference':int(shared_doc),
            'known_positive_pairs_shared_assay_with_top_same_target_positive_reference':int(shared_assay)},
        'direct_support_leakage_check':{'valid_support_entries_checked':valid_count,
            'nontraining_or_wrong_target_or_wrong_label':violations,'same_entity_support':self_support,
            'nearest_predictions_equal_cached_max_positive_similarity':True},
        'interpretation':'No direct support-label leak found. Chemical series/assay overlap and sparse positive-unlabeled evaluation limit the claim. This audit does not establish a corrected model ranking.'}
    write_json(out/'TESTSET_AUDIT_V4.json',audit)
    pd.DataFrame(examples).to_csv(out/'POSITIVE_NEAREST_PROVENANCE_V4.csv',index=False)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
