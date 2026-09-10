#!/usr/bin/env python3
"""Prepare independent early rolls and dated, strictly within-assay contrasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.temporal_relations import aggregate_window, first_observations, new_relations
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json

SOURCE = ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
BASE = ROOT / 'outputs/biomaster_unified_interaction_20260906/features'
OUT = ROOT / 'outputs/biomaster_best_model_20260906/data'
DB = ROOT / 'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'


def rolls():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'ROLL_MANIFEST.json').exists():
        return
    annual = pd.read_csv(SOURCE / 'ANNUAL_STANDARDIZED.csv.gz')
    first = first_observations(annual)
    old = pd.read_csv(SOURCE / 'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets = pd.read_csv(SOURCE / 'TARGET_INDEX.csv.gz')
    mapping = old.set_index('drug_feature_index').old_drug_index
    required = []
    counts = {}
    for cutoff in [2018, 2020]:
        folder = OUT / f'roll_{cutoff}'; folder.mkdir(exist_ok=True)
        train = aggregate_window(annual, end=cutoff)
        val = new_relations(aggregate_window(annual, start=cutoff+1, end=cutoff+2), first, cutoff)
        risk = np.ones((len(old), len(targets)), bool)
        prior = first[(first.first_document_year.le(cutoff) | first.has_undated)
                      & first.drug_feature_index.isin(mapping.index)]
        risk[prior.drug_feature_index.map(mapping).to_numpy(int), prior.target_feature_index.to_numpy(int)] = False
        np.save(folder / 'RISK.npy', risk)
        counts[str(cutoff)] = {}
        for name, frame in [('TRAIN', train), ('VALIDATION', val)]:
            frame = frame.loc[frame.binary_label.notna()].copy()
            frame.binary_label = frame.binary_label.astype(int)
            frame['is_project_old_drug'] = frame.drug_feature_index.isin(mapping.index)
            assert not frame.duplicated(['drug_feature_index', 'target_feature_index']).any()
            if name == 'TRAIN':
                assert frame.max_document_year.max() <= cutoff
            else:
                assert frame.min_document_year.min() > cutoff
                assert frame.max_document_year.max() <= cutoff+2
                assert not frame.has_undated.any()
                positives = frame[frame.is_project_old_drug & frame.binary_label.eq(1)]
                assert risk[positives.drug_feature_index.map(mapping).to_numpy(int),
                            positives.target_feature_index.to_numpy(int)].all()
            frame.to_csv(folder / f'{name}.csv.gz', index=False)
            required.extend(frame.drug_feature_index.to_numpy(int))
            o = frame[frame.is_project_old_drug]
            op = o[o.binary_label.eq(1)]
            counts[str(cutoff)][name] = dict(rows=len(frame), positive=int(frame.binary_label.sum()),
                old_rows=len(o), old_positive=int(o.binary_label.sum()),
                positive_old_drug_queries=op.drug_feature_index.nunique(),
                positive_old_target_queries=op.target_feature_index.nunique())
        tr = pd.read_csv(folder / 'TRAIN.csv.gz'); va = pd.read_csv(folder / 'VALIDATION.csv.gz')
        assert tr.merge(va, on=['drug_feature_index','target_feature_index']).empty
        if cutoff == 2020:
            original = pd.read_csv(SOURCE / 'DEVELOPMENT_TRAIN.csv.gz')
            pd.testing.assert_frame_equal(tr[original.columns.intersection(tr.columns)],
                                          original[original.columns.intersection(tr.columns)], check_dtype=False)
            assert np.array_equal(risk, np.load(SOURCE / 'RISK_SETS.npz')['development'])
    required = np.unique(np.r_[required, old.drug_feature_index]).astype(int)
    missing = np.setdiff1d(required, np.load(BASE / 'REQUIRED_MOLECULE_IDS.npy'))
    np.save(OUT / 'SUPPLEMENTAL_MOLECULE_IDS.npy', missing)
    write_json(OUT / 'ROLL_MANIFEST.json', dict(status='COMPLETE', counts=counts,
        cutoff_before_aggregation=True, exclude_all_prior_and_undated_from_risk=True,
        supplemental_molecules=len(missing), no_label_based_feature_exclusion=True,
        sources=[file_identity(SOURCE / n) for n in ['ANNUAL_STANDARDIZED.csv.gz','OLD_DRUG_INDEX.csv','TARGET_INDEX.csv.gz']],
        producer=file_identity(Path(__file__)), files=[file_identity(p) for p in sorted(OUT.glob('roll_*/*'))]))
    print(json.dumps(dict(event='rolls_complete', counts=counts, supplemental_molecules=len(missing))), flush=True)


def assays():
    """Only measured active/inactive compounds from an identical assay/endpoint."""
    if (OUT / 'ASSAY_MANIFEST.json').exists():
        return
    raw_path = OUT / 'ASSAY_ANNUAL.csv.gz'
    if not raw_path.exists():
        con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
        con.execute('PRAGMA temp_store=MEMORY')
        con.execute('CREATE TEMP TABLE target_scope (target_index INTEGER, tid INTEGER PRIMARY KEY)')
        target = pd.read_csv(SOURCE / 'TARGET_INDEX.csv.gz')
        pairs = [(int(r.target_feature_index), con.execute('SELECT tid FROM target_dictionary WHERE chembl_id=?',
                 (r.target_chembl_id,)).fetchone()[0]) for r in target.itertuples()]
        con.executemany('INSERT INTO target_scope VALUES (?,?)', pairs)
        original = (SOURCE / 'EXTRACTION.sql').read_text()
        from_where = original[original.index('    FROM target_scope'):original.index('    GROUP BY')]
        sql = '''SELECT t.target_index AS target_feature_index,
          COALESCE(mh.parent_molregno,a.molregno) AS parent_molregno,
          ass.assay_id, a.standard_type, a.standard_units, d.year AS document_year,
          COUNT(*) AS numeric_rows, SUM(a.pchembl_value) AS pchembl_sum,
          MIN(a.pchembl_value) AS min_pchembl, MAX(a.pchembl_value) AS max_pchembl
        ''' + from_where + '''
          AND d.year IS NOT NULL AND d.year<=2022
          AND a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='='
          AND a.pchembl_value IS NOT NULL AND a.standard_units IS NOT NULL
        GROUP BY t.target_index, COALESCE(mh.parent_molregno,a.molregno),
          ass.assay_id,a.standard_type,a.standard_units,d.year'''
        (OUT / 'ASSAY_EXTRACTION.sql').write_text(sql)
        print(json.dumps(dict(event='assay_extract_started')), flush=True)
        raw = pd.read_sql_query(sql, con); con.close()
        parent = pd.read_csv(SOURCE / 'PARENT_MAPPING.csv.gz')
        raw['drug_feature_index'] = raw.parent_molregno.map(parent.set_index('parent_molregno').drug_feature_index)
        missing = raw.drug_feature_index.isna()
        if missing.any():
            raw.loc[missing].to_csv(OUT / 'ASSAY_UNRESOLVED.csv.gz', index=False)
        raw = raw.loc[~missing].copy(); raw.drug_feature_index = raw.drug_feature_index.astype(int)
        raw.to_csv(raw_path, index=False)
    raw = pd.read_csv(raw_path)
    counts = {}
    for cutoff in [2018, 2020, 2022]:
        # Aggregate replicates only after date filtering. Exact assay, endpoint,
        # unit and target identity are preserved through salt standardization.
        keys = ['assay_id','standard_type','standard_units','target_feature_index','drug_feature_index']
        rows = raw[raw.document_year.le(cutoff)].groupby(keys, sort=True).agg(
            numeric_rows=('numeric_rows','sum'),pchembl_sum=('pchembl_sum','sum'),
            min_pchembl=('min_pchembl','min'),max_pchembl=('max_pchembl','max'),
            max_document_year=('document_year','max')).reset_index()
        trainpath = OUT / f'roll_{cutoff}/TRAIN.csv.gz' if cutoff < 2022 else SOURCE / 'FINAL_TRAIN.csv.gz'
        train = pd.read_csv(trainpath, usecols=['drug_feature_index','target_feature_index','binary_label'])
        rows = rows.merge(train, on=['drug_feature_index','target_feature_index'], validate='many_to_one')
        # A conservative tenfold threshold gap AND agreement with cutoff-only
        # relation labels. Grey/unreported compounds never become negatives.
        valid = (rows.min_pchembl.ge(6) & rows.binary_label.eq(1)) | (rows.max_pchembl.le(5) & rows.binary_label.eq(0))
        rows = rows.loc[valid].copy()
        rows['assay_group'] = rows.groupby(keys[:-1], sort=True).ngroup()
        eligible = rows.groupby('assay_group').binary_label.nunique().eq(2)
        rows = rows[rows.assay_group.isin(eligible[eligible].index)].copy()
        assert rows.max_document_year.max() <= cutoff
        assert rows.groupby('assay_group').target_feature_index.nunique().max() == 1
        assert not rows.duplicated(['assay_group','drug_feature_index']).any()
        rows.to_csv(OUT / f'ASSAY_CONTRASTS_{cutoff}.csv.gz', index=False)
        old = set(pd.read_csv(SOURCE / 'OLD_DRUG_INDEX.csv').drug_feature_index)
        counts[str(cutoff)] = dict(rows=len(rows), groups=rows.assay_group.nunique(),
            targets=rows.target_feature_index.nunique(), drugs=rows.drug_feature_index.nunique(),
            old_rows=int(rows.drug_feature_index.isin(old).sum()),
            positives=int(rows.binary_label.sum()))
    write_json(OUT / 'ASSAY_MANIFEST.json', dict(status='COMPLETE', counts=counts,
        max_extracted_year=2022, target_to_drug_only=True,
        group_definition='exact assay_id + target + standard_type + standard_units',
        eligibility='cutoff-only observed binary pairs, every replicate >=6 for positives or <=5 for negatives',
        no_cross_assay_potency_comparison=True, unknown_as_inactive=False,
        database=dict(path=str(DB),size=DB.stat().st_size,mtime_ns=DB.stat().st_mtime_ns),
        sources=[file_identity(SOURCE / 'PARENT_MAPPING.csv.gz'),file_identity(SOURCE / 'EXTRACTION.sql')],
        producer=file_identity(Path(__file__)), files=[file_identity(p) for p in sorted(OUT.glob('ASSAY_*.csv.gz'))]
        + [file_identity(OUT / 'ASSAY_EXTRACTION.sql')]))
    print(json.dumps(dict(event='assay_complete', counts=counts)), flush=True)


def supplement():
    import prepare_biomaster_unified_interaction as original
    folder = OUT / 'supplemental_features'; folder.mkdir(exist_ok=True)
    if (folder / 'ATOM_MANIFEST.json').exists():
        return
    # The original feature producer is unchanged. A separate source view only
    # requests the additional molecule IDs, without modifying its frozen bank.
    view = OUT / 'supplemental_source'; view.mkdir(exist_ok=True)
    molecule_link = view / 'MOLECULES.csv.gz'
    if not molecule_link.exists():
        molecule_link.symlink_to(SOURCE / 'MOLECULES.csv.gz')
    ids = np.load(OUT / 'SUPPLEMENTAL_MOLECULE_IDS.npy')
    pd.DataFrame({'drug_feature_index':ids}).to_csv(view / 'QUERY_POOL.csv.gz', index=False)
    pd.DataFrame({'drug_feature_index':[]}).to_csv(view / 'OLD_DRUG_INDEX.csv', index=False)
    original.SOURCE = view
    original.prepare_conformers(folder, workers=8)
    original.encode_atoms(folder)
    write_json(folder / 'SUPPLEMENT_PROVENANCE.json', dict(status='COMPLETE',
        base_manifest=file_identity(BASE / 'ATOM_MANIFEST.json'),
        requested_ids=file_identity(OUT / 'SUPPLEMENTAL_MOLECULE_IDS.npy'),
        producer=file_identity(Path(__file__)),original_producer=file_identity(Path(original.__file__)),
        labels_used=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['rolls','assays','supplement','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['rolls','all']: rolls()
    if args.stage in ['assays','all']: assays()
    if args.stage in ['supplement','all']: supplement()
