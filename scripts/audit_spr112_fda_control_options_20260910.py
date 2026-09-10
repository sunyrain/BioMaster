"""Read-only FDA-first control feasibility audit; does not replace frozen controls.

FDA names come from the local official product snapshot, not ChEMBL max_phase.
Raw activity retrieval is by exact ChEMBL entity, without parent activity pooling.
Name-only FDA matches remain provisional until chemical identity is reconciled.
"""
from pathlib import Path
import json
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/spr112_fda_control_audit_20260910'
FDA = ROOT / 'outputs/drug_universe_rebuild_v1/official_fda_source_audit_20260803/FDA_PRODUCT_INGREDIENTS_OPENFDA_20260803.csv'
MOIETY = ROOT / 'outputs/fda_drug_universe_2005_2026_v1/02_rxnorm_active_moiety/FDA_ACTIVE_MOIETY_TEMPORAL_MASTER_RXNORM_RESOLVED.csv'
REG = ROOT / 'outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv'
CONTROLS = ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR112_REFERENCE_CONTROLS.csv'
BASE = ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv'


def norm(s):
    return ' '.join(str(s).upper().split())


def main():
    OUT.mkdir(exist_ok=True)
    controls = pd.read_csv(CONTROLS).fillna('')
    products = pd.read_csv(FDA).fillna('')
    products = products[products.marketing_status.isin(['Prescription', 'Over-the-counter'])]
    products = products[products.application_type.isin(['NDA', 'ANDA'])]
    products = products[products.original_approval_date_from_submissions.ne('')]
    by_name = {}
    for name, group in products.groupby('ingredient_name_fda'):
        by_name[norm(name)] = set(group.application_number)
    for r in pd.read_csv(MOIETY).fillna('').itertuples():
        apps = set()
        for ingredient in r.precise_ingredient_names.split(';'):
            apps |= by_name.get(norm(ingredient), set())
        if apps:
            by_name.setdefault(norm(r.active_moiety_name), set()).update(apps)
    registry = pd.read_csv(REG).fillna('')
    by_key = {}
    live_apps = set(products.application_number)
    for r in registry.itertuples():
        if str(r.exact_structure_in_fda_registry).lower() != 'true':
            continue
        apps = set(r.all_fda_nda_application_numbers.split(';')) & live_apps
        # Some registry lists contain spaces after separators.
        apps |= {x.strip() for x in r.all_fda_nda_application_numbers.split(';')} & live_apps
        if apps:
            by_key[r.ligand_inchikey] = apps
    db = sqlite3.connect(f'file:{ROOT}/downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db?mode=ro', uri=True)
    molecules = pd.read_sql_query('''select md.molregno,md.chembl_id,md.pref_name,
        md.withdrawn_flag,cs.standard_inchi_key from molecule_dictionary md
        join compound_structures cs using(molregno)
        where md.molecule_type='Small molecule' ''', db).fillna('')
    drugs = []
    for r in molecules.itertuples():
        if r.withdrawn_flag == 1:
            continue
        exact = r.standard_inchi_key in by_key
        apps = by_key.get(r.standard_inchi_key, by_name.get(norm(r.pref_name), set()))
        if not apps or not r.pref_name:
            continue
        drugs.append(dict(molregno=r.molregno, control_chembl_id=r.chembl_id,
            control_name=r.pref_name, control_inchikey=r.standard_inchi_key,
            fda_identity_tier='项目FDA注册表完整InChIKey匹配' if exact else 'FDA成分/活性母体名称匹配；结构身份待复核',
            exact_fda_structure_match=exact, fda_applications='; '.join(sorted(apps)),
            fda_urls='; '.join('https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo='+''.join(ch for ch in x if ch.isdigit()) for x in sorted(apps)[:3])))
    drugs = pd.DataFrame(drugs)
    drugs.to_csv(OUT/'FDA_DRUG_IDENTITY_CANDIDATES.csv', index=False, encoding='utf-8-sig')
    db.execute('create temp table fda_molecules(molregno integer primary key)')
    db.executemany('insert into fda_molecules values (?)', [(int(x),) for x in drugs.molregno])
    tids = controls['新靶点ChEMBL编号'].tolist()
    query = '''select td.chembl_id target_chembl_id, ac.molregno,ac.activity_id,
        ac.standard_type,ac.standard_relation,ac.standard_value,ac.standard_units,
        ac.data_validity_comment,a.chembl_id assay_chembl_id,a.assay_type,
        a.confidence_score,a.relationship_type,a.description,vs.mutation,vs.isoform,
        d.doi,d.pubmed_id from fda_molecules fm join activities ac on ac.molregno=fm.molregno
        join assays a on a.assay_id=ac.assay_id join target_dictionary td on td.tid=a.tid
        left join variant_sequences vs on vs.variant_id=a.variant_id
        left join docs d on d.doc_id=ac.doc_id
        where td.chembl_id in ('''+','.join('?' for _ in tids)+''')
        and td.organism='Homo sapiens' and td.target_type='SINGLE PROTEIN'
        and ac.standard_type in ('Kd','Ki','IC50') and ac.standard_units='nM'
        and ac.standard_value>0 and a.confidence_score=9 and a.relationship_type='D'
        and a.assay_type='B'
        and (ac.data_validity_comment is null or ac.data_validity_comment='')
        and (vs.mutation is null or vs.mutation='')'''
    raw = pd.read_sql_query(query, db, params=tids).merge(drugs, on='molregno')
    raw.to_csv(OUT/'RAW_EXACT_ENTITY_ACTIVITY.csv', index=False, encoding='utf-8-sig')
    options = []
    for (target, mol), g in raw.groupby(['target_chembl_id', 'molregno']):
        for endpoint in ['Kd', 'Ki', 'IC50']:
            e = g[g.standard_type.eq(endpoint)]
            numeric = e[e.standard_relation.eq('=')].standard_value
            positive = e[e.standard_relation.isin(['=', '<', '<=']) & e.standard_value.le(1000)]
            weak = e[e.standard_relation.isin(['=', '>', '>=']) & e.standard_value.ge(10000)]
            if positive.empty:
                continue
            r = positive.sort_values('activity_id').iloc[0]
            info = {k: r[k] for k in ['control_chembl_id','control_name','control_inchikey','fda_identity_tier','exact_fda_structure_match','fda_applications','fda_urls']}
            options.append(dict(target_chembl_id=target, **info,
                endpoint=endpoint, numeric_median_nM=float(numeric.median()) if len(numeric) else None,
                positive_rows=len(positive), weak_rows=len(weak),
                affinity_conflict=bool(len(weak)),
                representative_value_nM=float(r.standard_value), representative_relation=r.standard_relation,
                assay_url='https://www.ebi.ac.uk/chembl/explore/assay/'+r.assay_chembl_id,
                assay_description=r.description, doi=r.doi, pubmed_id=r.pubmed_id,
                status='存在强弱矛盾，先人工复核' if len(weak) else '直接亲和候选，构建与SPR适用性待确认' if endpoint in ['Kd','Ki'] else '仅结合类IC50，SPR适用性需另证'))
            # Keep endpoint-specific evidence; choose per target below.
    opt = pd.DataFrame(options)
    opt['endpoint_order'] = opt.endpoint.map({'Kd':0,'Ki':1,'IC50':2})
    opt = opt.sort_values(['affinity_conflict','endpoint_order','exact_fda_structure_match','numeric_median_nM'], ascending=[True,True,False,True])
    opt.to_csv(OUT/'FDA_CONTROL_OPTIONS.csv', index=False, encoding='utf-8-sig')
    rows = []
    for r in controls.to_dict('records'):
        g = opt[opt.target_chembl_id.eq(r['新靶点ChEMBL编号']) & ~opt.affinity_conflict]
        direct = g[g.endpoint.isin(['Kd','Ki'])]
        exact = direct[direct.exact_fda_structure_match]
        pool = exact if len(exact) else direct if len(direct) else g
        row = dict(新靶点名称=r['新靶点名称'], 新靶点基因=r['新靶点基因'], 新靶点ChEMBL编号=r['新靶点ChEMBL编号'],
            原拟用对照=r['新靶点的已知药物名称'], 原对照ChEMBL编号=r['对照ChEMBL编号'])
        if len(pool):
            chosen = pool.iloc[0].to_dict()
            row.update(chosen)
            row['覆盖类别'] = ('A_完整FDA结构匹配且有Kd或Ki线索' if len(exact) else
                'B_FDA名称匹配且有Kd或Ki线索_身份待复核' if len(direct) else 'C_仅结合类IC50线索')
        else:
            row['覆盖类别'] = 'D_本次未找到合格FDA线索_不代表不存在'
            conflict = opt[opt.target_chembl_id.eq(r['新靶点ChEMBL编号'])]
            row['未入选原因'] = ('有达标活性记录，但同端点也有弱活性记录，需按实验体系拆分复核：'+
                '; '.join(sorted(set(conflict.control_name))) if len(conflict) else
                '在本次身份、单人源蛋白、端点及阈值口径下未找到；不是无FDA药的结论')
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT/'SPR112_FDA_CONTROL_FEASIBILITY.csv', index=False, encoding='utf-8-sig')
    base = pd.read_csv(BASE)
    candidate = base.merge(audit[['新靶点ChEMBL编号','覆盖类别']], on='新靶点ChEMBL编号', validate='many_to_one')
    checks = dict(targets=len(audit), candidate_rows=len(candidate),
        target_categories=audit['覆盖类别'].value_counts().to_dict(),
        candidate_categories=candidate['覆盖类别'].value_counts().to_dict(),
        fda_identity_candidates=len(drugs), raw_activity_rows=len(raw),
        qualified_option_rows=len(opt), fda_snapshot='2026-08-03',
        replaces_frozen_controls=False,
        limitations=['FDA当前状态依据本地官方快照，未逐一在线核验近期变更',
          'B类仅名称映射，不作为已确认FDA精确结构',
          'Kd/Ki线索不等于SPR已测通；未逐篇排查所有描述中未结构化标注的突变、构建及测量问题',
          '未覆盖不等于无FDA药；不含抗体、复合物靶点、代谢活性物种替代或功能实验外推',
          '阈值Kd/Ki/IC50不超过1µM；同端点若出现不低于10µM反例，则不纳入初选'])
    assert len(audit)==112 and len(candidate)==384
    (OUT/'AUDIT.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
