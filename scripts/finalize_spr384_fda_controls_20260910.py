"""Publish curated FDA-first control revision, preserving candidate identities/order."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/spr384_final_experiment_table_20260910'
ARCHIVE = ROOT / 'outputs/spr384_final_experiment_table_before_fda_v2_20260910'
SPEC = ROOT / 'data/curation/spr112_fda_control_replacements_20260910.json'
AUDIT = ROOT / 'outputs/spr112_fda_control_audit_20260910'
VERSION = '20260910_FDA_V2'
FIELDS = ['小分子药物名称','旧靶点名称','新靶点名称','新靶点的已知药物名称']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not ARCHIVE.exists():
        assert json.loads((OUT/'MANIFEST.json').read_text()).get('version') != VERSION
        shutil.copytree(OUT, ARCHIVE)
    old_manifest = json.loads((ARCHIVE/'MANIFEST.json').read_text())
    for name in ['SPR384_FINAL_DETAILED.csv','SPR112_REFERENCE_CONTROLS.csv','CONTROL_SOURCE_ACTIVITY_EVIDENCE.csv']:
        assert sha(ARCHIVE/name) == old_manifest['output_sha256'][name]
    stage = OUT / '.fda_v2_stage'
    stage.mkdir(exist_ok=True)
    original = pd.read_csv(ARCHIVE/'SPR384_FINAL_DETAILED.csv').fillna('')
    d = original.copy()
    ct = pd.read_csv(ARCHIVE/'SPR112_REFERENCE_CONTROLS.csv').fillna('')
    raw = pd.read_csv(AUDIT/'RAW_EXACT_ENTITY_ACTIVITY.csv').fillna('')
    identities = pd.read_csv(AUDIT/'FDA_DRUG_IDENTITY_CANDIDATES.csv').fillna('')
    options = pd.read_csv(AUDIT/'FDA_CONTROL_OPTIONS.csv').fillna('')
    plans = json.loads(SPEC.read_text())
    registry = pd.read_csv(ROOT/'outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv').fillna('').set_index('ligand_inchikey')
    db = sqlite3.connect(f'file:{ROOT}/downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db?mode=ro',uri=True)
    ct['pair_id'] = ct['原pair_id']
    ct['原对照编号'] = ct['对照编号']
    ct['替换前对照名称'] = ct['新靶点的已知药物名称']
    ct['替换前对照ChEMBL编号'] = ct['对照ChEMBL编号']
    ct['对照修订版本'] = VERSION
    ct['本次FDA替换'] = False
    for col in ['FDA身份状态','FDA审批申请号','FDA审批来源','对照构建适用条件','替换依据','选用参考activity_id','选用参考端点','选用参考值_nM','选用参考实验描述','对照原靶点名称','对照原靶点编号','对照原作用机制']:
        ct[col] = ''
    new_evidence = []
    for plan in plans:
        row = raw[raw.activity_id.eq(plan['activity_id'])]
        assert len(row) == 1
        r = row.iloc[0]
        i = ct.index[ct['新靶点基因'].eq(plan['gene'])].item()
        assert r.target_chembl_id == ct.at[i,'新靶点ChEMBL编号']
        assert r.control_name == plan['name'] and r.standard_type == plan['endpoint']
        assert r.exact_fda_structure_match and r.standard_relation == '=' and 0 < r.standard_value <= 1000
        op = options[options.target_chembl_id.eq(r.target_chembl_id)&options.control_chembl_id.eq(r.control_chembl_id)&options.endpoint.eq(r.standard_type)]
        assert len(op) and not op.affinity_conflict.any()
        assert 'kinobead' not in r.description.lower() and 'lysate' not in r.description.lower()
        assert ct.at[i,'对照完整InChIKey'] != r.control_inchikey
        structure = db.execute('select canonical_smiles,standard_inchi_key from compound_structures where molregno=?',(int(r.molregno),)).fetchone()
        assert structure[1] == r.control_inchikey
        assay = db.execute('select a.assay_id,a.chembl_id,doc.chembl_id from activities ac join assays a using(assay_id) left join docs doc on ac.doc_id=doc.doc_id where activity_id=?',(int(r.activity_id),)).fetchone()
        category = 'Kd_RECORD_PRESENT' if r.standard_type == 'Kd' else 'Ki_NO_Kd_RECORD'
        status = 'FDA药物拟用对照；' + ('有选用Kd结合记录' if r.standard_type == 'Kd' else '本次依据Ki记录，非SPR实测Kd') + '；待实验室确认构建并测通'
        updates = {'对照编号':ct.at[i,'原对照编号'].replace('CTRL-','CTRL-FDA-'),
            '新靶点的已知药物名称':r.control_name,'对照身份':'FDA获批药物精确结构已匹配',
            '对照证据类型':r.standard_type,'对照确认状态':status,'对照ChEMBL编号':r.control_chembl_id,
            '对照完整InChIKey':r.control_inchikey,'对照SMILES':structure[0],
            '精确对照原始证据已匹配':True,'参考证据层级':category,
            '真实ChEMBL实验编号':assay[1],'真实ChEMBL文献编号':assay[2] or '',
            '证据文献DOI':r.doi,'pair_id':f'{r.control_inchikey}__{r.target_chembl_id}',
            '本次FDA替换':True,'对照构建适用条件':plan['construct_requirement'],
            '替换依据':plan['decision'],'选用参考activity_id':str(plan['activity_id']),
            '选用参考端点':r.standard_type,'选用参考值_nM':str(r.standard_value),'选用参考实验描述':r.description}
        for k,v in updates.items():
            ct.at[i,k] = v
        new_evidence.append(dict(target_chembl_id=r.target_chembl_id,control_chembl_id=r.control_chembl_id,
            control_name=r.control_name,activity_id=int(r.activity_id),standard_type=r.standard_type,
            standard_relation=r.standard_relation,standard_value=r.standard_value,standard_units='nM',
            assay_id=assay[0],assay_chembl_id=assay[1],document_chembl_id=assay[2],description=r.description,
            doi=r.doi,pubmed_id=r.pubmed_id,assay_type=r.assay_type,confidence_score=r.confidence_score,
            relationship_type=r.relationship_type,selected_for_fda_revision=True))
    for i,r in ct.iterrows():
        hits = identities[identities.control_inchikey.eq(r['对照完整InChIKey'])]
        exact = hits[hits.exact_fda_structure_match]
        hit = exact.iloc[0] if len(exact) else hits.iloc[0] if len(hits) else None
        ct.at[i,'FDA身份状态'] = 'FDA获批身份已匹配（完整结构；2026-08-03官方快照）' if len(exact) else 'FDA名称匹配；精确结构身份待复核' if len(hits) else '本轮未确认FDA批准身份'
        if hit is not None:
            ct.at[i,'FDA审批申请号'] = hit.fda_applications
            ct.at[i,'FDA审批来源'] = hit.fda_urls
        if r['对照完整InChIKey'] in registry.index:
            origin = registry.loc[r['对照完整InChIKey']]
            assert isinstance(origin,pd.Series)
            for dest,source in [('对照原靶点名称','known_target_names'),('对照原靶点编号','known_target_chembl_ids'),('对照原作用机制','known_mechanism_of_action')]:
                ct.at[i,dest] = origin[source]
    ct.to_csv(stage/'SPR112_REFERENCE_CONTROLS.csv',index=False,encoding='utf-8-sig')
    evidence = pd.read_csv(ARCHIVE/'CONTROL_SOURCE_ACTIVITY_EVIDENCE.csv')
    changed = ct[ct['本次FDA替换']]
    evidence = evidence[~evidence.target_chembl_id.isin(changed['新靶点ChEMBL编号'])]
    pd.concat([evidence,pd.DataFrame(new_evidence)],ignore_index=True).to_csv(stage/'CONTROL_SOURCE_ACTIVITY_EVIDENCE.csv',index=False,encoding='utf-8-sig')
    lookup = ct.set_index('新靶点ChEMBL编号')
    columns = ['新靶点的已知药物名称','对照编号','对照身份','对照证据类型','对照确认状态','对照ChEMBL编号','对照完整InChIKey','真实ChEMBL实验编号','真实ChEMBL文献编号','证据文献DOI','FDA身份状态','FDA审批申请号','FDA审批来源','对照构建适用条件','替换依据','本次FDA替换','对照修订版本','选用参考activity_id','选用参考端点','选用参考值_nM','选用参考实验描述']
    for col in columns:
        d[col] = d['新靶点ChEMBL编号'].map(lookup[col])
    d['对照pair_id'] = d['新靶点ChEMBL编号'].map(lookup['pair_id'])
    d['新靶点实验参考化合物名称'] = d['新靶点的已知药物名称']
    d['第四列用途'] = 'FDA优先修订版拟用对照；具体构建及SPR实测活性仍需实验室确认'
    short = list(pd.read_csv(ARCHIVE/'SPR384_FINAL_EXPERIMENT_TABLE.csv',nrows=0).columns)
    short += ['FDA身份状态','FDA审批申请号','FDA审批来源','对照构建适用条件','本次FDA替换','对照修订版本','对照pair_id']
    d[short].to_csv(stage/'SPR384_FINAL_EXPERIMENT_TABLE.csv',index=False,encoding='utf-8-sig')
    d[FIELDS].to_csv(stage/'SPR384_FINAL_FOUR_COLUMNS.csv',index=False,encoding='utf-8-sig')
    d.to_csv(stage/'SPR384_FINAL_DETAILED.csv',index=False,encoding='utf-8-sig')
    changed.to_csv(stage/'FDA_CONTROL_REPLACEMENTS.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(stage/'SPR384_FINAL_EXPERIMENT_TABLE.xlsx') as writer:
        d[short].to_excel(writer,sheet_name='384候选_按优先级',index=False)
        ct.to_excel(writer,sheet_name='112对照_另计',index=False)
        changed.to_excel(writer,sheet_name='FDA对照替换记录',index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    checks = dict(candidate_rows=len(d),candidate_pairs=d.pair_id.nunique(),targets=ct['新靶点ChEMBL编号'].nunique(),
        control_rows=len(ct),control_pairs=ct.pair_id.nunique(),replaced_controls=len(changed),
        affected_candidate_rows=int(d['本次FDA替换'].sum()),
        same_frozen_candidate_set=set(d.pair_id)==set(original.pair_id),priority_unchanged=d.pair_id.tolist()==original.pair_id.tolist(),
        candidate_drug_target_fields_unchanged=d[FIELDS[:3]].equals(original[FIELDS[:3]]),
        control_candidate_pairs_disjoint=not bool(set(ct.pair_id)&set(d.pair_id)),
        new_control_ids_unique=ct['对照编号'].nunique()==112,
        fda_status_counts=ct['FDA身份状态'].value_counts().to_dict(),
        replacement_endpoints=changed['对照证据类型'].value_counts().to_dict(),
        all_replacements_exact_fda_identity=bool(changed['FDA身份状态'].str.startswith('FDA获批身份已匹配').all()),
        all_replacement_entities_have_selected_raw_activity=bool(changed['精确对照原始证据已匹配'].all()),
        unique_control_compounds=ct['对照完整InChIKey'].nunique(),
        exact_control_activity_unresolved=int((ct['精确对照原始证据已匹配'].astype(str).str.lower()!='true').sum()))
    assert checks['candidate_rows']==checks['candidate_pairs']==384
    assert checks['control_rows']==checks['control_pairs']==checks['targets']==112
    assert checks['replaced_controls']==len(plans)==31
    for key in ['same_frozen_candidate_set','priority_unchanged','candidate_drug_target_fields_unchanged','control_candidate_pairs_disjoint','new_control_ids_unique','all_replacements_exact_fda_identity','all_replacement_entities_have_selected_raw_activity']:
        assert checks[key],key
    (stage/'README.md').write_text(f'''# SPR384 最终实验表 · FDA优先修订版

版本：{VERSION}。384候选及其优先顺序保持原基线；本次将31个靶点的拟用对照替换为精确身份匹配FDA注册表且有选定Kd/Ki记录的药物，涉及{checks['affected_candidate_rows']}行候选。FDA药物身份主要依据2026-08-03官方产品快照及项目精确结构注册表；不以ChEMBL max_phase代替FDA审批。

本次选定证据：{checks['replacement_endpoints']}。Kd可能来自竞争结合或ITC，Ki可能为酶学或竞争结合；这些不等于本批SPR实测结果。表中构建条件是使用要求，当前未收到实验室实际蛋白边界、亚型和批次活性确认，因此不能称为已测通或已放行。

使用SPR384_FINAL_EXPERIMENT_TABLE.csv或Excel（384候选、112另计对照、31项替换记录）。第四列为本行新靶点拟用对照。仅四列CSV不包含确认条件，采购和实验安排应使用主表。仍保留原对照的靶点不代表已确认FDA身份或已验证SPR适用性。

替换对照使用新的CTRL-FDA-编号及精确pair_id；原编号、原pair_id和替换前名称保留在对照表，旧版完整归档于{ARCHIVE.relative_to(ROOT)}。历史实测记录不重写。网站及新结果模板使用新编号；之前下载的对照模板应重新下载。

每项替换只采用人工选定的activity及其真实ChEMBL assay/document编号，见CONTROL_SOURCE_ACTIVITY_EVIDENCE.csv和FDA_CONTROL_REPLACEMENTS.csv。没有将同药其他靶点、细胞裂解液Kinobeads、JH2域或含前药活化条件的活性直接移用。本次JAK1选择ruxolitinib的JH1构建837–1142酶学Ki；TYK2同样要求JH1；PPARG使用配体竞争Ki，未移用共激活肽结合读数。

生成脚本：scripts/finalize_spr384_fda_controls_20260910.py。人工选择：data/curation/spr112_fda_control_replacements_20260910.json。
''')
    manifest = dict(version=VERSION,checks=checks,baseline_sha256=old_manifest['baseline_sha256'],
        previous_manifest_sha256=sha(ARCHIVE/'MANIFEST.json'),curation_sha256=sha(SPEC),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SPEC,AUDIT/'RAW_EXACT_ENTITY_ACTIVITY.csv',AUDIT/'FDA_DRUG_IDENTITY_CANDIDATES.csv',AUDIT/'FDA_CONTROL_OPTIONS.csv']},
        output_sha256={p.name:sha(p) for p in stage.iterdir() if p.name!='MANIFEST.json'})
    (stage/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # Readers verify both tables against the manifest; publish the manifest last.
    for p in stage.iterdir():
        if p.name!='MANIFEST.json':os.replace(p,OUT/p.name)
    os.replace(stage/'MANIFEST.json',OUT/'MANIFEST.json')
    stage.rmdir()
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
