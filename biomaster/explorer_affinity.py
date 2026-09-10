"""Read-only, evidence-based atlas. No prediction is turned into a measurement."""
from collections import Counter, defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

DB = 'data/processed/biomaster_affinity_evidence_20260910.sqlite'


def classify(relation, value, threshold):
    """Classify bounds against <= threshold without treating missing as negative."""
    if value is None or not math.isfinite(value) or value <= 0: return 'uncertain'
    if relation == '=': return 'lower' if value <= threshold else 'higher'
    if relation in ('<', '<=') and value <= threshold: return 'lower'
    if relation == '>' and value >= threshold: return 'higher'
    if relation == '>=' and value > threshold: return 'higher'
    return 'uncertain'


class AffinityAtlas:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / DB

    def connect(self):
        if not self.path.exists(): raise FileNotFoundError('亲和证据数据库尚未就绪')
        conn = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def inventory(self):
        with closing(self.connect()) as db:
            sources = [dict(r) for r in db.execute('SELECT source,endpoint,count(*) AS rows FROM measurements GROUP BY source,endpoint')]
            auxiliary = db.execute('SELECT count(*) FROM auxiliary_observations').fetchone()[0]
            qc = [dict(r) for r in db.execute('SELECT * FROM evidence_quality_reviews')]
            sizes = dict(db.execute('SELECT source,count(*) FROM measurements GROUP BY source').fetchall())
            evidence = list(db.execute('''SELECT ligand_key,target_chembl_id,endpoint,relation FROM comparable_numeric_evidence m
                WHERE NOT EXISTS (SELECT 1 FROM evidence_quality_reviews q WHERE q.source=m.source AND q.record_id=m.record_id)'''))
        pairs = lambda rows: len({(r['ligand_key'], r['target_chembl_id']) for r in rows})
        legacy = []
        definitions = [
            ('BindingDB 202608 历史全包', 'outputs/biomaster_bindingdb_full_training_subset_v1/BINDINGDB_TRAINING_SUBSET_AUDIT_V1.json', ('row_counts','raw_rows'), '原始物理行；与 202609 重叠，不累加'),
            ('BindingDB 历史筛选配对', 'outputs/biomaster_bindingdb_full_training_subset_v1/BINDINGDB_TRAINING_SUBSET_AUDIT_V1.json', ('pair_counts','all_unique_pairs_after_identity_filter'), '身份筛选后的配对；非全部亲和阳性'),
            ('BindingDB 直接 Ki/Kd 训练子集', 'outputs/biomaster_bindingdb_full_training_subset_v1/BINDINGDB_TRAINING_SUBSET_AUDIT_V1.json', ('pair_counts','high_confidence_direct_ki_kd_pairs'), '上级数据的 9,778 对子集，不累加'),
            ('Davis 本地完整配对清单', 'outputs/biomaster_odti_davis_entity_cold_v1/DAVIS_ENTITY_COLD_SCORING_SUMMARY_V1.json', ('candidate','raw_manifest_unique_pairs'), '25,772 对；排除旧库重叠后 24,962 对；非明确阴性基准'),
            ('GtoPdb 外部严格阳性集', 'outputs/biomaster_odti_gtopdb_entity_cold_gap_v1/GTOPDB_ENTITY_COLD_GAP_AUDIT_V1.json', ('counts','strict_positive_pairs'), '项目外部评估子集，不是 GtoPdb 全库规模'),
        ]
        for name, path, keys, note in definitions:
            p = self.root / path
            if p.exists():
                d = json.loads(p.read_text())
                for key in keys: d = d[key]
                legacy.append(dict(name=name, count=d, note=note, path=path))
        candidates = self.root / 'outputs/biomaster_odti_local_external_candidates_v1/LOCAL_EXTERNAL_ENTITY_COLD_CANDIDATES_V1.json'
        if candidates.exists():
            source_info = json.loads(candidates.read_text())['sources']
            for key, name, note in [('davis_complete_secondary_hf', 'Davis 原始数值行', '30,056 行 / 25,772 唯一配对；地板值不是明确不结合'),
                                    ('platinum_official', 'Platinum 突变资源', '突变亲和变化；身份未完整，不并入野生型矩阵'),
                                    ('mdrdb', 'MdrDB 本地 CoreSet', '突变 ΔΔG 资源；非完整 MdrDB，不能当作野生型 Kd')]:
                if key in source_info:
                    item = source_info[key]
                    legacy.append(dict(name=name,count=item['rows'],note=note,path=item['source']))
        acquisition = self.root / 'outputs/evidence_routing_compute_execution_20260808_v1/gtopdb_external_v11/GTOPDB_RAW_ACQUISITION_MANIFEST_V11.json'
        if acquisition.exists():
            for path, info in json.loads(acquisition.read_text())['files'].items():
                if path.endswith('/interactions.csv'):
                    legacy.append(dict(name='GtoPdb 2026.2 原始相互作用',count=info['data_rows'],note='全部相互作用行，不全是数值亲和；17 对为其严格外部阳性子集',path=path))
        notes_path = self.root / 'downloads/chembl_37/chembl_37_release_notes.txt'
        if notes_path.exists():
            import re
            match = re.search(r'([\d,]+) activities', notes_path.read_text())
            if match: legacy.append(dict(name='ChEMBL37 全库活性（发行说明）',count=int(match[1].replace(',','')),note='活性记录，不全是亲和常数；59,798 行为当前证据库的项目子集',path=str(notes_path.relative_to(self.root))))
        chembl_path = self.root / 'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
        if chembl_path.exists():
            with closing(sqlite3.connect(chembl_path.resolve().as_uri() + '?mode=ro', uri=True)) as chembl:
                actual = chembl.execute('SELECT count(*) FROM activities').fetchone()[0]
            legacy = [r for r in legacy if not r['name'].startswith('ChEMBL37 全库活性')]
            legacy.append(dict(name='ChEMBL37 全库活性（本地核验）',count=actual,note='活性记录，不全是亲和常数；59,798 行为当前证据库的项目子集',path=str(chembl_path.relative_to(self.root))))
        old_gtop = self.root / 'outputs/evidence_routing_compute_execution_20260808_v1/gtopdb_release_delta_v12/GTOPDB_RELEASE_DELTA_SUMMARY_V12.json'
        if old_gtop.exists():
            g = json.loads(old_gtop.read_text())
            legacy.append(dict(name='GtoPdb 2025.4 历史匹配行',count=g['old_release_identity_coverage']['matched_old_interaction_rows'],note='对当前 17 对的历史抽取；17 对在旧版均已有数值阳性，不作为新版新增实验',path=str(old_gtop.relative_to(self.root))))
        raw_audit = self.root / 'outputs/affinity_evidence_refresh_20260910/BINDINGDB_RAW_LINE_AUDIT.json'
        if raw_audit.exists():
            raw_counts = json.loads(raw_audit.read_text())
            legacy.append(dict(name='BindingDB 202609 全包有效首字段行',count=raw_counts['physical_data_lines']-raw_counts['non_numeric_first_field_count'],note='本地物理行审计；70,585 为项目邻域端点行。分包补充、历史版本与全包不能相加',path=str(raw_audit.relative_to(self.root))))
        return dict(source_rows=sizes, endpoint_inventory=sources, total_rows=sum(sizes.values()), auxiliary_rows=auxiliary,
                    exact_rows=len(evidence), exact_pairs=pairs(evidence), kd_ki_pairs=pairs(r for r in evidence if r['endpoint'] in ('Kd','Ki')),
                    kd_equal_pairs=pairs(r for r in evidence if r['endpoint']=='Kd' and r['relation']=='='),
                    quality_reviews=qc, legacy=legacy, database=DB,
                    notes=['来源记录数不是独立实验次数；历史版本、训练子集和结构附带标签不能相加。',
                           'TxGNN、Open Targets、DrugCLIP、DTIAM、ConPlex 与 ReTargetMap 不计入实测亲和总数。',
                           'GatorAffinity 元数据未入库；预测结构及重复 BindingDB 标签不计入新增实验。'])

    def matrix(self, data, spr_store, endpoint='Kd', threshold=1000., source='all'):
        if endpoint not in ('Kd', 'Ki'): raise ValueError('亲和判定只支持单独查看 Kd 或 Ki')
        if not math.isfinite(threshold) or not 0 < threshold <= 1e9: raise ValueError('阈值须为 0 到 1e9 之间的有限正数（nM）')
        with closing(self.connect()) as db:
            sources = [r[0] for r in db.execute('SELECT DISTINCT source FROM measurements ORDER BY source')]
            if source not in ['all', *sources]: raise ValueError('未知的证据来源')
            drugs = [dict(id=r[0], name=r[1]) for r in db.execute('SELECT ligand_inchikey,drug_names FROM project_drugs ORDER BY lower(drug_names),ligand_inchikey')]
            targets = [dict(id=r[0], name=r[1]) for r in db.execute('SELECT target_chembl_id,gene_symbol FROM project_targets ORDER BY gene_symbol,target_chembl_id')]
            query = '''SELECT m.id,m.ligand_key,m.target_chembl_id,m.endpoint,m.relation,m.value_nM,m.source
                FROM comparable_numeric_evidence m WHERE NOT EXISTS
                (SELECT 1 FROM evidence_quality_reviews q WHERE q.source=m.source AND q.record_id=m.record_id)'''
            records = db.execute(query + ('' if source=='all' else ' AND m.source=?'), () if source=='all' else (source,)).fetchall()
        di = {r['id']: i for i,r in enumerate(drugs)}; ti = {r['id']: i for i,r in enumerate(targets)}
        cells = defaultdict(lambda: [0,0,0,0,0,0]) # evidence rows, <=, >, uncertain, planned, submitted
        for r in records:
            key=(di[r['ligand_key']], ti[r['target_chembl_id']]);cell=cells[key];cell[0]+=1
            if r['endpoint']==endpoint: cell[{'lower':1,'higher':2,'uncertain':3}[classify(r['relation'],r['value_nM'],threshold)]]+=1
        data.ensure_loaded()
        planned = {r['pair_id']: r for t in data.targets.values() for r in t.get('experiments', [])}
        outside = 0
        for r in planned.values():
            if r['drug_id'] not in di or r['target_id'] not in ti: outside+=1;continue
            cells[(di[r['drug_id']],ti[r['target_id']])][4] = 1
        submitted = spr_store.results(export=True)['items'];outside_upload=0
        for r in submitted:
            if r['drug_id'] not in di or r['target_id'] not in ti: outside_upload+=1;continue
            cells[(di[r['drug_id']],ti[r['target_id']])][5]+=1
        counts=Counter()
        for v in cells.values():
            if v[0]:counts['external_measured']+=1
            if v[5]:counts['uploaded_pairs']+=1
            if v[4] and not v[5]:counts['planned_without_upload']+=1
            if v[1] and v[2]:counts['conflicting']+=1
            elif v[1]:counts['lower']+=1
            elif v[2]:counts['higher']+=1
            elif v[0] or v[5]:counts['uncertain']+=1
        return dict(drugs=drugs,targets=targets,cells=[[d,t,*v] for (d,t),v in sorted(cells.items())],
                    cell_fields=['drug_index','target_index','numeric_rows','lower','higher','uncertain','planned','uploaded'],
                    counts=dict(counts),sources=sources,endpoint=endpoint,threshold_nM=threshold,source=source,
                    outside_planned_pairs=outside,outside_uploaded_rows=outside_upload,uploaded_rows=len(submitted),
                    inventory=self.inventory())

    def pair(self, drug, target, spr_store):
        with closing(self.connect()) as db:
            rows = [dict(r) for r in db.execute('''SELECT m.id,m.source,m.record_id,m.endpoint,m.relation,m.value_nM,m.method,m.quality,m.doi,m.pmid,m.source_url,m.target_scope,
                (SELECT group_concat(q.decision || ': ' || q.reason, '; ') FROM evidence_quality_reviews q WHERE q.source=m.source AND q.record_id=m.record_id) AS quality_review
                FROM comparable_numeric_evidence m WHERE ligand_key=? AND target_chembl_id=? ORDER BY endpoint,source,m.id''', (drug,target))]
        return dict(items=rows,total=len(rows),uploaded=spr_store.results(pair=f'{drug}__{target}',export=True)['items'])
