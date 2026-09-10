"""Additional review pool and actionable baseline opinions; never changes frozen design."""
import csv
import hashlib
import json
from pathlib import Path

DIR='outputs/spr_expanded_review_20260910'
ACTIONS={'EXPLORABLE':'仍可探索','FIRST_RESOLVE':'先解决具体问题','EVIDENCE_DEPRIORITIZE':'有证据支持降低投入'}
TAGS={'ONLY_EVIDENCE_GAP':'仅额外证据不足','ADVERSE_ACTIVITY':'不利活性或选择性','PRIOR_ACTIVITY':'已有活动记录','NOVELTY_ONLY':'新颖性取舍','ENTITY_MISMATCH':'实体需核实','ASSAY_UNRESOLVED':'实验体系待解决','DISEASE_DIRECTION':'疾病作用方向','INDIRECT_OR_WRONG_ENTITY_EVIDENCE':'间接或其他实体证据'}

def _load(path):
    return json.loads(path.read_text()) if path.exists() else []

def validate_reason(row):
    if row.get('action_class') not in ACTIONS or not isinstance(row.get('reason_tags'),list):
        raise ValueError('SPR action review missing classification')
    if any(not row.get(k) for k in ['reason_detail','can_test','critical_condition']):
        raise ValueError('SPR action review missing explanation')
    return {**row,'action_label':ACTIONS[row['action_class']],'reason_labels':[TAGS.get(t,t) for t in row['reason_tags']]}

def baseline_reasons(root):
    root=Path(root);path=root/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv'
    if not path.exists():return {}
    expected={r['candidate_id']:r['pair_id'] for r in csv.DictReader(path.open())};result={}
    for i in (1,2):
        for row in _load(root/DIR/f'BASELINE_REASON_{i}.json'):
            if expected.get(row['candidate_id'])!=row['pair_id'] or row['pair_id'] in result:raise ValueError('SPR action review identity mismatch')
            result[row['pair_id']]=validate_reason(row)
    return result

def expanded_items(root):
    from .explorer_spr_reviews import LABELS
    root=Path(root);directory=root/DIR;path=directory/'NEW_CANDIDATES_FOR_LLM.csv'
    if not (directory/'PUBLISH_READY.json').exists():return []
    manifest=json.loads((directory/'INPUT_MANIFEST.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['new_queue_sha256']:raise ValueError('Expanded SPR pool snapshot mismatch')
    rows=list(csv.DictReader(path.open()));expected={r['candidate_id']:r['pair_id'] for r in rows};reviews={}
    for i in (1,2,3):
        for r in _load(directory/f'EXPANDED_REVIEW_{i}.json'):
            if expected.get(r['candidate_id'])!=r['pair_id'] or r['pair_id'] in reviews:raise ValueError('Expanded review identity mismatch')
            if r.get('verdict') not in LABELS or not isinstance(r.get('search_log'),list) or not r['search_log']:raise ValueError('Expanded review missing verdict/search')
            if any(not r.get(k) for k in ['summary','support','concern','next_step','evidence_scope']):raise ValueError('Expanded review missing text')
            validate_reason(r)
            reviews[r['pair_id']]={**r,'verdict_label':LABELS[r['verdict']],'scope_note':'扩展证据候选的LLM意见，不是已纳入实验设计或实验放行。'}
    output=[]
    for r in rows:
        disease=r.get('recommended_disease') or '未形成达标共同疾病假设；结合探索不强制图谱交集'
        item={'id':r['candidate_id'],'experiment_id':r['candidate_id'],'pair_id':r['pair_id'],'drug_id':r['ligand_inchikey'],'target_id':r['target_chembl_id'],
          'drug_name':r['modeled_entity_name'],'gene_symbol':r['gene_symbol'],'is_control':False,'drug_in_catalog':True,'target_in_catalog':True,
          'design_id':'EXPANDED_EVIDENCE_POOL_20260910','design_context':'新增证据候选 · 尚未纳入冻结实验设计','role':'EXPANDED_REVIEW_CANDIDATE',
          'role_label':{'BOTH':'双通道','RELAXED_JOINT':'放宽联合门槛','BINDING_CHEMISTRY_WITHOUT_GRAPH_GATE':'结合与化学支持通道'}.get(r['route'],r['route']),
          'original_indications':r['origin_summary'],'original_targets':r['known_target_names'],'original_mechanism':r['known_mechanism_of_action'],
          'original_indication_sources':[u.strip() for u in r.get('source_urls','').split(';') if u.strip().startswith('http')],
          'proposed_target':f"{r['gene_symbol']} ({r['target_chembl_id']})",'proposed_indication':disease,'proposed_disease_id':r.get('recommended_disease_id',''),
          'ranks':{'retargetmap_rank_384':float(r['binding_rank_384'])},'status_label':'证据审查池 · 未纳入实验设计',
          'disease_evidence':{'TxGNN疾病排名':r.get('recommended_txgnn_rank') or '无达标交集','OT关联分':r.get('recommended_ot_score') or '无达标交集'},
          'review':{'阳性近邻相似度':r.get('positive_max_tanimoto'),'阴性近邻相似度':r.get('negative_max_tanimoto'),'已知活动复查':'本地完整ChEMBL未见该精确配对记录；非全球查新证明'},
          'source_path':str(path.relative_to(root)), 'llm_review_status':'REVIEWED' if r['pair_id'] in reviews else 'PENDING'}
        if r['pair_id'] in reviews:
            item['llm_review']=reviews[r['pair_id']];item['resource_review']=validate_reason(reviews[r['pair_id']])
        output.append(item)
    return output
