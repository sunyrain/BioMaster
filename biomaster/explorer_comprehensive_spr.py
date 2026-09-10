"""Validated comprehensive SPR snapshot with exact-identity origin annotations."""
import hashlib
import json
from pathlib import Path
from .explorer_annotations import _rows, _text, _number

DESIGN = 'outputs/joint384_comprehensive_20260909'
REGISTRY = 'outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv'
BANDS = {
    'PRIORITY_CROSS_AREA_MECHANISM_REVIEW': '优先单靶点侦察',
    'ADDITIONAL_SOURCE_OR_CHEMICAL_SUPPORT': '有额外来源或化学支持',
    'EXPLORATORY_LIMITED_INDEPENDENT_SUPPORT': '独立支持有限的探索',
}

def load_comprehensive(root: Path, drugs: dict, targets: dict):
    directory = root / DESIGN
    names = ['RECOMMENDED_CANDIDATES_384.csv', 'REFERENCE_CONTROLS_EXTRA.csv', 'TARGET_ROSTER.csv']
    validation = json.loads((directory / 'VALIDATION.json').read_text())
    if validation.get('all_pass') is not True or not all(validation.get('checks', {}).values()):
        raise ValueError('Comprehensive SPR validation failed')
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    for name in names:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest['output_sha256'].get(name):
            raise ValueError(f'Comprehensive SPR snapshot hash mismatch: {name}')
    if hashlib.sha256((root / REGISTRY).read_bytes()).hexdigest() != manifest['input_sha256'].get(REGISTRY):
        raise ValueError('Comprehensive SPR origin registry hash mismatch')
    registry = {r['ligand_inchikey']: r for r in _rows(root / REGISTRY)}
    roster = {r['target_chembl_id']: r for r in _rows(directory / names[2])}
    candidates, controls = [list(_rows(directory / name)) for name in names[:2]]
    from .explorer_spr_reviews import load_spr_reviews
    llm_reviews = load_spr_reviews(root, directory / names[0], candidates)
    pairs = candidates + controls
    from .explorer_spr_final import load_final, final_items, DIRECTORY
    final_tables = load_final(root)
    current_control_drugs = set()
    if len(candidates) != 384 or len(controls) != 112 or len({r['pair_id'] for r in pairs}) != 496:
        raise ValueError('Comprehensive SPR requires 384 unique candidates and 112 separate controls')
    result = {'drugs': {k: {'experiments': []} for k in drugs}, 'targets': {k: {'experiments': []} for k in targets},
              'sources': [{'name': '综合 SPR 384 候选与另计对照', 'path': f'{DESIGN}/{n}', 'available': True} for n in names],
              'summary': json.loads((directory / 'SUMMARY.json').read_text())}
    for index, row in enumerate(pairs):
        control = index >= 384
        drug, target = row['ligand_inchikey'], row['target_chembl_id']
        if row['pair_id'] != f'{drug}__{target}':
            raise ValueError('Comprehensive SPR exact pair identity mismatch')
        origin = registry.get(drug, {})
        context = roster[target]
        identifier = f'CTRL-{index - 383:03d}' if control else row['candidate_id']
        name = _text(row.get('modeled_entity_name') or row.get('drug_names'))
        item = {
            'id': identifier, 'experiment_id': identifier, 'pair_id': row['pair_id'],
            'name': f'{name} → {row["gene_symbol"]}', 'drug_name': name,
            'drug_id': drug, 'target_id': target, 'gene_symbol': row['gene_symbol'],
            'drug_in_catalog': drug in drugs, 'target_in_catalog': target in targets,
            'design_id': 'COMPREHENSIVE384_20260909', 'design_date': '2026-09-09',
            'design_context': '112 靶点 / 384 候选 + 112 另计参考对照',
            'source': 'BioMaster 综合384审查', 'source_path': f'{DESIGN}/{names[1 if control else 0]}',
            'is_control': control, 'role': row['selection_role'],
            'role_label': '另计参考对照' if control else BANDS[row['recommendation_band']],
            'release_status': row['release_status'], 'status_label': '设计审核中 · 尚未放行',
            'result_status': 'NO_EXPERIMENTAL_RESULT_IN_DESIGN_ARTIFACT',
            'original_indications': '参考化合物；本设计未审查原适应症' if control else _text(row.get('origin_summary')) or '未收录，需核实',
            'original_targets': _text(origin.get('known_target_names')) or '当前机制库未收录，不能据此认定无已知靶点',
            'original_target_ids': _text(origin.get('known_target_chembl_ids')),
            'original_mechanism': _text(origin.get('known_mechanism_of_action')),
            'original_targets_source': REGISTRY if origin else '无精确分子身份匹配记录',
            'original_indication_sources': [] if control else [u.strip() for u in row.get('source_urls', '').split(';') if u.strip().startswith('https://')],
            'origin_scope': '参考对照不作为药物再定位尝试' if control else _text(row.get('scope_limits')),
            'proposed_target': f'{row["gene_symbol"]} ({target})',
            'proposed_indication': '不适用：用于靶点实验质控' if control else _text(row.get('recommended_disease')) or '未形成疾病假设',
            'proposed_disease_id': '' if control else _text(row.get('recommended_disease_id')),
            'cross_area': None if control else row.get('recommended_disease_is_cross_area') == 'True',
            'disease_evidence': {} if control else {
                'TxGNN 疾病排名（非结合概率）': _number(row.get('recommended_txgnn_rank')),
                'Open Targets 关联分（非疗效概率）': _number(row.get('recommended_ot_score')),
                '胚系遗传证据分': _number(row.get('recommended_germline_score')),
                '体细胞突变证据分': _number(row.get('recommended_somatic_score')),
                '疾病映射范围': _text(row.get('recommended_mapping_scope')),
            },
            'evidence': 'TxGNN 为药物—疾病推断，Open Targets 为靶点—疾病关联；共同疾病不能证明直接结合、作用方向或治疗效果。',
            'construct_status': context['construct_status'], 'reference_control': context['proposed_reference'],
            'endpoint_plan': _text(row.get('assay_notes')),
            'pair_review_status': _text(row.get('deep_reason') or row.get('reason')),
            'novelty_status': _text(row.get('novelty_reaudit_status')),
            'rank_context': '原完整 720×384 评分空间内的药物→靶点排名；不代表命中率',
            'ranks': {} if control else {'retargetmap_rank_384': _number(row.get('binding_rank_384'))},
            'review': {'关键待验证项': _text(row.get('deep_critical_gap')), '实验建议': _text(row.get('assay_notes'))},
        }
        if not control and row["pair_id"] in llm_reviews:
            item["llm_review"] = llm_reviews[row["pair_id"]]
        if control:
            item['control_evidence'] = {k: row.get(k, '') for k in ['control_chembl_id', 'reference_endpoint_category', 'reference_standard_types', 'reference_assay_ids', 'reference_doc_ids', 'mean_pchembl_mixed_endpoints']}
            item['control_evidence']['说明'] = '端点包含 Kd 也不表示已验证适用于 SPR；混合端点 pChEMBL 不能换算为本实验 Kd。'
        if control:
            item = final_items(root, [item], final_tables)[0]
            drug = item['drug_id']
            item['drug_in_catalog'] = drug in drugs
            current_control_drugs.add(drug)
        for key, collection in [(drug, 'drugs'), (target, 'targets')]:
            if key in result[collection]:
                result[collection][key]['experiments'].append(item)
    result['counts'] = {
        'spr_design_pairs': 496, 'spr_candidate_pairs': 384, 'spr_control_pairs': 112,
        'spr_design_targets': len(roster), 'spr_not_released_pairs': 496,
        'spr_catalog_drugs': sum(bool(r['experiments']) for r in result['drugs'].values()),
        'spr_catalog_targets': sum(bool(r['experiments']) for r in result['targets'].values()),
        'spr_off_catalog_compounds': len(({r['ligand_inchikey'] for r in candidates} | current_control_drugs) - drugs.keys()),
        'spr_off_catalog_targets': len(set(roster) - targets.keys()),
    }
    if final_tables[1]:
        result['sources'].append({'name':'最终SPR对照与FDA修订记录','path':DIRECTORY+'/SPR112_REFERENCE_CONTROLS.csv','available':True})
    return result
