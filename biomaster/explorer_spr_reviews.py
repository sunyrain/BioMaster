"""Live additive LLM reviews, validated against the frozen exact-pair baseline."""
import hashlib
import json
from pathlib import Path

REVIEW_DIR = 'outputs/spr384_llm_review_20260909'
REVIEW_PATH = f'{REVIEW_DIR}/REVIEWS_384.json'
LABELS = {'PRIORITIZE': '优先验证', 'CONDITIONAL': '有条件保留', 'DEFER': '建议后置', 'INSUFFICIENT': '信息不足'}
BASELINE_PATH = 'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv'


def load_spr_reviews(root, baseline, candidates):
    directory = root / REVIEW_DIR
    if not (directory / 'BASELINE.json').exists():
        return {}
    package = json.loads((directory / 'BASELINE.json').read_text())
    if package.get('sha256') != hashlib.sha256(baseline.read_bytes()).hexdigest():
        raise ValueError('SPR LLM review baseline mismatch')
    expected = {r['candidate_id']: r['pair_id'] for r in candidates}
    result = {}
    for batch in range(1, 4):
        path = directory / f'review_{batch}.json'
        if not path.exists():
            continue
        allowed = {r['candidate_id']: r['pair_id'] for r in json.loads((directory / f'batch_{batch}.json').read_text())}
        for row in json.loads(path.read_text()):
            identifier, pair = row['candidate_id'], row['pair_id']
            if expected.get(identifier) != pair or allowed.get(identifier) != pair or pair in result:
                raise ValueError('SPR LLM review duplicate or pair identity mismatch')
            if row['verdict'] not in LABELS or any(not str(row.get(k, '')).strip() for k in ['summary', 'support', 'concern', 'next_step', 'evidence_scope']):
                raise ValueError('SPR LLM review missing evaluation fields')
            if not isinstance(row.get('search_log'), list) or not row['search_log']:
                raise ValueError('SPR LLM review requires actual search log')
            if not isinstance(row.get('source_urls'), list):
                raise ValueError('SPR LLM review requires source list')
            result[pair] = {**row, 'verdict_label': LABELS[row['verdict']],
                'scope_note': 'LLM证据审查意见，不代表实测结合或实验放行；检索与全文核实范围见本条记录。'}
    return result


def live_spr_items(root, items):
    import csv
    root = Path(root)
    baseline = root / BASELINE_PATH
    if not baseline.exists():
        return items
    with baseline.open() as handle:
        candidates = list(csv.DictReader(handle))
    reviews = load_spr_reviews(root, baseline, candidates)
    from .explorer_spr_expansion import baseline_reasons
    reasons = baseline_reasons(root)
    output = []
    for item in items:
        item = {k: v for k, v in item.items() if k != 'llm_review'}
        if not item.get('is_control'):
            item['llm_review_status'] = 'REVIEWED' if item['pair_id'] in reviews else 'PENDING'
            if item['pair_id'] in reviews:
                item['llm_review'] = reviews[item['pair_id']]
        if item.get('pair_id') in reasons:
            item['resource_review'] = reasons[item['pair_id']]
        output.append(item)
    from .explorer_spr_final import final_items
    return final_items(root, output)
