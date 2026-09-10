import hashlib
import json

import pytest
from biomaster.explorer_spr_reviews import REVIEW_DIR, load_spr_reviews


def setup_snapshot(tmp_path):
    baseline = tmp_path / 'baseline.csv'
    baseline.write_text('frozen')
    directory = tmp_path / REVIEW_DIR
    directory.mkdir(parents=True)
    (directory / 'BASELINE.json').write_text(json.dumps({'sha256': hashlib.sha256(baseline.read_bytes()).hexdigest()}))
    candidates = [{'candidate_id': 'C384-001', 'pair_id': 'D__T'}, {'candidate_id': 'C384-002', 'pair_id': 'D2__T'}]
    (directory / 'batch_1.json').write_text(json.dumps(candidates))
    row = dict(candidates[0], verdict='CONDITIONAL', summary='评价', support='支持', concern='疑点', next_step='下一步', evidence_scope='摘要', source_urls=[], search_log=[{'query': 'drug target', 'outcome': 'no direct evidence'}])
    return baseline, directory, candidates, row


def test_live_partial_reviews_and_updates(tmp_path):
    baseline, directory, candidates, row = setup_snapshot(tmp_path)
    assert load_spr_reviews(tmp_path, baseline, candidates) == {}
    (directory / 'review_1.json').write_text(json.dumps([row]))
    result = load_spr_reviews(tmp_path, baseline, candidates)
    assert set(result) == {'D__T'}
    assert result['D__T']['verdict_label'] == '有条件保留'
    row['verdict'] = 'DEFER'
    (directory / 'review_1.json').write_text(json.dumps([row]))
    assert load_spr_reviews(tmp_path, baseline, candidates)['D__T']['verdict_label'] == '建议后置'


@pytest.mark.parametrize('bad', ['identity', 'duplicate', 'missing_search', 'baseline'])
def test_invalid_review_rejected(tmp_path, bad):
    baseline, directory, candidates, row = setup_snapshot(tmp_path)
    if bad == 'identity': row['pair_id'] = 'WRONG__T'
    if bad == 'missing_search': row['search_log'] = []
    if bad == 'baseline': baseline.write_text('changed')
    (directory / 'review_1.json').write_text(json.dumps([row, row] if bad == 'duplicate' else [row]))
    with pytest.raises(ValueError):
        load_spr_reviews(tmp_path, baseline, candidates)
