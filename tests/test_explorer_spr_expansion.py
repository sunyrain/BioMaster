import csv,json,hashlib
from pathlib import Path
import pytest
from biomaster.explorer_spr_expansion import baseline_reasons, expanded_items, validate_reason, DIR

ROOT=Path(__file__).resolve().parents[1]

def test_reason_requires_specific_explanation():
    with pytest.raises(ValueError):validate_reason({'action_class':'EXPLORABLE','reason_tags':[]})
    r=validate_reason({'action_class':'EXPLORABLE','reason_tags':['ONLY_EVIDENCE_GAP'],'reason_detail':'没有额外证据不是反证','can_test':'可探索，未确认体系','critical_condition':'先核实际条件'})
    assert r['action_label']=='仍可探索'


def test_real_expansion_is_disjoint_and_not_frozen_design():
    if not (ROOT/DIR/'PUBLISH_READY.json').exists():pytest.skip('Expanded snapshot absent')
    rows=expanded_items(ROOT)
    old=list(csv.DictReader((ROOT/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv').open()))
    assert len(rows)==384 and len({r['pair_id'] for r in rows})==384
    assert not {r['pair_id'] for r in rows}&{r['pair_id'] for r in old}
    assert all(not r['is_control'] and r['design_id']=='EXPANDED_EVIDENCE_POOL_20260910' for r in rows)
    assert rows[0]['drug_name']=='betamethasone'
    assert all(r['source_path'].endswith('NEW_CANDIDATES_FOR_LLM.csv') for r in rows)
    json.dumps(rows,allow_nan=False)


def test_real_reason_coverage_and_non_exclusion():
    if not (ROOT/DIR/'BASELINE_REASON_2.json').exists():pytest.skip('Reason snapshot absent')
    reasons=baseline_reasons(ROOT)
    assert len(reasons)==384
    assert sum(r['action_class']=='EXPLORABLE' for r in reasons.values())==223
    assert sum(r['action_class']=='EVIDENCE_DEPRIORITIZE' for r in reasons.values())==12


def test_queue_hash_tampering_rejected(tmp_path):
    d=tmp_path/DIR;d.mkdir(parents=True)
    (d/'PUBLISH_READY.json').write_text('{}')
    (d/'NEW_CANDIDATES_FOR_LLM.csv').write_text('changed')
    (d/'INPUT_MANIFEST.json').write_text(json.dumps({'new_queue_sha256':hashlib.sha256(b'original').hexdigest()}))
    with pytest.raises(ValueError,match='snapshot mismatch'):expanded_items(tmp_path)
