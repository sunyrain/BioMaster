"""Regression checks for replacing a control's identity across all entry points."""
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomaster.explorer_comprehensive_spr import DESIGN, REGISTRY, load_comprehensive
from biomaster.explorer_spr_final import DIRECTORY, final_items, load_final
from biomaster.explorer_spr_results import SPRResults

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def catalog():
    if not (ROOT/DIRECTORY/'MANIFEST.json').exists():
        pytest.skip('Final SPR artifact not installed')
    registry = list(csv.DictReader((ROOT/REGISTRY).open()))
    drugs = {r['ligand_inchikey']:{} for r in registry}
    targets = {r['target_chembl_id']:{} for r in csv.DictReader((ROOT/DESIGN/'TARGET_ROSTER.csv').open())}
    return load_comprehensive(ROOT, drugs, targets)


def test_replaced_controls_have_new_ids_and_correct_drug_membership(catalog):
    rows = [r for t in catalog['targets'].values() for r in t['experiments']]
    shown = final_items(ROOT, rows)
    assert shown == final_items(ROOT, shown)  # repeated live enrichment must be idempotent
    controls = [r for r in shown if r['is_control']]
    changed = [r for r in controls if r.get('control_replaced')]
    assert len(changed)==31 and len(controls)==112 and len(shown)==496
    current_ids = {r['experiment_id'] for r in controls}
    for r in changed:
        assert r['pair_id']==f"{r['drug_id']}__{r['target_id']}"
        assert r['pair_id']!=r['baseline_pair_id']
        assert r['previous_experiment_id'] not in current_ids
        assert r['control_compound_inchikey']==r['drug_id']
        assert r['control_evidence']['control_chembl_id']==r['control_compound_chembl_id']
        assert r['control_fda_urls'] and r['control_construct_requirement']
        for drug, entry in catalog['drugs'].items():
            assert all(x['drug_id']==drug for x in entry['experiments'])
    by_target = {r['target_id']:r for r in controls}
    for r in shown:
        if not r['is_control']:
            assert r['control_id']==by_target[r['target_id']]['experiment_id']
            assert r['control_compound_inchikey']==by_target[r['target_id']]['drug_id']
    assert [r['final_priority_order'] for r in shown if not r['is_control']]==list(range(1,385))


def test_result_templates_use_current_control_pairs(catalog,tmp_path):
    data = SimpleNamespace(root=ROOT,targets=catalog['targets'],ensure_loaded=lambda:None)
    registry = SPRResults.registry(data)
    controls = SPRResults.catalog(registry,scope='control')
    expected = load_final(ROOT)[1]
    assert len(controls)==112
    by_id = {r['experiment_id']:r for r in controls}
    for r in expected.values():
        item = by_id[r['对照编号']]
        assert item['pair_id']==r['pair_id']
        assert item['drug_name']==r['新靶点的已知药物名称']
    store = SPRResults(tmp_path)
    rows = list(csv.DictReader(io.StringIO(store.template(registry,scope='control').decode('utf-8-sig'))))
    assert {r['配对编号'] for r in rows}=={r['pair_id'] for r in controls}
    # A retired identifier must not silently acquire the new compound's identity.
    retired = next(r for r in expected.values() if r['本次FDA替换']=='True')
    checked = store.preview({'batch':'identity-test','rows':[{'experiment_id':retired['原对照编号'],
        'pair_id':retired['原pair_id'],'sample_id':'S1','experiment_date':'2026-09-10',
        'result':'not_detected','qc':'review'}]},registry,'test')
    assert checked['token'] is None and checked['error_count']==1
    assert store.results()['total']==0


def test_final_table_checksum_is_enforced(tmp_path):
    directory = tmp_path/DIRECTORY
    directory.mkdir(parents=True)
    (directory/'MANIFEST.json').write_text(json.dumps({'output_sha256':{'SPR384_FINAL_DETAILED.csv':'wrong'}}))
    (directory/'SPR384_FINAL_DETAILED.csv').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):
        load_final(tmp_path)
