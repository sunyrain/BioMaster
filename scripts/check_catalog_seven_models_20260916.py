#!/usr/bin/env python3
"""Verify main seven-model views against complete directory ranks and exact direction."""
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_auth import Auth
from biomaster.explorer_data import ExplorerData,MODELS
from biomaster.explorer_server import ExplorerHTTPServer
OUT=ROOT/'outputs/catalog_seven_models_20260916'

def main():
    logging.basicConfig(level=logging.ERROR)
    data=ExplorerData(ROOT,infer=False);data.ensure_loaded()
    ar=data.resolve('target','AR')
    original=data._rows('target',ar).sort_values('target_biomaster_rank').head(10)
    expected=list(original.ligand_inchikey)
    api=data.rankings('target',ar,'biomaster',page_size=10)
    assert api['catalog_count']==720 and api['denominator']==720 and api['total']==720
    assert [x['id'] for x in api['items']]==expected
    assert [x['rank'] for x in api['items']]==list(range(1,11))
    assert all(x['score']==float(y) for x,y in zip(api['items'],original.biomaster_reverse))
    assert len(api['items'][0]['scores'])==7
    (OUT/'AR_RETARGETMAP_TOP10.json').write_text(json.dumps(api,ensure_ascii=False,indent=2))
    with tempfile.TemporaryDirectory() as temp:
        password=secrets.token_urlsafe(24);salt=secrets.token_hex(16)
        path=Path(temp)/'users.json';path.write_text(json.dumps({'audit':{'salt':salt,'hash':hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),260000).hex()}}));path.chmod(0o600)
        server=ExplorerHTTPServer(('127.0.0.1',0),data,ROOT/'web/dist');server.auth=Auth(path)
        threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
        try:
            session=requests.Session();session.trust_env=False
            assert session.get(base+'/api/rankings?kind=target&id='+ar).status_code==401
            assert session.post(base+'/login',data={'username':'audit','password':password},allow_redirects=False).status_code==303
            result=session.get(base+'/api/rankings.csv',params={'kind':'target','id':ar,'model':'biomaster','page_size':10}).content
            rows=list(csv.DictReader(io.StringIO(result.decode('utf-8-sig'))))
            assert len(rows)==720 and [r['entity_id'] for r in rows[:10]]==expected
            assert all(f'{m}_rank' in rows[0] for m in MODELS)
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser=p.chromium.launch(args=['--no-sandbox']);page=browser.new_page(viewport={'width':1700,'height':1150})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(base+'/login');page.locator('input[name=username]').fill('audit');page.locator('input[name=password]').fill(password);page.locator('button[type=submit],button').first.click();page.wait_for_url(base+'/')
                page.goto(base+'/#/target/'+ar+'/rankings',wait_until='networkidle')
                page.locator('.rv-matrix').wait_for()
                assert page.locator('.rv-sort').count()==7
                assert page.locator('.ranking-panel .frontier-models').count()==0
                page.get_by_role('button',name='ReTargetMap 全目录 Top10',exact=True).click()
                page.wait_for_timeout(500);page.locator('.rv-matrix').wait_for()
                assert page.locator('.rv-matrix tbody tr').count()==10
                assert page.locator('.rv-entity strong').all_text_contents()==[x['name'] for x in api['items']]
                page.locator('.ranking-panel').scroll_into_view_if_needed();page.screenshot(path=str(OUT/'AR_SEVEN_MODEL_MATRIX.png'))
                page.get_by_role('button',name='模型分歧 · 七模型',exact=True).click()
                page.get_by_role('group',name='七模型排名位置平行坐标图',exact=True).wait_for()
                assert page.locator('[data-model-axis]').count()==7
                assert page.locator('.rv-trace').count()==10
                page.locator('.rv-trace').first.focus()
                assert api['items'][0]['name'] in page.locator('.rv-trace-inspector').inner_text()
                page.locator('.rv-parallel').scroll_into_view_if_needed();page.screenshot(path=str(OUT/'AR_SEVEN_MODEL_DISAGREEMENT.png'))
                page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),page.evaluate('[document.documentElement.scrollWidth,innerWidth]')
                page.set_viewport_size({'width':1700,'height':1150})
                page.locator('.model-tabs button').filter(has_text='DTBind').click();page.wait_for_timeout(700)
                assert page.locator('[data-model-axis]').count()==7
                for tab in ['overview','rankings']:
                    page.goto(base+'/#/drug/'+expected[0]+'/'+tab,wait_until='networkidle')
                    page.locator('.rv-matrix').wait_for();assert page.locator('.rv-sort').count()==7
                    page.get_by_role('button',name='模型分歧 · 七模型',exact=True).click()
                    assert page.locator('[data-model-axis]').count()==7
                    assert page.locator('.frontier-models').count()==0
                assert not errors,errors
                browser.close()
            report=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),ar_target=ar,ar_catalog=720,ar_top10_ids=expected,seven_model_matrix=True,seven_axes_main_plot=True,drug_overview_and_rankings=True,reverse_head_verified=True,csv_rows=720,page_errors=errors,scope='same production code, assets and data with ephemeral local authentication')
            (OUT/'WEBSITE_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
        finally:server.shutdown();server.server_close()
if __name__=='__main__':main()
