#!/usr/bin/env python3
"""Check new authenticated API, live snapshots, CSV joins and real browser views."""
import csv
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
from biomaster.explorer_data import ExplorerData
from biomaster.explorer_server import ExplorerHTTPServer
OUT=ROOT/'outputs/frontier_dti_20260916'

def main():
    logging.basicConfig(level=logging.ERROR)
    data=ExplorerData(ROOT,infer=False);data.ensure_loaded()
    report={}
    with tempfile.TemporaryDirectory() as temp:
        password=secrets.token_urlsafe(24);salt=secrets.token_hex(16)
        path=Path(temp)/'users.json';path.write_text(json.dumps({'audit':{'salt':salt,'hash':hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),260000).hex()}}));path.chmod(0o600)
        server=ExplorerHTTPServer(('127.0.0.1',0),data,ROOT/'web/dist');server.auth=Auth(path)
        threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
        try:
            session=requests.Session();session.trust_env=False
            for route in ['/api/frontier-dti','/api/frontier-dti.csv']:
                assert session.get(base+route,timeout=10).status_code==401
            assert session.post(base+'/login',data={'username':'audit','password':password},allow_redirects=False,timeout=10).status_code==303
            payload=session.get(base+'/api/frontier-dti',timeout=10).json();assert len(payload['items'])==384
            assert {m['id'] for m in payload['models']}=={'nesso','probematch','dtbind'}
            first=payload['items'][0];drug=first['drug_id']
            filtered=session.get(base+'/api/frontier-dti',params={'kind':'drug','id':drug},timeout=10).json()
            assert all(x['drug_id']==drug for x in filtered['items'])
            response=session.get(base+'/api/frontier-dti.csv',timeout=10);assert response.status_code==200
            rows=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))));assert len(rows)==384
            assert {r['pair_id'] for r in rows}=={r['pair_id'] for r in payload['items']}
            final=session.get(base+'/api/spr-final.csv',timeout=10)
            assert final.content==(ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv').read_bytes()
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser=p.chromium.launch(args=['--no-sandbox']);page=browser.new_page(viewport={'width':1500,'height':1100})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(base+'/login');page.locator('input[name=username]').fill('audit');page.locator('input[name=password]').fill(password);page.locator('button[type=submit],button').first.click();page.wait_for_url(base+'/')
                page.goto(base+'/#/spr/models',wait_until='networkidle');page.get_by_role('heading',name='新模型复核与分歧').wait_for()
                page.get_by_label('筛选新模型复核候选').fill(first['candidate_id']);page.wait_for_timeout(200)
                assert page.locator('.frontier-pairs tbody tr').count()==1
                page.get_by_label('筛选新模型复核候选').fill('');page.get_by_label('新模型复核排序').select_option('disagreement')
                page.get_by_label('新模型测试集合').select_option('BINDINGDB479_probematch_AVAILABLE')
                assert '0.293' in page.locator('.frontier-validation').inner_text()
                page.screenshot(path=str(OUT/'WEBSITE_MODEL_REVIEW.png'),full_page=False)
                page.get_by_role('button',name='七模型分歧轨迹',exact=True).click()
                assert page.locator('.frontier-plot svg g[role=button]').count()>0
                page.locator('.frontier-plot svg g[role=button]').first.focus()
                assert '→' in page.locator('.frontier-inspector').inner_text()
                page.locator('.frontier-plot').scroll_into_view_if_needed()
                page.screenshot(path=str(OUT/'WEBSITE_SEVEN_MODEL_DISAGREEMENT.png'),full_page=False)
                page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),page.evaluate('[document.documentElement.scrollWidth,innerWidth]')
                page.set_viewport_size({'width':1500,'height':1100})
                page.goto(base+'/#/drug/'+drug+'/rankings',wait_until='networkidle')
                page.get_by_role('button',name='模型分歧',exact=True).click()
                page.get_by_role('heading',name='新模型复核与分歧').wait_for()
                assert page.locator('.frontier-pairs tbody tr').count()>0
                assert not errors,errors
                browser.close()
            report=dict(status='PASS',candidate_rows=384,auth_gates='401 without login',csv_exact_pair_join=True,original_final_csv_unchanged=True,browser_checks=['SPR model tab','candidate search','disagreement sorting','completed ProbeMatch metric selector','seven-model plot keyboard inspection','mobile no page overflow','drug model disagreement integration'],page_errors=errors,scope='same code/assets/data on ephemeral loopback authenticated server')
        finally:server.shutdown();server.server_close()
    (OUT/'WEBSITE_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
