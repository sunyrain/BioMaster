"""Read-only live gates and isolated real-data API/browser checks; no result uploads."""
import hashlib
import json
import logging
import secrets
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.explorer_auth import Auth
from biomaster.explorer_data import ExplorerData
from biomaster.explorer_server import ExplorerHTTPServer

OUT = ROOT / 'outputs/platform_frontier_audit_20260916'


def main():
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'live': [],
              'isolated_api': {}, 'production_auth_changed': False,
              'uploaded_results': False, 'new_model_inference': False}
    for base, trust in [('http://127.0.0.1:18765', False), ('https://palinova.xyz', True)]:
        session = requests.Session()
        session.trust_env = trust
        for route, expected in [('/login', 200), ('/api/summary', 401), ('/api/spr-final.csv', 401)]:
            response = session.get(base + route, timeout=25)
            report['live'].append({'url': base+route, 'status': response.status_code,
                                   'expected': expected, 'ok': response.status_code == expected})
    assert all(x['ok'] for x in report['live']), report['live']
    data = ExplorerData(ROOT, infer=False)
    data.ensure_loaded()
    report['summary'] = data.summary()
    # Same server, data and static assets; ephemeral login on a loopback-only port.
    # Production credentials are neither changed nor read by this check.
    with tempfile.TemporaryDirectory() as temp:
        password = secrets.token_urlsafe(24)
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 260000).hex()
        auth = Path(temp)/'users.json'
        auth.write_text(json.dumps({'audit': {'salt': salt, 'hash': digest}}))
        auth.chmod(0o600)
        server = ExplorerHTTPServer(('127.0.0.1', 0), data, ROOT/'web/dist')
        server.auth = Auth(auth)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            session = requests.Session()
            session.trust_env = False
            assert session.post(base+'/login', data={'username':'audit','password':password},
                                allow_redirects=False, timeout=20).status_code == 303
            for route in ['/api/health', '/api/summary', '/api/spr-design',
                          '/api/spr-results/catalog', '/api/spr-results',
                          '/api/search?kind=target&q=LYVE1', '/api/search?kind=target&q=SLC8A1']:
                r = session.get(base+route, timeout=60)
                assert r.status_code == 200, (route, r.status_code)
                payload = r.json()
                report['isolated_api'][route] = {'status': r.status_code, 'bytes':len(r.content),
                    'items':len(payload.get('items',[])), 'total':payload.get('total')}
            drug = report['summary']['featured']['drugs'][0]['id']
            r = session.get(base+'/api/rankings', params={'kind':'drug','id':drug}, timeout=30)
            assert r.status_code == 200 and len(r.json()['items']) > 0
            report['ranking_sample'] = {'drug':drug, 'rows':len(r.json()['items']),
                                       'total':r.json().get('total')}
            for route, name in [('/api/spr-final.csv','SPR384_FINAL_EXPERIMENT_TABLE.csv'),
                                ('/api/spr-final-controls.csv','SPR112_REFERENCE_CONTROLS.csv')]:
                r = session.get(base+route, timeout=30)
                source = ROOT/'outputs/spr384_final_experiment_table_20260910'/name
                assert r.status_code == 200 and r.content == source.read_bytes()
                report['isolated_api'][route] = {'status':200, 'identical_to_handoff':True,
                    'sha256':hashlib.sha256(r.content).hexdigest()}
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(args=['--no-sandbox'])
                context = browser.new_context(viewport={'width':1440,'height':1000})
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(base+'/login')
                page.locator('input[name=username]').fill('audit')
                page.locator('input[name=password]').fill(password)
                page.locator('button[type=submit], button').first.click()
                page.wait_for_url(base+'/')
                for route, expected in [('/','Palinova'),('/#/spr/baseline','384'),
                                        ('/#/drug/'+drug+'/rankings','靶点')]:
                    page.goto(base+route, wait_until='networkidle', timeout=60000)
                    page.wait_for_function('(s) => document.body.innerText.includes(s)', arg=expected)
                page.goto(base+'/#/spr/baseline', wait_until='networkidle')
                page.screenshot(path=str(OUT/'SPR_RESTORED.png'), full_page=False)
                page.set_viewport_size({'width':390,'height':844})
                page.wait_for_timeout(500)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert not errors, errors
                report['browser'] = {'status':'passed','page_errors':errors,
                    'checks':['ephemeral_login','home','SPR384','drug_ranking','mobile_no_overflow'],
                    'scope':'isolated loopback instance with production code, assets and data'}
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
    report['status'] = 'PASS'
    (OUT/'PLATFORM_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','live','isolated_api','ranking_sample','browser']},ensure_ascii=False))
    print(json.dumps(report['summary']['counts'],ensure_ascii=False))


if __name__ == '__main__':
    main()
