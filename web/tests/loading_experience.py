"""Slow-network, recovery, caching and public-display regressions."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:18891')
args = parser.parse_args()
out = Path('outputs/loading_optimization_20260909')
out.mkdir(parents=True, exist_ok=True)
report = {}
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    session = context.new_cdp_session(page)
    session.send('Network.enable')
    session.send('Network.emulateNetworkConditions', {'offline': False, 'latency': 180, 'downloadThroughput': 190000, 'uploadThroughput': 100000})
    requests = []
    page.on('request', lambda r: requests.append(r.url))
    page.goto(args.url, wait_until='domcontentloaded')
    expect(page.locator('.rh-intro h1')).to_be_visible(timeout=30000)
    assert not any('3Dmol-' in u or 'StructureViewer-' in u for u in requests)
    report['slow_home_ms'] = page.evaluate('performance.now()')
    report['home_transfer_bytes'] = page.evaluate('performance.getEntriesByType("resource").reduce((n,r)=>n+r.transferSize,0)')
    page.evaluate('location.hash="/target/CHEMBL2326/experiments"')
    expect(page.locator('.loading-state').first).to_be_visible()
    page.screenshot(path=str(out/'loading-desktop.png'))
    expect(page.locator('.experiment-record')).to_have_count(8, timeout=30000)
    key = '/api/entity/target/CHEMBL2326'
    before = sum(key in u for u in requests)
    page.evaluate('location.hash="/targets"')
    expect(page.locator('.catalog-table')).to_be_visible()
    page.evaluate('location.hash="/target/CHEMBL2326/experiments"')
    expect(page.locator('.experiment-record')).to_have_count(8)
    assert sum(key in u for u in requests) == before
    report['entity_return_reused_cache'] = True
    assert page.locator('.sidebar').evaluate('e=>getComputedStyle(e).backgroundColor') == 'rgb(11, 23, 43)'
    context.set_offline(True)
    expect(page.locator('.connection-status')).to_be_visible()
    context.set_offline(False)
    expect(page.locator('.connection-status')).to_have_count(0)
    assert not errors, errors
    context.close()

    # A failed request must expose an actionable retry, not a permanently spinning page.
    context = browser.new_context(viewport={'width':390,'height':844}, reduced_motion='reduce')
    page = context.new_page()
    page.route('**/api/summary', lambda r: r.fulfill(status=500, content_type='application/json', body='{"error":"演示连接测试"}'))
    page.goto(args.url)
    expect(page.get_by_role('button', name='重新加载', exact=True)).to_be_visible()
    page.unroute('**/api/summary')
    page.get_by_role('button', name='重新加载', exact=True).click()
    expect(page.locator('.rh-intro h1')).to_be_visible(timeout=30000)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
    page.screenshot(path=str(out/'home-mobile.png'))
    report['mobile_failure_recovery'] = True
    context.close()
    browser.close()
(out/'LOADING_VALIDATION.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
