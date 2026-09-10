"""Browser acceptance against the real local project API (no mocked scores).

Run after building/starting the service:
    python web/tests/smoke.py --url http://127.0.0.1:18765
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18765')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[2] / 'outputs/biomaster_explorer/screenshots'
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        context = browser.new_context(viewport={'width':1440, 'height':1050}, accept_downloads=True)
        page = context.new_page()
        failures = []
        page.on('pageerror', lambda error: failures.append(str(error)))
        page.set_default_timeout(25000)
        page.goto(args.url, wait_until='networkidle')
        expect(page.locator('.rh-intro h1')).to_be_visible()
        summary = context.request.get(args.url + '/api/summary').json()
        assert summary['counts']['drugs'] == 720
        assert summary['counts']['scored_targets'] == 384
        page.screenshot(path=str(out / 'home.png'), full_page=True)

        page.goto(args.url + '/#/drugs', wait_until='networkidle')
        expect(page.get_by_text('共 720 条记录', exact=True)).to_be_visible()
        assert page.locator('.catalog-table tbody tr').count() == 24
        page.get_by_label('下一页', exact=True).click()
        expect(page.locator('.pagination')).to_contain_text('2')
        page.locator('.topbar input').fill('imatinib')
        page.get_by_role('option').filter(has_text='imatinib').first.click()
        expect(page.locator('.entity-heading h1')).to_have_text('imatinib')
        expect(page.locator('.molecule-canvas img')).to_be_visible()
        page.get_by_role('button', name='靶点排名', exact=True).click()
        expect(page.locator('.ranking-table tbody tr')).to_have_count(20)
        drug_id = summary['featured']['drugs'][0]['id']
        scores = context.request.get(args.url + f'/api/rankings?kind=drug&id={drug_id}').json()
        expect(page.locator('.ranking-table tbody tr').first.locator('.table-entity strong')).to_have_text(scores['items'][0]['name'])
        page.locator('.ranking-table tbody tr').first.click()
        expect(page.get_by_role('dialog')).to_be_visible()
        expect(page.locator('.rank-chart')).to_contain_text('384')
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / 'model-comparison.png'), full_page=False)
        page.get_by_label('关闭配对面板').click()
        page.get_by_role('button', name='DrugCLIP', exact=True).click()
        expect(page.locator('.ranking-context')).to_contain_text('382')
        page.get_by_label('筛选排名').fill('ABL1')
        expect(page.locator('.ranking-table tbody tr')).to_have_count(1)
        expect(page.locator('.ranking-context')).to_contain_text('382')
        page.get_by_label('筛选排名').fill('')
        expect(page.locator('.ranking-table tbody tr')).to_have_count(20)
        with page.expect_download() as download:
            page.get_by_role('link', name='导出 CSV', exact=True).click()
        records = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
        assert len(records) == 384, 'CSV must export the complete catalog, not the visible page'
        assert records[0]['drugclip_denominator'] == '382'

        page.get_by_role('button', name='疾病与机制', exact=True).click()
        expect(page.locator('.evidence-browser').first.locator('.browser-row')).to_have_count(15)
        page.get_by_label('下一页证据').first.click()
        expect(page.locator('.evidence-browser').first.locator('.pagination')).to_contain_text('2 /')
        expect(page.get_by_text('当前快照展示已计算的疾病关联', exact=False)).to_be_visible()
        page.get_by_role('button', name='证据网络', exact=True).click()
        expect(page.locator('.evn-node').first).to_be_visible()
        page.locator('.evn-node').first.click()
        expect(page.locator('.evn-inspector')).to_contain_text('来源')
        page.screenshot(path=str(out / 'evidence-network.png'), full_page=True)

        page.goto(args.url + '/#/target/CHEMBL203/overview', wait_until='networkidle')
        expect(page.locator('.entity-heading h1')).to_have_text('EGFR')
        expect(page.locator('.rv-matrix tbody tr')).to_have_count(8)
        expect(page.locator('.ranking-context')).to_contain_text('720')
        page.screenshot(path=str(out / 'target.png'), full_page=True)
        page.get_by_role('button', name='完整档案', exact=True).click()
        expect(page.locator('.metadata-panel')).to_contain_text('组织表达')
        page.goto(args.url + '/#/target/CHEMBL2326/experiments', wait_until='networkidle')
        expect(page.locator('.experiment-record')).to_have_count(8)
        page.locator('.experiment-record summary').first.click()
        expect(page.locator('.experiment-content').first).to_contain_text('NO_EXPERIMENTAL_RESULT_IN_DESIGN_ARTIFACT')
        expect(page.locator('.experiment-panel')).to_contain_text('尚未放行')
        page.screenshot(path=str(out / 'spr-design.png'), full_page=True)
        # A registered GPCR is searchable even though it has no core rank.
        lookup = context.request.get(args.url + '/api/search?kind=target&q=ADRB2').json()
        assert lookup['items'], 'Registered targets outside the core must remain discoverable'
        gpcr = lookup['items'][0]
        page.goto(args.url + f'/#/target/{gpcr["id"]}/rankings', wait_until='networkidle')
        expect(page.get_by_text('当前模型没有可用排名', exact=True)).to_be_visible()
        page.goto(args.url + '/#/target/NO_SUCH_TARGET/overview', wait_until='networkidle')
        expect(page.get_by_text('数据暂时无法加载', exact=True)).to_be_visible()

        page.set_viewport_size({'width':390,'height':844})
        page.goto(args.url, wait_until='networkidle')
        expect(page.locator('.rh-intro h1')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile layout must not overflow horizontally'
        page.get_by_label('展开导航').click()
        page.locator('.sidebar nav button').filter(has_text='靶点图谱').click()
        expect(page.locator('.catalog-table tbody tr')).to_have_count(24)
        page.screenshot(path=str(out / 'mobile.png'), full_page=True)
        page.goto(args.url + '/#/target/CHEMBL203/structure', wait_until='networkidle')
        expect(page.get_by_label('选择结构口袋')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile structure controls must not overflow'
        assert not failures, failures
        (out.parent / 'BROWSER_VALIDATION.json').write_text(json.dumps({'status':'passed','url':args.url,'counts':summary['counts'],'page_errors':failures,'checks':['home','full_catalog','search','molecule','forward_rank','reverse_rank','model_comparison','filtered_rank_denominator','full_csv_export','evidence_pagination','network','metadata','SPR','unscored_registry','404','mobile']}, indent=2, ensure_ascii=False))
        browser.close()
        print('PASS: real-data browser acceptance; screenshots in', out)


if __name__ == '__main__':
    main()
