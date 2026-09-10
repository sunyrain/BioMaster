"""End-to-end research-workspace acceptance using the actual project API."""
import argparse
import csv
import io
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18765')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[2] / 'outputs/biomaster_explorer/studio'
    out.mkdir(parents=True, exist_ok=True)
    checks, errors, geometry = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(25000)
        page.on('pageerror', lambda e: errors.append(str(e)))
        drug = 'KTUFNOKKBVMGRW-UHFFFAOYSA-N'

        def goto(path):
            page.goto(args.url + '/#/' + path, wait_until='networkidle')
            expect(page.locator('main h1')).to_be_visible()

        goto('home')
        expect(page.locator('.rh-network-svg')).to_be_visible()
        checks.append('real_ranking_home_network')
        goto(f'drug/{drug}/rankings')
        expect(page.locator('.ranking-table tbody tr')).to_have_count(20)
        expected = context.request.get(args.url + f'/api/rankings?kind=drug&id={drug}&relationship=known').json()
        page.locator('.relationship-filters').get_by_role('button', name='已知关系', exact=True).click()
        expect(page.locator('.ranking-table tbody tr')).to_have_count(expected['total'])
        actual = page.locator('.ranking-table tbody tr .rank-number').all_text_contents()
        assert [int(x) for x in actual] == [int(x['rank']) for x in expected['items']]
        expect(page.locator('.ranking-context')).to_contain_text('384')
        with page.expect_download() as download:
            page.get_by_role('link', name='导出 CSV', exact=True).click()
        records = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
        assert len(records) == expected['total'] and all(row['known_relation'] == 'True' for row in records)
        checks.append('known_filter_and_full_filtered_export_preserve_ranks')
        page.locator('.relationship-filters').get_by_role('button', name='全部候选', exact=True).click()
        page.get_by_role('button', name='排名矩阵', exact=True).click()
        expect(page.locator('.rv-matrix tbody tr')).to_have_count(20)
        expect(page.locator('.rv-cell').first).to_contain_text('#1')
        page.locator('.rv-cell').first.click()
        expect(page.get_by_role('dialog')).to_be_visible()
        page.keyboard.press('Escape')
        expect(page.get_by_role('dialog')).to_have_count(0)
        expect(page.locator('.rv-cell').first).to_be_focused()
        checks.append('matrix_to_pair_evidence_and_keyboard_focus_restore')
        page.get_by_role('button', name='模型分歧', exact=True).click()
        expect(page.locator('.rv-trace')).to_have_count(20)
        trace = page.locator('.rv-trace').first
        trace.focus()
        expect(page.locator('.rv-trace-inspector strong')).to_be_visible()
        trace.press('Enter')
        expect(page.get_by_role('dialog')).to_be_visible()
        page.keyboard.press('Escape')
        checks.append('parallel_rank_plot_keyboard_inspection')
        page.screenshot(path=str(out / 'rank-disagreement-1440.png'), full_page=True)

        goto('target/CHEMBL203/diseases')
        expect(page.locator('.bio-bar-row')).to_have_count(6)
        source = context.request.get(args.url + '/api/entity/target/CHEMBL203').json()
        scored = sorted((r for r in source['target_diseases'] if isinstance(r.get('score'), (int, float))), key=lambda r: -r['score'])
        expect(page.locator('.bio-bar-row > strong').first).to_have_text(f"{scored[0]['score']:.3f}")
        page.locator('.bio-bar-row').nth(1).click()
        expect(page.locator('.bio-selected-evidence > strong')).to_have_text(scored[1]['name'])
        page.locator('.bio-expand').click()
        expect(page.locator('.bio-bar-row')).to_have_count(12)
        checks.append('disease_chart_matches_loaded_evidence_and_selection')

        goto('target/CHEMBL203/network')
        expect(page.locator('.evn-node')).to_have_count(14)
        page.get_by_label('搜索当前网络节点').fill('Cargo')
        expect(page.locator('.evn-node')).to_have_count(1)
        page.locator('.evn-node').focus()
        page.locator('.evn-node').press('Enter')
        expect(page.locator('.evn-inspector')).to_contain_text('Cargo')
        checks.append('network_filter_preserves_source_inspection')

        goto('targets')
        expect(page.locator('.catalog-table tbody tr')).to_have_count(24)
        page.get_by_label('按证据覆盖筛选目录').select_option('pathways')
        expect(page.locator('.catalog-table tbody tr')).to_have_count(24)
        assert page.locator('.catalog-evidence-badges .has-data').count() > 0
        page.get_by_role('button', name='卡片', exact=True).click()
        expect(page.locator('.catalog-card')).to_have_count(24)
        page.get_by_role('button', name='列表', exact=True).click()
        expect(page.locator('.catalog-table tbody tr')).to_have_count(24)
        checks.append('catalog_evidence_facet_and_list_grid_views')

        goto('target/CHEMBL203/overview')
        save = page.locator('.save-entity')
        expect(save).to_have_attribute('aria-pressed', 'false')
        save.click()
        expect(save).to_have_attribute('aria-pressed', 'true')
        page.reload(wait_until='networkidle')
        expect(page.locator('.save-entity')).to_have_attribute('aria-pressed', 'true')
        goto('workspace')
        expect(page.locator('.workspace-entry')).to_have_count(1)
        expect(page.locator('.workspace-entry')).to_contain_text('EGFR')
        page.locator('.workspace-entry').get_by_role('button', name='模型排名', exact=False).click()
        expect(page.get_by_role('button', name='老药排名', exact=True)).to_have_attribute('aria-current', 'page')
        goto('workspace')
        page.get_by_label('从清单移除 EGFR').click()
        expect(page.locator('.workspace-empty')).to_be_visible()
        checks.append('research_list_save_reload_resume_remove')

        for width in (1440, 1920, 390):
            page.set_viewport_size({'width': width, 'height': 844 if width == 390 else 1000})
            for name, route in [('home', 'home'), ('target', 'target/CHEMBL203/overview'), ('rankings', f'drug/{drug}/rankings'), ('catalog', 'targets'), ('workspace', 'workspace'), ('structure', 'target/CHEMBL203/structure'), ('diseases', 'target/CHEMBL203/diseases')]:
                goto(route)
                if name == 'structure':
                    expect(page.locator('.mol-stage canvas').first).to_be_visible()
                    bounds = page.locator('.mol-stage').first.bounding_box()
                    assert bounds and 280 <= bounds['height'] <= 800, f'3D viewport must remain bounded: {bounds}'
                page.wait_for_timeout(700)
                result = page.evaluate('''() => ({width:innerWidth, scrollWidth:document.documentElement.scrollWidth,
                    heading:document.querySelector('main h1')?.textContent,
                    dataTop:document.querySelector('.ranking-table tbody tr')?.getBoundingClientRect().top,
                    canvasTop:document.querySelector('.mol-stage canvas')?.getBoundingClientRect().top})''')
                assert result['scrollWidth'] <= width + 1, (name, width, result)
                geometry.append(dict(page=name, **result))
                page.screenshot(path=str(out / f'{name}-{width}.png'), full_page=True)
        checks.append('twenty_one_real_pages_responsive_without_body_overflow')
        page.emulate_media(reduced_motion='reduce')
        goto('home')
        page.wait_for_timeout(700)
        running = page.evaluate('''() => document.getAnimations().filter(a => a.playState === 'running' && a.effect?.getTiming().iterations === Infinity).length''')
        assert running == 0
        checks.append('reduced_motion_no_perpetual_animation')
        assert not errors, errors
        (out / 'STUDIO_VALIDATION.json').write_text(json.dumps(dict(status='passed', url=args.url, checks=checks, geometry=geometry, page_errors=errors), ensure_ascii=False, indent=2))
        browser.close()
        print('PASS:', len(checks), 'studio acceptance checks;', len(geometry), 'responsive page captures')


if __name__ == '__main__':
    main()
