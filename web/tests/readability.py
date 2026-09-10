"""Real-browser visual accessibility and interaction acceptance.

Run against a built, running BioMaster service:
    python web/tests/readability.py --url http://127.0.0.1:18765

Reports calculated text contrast, font sizes, responsive geometry, and keyboard
interactions. Gradient/canvas backgrounds require the accompanying screenshots;
their contrast is deliberately recorded as a manual check, never a false pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


TEXT_AUDIT = r"""() => {
  const rgb = value => {
    const values = value.match(/[\d.]+/g);
    if (!values) return [0, 0, 0, 0];
    return [...values.slice(0, 3).map(Number), values.length > 3 ? +values[3] : 1];
  };
  const blend = (front, back) => [0, 1, 2].map(i =>
    front[i] * front[3] + back[i] * (1 - front[3]));
  const luminance = c => c.slice(0, 3).map(v => {
    v /= 255;
    return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
  }).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
  const rows = [];
  // Content behind a modal overlay is visually obscured and outside its
  // reading/focus scope; auditing its uncomposited colors would be misleading.
  const scope = document.querySelector('[role="dialog"][aria-modal="true"]') || document.body;
  for (const e of scope.querySelectorAll('*')) {
    if (e.closest('svg, [aria-hidden="true"], [disabled], script, style')) continue;
    if (!e.checkVisibility({opacityProperty: true, visibilityProperty: true,
                           contentVisibilityAuto: true})) continue;
    const text = [...e.childNodes].filter(n => n.nodeType === Node.TEXT_NODE)
      .map(n => n.textContent.trim()).filter(Boolean).join(' ');
    if (!text) continue;
    const rect = e.getBoundingClientRect(), style = getComputedStyle(e);
    if (!rect.width || !rect.height || rect.right <= 0 || rect.left >= innerWidth ||
        style.visibility !== 'visible' || style.display === 'none') continue;
    const layers = [];
    let imageBackground = false, opacity = 1;
    for (let p = e; p; p = p.parentElement) {
      const s = getComputedStyle(p);
      opacity *= +s.opacity;
      const c = rgb(s.backgroundColor);
      imageBackground ||= s.backgroundImage !== 'none';
      layers.push(c);
      if (c[3] >= 1) break;
    }
    if (opacity < .01) continue;
    let background = [255, 255, 255];
    for (const layer of layers.reverse()) background = blend(layer, background);
    const ink = rgb(style.color);
    ink[3] *= opacity;
    const foreground = blend(ink, background);
    const lum = [luminance(foreground), luminance(background)].sort((a,b) => b-a);
    const contrast = (lum[0] + .05) / (lum[1] + .05);
    const fontSize = parseFloat(style.fontSize), weight = parseFloat(style.fontWeight);
    const minimum = fontSize >= 24 || (fontSize >= 18.66 && weight >= 700) ? 3 : 4.5;
    const primary = e.matches('p, .entity-heading p, .browser-row-title strong, '
      + '.evidence-row strong, .table-entity strong, .score-cell strong, '
      + '.detail-tabs button, .sidebar nav button, .catalog-card h3');
    rows.push({
      text: text.slice(0, 110), tag: e.tagName, class: e.className,
      font_size: fontSize, primary, color: style.color,
      background: background.map(Math.round),
      contrast: +contrast.toFixed(2), required_contrast: minimum,
      manual_background_check: imageBackground,
      small_text: fontSize < (primary ? 14 : 12),
      low_contrast: !imageBackground && contrast + .01 < minimum,
    });
  }
  return {
    body_width: document.documentElement.scrollWidth,
    viewport_width: innerWidth,
    overflow: document.documentElement.scrollWidth > innerWidth + 1,
    text_count: rows.length,
    small_text: rows.filter(r => r.small_text),
    low_contrast: rows.filter(r => r.low_contrast),
    manual_background_checks: rows.filter(r => r.manual_background_check),
    rows,
  };
}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18765')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--skip-interactions', action='store_true')
    parser.add_argument('--pages', nargs='+', default=['home', 'catalog', 'target', 'rankings', 'diseases'],
                        choices=['home', 'catalog', 'target', 'rankings', 'diseases',
                                 'metadata', 'spr', 'sources', 'network'])
    parser.add_argument('--viewports', nargs='+', type=int, default=[1440, 1920, 390])
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2]
                        / 'outputs/biomaster_explorer/readability')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports, errors, checks = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox', '--use-gl=angle',
                                          '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1050})
        page = context.new_page()
        page.set_default_timeout(25000)
        page.on('pageerror', lambda error: errors.append(str(error)))
        routes = {'home': '/', 'catalog': '/#/drugs',
                  'target': '/#/target/CHEMBL203/overview',
                  'rankings': '/#/target/CHEMBL203/rankings',
                  'diseases': '/#/target/CHEMBL203/diseases',
                  'metadata': '/#/target/CHEMBL203/metadata',
                  'spr': '/#/target/CHEMBL2326/experiments',
                  'sources': '/#/sources',
                  'network': '/#/target/CHEMBL203/network'}

        def capture(name: str, width: int) -> None:
            # Wait for finite entrance transitions before evaluating opacity.
            page.wait_for_timeout(650)
            result = page.evaluate(TEXT_AUDIT)
            result.update(page=name, viewport=width)
            reports.append(result)
            page.screenshot(path=str(args.out / f'{name}-{width}.png'), full_page=True)
            # Preserve audit evidence if a later interaction assertion fails.
            (args.out / 'READABILITY_VALIDATION.json').write_text(json.dumps(
                {'status': 'running', 'url': args.url, 'checks': checks,
                 'page_errors': errors, 'reports': reports}, indent=2, ensure_ascii=False))
            print(f'{name:16s} {width}: text={result["text_count"]} '
                  f'small={len(result["small_text"])} '
                  f'contrast={len(result["low_contrast"])} overflow={result["overflow"]}', flush=True)

        for width in args.viewports:
            height = 844 if width < 600 else 1080 if width >= 1920 else 1050
            page.set_viewport_size({'width': width, 'height': height})
            for name in args.pages:
                route = routes[name]
                page.goto(args.url + route, wait_until='networkidle')
                expect(page.locator('main')).to_be_visible()
                if name == 'catalog':
                    expect(page.locator('.catalog-table tbody tr')).to_have_count(24)
                elif name == 'rankings':
                    expect(page.locator('.ranking-table tbody tr')).to_have_count(20)
                capture(name, width)

        if not args.skip_interactions:
            page.set_viewport_size({'width': 1440, 'height': 1050})
            page.goto(args.url + '/#/drugs', wait_until='networkidle')
            page.get_by_label('筛选目录', exact=True).fill('imatinib')
            expect(page.locator('.catalog-table tbody tr').first).to_contain_text('imatinib')
            checks.append('catalog_filter')

            search = page.locator('.topbar input[role="combobox"]')
            page.keyboard.press('Control+k')
            expect(search).to_be_focused()
            search.fill('EGFR')
            expect(page.get_by_role('option').first).to_be_visible()
            expect(search).to_have_attribute('aria-expanded', 'true')
            active_id = search.get_attribute('aria-activedescendant')
            assert active_id and page.locator(f'[id="{active_id}"]').count(), 'Search active option must be announced'
            search.press('Enter')
            expect(page.locator('.entity-heading h1')).to_have_text('EGFR')
            expect(page.get_by_role('listbox')).to_have_count(0)
            checks.append('search_shortcut_keyboard_selection')
            overview_tab = page.get_by_role('button', name='综合概览', exact=True)
            overview_tab.focus()
            overview_tab.press('ArrowRight')
            expect(page.get_by_role('button', name='老药排名', exact=True)).to_have_attribute('aria-current', 'page')
            expect(page.locator('.ranking-table tbody tr')).to_have_count(20)
            checks.append('detail_tabs_arrow_keys')
            model = page.get_by_role('button', name='DrugCLIP', exact=True)
            model.click()
            expect(page.locator('.ranking-context')).to_contain_text('720')
            page.get_by_label('筛选排名', exact=True).fill('imatinib')
            expect(page.locator('.ranking-table tbody tr')).to_have_count(1)
            expect(page.locator('.ranking-context')).to_contain_text('720')
            checks.append('model_switch_filter_denominator')
            cell = page.locator('.ranking-table tbody tr').first.locator('.score-cell').first
            assert cell.locator('strong').inner_text().startswith('#')
            page.get_by_role('button', name='分数优先', exact=True).click()
            assert not cell.locator('strong').inner_text().startswith('#')
            expect(cell.locator('span')).to_contain_text('/ 720')
            page.get_by_role('button', name='排名优先', exact=True).click()
            assert cell.locator('strong').inner_text().startswith('#')
            checks.append('rank_score_reading_mode')

            row = page.locator('.ranking-table tbody tr').first
            row.focus()
            row.press('Enter')
            dialog = page.get_by_role('dialog')
            expect(dialog).to_be_visible()
            assert page.evaluate('!!document.activeElement.closest("[role=dialog]")'), 'Drawer must receive keyboard focus'
            page.keyboard.press('Shift+Tab')
            assert page.evaluate('!!document.activeElement.closest("[role=dialog]")'), 'Drawer must contain reverse tab'
            page.keyboard.press('Tab')
            assert page.evaluate('!!document.activeElement.closest("[role=dialog]")'), 'Drawer must contain forward tab'
            capture('drawer', 1440)
            page.keyboard.press('Escape')
            expect(dialog).to_have_count(0)
            expect(row).to_be_focused()
            checks.append('drawer_keyboard_focus_trap_escape_restore')
            search.focus()
            search.press('Tab')
            focus = page.evaluate('''() => { const s=getComputedStyle(document.activeElement);
              return {outline: s.outlineStyle, width: parseFloat(s.outlineWidth), shadow: s.boxShadow}; }''')
            assert (focus['outline'] != 'none' and focus['width'] >= 2) or focus['shadow'] != 'none', focus
            checks.append('visible_keyboard_focus')

            page.goto(args.url + '/#/target/NO_SUCH_TARGET/overview', wait_until='networkidle')
            expect(page.get_by_text('数据暂时无法加载', exact=True)).to_be_visible()
            capture('error', 1440)
            lookup = context.request.get(args.url + '/api/search?kind=target&q=ADRB2').json()['items'][0]
            page.goto(args.url + f'/#/target/{lookup["id"]}/rankings', wait_until='networkidle')
            expect(page.get_by_text('当前模型没有可用排名', exact=True)).to_be_visible()
            capture('empty', 1440)
            checks.append('readable_empty_and_error_states')

            page.set_viewport_size({'width': 390, 'height': 844})
            page.goto(args.url + '/', wait_until='networkidle')
            page.get_by_label('展开导航', exact=True).click()
            nav = page.locator('.sidebar nav button').filter(has_text='靶点图谱')
            expect(nav).to_be_visible()
            nav.click()
            expect(page.locator('.catalog-table tbody tr')).to_have_count(24)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            page.goto(args.url + '/#/target/CHEMBL203/structure', wait_until='networkidle')
            expect(page.get_by_label('选择结构口袋', exact=True)).to_be_visible()
            capture('structure', 390)
            checks.append('mobile_navigation_and_structure')

            page.emulate_media(reduced_motion='reduce')
            page.goto(args.url + '/#/target/CHEMBL203/overview', wait_until='networkidle')
            page.wait_for_timeout(650)
            running = page.evaluate('''() => document.getAnimations().filter(a =>
                a.playState === 'running' && a.effect?.getTiming().iterations === Infinity
            ).map(a=>({name:a.animationName, duration:a.effect.getTiming().duration}))''')
            assert not running, f'Reduced motion leaves perpetual animations: {running}'
            checks.append('prefers_reduced_motion')

        failures = [{'page': r['page'], 'viewport': r['viewport'],
                     'overflow': r['overflow'], 'small_text': r['small_text'],
                     'low_contrast': r['low_contrast']} for r in reports
                    if r['overflow'] or r['small_text'] or r['low_contrast']]
        result = {'status': 'passed' if not failures and not errors else 'needs_review',
                  'url': args.url, 'checks': checks, 'page_errors': errors,
                  'failures': failures, 'reports': reports}
        (args.out / 'READABILITY_VALIDATION.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        browser.close()
        print(f'Visual audit: {result["status"]}; report: {args.out / "READABILITY_VALIDATION.json"}')
        if not args.report_only:
            assert not errors, errors
            assert not failures, f'{len(failures)} page/viewport combinations need readability fixes; see report'


if __name__ == '__main__':
    main()
