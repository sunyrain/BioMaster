"""Homepage evidence entrances must yield searchable records before entity selection."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:18891');args=parser.parse_args()
out=Path('outputs/browse_refresh_20260909');out.mkdir(exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(args=['--no-sandbox'])
 context=browser.new_context(viewport={'width':1440,'height':1000},reduced_motion='reduce')
 page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 report={}
 for text,slug in [('ChEMBL 37','chembl'),('Open Targets / Reactome','pathways'),('AlphaFold / P2Rank','structures'),('TxGNN','txgnn')]:
  page.goto(args.url,wait_until='networkidle')
  page.locator('.rh-evidence-card').filter(has_text=text).click()
  expect(page).to_have_url(args.url+'/#/browse/'+slug)
  expect(page.locator('.browse-record')).to_have_count(20,timeout=30000)
  page.locator('.browse-record details').first.locator('summary').click()
  expect(page.locator('.browse-record-details').first).to_be_visible()
  page.screenshot(path=str(out/(slug+'.png')),full_page=False)
  report[slug]=page.locator('.browse-stats').inner_text()
  page.get_by_label('搜索浏览目录').fill('no-such-record-zzzz')
  expect(page.get_by_text('没有匹配的记录',exact=True)).to_be_visible()
  page.get_by_role('button',name='清除筛选',exact=True).click()
  expect(page.locator('.browse-record')).to_have_count(20)
 page.goto(args.url,wait_until='networkidle')
 page.locator('.rh-metric').filter(has_text='结构与口袋').click()
 expect(page).to_have_url(args.url+'/#/browse/structures')
 page.get_by_label('搜索浏览目录').fill('CA7')
 expect(page.locator('.browse-record')).to_have_count(1)
 expect(page.locator('.browse-record')).to_contain_text('CHEMBL2326')
 page.get_by_role('button',name='打开三维工作台',exact=True).click()
 expect(page).to_have_url(args.url+'/#/target/CHEMBL2326/structure')
 page.goto(args.url,wait_until='networkidle')
 page.locator('.rh-metric').filter(has_text='配对检索空间').click()
 expect(page).to_have_url(args.url+'/#/browse/pairs')
 expect(page.locator('.browse-record')).to_have_count(20)
 page.get_by_role('button',name='从靶点出发',exact=True).click()
 expect(page.locator('.browse-stats')).to_contain_text('384')
 page.get_by_label('搜索浏览目录').fill('CA7')
 expect(page.locator('.browse-record')).to_have_count(1)
 page.get_by_role('button',name='查看完整排名',exact=True).click()
 expect(page.locator('.ranking-table tbody tr')).to_have_count(20,timeout=30000)
 page.goto(args.url+'/#/browse/chembl',wait_until='networkidle')
 page.get_by_label('筛选证据类型').select_option('approved')
 expect(page.locator('.browse-match')).to_contain_text('3,017')
 page.get_by_role('button',name='作用机制',exact=True).click()
 expect(page.locator('.browse-record')).to_have_count(20)
 page.get_by_label('下一页浏览记录').click()
 expect(page.locator('.browse-pagination')).to_contain_text('第 2')
 page.set_viewport_size({'width':390,'height':844})
 for slug in ['chembl','pathways','structures','txgnn','pairs']:
  page.goto(args.url+'/#/browse/'+slug,wait_until='networkidle')
  expect(page.locator('.browse-record')).to_have_count(20)
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),slug
 page.screenshot(path=str(out/'mobile.png'),full_page=False)
 assert not errors,errors
 report['page_errors']=errors
 (out/'BROWSE_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 browser.close();print('PASS evidence entrances, records, search, filters, pagination, explicit selection and mobile layout')
