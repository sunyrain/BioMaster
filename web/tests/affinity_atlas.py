"""Read-only browser acceptance for sparse affinity overview and exact cells."""
import argparse,json,math
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:18767');args=parser.parse_args()
out=Path('outputs/affinity_atlas_inventory_20260910');out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100},reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(args.url+'/#/affinity',wait_until='networkidle')
 expect(page.get_by_role('heading',name='亲和证据 · 点亮矩阵')).to_be_visible(timeout=30000)
 d=page.request.get(args.url+'/api/affinity-matrix').json()
 assert len(d['drugs'])==720 and len(d['targets'])==888
 assert d['inventory']['exact_pairs']==8456 and d['inventory']['exact_rows']==39578
 expect(page.locator('.aa-grid button')).to_have_count(576)
 measured=next(c for c in d['cells'] if c[2]>0 and c[3]>0)
 page.get_by_label('选择药物块').select_option(str(measured[0]//24))
 page.get_by_label('选择靶点块').select_option(str(measured[1]//24))
 cols=min(24,888-measured[1]//24*24)
 index=(measured[0]%24)*cols+measured[1]%24
 page.locator('.aa-grid button').nth(index).click()
 evidence=page.get_by_role('region',name='矩阵配对证据')
 expect(evidence).to_contain_text('来源数值记录',timeout=30000)
 expect(evidence.locator('tbody tr').first).to_be_visible()
 page.get_by_role('button',name='亲和结果 · 阈值分层').click()
 expect(page.get_by_label('亲和端点')).to_be_visible()
 page.get_by_label('亲和分层阈值').select_option('100')
 expect(page.get_by_text('正在更新筛选',exact=False)).to_have_count(0,timeout=30000)
 expect(page.locator('.aa-legend')).to_contain_text('阈值结论冲突')
 page.get_by_label('矩阵证据来源').select_option('CACHE3')
 expect(page.get_by_text('正在更新筛选',exact=False)).to_have_count(0,timeout=30000)
 expect(page.locator('.aa-legend')).to_contain_text('满足所选阈值 · 0')
 page.get_by_label('矩阵证据来源').select_option('all')
 expect(page.get_by_text('正在更新筛选',exact=False)).to_have_count(0,timeout=30000)
 for width in [1440,390]:
  page.set_viewport_size({'width':width,'height':1100 if width==1440 else 844})
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),width
  page.locator('.aa-controls').scroll_into_view_if_needed()
  page.screenshot(path=str(out/f'affinity-atlas-{width}.png'),full_page=True)
 assert not errors,errors
 (out/'BROWSER_VALIDATION.json').write_text(json.dumps({'axes':[720,888],'overview_blocks':[30,37],'local_cells':576,'exact_pairs':8456,'checks':['source filter','threshold switching','cell evidence','mobile overflow','no JavaScript errors'],'page_errors':errors},ensure_ascii=False,indent=2))
 b.close()
print('PASS affinity atlas overview, exact evidence, filters and responsive layout')
