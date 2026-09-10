"""Run only against an isolated result database: writes synthetic observations."""
import argparse
import csv
import io
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from biomaster.explorer_spr_results import LABELS
from playwright.sync_api import sync_playwright, expect

parser=argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:18767')
parser.add_argument('--isolated-results-db',action='store_true',required=True)
args=parser.parse_args()
out=Path('outputs/biomaster_explorer/spr_upload_validation');out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1080},reduced_motion='reduce')
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(args.url+'/#/spr',wait_until='networkidle')
    page.get_by_role('button',name='实验结果 · 上传与记录').click()
    expect(page.get_by_role('heading',name='让实验结果回到研究配对')).to_be_visible()
    response=page.request.get(args.url+'/api/spr-results/template.csv')
    assert response.status==200
    template=list(csv.DictReader(io.StringIO(response.body().decode('utf-8-sig'))))
    assert len(template)==384
    assert [r['实验编号'] for r in template[:2]]==['C384-001','C384-040']
    assert [r['模板序号'] for r in template[:2]]==['1','2']
    aliases={v:k for k,v in LABELS.items()}
    template=[{aliases.get(k,k):v for k,v in r.items()} for r in template]
    page.get_by_role('button',name='CSV 批量上传',exact=False).click()
    record=next(row for row in template if row['experiment_id']=='C384-001')
    row={**record,'sample_id':'TEST-1','experiment_date':'2026-09-09','result':'detected','qc':'pass','KD':'0.25','KD_unit':'uM','notes':'UI integration test, isolated database only'}
    def content(rows):
        out=io.StringIO();writer=csv.DictWriter(out,fieldnames=[LABELS[k] for k in record]);writer.writeheader()
        for row in rows:
            row={**row,'result':{'detected':'检出响应','not_detected':'未检出响应','inconclusive':'无法判定'}.get(row.get('result'),row.get('result','')),'qc':{'pass':'通过','fail':'未通过','review':'待确认'}.get(row.get('qc'),row.get('qc',''))}
            writer.writerow({LABELS[k]:v for k,v in row.items()})
        return out.getvalue().encode()
    batch='UI-ISOLATED-'+str(time.time_ns())
    page.get_by_label('实验批次',exact=False).fill(batch)
    page.locator('#spr-result-file').set_input_files({'name':'bad.csv','mimeType':'text/csv','buffer':content([{**row,'pair_id':'WRONG'}])})
    page.get_by_role('button',name='校验并预览',exact=True).click()
    expect(page.get_by_text('配对编号与设计不一致',exact=False)).to_be_visible()
    expect(page.get_by_role('button',name='确认导入 0 条')).to_be_disabled()
    page.locator('#spr-result-file').set_input_files({'name':'good.csv','mimeType':'text/csv','buffer':content([row,{**row,'sample_id':'TEST-2','KD':'','KD_unit':'','result':'not_detected'}])})
    page.get_by_role('button',name='校验并预览',exact=True).click()
    expect(page.get_by_role('button',name='确认导入 2 条')).to_be_enabled()
    preview=page.get_by_role('region',name='上传校验预览')
    expect(preview).to_contain_text('250')
    page.screenshot(path=str(out/'upload-preview-1440.png'),full_page=True)
    page.get_by_role('button',name='确认导入 2 条').click()
    expect(page.get_by_text('条结果已保存',exact=False)).to_be_visible()
    history=page.get_by_role('region',name='已上传 SPR 结果')
    page.get_by_label('搜索已上传结果').fill(batch)
    expect(history.locator('tbody tr')).to_have_count(2)
    history.get_by_role('button',name='查看记录').first.click()
    expect(history).to_contain_text('UI integration test')
    response=page.request.get(args.url+'/api/spr-results.csv?search='+batch)
    exported=list(csv.DictReader(io.StringIO(response.body().decode('utf-8-sig'))))
    assert len(exported)==2 and exported[0]['review_status']=='pending'
    page.reload(wait_until='networkidle');page.get_by_role('button',name='实验结果 · 上传与记录').click()
    page.get_by_label('搜索已上传结果').fill(batch)
    expect(page.get_by_role('region',name='已上传 SPR 结果').locator('tbody tr')).to_have_count(2)
    # Online entry, editing and commit use the same server validation.
    page.get_by_role('button',name='在线填写',exact=False).click()
    page.get_by_label('实验批次',exact=False).fill(batch+'-ONLINE')
    page.get_by_label('搜索配对',exact=False).fill('C384-001')
    pair=page.locator('.sri-pair-list button').first
    expect(pair).to_contain_text('tecovirimat')
    pair.click()
    page.get_by_label('样本编号 *',exact=True).fill('ONLINE-1')
    page.get_by_label('响应结果 *',exact=True).select_option('detected')
    page.get_by_label('质控结果 *',exact=True).select_option('pass')
    page.get_by_label('KD（可选）',exact=True).fill('0.25')
    page.get_by_label('KD 单位',exact=True).select_option('uM')
    page.get_by_role('button',name='加入待提交列表',exact=True).click()
    page.get_by_role('button',name='编辑 ONLINE-1',exact=True).click()
    page.get_by_label('KD（可选）',exact=True).fill('0.5')
    page.get_by_role('button',name='保存本条修改',exact=True).click()
    page.get_by_role('button',name='校验在线记录并预览',exact=True).click()
    expect(page.get_by_role('region',name='上传校验预览')).to_contain_text('500')
    page.get_by_role('button',name='确认导入 1 条').click()
    expect(page.get_by_text('1 条结果已保存',exact=False)).to_be_visible()
    # Switching result to not detected clears the previously entered KD.
    page.locator('.sri-pair-list button').first.click()
    page.get_by_label('响应结果 *',exact=True).select_option('detected')
    page.get_by_label('KD（可选）',exact=True).fill('1')
    page.get_by_label('KD 单位',exact=True).select_option('nM')
    page.get_by_label('响应结果 *',exact=True).select_option('not_detected')
    expect(page.get_by_label('KD（可选）',exact=True)).to_have_count(0)
    page.get_by_role('button',name='清空本条 / 重选配对').click()
    page.get_by_label('搜索配对',exact=False).fill('')
    for width in [1440,390]:
        page.set_viewport_size({'width':width,'height':1000 if width==1440 else 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),width
        page.screenshot(path=str(out/f'upload-records-{width}.png'),full_page=True)
    assert not errors,errors
    (out/'VALIDATION.json').write_text(json.dumps({'template_rows':len(template),'checks':['identity mismatch blocked','preview unit conversion','two records committed','full filtered CSV export','reload persistence','desktop and mobile no page overflow','online select, edit, preview and commit','conditional KD clearing','priority order and continuous template numbering'],'page_errors':errors},ensure_ascii=False,indent=2))
    browser.close()
print('PASS SPR upload browser workflow (isolated database)')
