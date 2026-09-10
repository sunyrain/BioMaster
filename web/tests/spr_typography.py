"""Authenticated visual regression for the SPR detail hierarchy.
Set BIOMASTER_TEST_PASSWORD or enter a password at the prompt.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='https://palinova.xyz')
parser.add_argument('--user',default='txc')
args=parser.parse_args()
password=os.environ.get('BIOMASTER_TEST_PASSWORD') or getpass.getpass('Test account password: ')
out=Path('outputs/spr_visual_design_20260910');out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(args=['--no-sandbox'])
    context=browser.new_context(reduced_motion='reduce')
    page=context.new_page();errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(args.url+'/login',timeout=60000)
    page.get_by_label('用户名',exact=True).fill(args.user)
    page.get_by_label('密码',exact=True).fill(password)
    page.get_by_role('button',name='登录',exact=True).click()
    expect(page.locator('.rh-intro h1')).to_be_visible(timeout=45000)
    checks=[]
    for width in [1440,1920,390]:
        page.set_viewport_size({'width':width,'height':1050 if width>700 else 844})
        page.goto(args.url+'/#/spr',wait_until='domcontentloaded')
        row=page.locator('.experiment-record').first
        expect(row).to_be_visible(timeout=45000)
        if row.get_attribute('open') is None:row.locator(':scope > summary').click()
        first=row.locator('.experiment-content > :first-child')
        expect(first).to_have_class('spr-attempt-chain spr-primary-summary')
        assert first.locator('small').all_text_contents()==['原适应症','原靶点','目标靶点','目标适应症']
        review=row.locator('.spr-review-details')
        assert review.get_attribute('open') is None
        toggle=review.locator(':scope > summary')
        assert toggle.bounding_box()['height']<=56
        fonts=first.locator('p').evaluate_all('els=>els.map(e=>({size:parseFloat(getComputedStyle(e).fontSize),line:getComputedStyle(e).lineHeight}))')
        assert len({v['size'] for v in fonts})==1,fonts
        assert 14<=fonts[0]['size']<=16
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        first.scroll_into_view_if_needed()
        page.screenshot(path=str(out/f'detail-{width}.png'))
        toggle.focus();page.keyboard.press('Enter');expect(review).to_have_attribute('open','')
        page.keyboard.press('Enter');assert review.get_attribute('open') is None
        source=row.locator('.spr-source');source.locator(':scope > summary').click()
        expect(source.get_by_text('原用途范围',exact=True)).to_be_visible()
        source.locator(':scope > summary').click()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        checks.append({'width':width,'primary_fonts':fonts,'compact_disclosure':True,'keyboard_and_provenance':True})
    assert not errors,errors
    page.get_by_role('button',name='退出登录',exact=True).click()
    browser.close()
    (out/'VALIDATION.json').write_text(json.dumps({'status':'passed','checks':checks,'page_errors':errors},indent=2))
    print('PASS SPR typography, disclosure, keyboard, original data and responsive layout')
