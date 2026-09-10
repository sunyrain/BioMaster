"""Browser regression for the SPR layout and its new filtering controls."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18765")
    parser.add_argument("--out", type=Path, default=Path("outputs/ui_refresh_20260909"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        for width in (1440, 1920, 390):
            context = browser.new_context(viewport={"width": width, "height": 1000 if width > 700 else 844}, reduced_motion="reduce")
            # A display package must not depend on a font, script or API CDN.
            external = []
            def route(request):
                if request.request.url.startswith((args.url, "data:", "blob:")):
                    request.continue_()
                else:
                    external.append(request.request.url)
                    request.abort()
            context.route("**/*", route)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(args.url + "/#/target/CHEMBL2326/experiments", wait_until="networkidle")
            rows = page.locator(".experiment-record")
            expect(rows).to_have_count(8)
            summary = page.locator(".experiment-summary")
            numeric_size = summary.locator("strong").first.evaluate("e => parseFloat(getComputedStyle(e).fontSize)")
            assert 24 <= numeric_size <= 36, numeric_size
            assert summary.bounding_box()["height"] < (150 if width > 700 else 240)
            assert page.locator(".spr-heading h2").bounding_box()["height"] < 40
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            page.screenshot(path=str(args.out / f"spr-{width}.png"), full_page=True)
            filters = page.get_by_role("group", name="实验分组筛选")
            filters.get_by_role("button", name="参考对照", exact=False).click()
            expect(rows).to_have_count(1)
            expect(rows.first).to_contain_text("Ethoxzolamide")
            filters.get_by_role("button", name="模型候选", exact=False).click()
            expect(rows).to_have_count(7)
            filters.get_by_role("button", name="全部配对", exact=False).click()
            page.get_by_label("搜索实验配对", exact=True).fill("crisaborole")
            expect(rows).to_have_count(1)
            page.get_by_label("搜索实验配对", exact=True).fill("no-such-experiment")
            expect(page.get_by_text("没有匹配的实验配对", exact=True)).to_be_visible()
            page.get_by_role("button", name="清除筛选", exact=True).click()
            expect(rows).to_have_count(8)
            first = rows.first.locator(":scope > summary")
            first.focus()
            page.keyboard.press("Enter")
            expect(rows.first).to_have_attribute("open", "")
            expect(rows.first.get_by_role("heading", name="实验准备", exact=True)).to_be_visible()
            expect(rows.first).to_contain_text("可溶性全长胞内酶")
            rows.first.locator(".spr-source > summary").click()
            expect(rows.first.locator(".spr-source")).to_have_attribute("open", "")
            expect(rows.first.locator(".spr-source")).to_contain_text("NO_EXPERIMENTAL_RESULT_IN_DESIGN_ARTIFACT")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            assert not errors, errors
            assert not external, external
            reports.append({"width": width, "numeric_font_px": numeric_size, "summary_height": summary.bounding_box()["height"], "filters": "passed", "keyboard": "passed", "horizontal_overflow": False, "external_requests": external})
            context.close()
        browser.close()
    (args.out / "DESIGN_VALIDATION.json").write_text(json.dumps({"status": "passed", "reports": reports}, ensure_ascii=False, indent=2))
    print("PASS: desktop, wide desktop and mobile SPR layout, filters, keyboard and provenance")


if __name__ == "__main__":
    main()
