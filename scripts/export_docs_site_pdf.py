from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pypdf import PdfWriter
from playwright.async_api import async_playwright


DEFAULT_PAGES = [
    ("index.html", "01-home"),
    ("results-report.html", "02-results-report"),
    ("status-report.html", "03-status-report"),
    ("data-summary.html", "04-data-summary"),
    ("sota-readiness.html", "05-sota-readiness"),
    ("artifacts.html", "06-artifacts"),
    ("main-figure.html", "07-main-figure"),
    ("formal-report.html", "08-formal-report"),
]


PRINT_CSS = """
@page {
  size: A3 landscape;
  margin: 9mm;
}

html,
body {
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
  overflow: visible !important;
  background: #ffffff !important;
}

* {
  -webkit-print-color-adjust: exact !important;
  print-color-adjust: exact !important;
  text-shadow: none !important;
}

.site-header,
.top-nav {
  position: static !important;
}

.site-header {
  break-after: avoid;
}

main,
.section-band,
.hero-band {
  width: 100% !important;
  max-width: none !important;
}

.section-band,
.hero-band {
  break-inside: auto;
  page-break-inside: auto;
  padding-top: 18px !important;
  padding-bottom: 18px !important;
}

.hero-layout,
.runway-layout,
.compute-status-grid,
.architecture-grid,
.candidate-layout,
.quality-grid,
.sota-grid,
.validation-layout,
.artifact-grid {
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
}

.metric-grid,
.executive-grid,
.presentation-grid,
.direction-grid,
.candidate-grid,
.analytics-grid,
.gpu-telemetry-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
}

.candidate-detail,
.wide-panel,
.data-table-wrap,
.table-wrap,
table {
  max-width: 100% !important;
  overflow: visible !important;
}

table {
  width: 100% !important;
  font-size: 8.5px !important;
}

pre,
code {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
}

details {
  display: block !important;
}

details > * {
  display: block !important;
}

details summary {
  display: list-item !important;
}

.button,
.nav-action {
  color: inherit !important;
  text-decoration: none !important;
}

canvas,
svg,
img {
  max-width: 100% !important;
}
"""


async def auto_scroll(page) -> None:
    await page.evaluate(
        """
        async () => {
          await new Promise((resolve) => {
            let totalHeight = 0;
            const distance = 700;
            const timer = setInterval(() => {
              window.scrollBy(0, distance);
              totalHeight += distance;
              if (totalHeight >= document.body.scrollHeight) {
                clearInterval(timer);
                window.scrollTo(0, 0);
                resolve();
              }
            }, 40);
          });
        }
        """
    )


async def render_page(context, base_url: str, html_name: str, out_pdf: Path) -> None:
    page = await context.new_page()
    await page.goto(f"{base_url.rstrip('/')}/{html_name}", wait_until="networkidle", timeout=120_000)
    await page.wait_for_timeout(1500)
    await page.evaluate(
        """
        () => {
          document.querySelectorAll('details').forEach((el) => { el.open = true; });
          document.querySelectorAll('[hidden]').forEach((el) => { el.hidden = false; });
          document.querySelectorAll('.is-hidden, .hidden').forEach((el) => {
            el.classList.remove('is-hidden', 'hidden');
          });
        }
        """
    )
    await auto_scroll(page)
    await page.emulate_media(media="screen")
    await page.add_style_tag(content=PRINT_CSS)
    await page.pdf(
        path=str(out_pdf),
        format="A3",
        landscape=True,
        print_background=True,
        prefer_css_page_size=True,
        display_header_footer=True,
        header_template="<div></div>",
        footer_template=(
            "<div style=\"font-size:7px;color:#52606d;width:100%;padding:0 10mm;"
            "display:flex;justify-content:space-between;\">"
            "<span>BioMaster website snapshot</span>"
            "<span><span class=\"pageNumber\"></span> / <span class=\"totalPages\"></span></span>"
            "</div>"
        ),
    )
    await page.close()


def merge_pdfs(parts: list[Path], output: Path) -> None:
    writer = PdfWriter()
    for part in parts:
        writer.append(str(part))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    executable = Path(args.chromium)
    if not executable.exists():
        raise FileNotFoundError(f"Chromium executable not found: {executable}")

    rendered_parts: list[Path] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=str(executable),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            locale="zh-CN",
            viewport={"width": 1800, "height": 1200},
            device_scale_factor=1,
        )
        for html_name, stem in DEFAULT_PAGES:
            out_pdf = parts_dir / f"{stem}.pdf"
            await render_page(context, args.base_url, html_name, out_pdf)
            rendered_parts.append(out_pdf)
            print(f"rendered {html_name} -> {out_pdf}")
        await browser.close()

    output = Path(args.output)
    merge_pdfs(rendered_parts, output)
    print(f"merged {len(rendered_parts)} pages -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the static docs website to a merged PDF.")
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument(
        "--chromium",
        default="/root/autodl-tmp/ms-playwright/chromium-1223/chrome-linux64/chrome",
    )
    parser.add_argument("--output-dir", default="outputs/site_pdf")
    parser.add_argument("--output", default="docs/assets/biomaster-website-full-snapshot.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
