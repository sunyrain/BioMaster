"""Cancellation, retry, expiry and timeout tests for the shared request client.
Build the fixture with web/node_modules/.bin/esbuild web/src/request.ts --bundle
--format=iife --global-name=RequestClient --outfile=/tmp/biomaster_request_test.js
"""
import subprocess
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright
fixture = Path(tempfile.gettempdir()) / "biomaster_request_test.js"
subprocess.run(["web/node_modules/.bin/esbuild", "web/src/request.ts", "--bundle", "--format=iife", "--global-name=RequestClient", f"--outfile={fixture}"], check=True)
with sync_playwright() as p:
    browser = p.chromium.launch(args=['--no-sandbox'])
    page = browser.new_page()
    page.add_script_tag(path=str(fixture))
    result = page.evaluate('''async () => {
      const assert = (condition, message) => { if (!condition) throw Error(message); };
      let calls = 0;
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      window.fetch = async () => { calls++; await sleep(40); return new Response('{"value":7}'); };
      const c = new AbortController();
      const cancelled = RequestClient.api('/shared', c.signal).catch(e => e.name);
      const active = RequestClient.api('/shared');
      c.abort();
      assert(await cancelled === 'AbortError', 'cancelled consumer');
      assert((await active).value === 7 && calls === 1, 'independent shared consumer');
      await RequestClient.api('/shared');
      assert(calls === 1, 'cached return');
      const now = Date.now;
      Date.now = () => now() + 61000;
      await RequestClient.api('/shared');
      assert(calls === 2, 'expired snapshot reloaded');
      Date.now = now;
      calls = 0;
      window.fetch = async () => { calls++; return calls === 1 ? new Response('gateway', {status:502}) : new Response('{"ok":true}'); };
      assert((await RequestClient.api('/retry')).ok && calls === 2, 'transient retry');
      calls = 0;
      window.fetch = async () => { calls++; return new Response('{"error":"missing"}', {status:404}); };
      await RequestClient.api('/failure').catch(e => assert(e.message === 'missing', 'error message'));
      window.fetch = async () => { calls++; return new Response('{"ok":true}'); };
      assert((await RequestClient.api('/failure')).ok && calls === 2, 'failures not cached');
      const timer = window.setTimeout;
      window.setTimeout = (fn, ms) => timer(fn, ms === 30000 ? 20 : ms);
      window.fetch = (_, {signal}) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted','AbortError'))));
      const message = await RequestClient.api('/timeout').catch(e => e.message);
      assert(message.includes('超时'), 'timeout has actionable feedback');
      return {cancel: true, dedup: true, cache: true, expiry: true, retry: true, failureRecovery: true, timeout: true};
    }''')
    print(result)
    browser.close()
