"""Password login and bounded server-side sessions for the hosted explorer."""
import hashlib
import hmac
import html
import json
import secrets
import threading
import time
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlsplit

COOKIE = 'biomaster_session'
TTL = 12 * 3600

class Auth:
    def __init__(self, path):
        self.users = json.loads(path.read_text())
        self.sessions = {}
        self.attempts = {}
        self.lock = threading.Lock()

    def username(self, handler):
        try:
            cookie = SimpleCookie(handler.headers.get('Cookie', ''))
            token = cookie[COOKIE].value
        except Exception:
            return None
        with self.lock:
            entry = self.sessions.get(token)
            if entry and entry[1] > time.time(): return entry[0]
            self.sessions.pop(token, None)
        return None

    def verify(self, username, password, address):
        now = time.time()
        with self.lock:
            self.attempts = {k:v for k,v in self.attempts.items() if v[1] > now}
            key = (address, username)
            count, expiry = self.attempts.get(key, (0, now + 300))
            if count >= 8 or len(self.attempts) >= 10000: return None, 429
            self.attempts[key] = (count + 1, expiry)
        record = self.users.get(username)
        salt = record['salt'] if record else '0' * 32
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 260000).hex()
        if not record or not hmac.compare_digest(digest, record['hash']): return None, 401
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.attempts.pop(key, None)
            self.sessions = {k:v for k,v in self.sessions.items() if v[1] > now}
            if len(self.sessions) >= 1000: self.sessions.pop(next(iter(self.sessions)))
            self.sessions[token] = (username, now + TTL)
        return token, 200

    def redirect(self, h, destination, token=None):
        h.send_response(303)
        h.send_header('Location', destination)
        h.send_header('Cache-Control', 'no-store')
        h.send_header('Content-Length', '0')
        if token is not None:
            secure = '; Secure' if h.headers.get('X-Forwarded-Proto', '').lower() == 'https' else ''
            h.send_header('Set-Cookie', f'{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={TTL if token else 0}{secure}')
        h.end_headers()

    def get(self, h):
        path = urlsplit(h.path).path
        if path == '/login':
            if self.username(h): self.redirect(h, '/')
            else: self.login_page(h)
            return True
        if self.username(h): return False
        if path.startswith('/api/'):
            h._json({'error':'请先登录后继续浏览。'}, 401)
        else: self.redirect(h, '/login')
        return True

    def post(self, h):
        origin = h.headers.get('Origin')
        if (origin and urlsplit(origin).netloc != h.headers.get('Host')) or h.headers.get('Sec-Fetch-Site') == 'cross-site':
            h._json({'error':'请求来源不匹配。'}, 403); return
        path = urlsplit(h.path).path
        if path == '/logout':
            try:
                cookie = SimpleCookie(h.headers.get('Cookie', ''))
                with self.lock: self.sessions.pop(cookie[COOKIE].value, None)
            except Exception: pass
            self.redirect(h, '/login', ''); return
        if path != '/login': h._json({'error':'Not found'}, 404); return
        try: size = int(h.headers.get('Content-Length', '0'))
        except ValueError: size = -1
        if not 0 < size <= 4096: h._json({'error':'无效的登录请求。'}, 400); return
        fields = parse_qs(h.rfile.read(size).decode('utf-8', errors='replace'))
        username = fields.get('username', [''])[0].strip().lower()
        password = fields.get('password', [''])[0]
        token, status = self.verify(username, password, h.headers.get('CF-Connecting-IP', h.client_address[0]))
        if token: self.redirect(h, '/', token)
        else: self.login_page(h, '尝试次数过多，请 5 分钟后重试。' if status == 429 else '用户名或密码不正确，请重试。', status)

    def login_page(self, h, message='', status=200):
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>登录 · Palinova</title>
<style>*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#f2f5f9;color:#142338;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}.card{width:100%;max-width:420px;background:white;border:1px solid #dbe3ee;border-radius:18px;overflow:hidden;box-shadow:0 12px 48px #0b172b0d}.brand{background:#0b172b;padding:28px 32px;color:white}.brand strong{font-size:27px}.brand p{margin:4px 0 0;color:#b9c8df;font-size:12px;letter-spacing:2px}form{padding:28px 32px}h1{font-size:22px;margin:0 0 6px}.intro{color:#526176;margin:0 0 24px}label{display:block;margin:16px 0 6px}input{width:100%;padding:12px;border:1px solid #cbd6e6;border-radius:8px;font:inherit}input:focus{outline:2px solid #2459e8;outline-offset:2px}button{margin-top:24px;width:100%;padding:12px;background:#2459e8;color:white;border:0;border-radius:8px;font:600 15px system-ui;cursor:pointer}.error{color:#a12d2d;margin:12px 0}.foot{font-size:12px;color:#526176;margin-top:18px;text-align:center}</style>
<div class="card"><div class="brand"><strong>Palinova</strong><p>DISCOVERY ATLAS</p></div><form method="post" action="/login"><h1>登录研究工作区</h1><p class="intro">使用你的账号，浏览研究图谱与 SPR 设计。</p>MESSAGE<label for="username">用户名</label><input id="username" name="username" autocomplete="username" required maxlength="64" autofocus><label for="password">密码</label><input id="password" name="password" type="password" autocomplete="current-password" required maxlength="256"><button type="submit">登录</button><div class="foot">研究数据 · 仅限授权用户访问</div></form></div></html>'''
        page = page.replace('MESSAGE', f'<p class="error" role="alert">{html.escape(message)}</p>' if message else '')
        h._send(page.encode(), 'text/html; charset=utf-8', status)
