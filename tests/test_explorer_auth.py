import hashlib
import http.client
import json
import threading
from pathlib import Path
from biomaster.explorer_auth import Auth
from biomaster.explorer_server import ExplorerHTTPServer


def test_login_protection_sessions_and_logout(tmp_path):
    salt = 'ab' * 16
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'tester': {'salt':salt, 'hash':hashlib.pbkdf2_hmac('sha256', b'test-pass', bytes.fromhex(salt),260000).hex()}}))
    static = tmp_path / 'static'; static.mkdir(); (static/'index.html').write_text('private website')
    class Data:
        _ready=True
        root=tmp_path
    server=ExplorerHTTPServer(('127.0.0.1',0),Data(),static)
    server.auth=Auth(users)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def call(method,path,body=None,cookie=None,origin=None):
        c=http.client.HTTPConnection('127.0.0.1',server.server_port)
        headers={'X-Forwarded-Proto':'https','Content-Type':'application/x-www-form-urlencoded'}
        if cookie:headers['Cookie']=cookie
        if origin:headers['Origin']=origin
        c.request(method,path,body,headers);r=c.getresponse();result=(r.status,dict(r.getheaders()),r.read());c.close();return result
    try:
        assert call('GET','/')[0]==303
        for path in ['/api/health','/api/rankings.csv','/api/structure/anything','/api/browse?section=structures']:
            assert call('GET',path)[0]==401
        assert call('HEAD','/api/health')[0]==401
        assert call('GET','/login')[0]==200
        assert call('POST','/login','username=tester&password=wrong')[0]==401
        assert call('POST','/login','username=tester&password=test-pass',origin='https://evil.example')[0]==403
        status,headers,_=call('POST','/login','username=tester&password=test-pass')
        assert status==303
        cookie=headers['Set-Cookie'];assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=Lax' in cookie
        cookie=cookie.split(';')[0]
        assert call('GET','/',cookie=cookie)[2]==b'private website'
        assert call('GET','/api/health',cookie=cookie)[0]==200
        assert 'no-store' in call('GET','/',cookie=cookie)[1]['Cache-Control']
        assert call('POST','/logout',cookie=cookie)[0]==303
        assert call('GET','/api/health',cookie=cookie)[0]==401
        for _ in range(8):call('POST','/login','username=tester&password=bad')
        assert call('POST','/login','username=tester&password=test-pass')[0]==429
    finally:
        server.shutdown();server.server_close();thread.join()
