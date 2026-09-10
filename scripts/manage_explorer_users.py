#!/usr/bin/env python3
"""Manage local website accounts; passwords are entered without terminal echo."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=Path.home()/'.config/biomaster/users.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='列出账号（不显示密码哈希）')
    for command, help_text in [('add','添加账号'),('delete','删除账号'),('password','修改密码')]:
        item = sub.add_parser(command, help=help_text)
        item.add_argument('username')
    args = parser.parse_args()
    path = args.file.expanduser()
    if not path.is_file(): parser.error(f'账号文件不存在：{path}')
    username = getattr(args, 'username', '').strip().lower()
    if username and not re.fullmatch(r'[a-z0-9_-]{1,64}', username):
        parser.error('用户名仅允许 1–64 个小写字母、数字、下划线或连字符')
    replacement = None
    if args.command in ('add','password'):
        # Check obvious mistakes before prompting; recheck under the write lock.
        users = json.loads(path.read_text())
        if args.command == 'add' and username in users: parser.error('账号已存在，请使用 password 修改密码')
        if args.command == 'password' and username not in users: parser.error('账号不存在')
        password = getpass.getpass('新密码（输入不显示）：')
        if not password: parser.error('密码不能为空')
        if len(password) > 256: parser.error('密码不能超过 256 个字符')
        if password != getpass.getpass('再次输入密码：'): parser.error('两次密码不一致，未修改')
        salt = secrets.token_hex(16)
        replacement = {'salt':salt, 'hash':hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),260000).hex()}
    lock_fd = os.open(str(path)+'.lock',os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(lock_fd,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        users = json.loads(path.read_text())
        if args.command == 'list':
            print('\n'.join(sorted(users))); print(f'共 {len(users)} 个账号'); return
        if args.command == 'add':
            if username in users: parser.error('账号已存在')
            users[username] = replacement
        else:
            if username not in users: parser.error('账号不存在')
            if args.command == 'delete':
                if len(users) == 1: parser.error('不能删除最后一个账号')
                del users[username]
            else: users[username] = replacement
        fd, temp = tempfile.mkstemp(prefix='.users-',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as out:
                json.dump(users,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
            os.replace(temp,path)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    print(f'已更新账号 {username}。需重启网站服务后生效；重启会注销已有登录会话。')

if __name__ == '__main__':
    main()
