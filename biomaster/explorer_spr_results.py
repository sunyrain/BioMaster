"""Append-only SPR observations, validated independently of prediction artifacts."""
from __future__ import annotations
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import secrets
import sqlite3
import time

FIELDS = ['experiment_id', 'pair_id', 'sample_id', 'experiment_date', 'result', 'qc', 'KD', 'KD_unit', 'ka_M_inv_s_inv', 'kd_s_inv', 'Rmax_RU', 'chi2', 'construct', 'max_concentration_uM', 'notes']
TEMPLATE_VERSION = 'SPR_RESULTS_V2'
CONTEXT_FIELDS = ['template_version', 'template_order', 'group', 'priority_rank', 'priority', 'drug_name', 'target_name']
LABELS = dict(zip(CONTEXT_FIELDS + FIELDS, ['模板版本', '模板序号', '实验分组', '优先顺序', '优先级', '药物名称', '靶点名称', '实验编号', '配对编号', '样本编号', '实验日期', '响应结果', '质控结果', 'KD', 'KD单位', 'ka_M_inv_s_inv', 'kd_s_inv', 'Rmax_RU', 'chi2', '蛋白构建', '最高浓度_uM', '备注']))
GROUPS = {'baseline': '基线候选', 'control': '参考对照', 'expanded': '扩展候选'}
REQUIRED = {'experiment_id', 'sample_id', 'experiment_date', 'result', 'qc'}
UNITS = {'M': 1e9, 'mM': 1e6, 'uM': 1e3, 'µM': 1e3, 'μM': 1e3, 'nM': 1, 'pM': .001}
MAX_BYTES = 2 * 1024 * 1024


def csv_bytes(rows, fields):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v) for k, v in row.items()})
    return out.getvalue().encode('utf-8-sig')


class SPRResults:
    def __init__(self, root):
        directory = Path(root) / 'outputs/biomaster_explorer/spr_results'
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'results.sqlite3'
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, batch TEXT NOT NULL, filename TEXT NOT NULL,
                digest TEXT NOT NULL, created REAL NOT NULL, committed TEXT, payload TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS uploaded_content ON uploads(batch,digest) WHERE committed IS NOT NULL;
            CREATE TABLE IF NOT EXISTS observations (
                batch TEXT NOT NULL, sample TEXT NOT NULL, upload_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(batch,sample));
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def registry(data):
        data.ensure_loaded()
        records = [r for t in data.targets.values() for r in t.get('experiments', [])]
        from .explorer_spr_expansion import expanded_items
        records += expanded_items(data.root)
        result = {}
        for row in records:
            identifier = row['experiment_id']
            if identifier in result and result[identifier]['pair_id'] != row['pair_id']:
                raise ValueError('实验设计编号重复，请先修复设计目录。')
            result[identifier] = dict(row)
        priority_path = Path(data.root) / 'outputs/spr384_priority_export_20260910/SPR384_PRIORITY_DETAILED.csv'
        if priority_path.exists():
            priority = list(csv.DictReader(io.StringIO(priority_path.read_text(encoding='utf-8-sig'))))
            ordered = {}
            for row in priority:
                identifier = row['原候选编号']
                if identifier not in result or result[identifier]['pair_id'] != row['pair_id']:
                    raise ValueError('SPR 优先清单与当前设计配对不一致。')
                result[identifier].update(priority_rank=int(row['排序']), priority=row['优先级'])
                ordered[identifier] = result[identifier]
            result = {**ordered, **result}
        for row in result.values():
            row['group'] = 'control' if row.get('is_control') else 'expanded' if row.get('design_id', '').startswith('EXPANDED') else 'baseline'
            row['target_name'] = row.get('gene_symbol', '')
        return result

    @staticmethod
    def catalog(registry, scope='baseline', order='priority', target='', search=''):
        if scope not in (*GROUPS, 'all') or order not in ('priority', 'id', 'target'):
            raise ValueError('无效的模板分组或排序。')
        rows = [dict(row) for row in registry.values() if (scope == 'all' or row.get('group', 'baseline') == scope)
                and (not target or row['target_id'] == target)
                and search.lower() in ' '.join(str(row.get(k, '')) for k in ('experiment_id', 'drug_name', 'target_name', 'target_id', 'drug_id')).lower()]
        group_order = {'baseline': 0, 'control': 1, 'expanded': 2}
        def key(row):
            group = group_order.get(row.get('group'), 0)
            if order == 'id': return (group, row['experiment_id'])
            if order == 'target': return (group, row.get('target_name', ''), row.get('priority_rank', 10**9), row['experiment_id'])
            return (group, row.get('priority_rank', 10**9), row['experiment_id'])
        rows.sort(key=key)
        for index, row in enumerate(rows, 1): row['template_order'] = index
        return rows

    def template(self, registry, scope='baseline', order='priority', target='', search=''):
        fields = CONTEXT_FIELDS + FIELDS
        rows = self.catalog(registry, scope, order, target, search)
        output = []
        for row in rows:
            item = {key: row.get(key, '') if key in CONTEXT_FIELDS or key in ('experiment_id', 'pair_id') else '' for key in fields}
            item['template_version'] = TEMPLATE_VERSION
            item['group'] = GROUPS.get(row.get('group'), '基线候选')
            output.append({LABELS[k]: v for k, v in item.items()})
        return csv_bytes(output, [LABELS[k] for k in fields])

    def preview(self, payload, registry, owner):
        batch = str(payload.get('batch', '')).strip()
        filename = str(payload.get('filename', 'results.csv')).replace('\\', '/').split('/')[-1]
        content = payload.get('content')
        entry_method = 'csv'
        if 'rows' in payload:
            entries = payload['rows']
            if not isinstance(entries, list) or not 1 <= len(entries) <= 2000:
                raise ValueError('在线填写需要 1–2,000 条记录。')
            if any(not isinstance(row, dict) or set(row) - set(FIELDS) or any(not isinstance(v, (str, int, float, type(None))) for v in row.values()) for row in entries):
                raise ValueError('在线结果字段无效。')
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(entries)
            content = stream.getvalue()
            filename = 'online-entry.csv'
            entry_method = 'online'

        if not batch or len(batch) > 120:
            raise ValueError('请填写实验批次（1–120 字）。')
        if not isinstance(content, str) or not content.strip() or len(content.encode()) > MAX_BYTES:
            raise ValueError('请选择非空 CSV 文件，最大 2 MB。')
        if len(filename) > 200 or not filename.lower().endswith('.csv'):
            raise ValueError('仅支持 CSV 文件。Excel 请另存为 UTF-8 CSV。')
        if '\ufffd' in content or '\x00' in content:
            raise ValueError('文件编码无效，请另存为 UTF-8 CSV。')
        reader = csv.DictReader(io.StringIO(content.lstrip('\ufeff')), strict=True)
        if entry_method == 'csv' and set(reader.fieldnames or []) != set(LABELS.values()):
            raise ValueError('旧模板已退役。请下载当前 SPR_RESULTS_V2 中文模板后填写；保留全部表头与模板版本列。')
        aliases = {v: k for k, v in LABELS.items()}
        headers = [aliases.get(h.strip(), h.strip()) for h in (reader.fieldnames or [])]
        reader.fieldnames = headers
        if len(set(headers)) != len(headers) or not REQUIRED.issubset(headers):
            raise ValueError('缺少必需字段或存在重复表头：' + ', '.join(sorted(REQUIRED)))
        if set(headers) - set(FIELDS + CONTEXT_FIELDS):
            raise ValueError('未识别的字段：' + ', '.join(sorted(set(headers) - set(FIELDS + CONTEXT_FIELDS))))
        rows, errors, seen, skipped = [], [], set(), 0
        with self.connect() as db:
            existing = {r[0] for r in db.execute('SELECT sample FROM observations WHERE batch=?', (batch,))}
        try:
            for line, raw in enumerate(reader, 2):
                if line > 2001:
                    raise ValueError('每次最多上传 2,000 行。')
                if None in raw or any(v is None for v in raw.values()):
                    errors.append({'line': line, 'message': '字段数量与表头不一致。'}); continue
                row = {k: v.strip() for k, v in raw.items()}
                if not any(row.get(k) for k in FIELDS if k not in ('experiment_id', 'pair_id')):
                    skipped += 1; continue
                if entry_method == 'csv' and row.get('template_version') != TEMPLATE_VERSION:
                    errors.append({'line': line, 'message': '模板版本无效，请使用 SPR_RESULTS_V2 模板。'}); continue
                row = {k: v for k, v in row.items() if k in FIELDS}
                if entry_method == 'csv' and (row.get('result') not in ('检出响应', '未检出响应', '无法判定') or row.get('qc') not in ('通过', '未通过', '待确认')):
                    errors.append({'line': line, 'message': '请使用中文响应结果与质控结果选项。'}); continue
                row['result'] = {'检出响应': 'detected', '未检出响应': 'not_detected', '无法判定': 'inconclusive'}.get(row.get('result'), row.get('result', ''))
                row['qc'] = {'通过': 'pass', '未通过': 'fail', '待确认': 'review'}.get(row.get('qc'), row.get('qc', ''))
                issues = []
                if any(len(v) > 4000 for v in row.values()): issues.append('单字段最多 4,000 字')
                design = registry.get(row.get('experiment_id'))
                if not design: issues.append('实验编号未在当前基线/扩展设计中找到')
                elif row.get('pair_id') and row['pair_id'] != design['pair_id']: issues.append('配对编号与设计不一致')
                sample = row.get('sample_id', '')
                if not sample or len(sample) > 120: issues.append('sample_id 必填，最多 120 字')
                if sample in seen or sample in existing: issues.append('此批次内 sample_id 重复；重复实验请使用不同样本编号')
                seen.add(sample)
                try:
                    d = date.fromisoformat(row.get('experiment_date', ''))
                    if d > datetime.now(timezone.utc).date(): raise ValueError()
                except ValueError: issues.append('实验日期须为有效 YYYY-MM-DD，且不能晚于今天')
                if row.get('result') not in ('detected', 'not_detected', 'inconclusive'): issues.append('result 仅允许 detected / not_detected / inconclusive')
                if row.get('qc') not in ('pass', 'fail', 'review'): issues.append('qc 仅允许 pass / fail / review')
                for field in ('KD', 'ka_M_inv_s_inv', 'kd_s_inv', 'Rmax_RU', 'chi2', 'max_concentration_uM'):
                    value = row.get(field, '')
                    row[field] = None
                    if value:
                        try:
                            number = float(value)
                            if not math.isfinite(number) or number < 0 or (field not in ('chi2', 'Rmax_RU') and number == 0): raise ValueError()
                            row[field] = number
                        except ValueError: issues.append(f'{field} 须为有限正数（chi2 / Rmax 可为 0）')
                if row.get('KD') is not None:
                    if row.get('KD_unit') not in UNITS: issues.append('KD_unit 须为 M / mM / uM / nM / pM')
                    else:
                        row['KD_nM'] = row['KD'] * UNITS[row['KD_unit']]
                        if not math.isfinite(row['KD_nM']): issues.append('KD 换算后超出有效范围')
                    if row.get('result') != 'detected': issues.append('未检出或无法判定的结果请勿填写拟合 KD')
                elif row.get('KD_unit'): issues.append('填写 KD_unit 时须同时填写 KD')
                if issues:
                    errors.append({'line': line, 'message': '；'.join(issues)}); continue
                row.update(pair_id=design['pair_id'], drug_id=design['drug_id'], target_id=design['target_id'],
                           drug_name=design.get('drug_name', ''), target_name=design.get('gene_symbol', ''),
                           design_id=design.get('design_id', ''), is_control=design.get('is_control', False),
                           review_status='pending', source_line=line, entry_method=entry_method)
                rows.append(row)
        except csv.Error as exc:
            raise ValueError(f'CSV 解析失败：{exc}') from exc
        if not rows and not errors: raise ValueError('没有已填写的结果。模板空白行会自动跳过。')
        token = None
        if not errors:
            token = secrets.token_urlsafe(24)
            with self.connect() as db:
                db.execute('DELETE FROM uploads WHERE committed IS NULL AND created < ?', (time.time() - 86400,))
                db.execute('INSERT INTO uploads VALUES (?,?,?,?,?,?,NULL,?)',
                           (token, owner, batch, filename, hashlib.sha256(content.encode()).hexdigest(), time.time(), json.dumps(rows, ensure_ascii=False)))
        return {'token': token, 'entry_method': entry_method, 'batch': batch, 'valid_count': len(rows), 'error_count': len(errors), 'skipped_count': skipped,
                'errors': errors, 'items': rows, 'expires_in': 1800, 'warnings': ['上传结果均标为待审核；响应检出不等于特异性结合。原始曲线与拟合报告请保留在实验档案中。']}

    def commit(self, token, owner):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT owner,batch,filename,digest,created,committed,payload FROM uploads WHERE id=?', (token,)).fetchone()
            if not row or row[0] != owner: raise ValueError('上传预览不存在或不属于当前用户，请重新校验。')
            if row[5]: return {'id': token, 'count': len(json.loads(row[6])), 'committed_at': row[5], 'already_saved': True}
            if time.time() - row[4] > 1800: raise ValueError('预览已过期，请重新校验文件。')
            items = json.loads(row[6])
            now = datetime.now(timezone.utc).isoformat()
            try:
                for item in items:
                    db.execute('INSERT INTO observations VALUES (?,?,?,?)', (row[1], item['sample_id'], token, json.dumps(item, ensure_ascii=False)))
                db.execute('UPDATE uploads SET committed=? WHERE id=?', (now, token))
            except sqlite3.IntegrityError as exc:
                raise ValueError('该批次已有相同样本或文件，未导入任何行。请刷新记录并核对批次。') from exc
        return {'id': token, 'count': len(items), 'committed_at': now, 'already_saved': False}

    def results(self, search='', pair='', page=1, page_size=20, export=False):
        if page < 1 or not 1 <= page_size <= 200: raise ValueError('无效的分页参数')
        # JSON values are returned intact; substring search treats user input literally.
        with self.connect() as db:
            records = db.execute('SELECT o.payload,o.batch,u.owner,u.committed,u.filename,o.upload_id FROM observations o JOIN uploads u ON u.id=o.upload_id ORDER BY u.committed DESC,o.sample').fetchall()
        items = []
        for payload, batch, owner, committed, filename, upload in records:
            item = {**json.loads(payload), 'batch': batch, 'uploaded_by': owner, 'uploaded_at': committed, 'filename': filename, 'upload_id': upload}
            if pair and item['pair_id'] != pair: continue
            if search.lower() not in ' '.join(str(v) for v in item.values()).lower(): continue
            items.append(item)
        return {'total': len(items), 'items': items if export else items[(page-1)*page_size:page*page_size], 'page': page, 'page_size': page_size}
