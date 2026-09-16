"""Live full-catalog model scores. Missing inputs never become negative labels."""
from __future__ import annotations
import json
import math
import sqlite3
from pathlib import Path

DIRECTORY = 'outputs/catalog_seven_models_20260916'
NEW_MODELS = ('nesso', 'probematch', 'dtbind')


def connect(root, *, write=False):
    path = Path(root) / DIRECTORY / 'scores.sqlite'
    if not write and not path.exists():
        return None
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=30)
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS predictions (
          model TEXT NOT NULL, drug_id TEXT NOT NULL, target_id TEXT NOT NULL,
          status TEXT NOT NULL, score REAL, reason TEXT NOT NULL DEFAULT '',
          PRIMARY KEY (model, target_id, drug_id));
        CREATE INDEX IF NOT EXISTS predictions_drug ON predictions(drug_id, model);
        CREATE TABLE IF NOT EXISTS target_status (
          model TEXT NOT NULL, target_id TEXT NOT NULL, status TEXT NOT NULL,
          reason TEXT NOT NULL DEFAULT '', PRIMARY KEY(model,target_id));
        ''')
    else:
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def save(db, rows):
    values=[]
    for r in rows:
        score=r.get('score')
        if r['status']=='completed' and (score is None or not math.isfinite(float(score))):
            raise ValueError('Completed predictions require finite scores')
        values.append((r['model'],r['drug_id'],r['target_id'],r['status'],score if r['status']=='completed' else None,r.get('reason') or ''))
    db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?) ON CONFLICT(model,target_id,drug_id) DO UPDATE SET status=excluded.status,score=excluded.score,reason=excluded.reason',values)
    db.commit()


def entity_scores(root, kind, identifier):
    db=connect(root)
    if db is None:return {},{}
    try:
        key='drug_id' if kind=='drug' else 'target_id'
        other='target_id' if kind=='drug' else 'drug_id'
        result={}
        for r in db.execute(f'SELECT * FROM predictions WHERE {key}=?',(identifier,)):
            result[(r['model'],r[other])]=dict(r)
        target_states={}
        for r in db.execute('SELECT * FROM target_status'+(' WHERE target_id=?' if kind=='target' else ''),(identifier,) if kind=='target' else ()):
            target_states[(r['model'],r['target_id'])]=dict(r)
        return result,target_states
    finally:db.close()


def progress(root):
    result={}
    for model in NEW_MODELS:
        path=Path(root)/DIRECTORY/model/'STATUS.json'
        try:result[model]=json.loads(path.read_text())
        except (OSError,ValueError):result[model]={'state':'not_started'}
    return result


def coverage(root):
    db=connect(root)
    if db is None:return {}
    try:
        return {r['model']:dict(r) for r in db.execute("SELECT model,COUNT(*) AS coverage,COUNT(DISTINCT target_id) AS targets FROM predictions WHERE status='completed' GROUP BY model")}
    finally:db.close()
