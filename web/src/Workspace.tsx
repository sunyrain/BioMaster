import { useEffect, useState } from "react";
import { ArrowRight, ArrowUpRight, Bookmark, BookmarkCheck, Clock3, Dna, FlaskConical, FolderOpen, Search, Trash2 } from "lucide-react";
import type { EntityRef } from "./types";
import "./workspace.css";

type Entry = EntityRef & { savedAt: string };
const SAVED = "biomaster.research-list.v1";
const RECENT = "biomaster.recent.v1";
const EVENT = "biomaster-workspace-update";
const same = (a: EntityRef, b: EntityRef) => a.kind === b.kind && a.id === b.id;
function read(key: string): Entry[] {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(value) ? value.filter(row => row && typeof row.id === "string" && typeof row.name === "string" && ["drug", "target"].includes(row.kind)) : [];
  } catch { return []; }
}
function write(key: string, entries: Entry[]): boolean {
  try { localStorage.setItem(key, JSON.stringify(entries)); window.dispatchEvent(new Event(EVENT)); return true; }
  catch { return false; }
}
function useEntries(key: string) {
  const [entries, setEntries] = useState(() => read(key));
  useEffect(() => {
    const refresh = () => setEntries(read(key));
    window.addEventListener(EVENT, refresh); window.addEventListener("storage", refresh);
    return () => { window.removeEventListener(EVENT, refresh); window.removeEventListener("storage", refresh); };
  }, [key]);
  return entries;
}
export function recordVisit(entity: EntityRef) {
  const entry = { id: entity.id, name: entity.name, subtitle: entity.subtitle, kind: entity.kind, scored: entity.scored, savedAt: new Date().toISOString() };
  write(RECENT, [entry, ...read(RECENT).filter(e => !same(e, entity))].slice(0, 12));
}
export function SaveEntityButton({ entity }: { entity: EntityRef }) {
  const saved = useEntries(SAVED).some(e => same(e, entity));
  const [error, setError] = useState(false);
  return <button className={`button small-button save-entity ${saved ? "is-saved" : ""}`} aria-pressed={saved} onClick={() => {
    const entries = read(SAVED);
    setError(!write(SAVED, saved ? entries.filter(e => !same(e, entity)) : [{ id: entity.id, kind: entity.kind, name: entity.name, subtitle: entity.subtitle, scored: entity.scored, savedAt: new Date().toISOString() }, ...entries]));
  }}>{saved ? <BookmarkCheck size={16} /> : <Bookmark size={16} />}{error ? "浏览器存储不可用" : saved ? "已加入清单" : "加入研究清单"}</button>;
}
export function RecentEntities({ onOpen }: { onOpen: (entity: EntityRef) => void }) {
  const entries = useEntries(RECENT);
  if (!entries.length) return null;
  return <section className="workspace-recents" aria-label="最近查看"><div className="workspace-recents-heading"><Clock3 size={13} /><span>最近查看</span></div>{entries.slice(0, 3).map(e => <button key={`${e.kind}:${e.id}`} onClick={() => onOpen(e)} title={`${e.name} · ${e.id}`}><i className={e.kind} /><span>{e.name}</span><small>{e.kind === "drug" ? "药物" : "靶点"}</small></button>)}</section>;
}
export default function Workspace({ onOpen, onNavigate }: { onOpen: (entity: EntityRef, tab?: string) => void; onNavigate: (path: string) => void }) {
  const entries = useEntries(SAVED);
  const recent = useEntries(RECENT);
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("all");
  const [error, setError] = useState(false);
  const filtered = entries.filter(e => (kind === "all" || e.kind === kind) && `${e.name} ${e.id}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="research-workspace">
    <div className="page-heading"><div><span className="eyebrow">PERSONAL RESEARCH COLLECTION</span><h1>研究清单</h1><p>保留值得跟进的药物与靶点，让探索可以继续。</p></div><span className="badge blue"><Bookmark size={15} />{entries.length} 个已保存实体</span></div>
    <div className="workspace-banner"><div className="workspace-banner-icon"><FolderOpen size={28} /></div><div><h2>你的下一轮研究，从这里开始</h2><p>在实体详情中点击「加入研究清单」，集中查看排名、机制和结构。</p></div><span>保存在当前浏览器 · 刷新后保留</span></div>
    <div className="catalog-toolbar"><div className="filter-search"><Search size={17} /><input aria-label="搜索研究清单" placeholder="搜索已保存的名称或编号…" value={query} onChange={e => setQuery(e.target.value)} /></div><div className="ranking-view-switch" aria-label="清单实体类型">{[["all", "全部"], ["drug", "药物"], ["target", "靶点"]].map(([id, text]) => <button key={id} className={kind === id ? "active" : ""} aria-pressed={kind === id} onClick={() => setKind(id)}>{text}</button>)}</div><span>{filtered.length} 个匹配</span></div>
    {error && <p role="alert" className="info-note">浏览器存储不可用，未能更新清单。</p>}
    {filtered.length ? <div className="workspace-entries">{filtered.map(e => <article className="workspace-entry" key={`${e.kind}:${e.id}`}><span className={`entity-icon ${e.kind}`}>{e.kind === "drug" ? <FlaskConical size={23} /> : <Dna size={23} />}</span><div className="workspace-entry-title"><span>{e.kind === "drug" ? "DRUG" : "TARGET"}</span><button onClick={() => onOpen(e)}><h2>{e.name}</h2><ArrowUpRight size={17} /></button><p>{e.id}</p></div><div className="workspace-entry-actions"><button className="text-button" onClick={() => onOpen(e, "rankings")}>模型排名 <ArrowRight size={14} /></button><button className="text-button" onClick={() => onOpen(e, e.kind === "target" ? "structure" : "diseases")}>{e.kind === "target" ? "三维口袋" : "疾病机制"}<ArrowRight size={14} /></button><button className="workspace-remove" aria-label={`从清单移除 ${e.name}`} title="从清单移除" onClick={() => setError(!write(SAVED, read(SAVED).filter(item => !same(item, e))))}><Trash2 size={17} /></button></div></article>)}</div> : <div className="empty workspace-empty"><Bookmark size={32} /><strong>{entries.length ? "没有匹配的保存实体" : "清单准备就绪，开始第一次收藏"}</strong><span>{entries.length ? "尝试其他名称或实体类型。" : "从药物或靶点详情中保存，逐步建立自己的研究候选清单。"}</span>{!entries.length && <div><button className="button primary" onClick={() => onNavigate("drugs")}>浏览药物 <ArrowRight size={16} /></button><button className="button secondary" onClick={() => onNavigate("targets")}>浏览靶点 <ArrowRight size={16} /></button></div>}</div>}
    {!!recent.length && <section className="workspace-recent-panel"><div className="section-title"><div><span className="eyebrow">CONTINUE EXPLORING</span><h2>继续上次的探索</h2></div><Clock3 size={20} /></div><div>{recent.slice(0, 6).map(e => <button key={`${e.kind}:${e.id}`} onClick={() => onOpen(e)}><span className={`entity-icon small ${e.kind}`}>{e.kind === "drug" ? <FlaskConical size={17} /> : <Dna size={17} />}</span><span><strong>{e.name}</strong><small>{e.kind === "drug" ? "药物档案" : "靶点档案"}</small></span><ArrowUpRight size={15} /></button>)}</div></section>}
  </div>;
}
