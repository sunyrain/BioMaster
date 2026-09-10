import { useEffect, useId, useMemo, useState } from "react";
import { ArrowUpRight, ChevronDown, Database, Network } from "lucide-react";
import { Entity, Evidence, label } from "./types";
import "./biology-insights.css";

type ScoredDisease = { row: Evidence; score: number; key: string };
const displayCount = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US") : "—";
const scoreOf = (row: Evidence) => {
  const value = row.score ?? row.association_score;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

/** A view of the entity response's loaded records, never a global disease ranking. */
export default function BiologyInsights({ entity }: { entity: Entity }) {
  const [expanded, setExpanded] = useState(false);
  const [selectedKey, setSelectedKey] = useState("");
  const listId = useId();
  const loaded = entity.target_diseases || [];
  const scored = useMemo<ScoredDisease[]>(() => loaded.flatMap((row, index) => {
    const score = scoreOf(row);
    return score == null ? [] : [{ row, score, key: `${row.id || index}:${index}` }];
  }).sort((a, b) => b.score - a.score || a.key.localeCompare(b.key)), [loaded]);
  useEffect(() => { setExpanded(false); setSelectedKey(""); }, [entity.id]);
  if (entity.kind !== "target" || !loaded.length) return null;
  const visible = scored.slice(0, expanded ? 12 : 6);
  const selected = visible.find(item => item.key === selectedKey) || visible[0];
  const minimum = Math.min(0, ...scored.map(item => item.score));
  const maximum = Math.max(1, ...scored.map(item => item.score));
  const scale = maximum - minimum;
  const sourceNames = [...new Set(visible.map(item => item.row.source).filter(Boolean))] as string[];
  const total = entity.target_diseases_total ?? entity.evidence_totals?.target_diseases ?? loaded.length;
  const url = selected?.row.url;
  const safeUrl = typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
  return <section className="bio-insights" aria-label={`${entity.name} 已加载疾病关联概览`}>
    <div className="bio-insights-heading">
      <div><span className="bio-kicker">DISEASE ASSOCIATION PROFILE</span><h2>疾病关联强度</h2></div>
      <span className="bio-source-badge"><Network size={14} />{sourceNames.every(name => /opentargets|open targets/i.test(name)) && sourceNames.length ? "Open Targets · 本地快照" : "本地疾病证据"}</span>
    </div>
    <div className="bio-insights-content">
      <div className="bio-chart-column">
        <div className="bio-chart-caption"><span>已加载子集内，按原始关联分数降序</span><span>分数轴 {minimum}–{maximum}</span></div>
        {visible.length ? <div id={listId} className="bio-bars" aria-label="已加载疾病关联分数">
          {visible.map(item => <button type="button" className={`bio-bar-row${selected?.key === item.key ? " selected" : ""}`}
            key={item.key} aria-pressed={selected?.key === item.key} aria-label={`${label(item.row)}，关联分数 ${item.score.toFixed(3)}，查看来源`}
            onClick={() => setSelectedKey(item.key)} title={`${label(item.row)} · ${item.row.id || "未提供疾病编号"}`}>
            <span className="bio-disease-name">{label(item.row)}</span>
            <span className="bio-bar-track" aria-hidden="true"><span style={{
              left: `${(Math.min(0, item.score) - minimum) / scale * 100}%`,
              width: `${Math.abs(item.score) / scale * 100}%`,
            }} /></span>
            <strong>{item.score.toFixed(3)}</strong>
          </button>)}
        </div> : <p className="bio-unscored">已加载 {displayCount(loaded.length)} 条关联记录，当前没有可用的数值分数。下方可查看原始证据。</p>}
        {scored.length > 6 && <button type="button" className="bio-expand" aria-controls={listId} aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>
          {expanded ? "收起至 6 条" : `展开至 ${Math.min(12, scored.length)} 条`}<ChevronDown size={15} className={expanded ? "rotated" : ""} />
        </button>}
      </div>
      <aside className="bio-association-detail" aria-live="polite">
        <dl className="bio-counts"><div><dt>本地关联记录</dt><dd>{displayCount(total)}</dd></div><div><dt>当前已加载</dt><dd>{displayCount(loaded.length)}</dd></div><div><dt>生物通路</dt><dd>{displayCount(entity.pathways_total ?? entity.pathways?.length)}</dd></div></dl>
        {selected && <div className="bio-selected-evidence">
          <span className="bio-selected-label">所选关联 · 证据来源</span><strong>{label(selected.row)}</strong>
          <span className="bio-disease-id">{selected.row.id || "未提供疾病编号"}</span>
          <span className="bio-source-name"><Database size={13} />{selected.row.source || "来源未标注"}</span>
          <div className="bio-selected-actions">
            {safeUrl && <a href={safeUrl} target="_blank" rel="noreferrer">查看来源页面 <ArrowUpRight size={14} /></a>}
            {selected.row.source_path && <details className="bio-source-path"><summary>本地来源文件</summary><code>{String(selected.row.source_path)}</code></details>}
          </div>
        </div>}
      </aside>
    </div>
    <p className="bio-insights-note">图中仅比较本页已加载的 {displayCount(loaded.length)} 条记录{scored.length !== loaded.length ? `（其中 ${displayCount(scored.length)} 条有数值分数）` : ""}，不代表全部疾病的前列排序。关联分数衡量数据库证据支持，不是结合概率、药效或治疗方向。</p>
  </section>;
}
