import { useState } from "react";
import { ArrowUpRight, CircleHelp, GitCompareArrows, Layers3 } from "lucide-react";
import { MODEL_COLORS, MODEL_NAMES, MODELS, type Model, type Ranking } from "./types";
import "./rank-visuals.css";

export type RankView = "matrix" | "parallel";
const position = (rank: number, denominator: number) => denominator <= 1 ? 0 : (rank - 1) / (denominator - 1);

export default function RankVisuals({ rows, model, view, order, onSort, onSelect }: {
  rows: Ranking[]; model: Model; view: RankView; order: "asc" | "desc"; onSort: (model: Model) => void; onSelect: (row: Ranking) => void;
}) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const active = rows.find(row => row.id === activeId);
  const known = rows.filter(row => row.known_relation).length;
  const supported = rows.filter(row => MODELS.filter(m => row.ranks[m] != null && row.ranks[m]! <= 20).length >= 3).length;
  return <div className="rank-visuals">
    <div className="rv-summary">
      <div><span>当前展示</span><strong>{rows.length}<small> 项候选</small></strong></div>
      <div><span>其中已知关系</span><strong>{known}<small> 项</small></strong></div>
      <div><span>至少 3 个模型排入 Top 20</span><strong>{supported}<small> 项</small></strong></div>
      <p><Layers3 size={17} />按 {MODEL_NAMES[model]} {order === "asc" ? "排名升序" : "排名降序"}<br />各模型使用自己的评分分母</p>
    </div>
    {view === "matrix" ? <div className="rv-matrix-scroll">
      <table className="rv-matrix">
        <caption className="sr-only">当前候选在四个模型中的排名；点击单元格查看配对证据</caption>
        <thead><tr><th>候选实体 / 关系证据</th>{MODELS.map(m => <th aria-sort={m === model ? (order === "asc" ? "ascending" : "descending") : "none"} className={m === model ? "is-sorted" : ""} key={m}><button className="rv-sort" onClick={() => onSort(m)}><i style={{ background: MODEL_COLORS[m] }} />{MODEL_NAMES[m]} {m === model ? (order === "asc" ? "↑" : "↓") : "↕"}<small>模型内排名 · 点击排序</small></button></th>)}<th>关系证据</th></tr></thead>
        <tbody>{rows.map(row => <tr key={row.id} onMouseEnter={() => setActiveId(row.id)} onMouseLeave={() => setActiveId(null)} className={activeId === row.id ? "is-active" : ""}>
          <th scope="row"><button className="rv-entity" onClick={() => onSelect(row)}><span className="rv-order">{row.rank == null ? "—" : String(row.rank).padStart(2, "0")}</span><span><strong title={row.name}>{row.name}</strong><small><i className={row.known_relation ? "known" : "unknown"} />{row.known_relation ? "已知关系" : "关系未标注"}</small></span><ArrowUpRight size={15} /></button></th>
          {MODELS.map(m => {
            const rank = row.ranks[m]; const den = row.denominators[m];
            const band = rank == null ? "missing" : rank <= 5 ? "first" : rank <= 20 ? "top" : rank <= den / 4 ? "upper" : "rest";
            return <td key={m}><button className={`rv-cell ${band}`} onClick={() => onSelect(row)} aria-label={`${row.name}，${MODEL_NAMES[m]}，${rank == null ? "暂无评分" : `第 ${rank} 名，共 ${den} 项`}，查看配对证据`}>
              <span><strong>{rank == null ? "—" : `#${rank}`}</strong><small>{rank == null ? "暂无结果" : `/ ${den}`}</small></span>
              <small className="rv-score">分数 {row.scores[m] == null ? "—" : Number(row.scores[m]).toPrecision(4)}</small><i className="rv-cell-meter"><b style={{ width: rank == null ? "0" : `${Math.max(2, (1 - position(rank, den)) * 100)}%` }} /></i>
            </button></td>;
          })}
          <td className="rv-evidence"><button onClick={() => onSelect(row)}><strong>{row.known_relation ? "已知关系" : "关系未标注"}</strong><span>{row.mechanism || (row.known_relation ? "已收录关系，机制待核对" : "暂无已知关系注释")}</span>{row.action && <small>作用类型：{row.action}</small>}<small>查看关系与来源 →</small></button></td>
        </tr>)}</tbody>
      </table>
    </div> : <div className="rv-parallel">
      <div className="rv-plot-title"><div><GitCompareArrows size={18} /><strong>同一候选，四种模型视角</strong></div><span>越靠上，模型内排名越靠前</span></div>
      <div className="rv-plot-scroll"><svg viewBox="0 0 900 335" role="group" aria-label="四模型排名位置平行坐标图">
        {[0, .25, .5, .75, 1].map(p => <g key={p}><line x1="92" x2="830" y1={55 + p * 225} y2={55 + p * 225} stroke="#d7e1ed" strokeDasharray="3 5" /><text x="76" y={60 + p * 225} textAnchor="end" fill="#526176" fontSize="13">{p === 0 ? "第 1 名" : `${p * 100}%`}</text></g>)}
        {MODELS.map((m, i) => <g key={m}><line x1={140 + i * 215} x2={140 + i * 215} y1="55" y2="280" stroke="#b7c8db" /><text x={140 + i * 215} y="28" textAnchor="middle" fontSize="15" fontWeight="700" fill={MODEL_COLORS[m]}>{MODEL_NAMES[m]}</text></g>)}
        {[...rows].sort((a, b) => Number(a.id === activeId) - Number(b.id === activeId)).map(row => {
          const points = MODELS.map((m, i) => row.ranks[m] == null ? null : { x: 140 + i * 215, y: 55 + position(row.ranks[m]!, row.denominators[m]) * 225, m });
          const focused = activeId === row.id;
          const color = row.known_relation ? "#007d68" : "#2459e8";
          return <g key={row.id} className="rv-trace" role="button" tabIndex={0} aria-label={`${row.name}，查看四模型证据`} onMouseEnter={() => setActiveId(row.id)} onMouseLeave={() => setActiveId(null)} onFocus={() => setActiveId(row.id)} onBlur={() => setActiveId(null)} onClick={() => onSelect(row)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(row); } }} opacity={activeId && !focused ? .15 : focused ? 1 : .6}>
            <title>{row.name}: {MODELS.map(m => `${MODEL_NAMES[m]} #${row.ranks[m] ?? "—"}/${row.denominators[m]}`).join(" · ")}</title>
            {points.map((point, i) => point && points[i + 1] ? <g key={`line-${i}`}><line x1={point.x} y1={point.y} x2={points[i + 1]!.x} y2={points[i + 1]!.y} stroke="transparent" strokeWidth="12" /><line x1={point.x} y1={point.y} x2={points[i + 1]!.x} y2={points[i + 1]!.y} stroke={color} strokeWidth={focused ? 3 : 1.8} /></g> : null)}
            {points.map(point => point && <circle key={point.m} cx={point.x} cy={point.y} r={focused ? 5 : 3} fill={color} stroke="white" strokeWidth="1.5" />)}
          </g>;
        })}
        <text x="470" y="321" textAnchor="middle" fontSize="12" fill="#526176">纵轴 = (排名 − 1) / (该模型评分数 − 1)；缺失评分处不连线</text>
      </svg></div>
      <div className="rv-trace-inspector" aria-live="polite">{active ? <><strong>{active.name}</strong>{MODELS.map(m => <span key={m}>{MODEL_NAMES[m]} <b>#{active.ranks[m] ?? "—"}</b></span>)}<button onClick={() => onSelect(active)}>查看证据 <ArrowUpRight size={15} /></button></> : <><CircleHelp size={16} /><span>悬停或用 Tab 选择轨迹，查看候选在四个模型中的排名；点击打开证据。</span></>}</div>
    </div>}
    <div className="rv-legend">{view === "matrix" ? <><span><i className="first" />Top 5</span><span><i className="top" />Top 20</span><span><i className="upper" />其余前 25%</span><span><i className="rest" />其他排名</span></> : <><span><i className="known" />已知关系</span><span><i className="prediction" />关系未标注</span></>}<p>颜色表达排名或关系注释，不代表结合概率。</p></div>
  </div>;
}
