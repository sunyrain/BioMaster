import { useEffect, useState } from "react";
import { ArrowUpRight, RefreshCw, GitCompareArrows } from "lucide-react";
import { api, fmt } from "./types";
import "./frontier-models.css";

type Row = {pair_id: string; drug_id: string; target_id: string; drug_name: string; gene: string; candidate_id: string; priority: number; scores: Record<string, number | null>; common_ranks: Record<string, number | null>; common_percentiles: Record<string, number | null>; common_denominator: number; disagreement_span: number | null; status: Record<string,string>; reasons: Record<string,string>; nesso_pic50: number | null};
type Metric = {scope: string; model: string; name: string; pairs: number; positive: number; negative: number; prevalence: number | null; ap: number | null; auroc: number | null};
type Snapshot = {available: boolean; updated_utc: string; provisional: boolean; models: {id: string; name: string; state: string; running: boolean; spr_scored: number; benchmark_scored: number; failed: number; unavailable: number; spr_unavailable?: number; spr_failed?: number}[]; items: Row[]; model_names: Record<string,string>; model_order: string[]; common_spr_pairs: number; common_benchmark_pairs: number; metrics: Metric[]; interpretation: string; benchmark_note: string; ranking_note: string; nesso_note: string; source_urls: Record<string,string>};
const stateText: Record<string,string> = {running:"推理中", inference:"推理中", encoding_protein:"生成蛋白表征",encoding_drug:"生成分子表征",waiting_encoders:"准备编码器",waiting_graphs:"下载特征并推理",preparing:"准备中",completed:"已完成",completed_with_missing:"结束，部分未覆盖",stopped_with_missing:"结束，部分未完成",stopped_before_completion:"中断，待恢复",not_running:"尚未运行"};
const reasonText: Record<string,string> = {publisher_protein_graph_not_available:"官方数据未覆盖该蛋白图特征",publisher_sequence_mismatch:"官方蛋白序列与本项目不一致",feature_download_incomplete:"特征下载未完成",long_protein_timeout_900s:"长蛋白单对推理超过15分钟",inference_missing_after_stage:"本阶段已结束，推理未产生有效结果"};
const missingText = (row: Row, model: string) => row.status[model] === "pending" ? "待完成" : row.status[model] === "unavailable" ? "未覆盖" : row.status[model] === "failed" ? "失败" : "无结果";

export default function FrontierModels({ kind, identifier, compact = false, initialView = "table", onShowTop10 }: {kind?: string; identifier?: string; compact?: boolean; initialView?: "table" | "plot"; onShowTop10?: () => void}) {
  const [data, setData] = useState<Snapshot | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState(initialView === "plot" ? "disagreement" : "priority");
  const [page, setPage] = useState(1);
  const [scope, setScope] = useState("BINDINGDB479_COMMON");
  const [view, setView] = useState(initialView);
  const [commonOnly, setCommonOnly] = useState(false);
  const [activePair, setActivePair] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => api<Snapshot>(`/api/frontier-dti?${new URLSearchParams({kind:kind || "",id:identifier || ""})}`,controller.signal).then(r => {setData(r);setError("");}).catch(e => {if(e.name!=="AbortError")setError(e.message);});
    refresh(); const timer = window.setInterval(refresh, 30000);
    return () => {controller.abort();window.clearInterval(timer);};
  },[kind,identifier,revision]);
  useEffect(() => setPage(1),[query,sort,kind,identifier,commonOnly,view]);
  const rows = [...(data?.items || [])].filter(r => (!(commonOnly || view === "plot") || r.disagreement_span!=null) && `${r.candidate_id} ${r.drug_name} ${r.gene} ${r.drug_id} ${r.target_id}`.toLowerCase().includes(query.toLowerCase()));
  rows.sort((a,b) => sort === "disagreement" ? (b.disagreement_span ?? -1) - (a.disagreement_span ?? -1) || a.priority-b.priority : sort === "priority" ? a.priority-b.priority : (b.scores[sort] ?? -Infinity)-(a.scores[sort] ?? -Infinity) || a.priority-b.priority);
  const visible = rows.slice((page-1)*20,page*20);
  const plotted = visible.filter(r => r.disagreement_span!=null);
  const active = plotted.find(r => r.pair_id===activePair);
  const metrics = data?.metrics.filter(m => m.scope===scope) || [];
  const gaps = (data?.models || []).flatMap(m => {
    const groups = new Map<string, {model: string; status: string; reason: string; count: number; genes: Set<string>}>();
    for (const row of data?.items || []) {
      if (row.scores[m.id] != null) continue;
      const status = missingText(row,m.id);
      const reason = reasonText[row.reasons[m.id]] || row.reasons[m.id] || (row.status[m.id] === "pending" ? "尚在队列中，或等待特征下载" : "尚无有效预测");
      const key = `${status}:${reason}`;
      const group = groups.get(key) || {model: m.name, status, reason, count: 0, genes: new Set<string>()};
      group.count++; group.genes.add(row.gene); groups.set(key,group);
    }
    return [...groups.values()];
  });
  return <section className="panel frontier-models">
    <div className="frontier-heading"><div><span className="eyebrow">SPR384 / MODEL REVIEW</span><h2><GitCompareArrows size={22}/> 新模型复核与分歧</h2></div><button className="button secondary" onClick={() => setRevision(x=>x+1)}><RefreshCw size={15}/>刷新</button></div>
    <p>原384候选的七模型复核：ReTargetMap、DrugCLIP、DTIAM A（九月加强版）、ConPLex，以及 Nesso-1、ProbeMatchDTI、DTBind。本轮新增模型尚未覆盖完整药物×靶点矩阵。</p>
    {error && <p className="error-panel" role="alert">{error}</p>}
    {!data ? <p role="status">正在读取模型结果…</p> : !data.available ? <p>本轮结果尚未生成。</p> : <>
      <div className="frontier-run"><strong>{data.provisional ? "正在运行 · 阶段性结果" : "本轮运行已结束"}</strong><span>更新 {new Date(data.updated_utc).toLocaleString()} · 每30秒刷新</span><a href="/api/frontier-dti.csv" download>下载384逐对复核 CSV <ArrowUpRight size={14}/></a></div>
      <div className="frontier-coverage">{data.models.map(m => <article key={m.id}><strong>{m.name}</strong><span>{stateText[m.state] || m.state}</span><b>{m.spr_scored}<small> / 384 候选已出分</small></b><small>待完成 {Math.max(0,384-m.spr_scored-(m.spr_unavailable || 0)-(m.spr_failed || 0))} · 未覆盖 {m.spr_unavailable || 0} · 失败 {m.spr_failed || 0}</small><span>有标签测试 {m.benchmark_scored} / 479</span><a href={data.source_urls[m.id]} target="_blank" rel="noreferrer">官方来源 ↗</a></article>)}</div>
      {gaps.length>0 && <details className="frontier-missing"><summary>为什么尚未覆盖？{kind ? "当前实体" : "原384候选"}的逐项原因</summary><p>“待完成”是运行进度；“未覆盖”是当前输入或特征缺口；“失败”是推理未成功。三者均不代表不结合。DTBind 的作者特征不全时，需要按同一流程生成蛋白表征、表面特征和几何图，不能补零或套用不同异构体；不必因此重新训练模型。</p><div className="frontier-scroll"><table><thead><tr><th>模型</th><th>状态</th><th>候选对数</th><th>原因 / 涉及靶点</th></tr></thead><tbody>{gaps.map(g=><tr key={`${g.model}:${g.status}:${g.reason}`}><th>{g.model}</th><td>{g.status}</td><td>{g.count}</td><td className="frontier-gap-reason">{g.reason}{g.status !== "待完成" && <small>{[...g.genes].sort().join("、")}</small>}</td></tr>)}</tbody></table></div></details>}
      <div className="frontier-note"><strong>七模型共同覆盖：{data.common_spr_pairs} / 384 候选</strong><p>{data.ranking_note}</p><p>{data.interpretation}</p></div>
      <div className="frontier-scope"><strong>{kind ? `当前范围：原384实验清单中，属于本${kind === "target" ? "靶点" : "药物"}的${data.items.length}对` : "当前范围：已冻结的384实验候选"}</strong><p>下方排序只重排这个候选范围，不会从完整目录重新选取Top10。表中的 #名次 / {data.common_spr_pairs} 是全部共同覆盖实验配对（跨药物、跨靶点）的池内位置，不是当前靶点的药物排名，也不是当前药物的靶点排名。</p><p>此处 ReTargetMap 使用药物→靶点的正向分数；靶点全目录药物排名使用反向输出。原384使用当时的历史组合评分、已知关系排除、疾病证据及人工审查选样；历史组合评分与当前网站ReTargetMap分数也不同。</p>{onShowTop10 && <button className="button secondary" onClick={onShowTop10}>查看 ReTargetMap 全目录Top10{kind === "target" ? "（靶点→药物）" : "（药物→靶点）"} <ArrowUpRight size={14}/></button>}</div>
      <div className="frontier-filters"><input aria-label="筛选新模型复核候选" placeholder="药物、基因或候选编号" value={query} onChange={e=>setQuery(e.target.value)}/><select aria-label="新模型复核排序" value={sort} onChange={e=>setSort(e.target.value)}><option value="priority">候选内 · 原实验优先顺序</option><option value="disagreement">候选内 · 模型分歧从大到小</option>{data.model_order.map(m=><option key={m} value={m}>候选内 · {data.model_names[m]}{m === "biomaster" ? " 正向" : ""}分数从高到低</option>)}</select><span>显示 {rows.length} / {data.items.length} 对冻结候选</span></div>
      <div className="frontier-view" role="group" aria-label="新模型分歧视图"><button className="button secondary" aria-pressed={view==="table"} onClick={()=>{setView("table");setCommonOnly(false);}}>逐对评分</button><button className="button secondary" aria-pressed={view==="plot"} onClick={()=>{setView("plot");setSort("disagreement");}}>七模型分歧轨迹</button><label><input type="checkbox" checked={view==="plot" || commonOnly} disabled={view==="plot"} onChange={e=>setCommonOnly(e.target.checked)}/>{view==="plot" ? "轨迹仅显示共同覆盖" : "仅看共同覆盖"}</label></div>
      {view==="plot" ? <div className="frontier-plot"><p>当前页 {plotted.length} 条轨迹 · 共同池 {data.common_spr_pairs} 对。越靠上表示排名越靠前；悬停或用Tab选择候选。</p><div className="frontier-scroll"><svg viewBox="0 0 1320 370" role="group" aria-label="原384候选七模型共同样本排名轨迹">
        {[0,.25,.5,.75,1].map(v=><g key={v}><line x1="90" x2="1290" y1={65+v*245} y2={65+v*245} stroke="#d8e2f0" strokeDasharray="4 5"/><text x="75" y={70+v*245} textAnchor="end" fill="#62728a" fontSize="12">{v===0?"前列":v===1?"末位":`${100*v}%`}</text></g>)}
        {data.model_order.map((m,i)=><g key={m} data-model-axis={m}><line x1={145+i*180} x2={145+i*180} y1="65" y2="310" stroke="#a3b7d1"/><text x={145+i*180} y="30" textAnchor="middle" fontSize="14" fontWeight="600" fill="#253953">{data.model_names[m].replace(/（.*）/g,"").replace(" · 结合预测","")}</text></g>)}
        {plotted.map(r=><g key={r.pair_id} role="button" tabIndex={0} aria-label={`${r.candidate_id} ${r.drug_name} → ${r.gene} 七模型排名`} onMouseEnter={()=>setActivePair(r.pair_id)} onMouseLeave={()=>setActivePair(null)} onFocus={()=>setActivePair(r.pair_id)} onBlur={()=>setActivePair(null)} onClick={()=>setActivePair(r.pair_id)} onKeyDown={e=>{if(e.key==="Enter")setActivePair(r.pair_id);}} opacity={activePair && activePair !== r.pair_id ? 0.18 : 0.8}>
          <title>{r.drug_name} → {r.gene} · 分位跨度 {((r.disagreement_span||0)*100).toFixed(1)} pp</title>
          <polyline points={data.model_order.map((m,i)=>`${145+i*180},${65+r.common_percentiles[m]!*245}`).join(" ")} fill="none" stroke="transparent" strokeWidth="12"/>
          <polyline points={data.model_order.map((m,i)=>`${145+i*180},${65+r.common_percentiles[m]!*245}`).join(" ")} fill="none" stroke={activePair===r.pair_id?"#006c5a":"#557cda"} strokeWidth={activePair===r.pair_id?3:1.5}/>
          {data.model_order.map((m,i)=><circle key={m} cx={145+i*180} cy={65+r.common_percentiles[m]!*245} r="3" fill="#2459e8" stroke="white"/>)}
        </g>)}
        <text x="680" y="350" textAnchor="middle" fontSize="12" fill="#62728a">所有坐标轴使用同一个共同候选池；缺失预测不画成低分。</text>
      </svg></div><div className="frontier-inspector" aria-live="polite">{active?<><strong>{active.candidate_id} · {active.drug_name} → {active.gene}</strong><span>{data.model_order.map(m=>`${data.model_names[m]} #${fmt(active.common_ranks[m],1)}`).join(" · ")}</span></>:<span>选择一条轨迹查看每个模型的具体排名。</span>}</div></div> : <div className="frontier-scroll"><table className="frontier-pairs"><thead><tr><th>原候选 / 药物 → 靶点</th>{data.model_order.map(m=><th key={m}>{data.model_names[m]}<small>{m === "biomaster" ? "正向分数 · 全SPR共同池排名" : "全SPR共同池排名"}</small></th>)}<th>分位跨度</th></tr></thead><tbody>{visible.map(r=><tr key={r.pair_id}><th><small>{r.candidate_id} · 原顺序 {r.priority}</small><a href={`#/drug/${r.drug_id}/rankings`}>{r.drug_name}</a><a href={`#/target/${r.target_id}/rankings`}>→ {r.gene}</a></th>{data.model_order.map(m=><td key={m}>{r.scores[m] == null ? <span title={reasonText[r.reasons[m]] || r.reasons[m] || "未完成评分，不作为阴性"}>{missingText(r,m)}</span> : <><strong>{r.common_ranks[m] == null ? "待共同覆盖" : `共同池 #${fmt(r.common_ranks[m],1)} / ${r.common_denominator}`}</strong><small>分数 {fmt(r.scores[m],4)}</small>{r.common_percentiles[m] != null && <div className="frontier-meter"><i style={{width:`${100*(1-r.common_percentiles[m]!)}%`}}/></div>}</>}</td>)}<td>{r.disagreement_span==null ? "—" : `${(100*r.disagreement_span).toFixed(1)} pp`}</td></tr>)}</tbody></table></div>}
      {!rows.length && <p role="status">{!data.items.length ? "此实体没有进入本次冻结384候选，暂无这三个模型的本轮复核结果。" : query ? "没有匹配当前搜索条件的候选。" : "这些候选尚无七模型共同覆盖结果，可切换“逐对评分”查看已完成模型和未覆盖原因。"}</p>}
      <div className="frontier-pages"><button className="button secondary" disabled={page<=1} onClick={()=>setPage(p=>p-1)}>上一页</button><span>{page} / {Math.max(1,Math.ceil(rows.length/20))}</span><button className="button secondary" disabled={page*20>=rows.length} onClick={()=>setPage(p=>p+1)}>下一页</button></div>
      {!compact && <details className="frontier-validation"><summary>真实标签上的共同样本测试</summary><p>{data.benchmark_note}</p><label>测试集合 <select aria-label="新模型测试集合" value={scope} onChange={e=>setScope(e.target.value)}><option value="BINDINGDB479_COMMON">479对中的七模型共同覆盖</option><option value="BINDINGDB175_COMMON">175对A/B未见子集中的共同覆盖</option>{data.models.flatMap(m => [<option key={m.id+"479"} value={`BINDINGDB479_${m.id}_AVAILABLE`}>{m.name} 已完成样本与原四模型 · 479集合</option>,<option key={m.id+"175"} value={`BINDINGDB175_${m.id}_AVAILABLE`}>{m.name} 已完成样本与原四模型 · 175子集</option>])}</select></label>{data.provisional && <p>当前共同样本受运行顺序影响，不能据此宣布哪个模型更强。</p>}<div className="frontier-scroll"><table><thead><tr><th>模型</th><th>同批样本</th><th>阳性 / 阴性</th><th>阳性率参照</th><th>AP</th><th>AUROC</th></tr></thead><tbody>{metrics.map(m=><tr key={m.model}><th>{m.name}</th><td>{m.pairs}</td><td>{m.positive} / {m.negative}</td><td>{fmt(m.prevalence)}</td><td>{fmt(m.ap)}</td><td>{fmt(m.auroc)}</td></tr>)}</tbody></table></div><p>AP越高，阳性越集中在前列；AUROC随机参照为0.5。没有两类标签时不计算指标。DTBind此处为结合预测；Nesso的pIC50为辅助通道，不当作实测Kd。</p></details>}
      {compact && <a href="#/spr/models">打开完整模型成绩与384分歧表 <ArrowUpRight size={14}/></a>}
    </>}
  </section>;
}
