import "./spr-detail.css";
import diseaseZh from "./generated/sprDiseaseZh.json";
import { LoadingState } from "./LoadingState";
import { useEffect, useState } from "react";
import {
  ArrowUpRight, ArrowRight, FlaskConical, Clock3, CalendarDays, CircleDashed, Search, X, Info, ShieldCheck,
  ChevronDown,
  ClipboardList,
} from "lucide-react";
import { api, Entity, Evidence } from "./types";
import { Value } from "./EvidenceBrowser";

const diseaseName = (value: unknown) => { const name = String(value || ""); return (diseaseZh as Record<string, string>)[name] || name || "未收录"; };

const fieldLabels: Record<string, string> = {
  original_indications: "原适应症", original_targets: "原靶点（已知机制库）", original_target_ids: "原靶点编号",
  original_mechanism: "原作用机制", origin_scope: "原用途范围", original_indication_sources: "原适应症来源",
  proposed_target: "目标靶点", proposed_indication: "目标适应症假设", proposed_disease_id: "疾病编号", disease_evidence: "联合疾病证据",
  name: "实验配对",
  experiment_id: "实验编号",
  pair_id: "配对标准编号",
  design_id: "设计版本",
  design_date: "设计日期",
  source: "来源",
  source_path: "来源文件",
  drug_name: "候选药物",
  gene_symbol: "靶点基因",
  drug_in_catalog: "药物在目录内",
  target_in_catalog: "靶点在目录内",
  role: "设计角色",
  role_label: "候选组别",
  is_control: "是否对照",
  experiment_arm: "实验分组",
  row_type: "行类型",
  logistics_batch: "物流批次",
  release_status: "放行状态",
  status_label: "审核状态",
  result_status: "实测结果状态",
  evidence: "证据说明",
  construct_recommendation: "蛋白构建建议",
  construct_status: "构建确认状态",
  special_system_review: "特殊体系审查",
  endpoint_plan: "实验端点规划",
  pair_review_status: "配对证据审查",
  novelty_status: "新颖性审查",
  reference_control: "参考对照",
  control_status: "对照状态",
  control_fda_status: "FDA身份核验",
  control_construct_requirement: "对照构建适用条件",
  previous_experiment_id: "替换前对照编号",
  previous_drug_name: "替换前对照药物",
  rank_context: "冻结排名口径",
  ranks: "设计冻结排名",
  review: "化学支持审查",
  exclusion_audit: "排除项审计",
  control_evidence: "对照原始证据",
};

const groups = [
  { title: "已知机制与来源", fields: ["original_mechanism", "original_indication_sources"] },
  { title: "疾病关联证据", fields: ["proposed_disease_id", "disease_evidence"] },
  { title: "实验准备", fields: ["construct_recommendation", "reference_control", "construct_status"] },
];
const conciseValue = (key: string, value: any) => {
  if (key === "construct_status" && String(value).includes("PENDING")) return "待实验室确认构建与采购条件";
  if (key === "disease_evidence" && value && typeof value === "object") return Object.fromEntries(Object.entries(value).filter(([k]) => !k.includes("映射范围")));
  return value;
};

function FDAControlDetails({ row }: { row: Evidence }) {
  if (!row.control_fda_status) return null;
  return <div className="spr-fda-control-details">
    <p><strong>{row.control_replaced ? "已换入 FDA 药物对照" : "对照 FDA 身份"}</strong> · {String(row.control_fda_status)}</p>
    {row.control_construct_requirement && <p><strong>适用构建：</strong>{String(row.control_construct_requirement)}</p>}
    {row.control_selected_endpoint && <p>选用文献记录：{String(row.control_selected_endpoint)} = {String(row.control_selected_value_nM)} nM（历史记录，非本次 SPR 测量）</p>}
    {row.control_selected_assay_description && <details><summary>选用实验方法与构建来源</summary><p>{String(row.control_selected_assay_description)}</p></details>}
    <div className="experiment-links">{((row.control_fda_urls || []) as string[]).map((url, i) => <a className="text-button" href={url} key={url} target="_blank" rel="noreferrer">FDA 审批记录 {i + 1}<ArrowUpRight size={13} /></a>)}</div>
    {row.is_control && row.control_replaced && <p>替换前：{String(row.previous_experiment_id)} · {String(row.previous_drug_name)}。本版编号：{String(row.experiment_id)}。</p>}
  </div>;
}

function ControlEvidence({ row }: { row: Evidence }) {
  const evidence = (row.control_evidence || {}) as Record<string, unknown>;
  const category = String(evidence.reference_endpoint_category || evidence.category || "");
  const endpoints = String(evidence.reference_standard_types || evidence.standard_types || "未收录");
  const explanation = category === "Kd_RECORD_PRESENT"
    ? "该化合物—靶点在本地 ChEMBL 数据中有阳性记录，端点包含 Kd，因此优先列为拟用结合参考对照。"
    : category === "Ki_NO_Kd_RECORD"
    ? "本版对照依据 Ki 记录；Ki 不能直接当作 SPR 实测 Kd，仍需确认直接结合实验适用性。"
    : category === "FUNCTIONAL_OR_OTHER_ONLY"
    ? "当前只有活性或其他端点记录，尚不足以确认可作 SPR 结合阳性对照；需核实或替换。"
    : "参考证据需进一步核查，当前尚未确认为可用的 SPR 阳性对照。";
  const links = (value: unknown, kind: "assay" | "document") => String(value || "").split(/[,;]/).map(v => v.trim()).filter(Boolean).map(id => {
    const chembl = id.startsWith("CHEMBL") ? id : `CHEMBL${id}`;
    return <a key={id} className="text-button" href={`https://www.ebi.ac.uk/chembl/explore/${kind}/${chembl}`} target="_blank" rel="noreferrer">{chembl}<ArrowUpRight size={13} /></a>;
  });
  return <section className="spr-detail-section spr-control-evidence">
    <h3>为什么选择 {row.drug_name} 作为 {row.gene_symbol} 的拟用参考对照</h3>
    <p>{explanation}</p><p>{String(row.control_status || "")}</p><FDAControlDetails row={row} />
    <div className="spr-attempt-chain">
      <div><small>对应靶点</small><p>{String(row.proposed_target || row.gene_symbol)}</p></div>
      <div><small>历史实验端点</small><p>{endpoints}</p></div>
      <div><small>对照确认状态</small><p>拟用参考 · 尚未验证 SPR 适用性</p></div>
    </div>
    <div className="experiment-field"><span>化合物来源</span><div className="field-value"><a className="text-button" href={`https://www.ebi.ac.uk/chembl/explore/compound/${String(evidence.control_chembl_id || evidence.chembl_id || "")}`} target="_blank" rel="noreferrer">{String(evidence.control_chembl_id || evidence.chembl_id || "未收录")}<ArrowUpRight size={13} /></a></div></div>
    <div className="experiment-field"><span>历史实验记录</span><div className="field-value">{links(evidence.reference_assay_ids || evidence.assay_ids, "assay")}</div></div>
    <div className="experiment-field"><span>实验文献索引</span><div className="field-value">{links(evidence.reference_doc_ids || evidence.document_ids, "document")}</div></div>
    <p>入选时已排除本地标记的阳性／阴性冲突，并核对结构身份。文献与实验编号可追溯，但尚未逐篇确认实验条件。</p>
    <p><strong>实验前待确认：</strong>原始数值与测定方法、蛋白构建与突变状态、溶解性及可采购性，并在实际 SPR 条件下验证信号。上述历史端点不是本项目实测结果。</p>
  </section>;
}

const reasonLabels: Record<string,string> = { ONLY_EVIDENCE_GAP: "仅额外证据不足", ADVERSE_ACTIVITY: "不利活性或选择性", PRIOR_ACTIVITY: "已有活动记录", NOVELTY_ONLY: "新颖性取舍", ENTITY_MISMATCH: "实体需核实", ASSAY_UNRESOLVED: "实验体系待解决", DISEASE_DIRECTION: "疾病作用方向", INDIRECT_OR_WRONG_ENTITY_EVIDENCE: "间接或其他实体证据" };
const actionLabels: Record<string, string> = { EXPLORABLE: "仍可探索", FIRST_RESOLVE: "先解决具体问题", EVIDENCE_DEPRIORITIZE: "有证据支持降低投入" };
const reviewLabels: Record<string, string> = { PRIORITIZE: "优先验证", CONDITIONAL: "有条件保留", DEFER: "建议后置", INSUFFICIENT: "信息不足" };
function LLMReview({ row }: { row: Evidence }) {
  const review = row.llm_review as Record<string, any> | undefined;
  if (!review) return <div className="spr-record-note">LLM审查待完成 · 当前保留基线评价</div>;
  const action = row.resource_review as Record<string, any> | undefined;
  return <section className="spr-detail-section spr-review-panel">{action && <div className="spr-attempt-chain spr-action-grid"><div><small>如何安排实验</small><p><strong>{action.action_label}</strong></p><p>{action.can_test}</p></div><div><small>具体原因</small><p>{action.reason_detail}</p><p className="spr-reason-tags">{(action.reason_labels || []).join(" · ")}</p></div><div><small>关键条件</small><p>{action.critical_condition}</p></div></div>}<div className="spr-review-verdict"><h3>LLM逐对审查 · {review.verdict_label}</h3>
    <p>{review.summary}</p></div>
    <details className="spr-review-details"><summary>审查依据与来源<ChevronDown size={14} /></summary>
    {[["支持依据", "support"], ["疑点与投入风险", "concern"], ["下一步", "next_step"], ["证据范围", "evidence_scope"], ["来源说明", "source_notes"]].map(([label, key]) => review[key] && <div className="experiment-field" key={key}><span>{label}</span><div className="field-value"><Value value={key === "evidence_scope" ? "本地记录＋逐对联网检索；各来源访问范围见下方说明和检索记录" : review[key]} /></div></div>)}
    <div className="experiment-links">{(review.source_urls || []).filter((u: string) => /^https?:\/\//.test(u)).map((u: string, i: number) => <a key={u} href={u} target="_blank" rel="noreferrer" className="text-button">依据 {i + 1}<ArrowUpRight size={13} /></a>)}</div>
    <details className="spr-search-history"><summary>检索记录与访问范围<ChevronDown size={14} /></summary><Value value={review.search_log} /></details><p>{review.scope_note}</p></details>
  </section>;
}

export default function Experiments({ entity, expanded = false }: { entity?: Entity; expanded?: boolean }) {
  const [rows, setRows] = useState<Evidence[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [reviewFilter, setReviewFilter] = useState("all");
  const [actionFilter, setActionFilter] = useState("all");
  const [reasonFilter, setReasonFilter] = useState("all");
  const [rankFilter, setRankFilter] = useState("all");
  useEffect(() => {
    const c = new AbortController();
    setBusy(true); setError(""); setRows([]); setFilter("all"); setQuery("");
    const url = entity ? `/api/evidence?kind=${entity.kind}&id=${encodeURIComponent(entity.id)}&section=experiments&page_size=200` : expanded ? "/api/spr-expanded" : "/api/spr-design";
    const fetchRows = (initial = false) => api<{ items: Evidence[] }>(url, c.signal)
      .then(d => { setRows([...d.items].sort((a, b) => Number(Boolean(a.is_control)) - Number(Boolean(b.is_control)) || Number(a.final_priority_order || 9999) - Number(b.final_priority_order || 9999))); setError(""); })
      .catch(e => { if (e.name !== "AbortError") setError(e.message); })
      .finally(() => { if (initial && !c.signal.aborted) setBusy(false); });
    fetchRows(true);
    const timer = window.setInterval(() => fetchRows(), 15000);
    return () => { c.abort(); window.clearInterval(timer); };
  }, [entity?.id, entity?.kind, expanded, revision]);
  const candidates = rows.filter(r => !r.is_control).length;
  const controls = rows.length - candidates;
  const rankMatches = (r: Evidence) => {
    if (rankFilter === "all") return true;
    if (r.is_control) return false;
    const rank = Number((r.ranks as Record<string, unknown>)?.retargetmap_rank_384);
    if (!Number.isFinite(rank) || rank < 1) return false;
    const [lo, hi] = rankFilter.split(":").map(Number);
    return rank >= lo && rank <= hi;
  };
  const visible = rows.filter(r => rankMatches(r) && (reasonFilter === "all" || (r.resource_review as Record<string, any>)?.reason_tags?.includes(reasonFilter)) && (actionFilter === "all" || (r.resource_review as Record<string, any>)?.action_class === actionFilter) && (reviewFilter === "all" || (!r.is_control && (reviewFilter === "PENDING" ? !r.llm_review : (r.llm_review as Record<string, any>)?.verdict === reviewFilter))) && (filter === "all" || (filter === "control" ? r.is_control : !r.is_control)) &&
    [r.drug_name, r.control_drug_name, r.gene_symbol, r.experiment_id, r.role_label, r.original_indications, r.original_targets, r.proposed_indication, diseaseName(r.proposed_indication), JSON.stringify(r.llm_review || {})].join(" ").toLowerCase().includes(query.trim().toLowerCase()));
  return (
    <section className="panel experiment-panel">
      <div className="spr-heading">
        <div className="spr-title-icon"><FlaskConical size={23} /></div>
        <div><span className="eyebrow">EXPERIMENT DESIGN / SPR</span><h2>{expanded ? "新增候选证据池" : "SPR 实验设计"}</h2><p>{expanded ? "本轮扩大范围的证据候选，尚未加入冻结实验设计或安排对照。" : "从计算候选到实验验证，追溯每一组配对。"}</p></div>
        <span className="spr-status"><Clock3 size={14} />设计审核中 · 尚未放行</span>
      </div>
      <div className="spr-context"><span><CalendarDays size={14} />2026.09.09 冻结快照</span><span>{rows[0]?.design_context || "综合384候选设计"}</span><span>排名来自设计版本</span></div>
      {busy ? <LoadingState text="正在读取设计记录…" />
        : error ? <div className="error-panel" role="alert">{error}<button className="button secondary" onClick={() => setRevision(v => v + 1)}>重新加载设计记录</button></div>
        : !rows.length ? <div className="empty"><ClipboardList size={27} /><strong>{expanded ? "扩展候选准备中" : "该实体未进入当前 SPR 设计"}</strong><span>可继续查看其完整模型排名和其他证据。</span></div>
        : <>
          <div className="experiment-summary">
            <div><span>关联实验配对</span><strong>{rows.length}<small>组</small></strong></div>
            <div><span><i className="spr-dot candidate" />模型候选</span><strong>{candidates}<small>组</small></strong></div>
            <div><span><i className="spr-dot control" />参考对照</span><strong>{controls}<small>组</small></strong></div>
            <div className="spr-result"><span>实验记录</span><strong><CircleDashed size={18} />独立归档</strong><small><a href="#/spr/results">上传与查看 SPR 实测结果 →</a></small></div>
          </div>
          <p className="spr-record-note">实验建议用于安排验证顺序；具体条件见配对详情。</p>
          <div className="spr-toolbar"><span>LLM已审查 {rows.filter(r => !r.is_control && r.llm_review).length} / {candidates} · 自动更新</span>
            <label>结合排名 <select aria-label="按结合排名筛选" value={rankFilter} onChange={e => setRankFilter(e.target.value)}><option value="all">全部</option><option value="1:5">前1–5</option><option value="6:10">第6–10</option><option value="1:10">前1–10：高排名利用</option><option value="11:20">第11–20：相对探索</option><option value="21:50">第21–50</option><option value="51:384">第51–384</option></select></label>
            <label>实验安排 <select aria-label="按实验安排筛选" value={actionFilter} onChange={e => setActionFilter(e.target.value)}><option value="all">全部</option>{Object.entries(actionLabels).map(([k,v]) => <option key={k} value={k}>{v}</option>)}</select></label>
            <label>具体原因 <select aria-label="按具体原因筛选" value={reasonFilter} onChange={e => setReasonFilter(e.target.value)}><option value="all">全部</option>{Object.entries(reasonLabels).map(([k,v]) => <option key={k} value={k}>{v}</option>)}</select></label>
            <label>LLM优先级 <select aria-label="按LLM审查建议筛选" value={reviewFilter} onChange={e => setReviewFilter(e.target.value)}><option value="all">全部</option><option value="PENDING">待审查</option>{Object.entries(reviewLabels).map(([k,v]) => <option key={k} value={k}>{v}</option>)}</select></label>
            <button className="button secondary" onClick={() => { const selected = visible.filter(r => !r.is_control); const blob = new Blob([JSON.stringify(selected, null, 2)], {type: "application/json"}); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "spr-candidate-review-subset.json"; a.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000); }}>导出当前筛选候选</button>
          </div>
          <div className="spr-toolbar">
            <div className="spr-filters" role="group" aria-label="实验分组筛选">
              {[["all", "全部配对", rows.length], ["candidate", "模型候选", candidates], ["control", "参考对照", controls]].map(([id, label, count]) =>
                <button key={id} className={filter === id ? "active" : ""} aria-pressed={filter === id} onClick={() => setFilter(String(id))}>{label}<span>{count}</span></button>)}
            </div>
            <label className="spr-search"><Search size={16} /><input aria-label="搜索实验配对" placeholder="搜索药物、原靶点、疾病或编号" value={query} onChange={e => setQuery(e.target.value)} />{query && <button aria-label="清除实验搜索" onClick={() => setQuery("")}><X size={14} /></button>}</label>
          </div>
          <div className="spr-list-heading"><span>实验配对 / {visible.length}</span><span>分组与冻结排名</span></div>
          <div className="spr-records">
          {visible.map(row => {
            const i = rows.indexOf(row);
            const rank = typeof row.ranks === "object" && row.ranks !== null ? (row.ranks as Record<string, unknown>).retargetmap_rank_384 : null;
            return <details className="experiment-record" key={row.id || i}>
              <summary>
                <span className={`experiment-number ${row.is_control ? "is-control" : ""}`}>{String(i + 1).padStart(2, "0")}</span>
                <div className="spr-pair"><strong>{row.drug_name}<ArrowRight size={14} /><span>{row.gene_symbol}</span></strong><small>{row.experiment_id || row.id} · {row.final_priority_label ? `${row.final_priority_label} · ` : ""} {!row.is_control && ((row.resource_review as Record<string, any>)?.action_label || (row.llm_review as Record<string, any>)?.verdict_label || "待审查")} · {row.is_control ? "靶点质控" : row.cross_area === true ? "跨领域假设" : row.cross_area === false ? "同领域或领域重叠" : "跨领域尚未核实"}</small></div>
                <span className={`badge ${row.is_control ? "blue" : "neutral"}`}>{row.role_label || row.role}</span>
                <span className="spr-rank">{row.is_control ? "参考对照" : rank != null ? <><b>#{String(rank)}</b><small> / 384</small></> : "未列排名"}</span>
                <ChevronDown size={16} />
              </summary>
              <div className="experiment-content">
                {!row.is_control && row.final_table_version && <section className="spr-detail-section"><h3>最终实验安排 · {String(row.final_priority_label || "")}</h3><div className="spr-attempt-chain spr-primary-summary"><div><small>小分子药物名称</small><p>{String(row.drug_name)}</p></div><div><small>旧靶点名称</small><p>{String(row.original_targets)}</p></div><div><small>新靶点名称</small><p>{String(row.proposed_target)}</p></div><div><small>新靶点的已知药物名称（拟用对照）</small><p>{String(row.control_drug_name)}</p><small>{String(row.control_id)} · {String(row.control_endpoint_types)}</small></div></div><p>{String(row.control_status)}</p><FDAControlDetails row={row} /><p>{String(row.identity_note || "")}</p><div className="experiment-links">{((row.control_assay_urls || []) as string[]).map((u, i) => <a className="text-button" href={u} key={u} target="_blank" rel="noreferrer">对照原始实验 {i + 1}<ArrowUpRight size={13} /></a>)}</div></section>}
                <div className="spr-attempt-chain spr-primary-summary"><div><small>原适应症</small><p>{String(row.original_indications || "未收录")}</p></div><div><small>原靶点</small><p>{String(row.original_targets || "未收录")}</p></div><div><small>目标靶点</small><p>{String(row.proposed_target || row.gene_symbol || "未收录")}</p></div><div><small>目标适应症</small><p title={String(row.proposed_indication || "")}>{row.is_control && !row.proposed_indication ? "参考对照，不设新适应症假设" : diseaseName(row.proposed_indication)}</p></div></div>
                {!row.is_control && <LLMReview row={row} />}
                <div className="spr-record-note"><Info size={15} /><span>设计记录，不代表已完成实验或实测结合。目标适应症为研究假设。</span></div>
                {row.is_control && <ControlEvidence row={row} />}
                <div className="experiment-links">
                  {row.drug_id && row.drug_in_catalog !== false && <a className="text-button" href={`#/drug/${encodeURIComponent(row.drug_id)}/overview`}>药物档案<ArrowUpRight size={13} /></a>}
                  {row.target_id && <a className="text-button" href={`#/target/${encodeURIComponent(row.target_id)}/overview`}>靶点档案<ArrowUpRight size={13} /></a>}
                </div>
                <div className="spr-detail-grid">{(row.is_control ? [{ title: "已知机制与用途记录", fields: ["original_indications", "original_targets", "original_target_ids", "original_mechanism"] }] : groups).map(group => <section className="spr-detail-section" key={group.title}><h3>{group.title}</h3>{group.fields.filter(key => row[key] != null && row[key] !== "").map(key => <div className="experiment-field" key={key}><span>{fieldLabels[key] || key}</span><div className="field-value">{key === "original_indication_sources" && Array.isArray(row[key]) ? (row[key] as string[]).map((url, j) => <a key={url} className="text-button" href={url} target="_blank" rel="noreferrer">官方用途来源 {j + 1}<ArrowUpRight size={13} /></a>) : <Value value={conciseValue(key, row[key])} />}</div></div>)}</section>)}</div>
                <details className="spr-source"><summary>完整记录与来源<ChevronDown size={14} /></summary><div>{Object.entries(row).filter(([k]) => !["id", "drug_id", "target_id"].includes(k)).map(([key, value]) => <div className="experiment-field" key={key}><span>{fieldLabels[key] || key.replace(/_/g, " ")}</span><div className="field-value"><Value value={value} /></div></div>)}</div></details>
              </div>
            </details>;
          })}
          </div>
          {!visible.length && <div className="empty"><Search size={23} /><strong>没有匹配的实验配对</strong><button className="text-button" onClick={() => { setQuery(""); setFilter("all"); setRankFilter("all"); setActionFilter("all"); setReasonFilter("all"); setReviewFilter("all"); }}>清除筛选</button></div>}
          <div className="spr-footnote"><ShieldCheck size={15} /><span>仅展示冻结设计与参考证据。所有配对均需完成实验审核后放行。</span></div>
        </>}
    </section>
  );
}
