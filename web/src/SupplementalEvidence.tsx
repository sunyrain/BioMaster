import { useEffect, useState } from "react";
import { ArrowDownToLine, ArrowUpRight, ChevronDown, ChevronLeft, ChevronRight, Database, FileSearch, LoaderCircle, Search, X } from "lucide-react";
import { api, Entity, Evidence, fmt } from "./types";
import "./supplemental-evidence.css";

type Result = { items: Evidence[]; total: number; all_total: number; page: number; page_size: number };
const truth = (value: unknown) => value === true || value === 1 || String(value).toLowerCase() === "true";
const classes: Record<string, { label: string; tone: string; explanation: string }> = {
  OTHER_SOURCE_ASSERTION_REVIEW: { label: "来源断言 · 待核验", tone: "teal", explanation: "保留来源记录，仍需检查实验对象、关系方向与测量端点。" },
  TEXT_MINING_OR_PREDICTION: { label: "文本挖掘／预测", tone: "violet", explanation: "来自文本挖掘、模型预测或含此类证据的混合来源。" },
  PRIMEKG_ONLY_REUSE: { label: "仅复用 PrimeKG", tone: "slate", explanation: "来源仅转载旧图，不构成独立新增证据。" },
  INFERRED_OR_UNSPECIFIED_HOLD: { label: "推断／来源不清 · 暂扣", tone: "amber", explanation: "推断性质或原始来源不足，保留待审查。" },
};
const predicateNames: Record<string, string> = {
  "biolink:treats": "治疗来源标注", "biolink:clinical_trials_for": "临床试验关系",
  "biolink:contraindicated_in": "禁忌来源标注", "biolink:affects": "影响／调控关系",
  "biolink:interacts_with": "相互作用标注", "biolink:physically_interacts_with": "物理相互作用标注",
  "biolink:correlated_with": "相关性关系", "biolink:associated_with": "关联来源标注",
  "biolink:related_to": "来源相关关系", "biolink:gene_associated_with_condition": "基因与病况关联",
};
const roles: Record<string, string> = { OUR_FROZEN_MODEL_HIGH: "高排名候选", OUR_FROZEN_MODEL_INTERMEDIATE: "中排名候选", OUR_FROZEN_MODEL_LOW_BACKGROUND: "低排名背景", POSITIVE_CONTROL: "参考对照" };
const fieldNames: Record<string, string> = {
  project_key: "项目实体", other_id: "关系另一端", relation_class: "关系类别", ec_subject: "EC 主体", ec_object: "EC 客体",
  predicate: "原始谓词", qualified_predicate: "限定谓词", object_direction_qualifier: "客体方向限定",
  primary_sources: "原始来源", source: "来源", knowledge_level: "知识层级", agent_type: "生成方式",
  evidence_class: "证据分类", mapping_ambiguous: "映射是否有歧义", increment_class: "相对旧图的覆盖状态",
  publications: "出版物标识", source_path: "来源文件", source_paths: "来源文件", mapping_scope: "疾病映射范围",
  ligand_inchikey: "模型分子 InChIKey", target_chembl_id: "靶点 ChEMBL", selection_role: "原设计角色",
  review_status: "EC 审查状态", txgnn_available: "TxGNN 分数可用", treatment_direction_established: "治疗方向已确立",
  ec_relation_rows: "EC 关系记录数", non_text_unambiguous_assertion_rows: "非文本且映射无歧义断言数",
  top50_ot_overlap: "Top50 疾病线索与 OT 交集", ot_status: "Open Targets 覆盖状态",
};
function valueText(value: unknown): string {
  if (value === true) return "是";
  if (value === false) return "否";
  if (value == null || value === "") return "未提供";
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
}
function Fields({ row, keys }: { row: Evidence; keys?: string[] }) {
  return <dl className="se-fields">{(keys || Object.keys(row).filter(key => !["id", "name", "disease_hypotheses", "disease_hypotheses_json"].includes(key)))
    .filter(key => key in row).map(key => <div key={key}><dt>{fieldNames[key] || key.replace(/_/g, " ")}</dt><dd>{valueText(row[key])}</dd></div>)}</dl>;
}
function EvidenceBadge({ classification }: { classification?: string }) {
  const item = classes[classification || ""] || { label: "证据类型未分类", tone: "slate", explanation: "当前未提供可识别的证据分类，需查看来源。" };
  return <span className={`se-badge ${item.tone}`} title={item.explanation}>{item.label}</span>;
}

function SourceRelations({ entity }: { entity: Entity }) {
  const [data, setData] = useState<Result | null>(null);
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => { setPage(1); setQuery(""); setData(null); }, [entity.id, entity.kind]);
  const params = new URLSearchParams({ kind: entity.kind, id: entity.id, section: "eckg", search: query, page: String(page), page_size: "15" });
  const exportParams = new URLSearchParams({ kind: entity.kind, id: entity.id, section: "eckg", search: query });
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true); setError("");
    const timer = window.setTimeout(() => {
      api<Result>(`/api/evidence?${params}`, controller.signal)
        .then(result => { if (!controller.signal.aborted) setData(result); })
        .catch(err => { if (!controller.signal.aborted) setError(err.message); })
        .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    }, query ? 180 : 0);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [entity.id, entity.kind, query, page, retry]);
  const pages = Math.max(1, Math.ceil((data?.total || 0) / 15));
  return <section className="se-panel">
    <div className="se-heading"><div><span className="se-kicker">EC-KG / SOURCE RELATION AUDIT</span><h2>来源关系与证据审查</h2></div>
      <a className="se-export" href={`/api/evidence.csv?${exportParams}`} download><ArrowDownToLine size={16} />导出 CSV</a>
    </div>
    <p className="se-intro">查看 {entity.name} 在 EC-KG 中的关系、来源和限定条件。来源断言、文本挖掘及旧图复用分别标注；这些记录不能自动认定为已验证结合或已批准适应症。</p>
    <div className="se-legend">{Object.entries(classes).map(([key, item]) => <span key={key} title={item.explanation}><i className={item.tone} />{item.label}</span>)}</div>
    <div className="se-toolbar"><label className="se-search"><Search size={17} /><input aria-label="检索 EC-KG 来源关系" value={query} placeholder="检索实体、关系、来源或文献编号…" onChange={event => { setPage(1); setQuery(event.target.value); }} />
      {query && <button type="button" aria-label="清空 EC-KG 检索" onClick={() => { setQuery(""); setPage(1); }}><X size={16} /></button>}{busy && <LoaderCircle size={16} className="se-spin" />}</label>
      <span className="se-result-count" role="status">{data ? `${fmt(data.total)} 条${query ? ` / 全部 ${fmt(data.all_total)}` : "来源记录"}` : "正在读取来源记录"}</span>
    </div>
    <div className="se-browser" aria-busy={busy}>
      {error ? <div className="se-state se-error" role="alert"><Database size={25} /><strong>来源关系暂时无法读取</strong><p>{error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>重试加载</button></div> : !data && busy ?
        <div className="se-state" role="status"><LoaderCircle size={26} className="se-spin" /><strong>正在加载真实来源关系…</strong></div> : !data?.items.length ?
          <div className="se-state"><FileSearch size={28} /><strong>{query ? "没有匹配的来源关系" : "当前没有已接入的 EC-KG 关系"}</strong><p>{query ? "尝试其他实体名称、谓词或来源标识。" : "未检出关系不代表不存在，也不证明候选具有新颖性。"}</p></div> :
          data.items.map((row, index) => <details className="se-relation" key={`${row.id || row.project_key}-${row.other_id || ""}-${page}-${index}`}>
            <summary><span className="se-row-number">{String((page - 1) * 15 + index + 1).padStart(2, "0")}</span><span className="se-relation-main"><strong>{row.name || row.other_name || row.other_id || row.id || "来源关系记录"}</strong>
              <span className="se-relation-meta"><span>{predicateNames[row.predicate] || row.predicate || "谓词未标注"}</span><span>{row.primary_sources || row.source || "来源未标注"}</span></span>
            </span><span className="se-relation-flags"><EvidenceBadge classification={row.evidence_class} />{truth(row.mapping_ambiguous) && <span className="se-badge amber">映射有歧义</span>}</span><ChevronDown size={17} className="se-chevron" /></summary>
            <div className="se-relation-body"><p className="se-class-note">{(classes[row.evidence_class] || { explanation: "需查看原始记录确认证据类型。" }).explanation}{truth(row.mapping_ambiguous) ? " 当前项目实体对应多个规范节点，应先解决映射歧义。" : ""}</p><Fields row={row} /></div>
          </details>)}
    </div>
    {data && data.total > 0 && !error && <div className="se-pagination"><span>第 {page} / {pages} 页 · 每页 15 条</span><div><button type="button" aria-label="上一页 EC-KG 关系" disabled={page <= 1 || busy} onClick={() => setPage(value => value - 1)}><ChevronLeft size={18} /></button><button type="button" aria-label="下一页 EC-KG 关系" disabled={page >= pages || busy} onClick={() => setPage(value => value + 1)}><ChevronRight size={18} /></button></div></div>}
  </section>;
}

function SprReview({ entity }: { entity: Entity }) {
  const rows = (entity.disease_review?.spr_records || []) as Evidence[];
  if (!rows.length) return null;
  return <section className="se-panel se-spr">
    <div className="se-heading"><div><span className="se-kicker">SPR / APPEND-ONLY DISEASE REVIEW</span><h2>补充疾病假设与既有来源审查</h2></div><span className="se-badge amber">附加审查 · 非实验结果</span></div>
    <p className="se-intro">以下 {fmt(rows.length)} 条附录保留原 SPR 配对和角色。疾病线索、靶点关联与 EC 来源记录用于解释和复核候选，原实验放行状态以设计档案为准。</p>
    <div className="se-spr-records">{rows.map((row, index) => {
      const hypotheses = Array.isArray(row.disease_hypotheses) ? row.disease_hypotheses as Evidence[] : [];
      return <article className="se-spr-record" key={row.id || `${row.ligand_inchikey}-${row.target_chembl_id}-${row.selection_role}-${index}`}>
        <div className="se-spr-title"><div><span className="se-row-number">{String(index + 1).padStart(2, "0")}</span><h3>{row.drug_name || row.drug_names || row.ligand_inchikey} <span>→</span> {row.target_name || row.gene_symbol || row.target_chembl_id}</h3></div><div className="se-spr-flags"><span className="se-badge slate">{roles[row.selection_role] || row.selection_role || "原设计角色未标注"}</span>{row.drug_in_catalog === false && <span className="se-badge amber">模型目录外分子</span>}</div></div>
        <div className="se-spr-columns"><div className="se-hypotheses"><div className="se-subheading"><strong>药物疾病假设 × 靶点疾病关联</strong><span>Top50 交集：{fmt(row.top50_ot_overlap)}</span></div>
          {row.hypothesis_rule && <p className="se-class-note">展示 OT 分数至少 0.1 的交集，再按药物内 TxGNN 排名查看；这是审查阈值，不是验证过的联合预测概率。Top50 交集数量仍按全部 OT 关联计算。</p>}
          {truth(row.known_relations_allowed_for_control) && <p className="se-class-note">阳性对照允许展示已有适应证，供核对使用，不计入新用途候选。</p>}
          {!truth(row.txgnn_available) ? <p className="se-inline-empty">该配对药物没有可用 TxGNN 分数；缺失不作为零分或阴性。</p> : !hypotheses.length ? <p className="se-inline-empty">当前附录没有可展示的配对疾病交集。疾病无交集不等于不结合。</p> :
            hypotheses.map((hypothesis, i) => <div className="se-hypothesis" key={`${hypothesis.txgnn_disease_id}-${i}`}>
              <strong>{hypothesis.txgnn_disease_name || hypothesis.ot_disease_name || "疾病名称未提供"}</strong>
              <div className="se-hypothesis-values"><span>TxGNN logit <b>{fmt(hypothesis.txgnn_logit)}</b></span><span>OT 关联分数 <b>{fmt(hypothesis.ot_score)}</b></span></div>
              {hypothesis.txgnn_rank_within_drug != null && <p className="se-class-note">该药疾病排名：{fmt(hypothesis.txgnn_rank_within_drug)}；存在交集不等于位于预测前列。</p>}
              <span className="se-ot-disease">对应 OT：{hypothesis.ot_disease_name || "未提供名称"} · {hypothesis.ot_disease_id || "未提供编号"}</span>
              {hypothesis.mapping_scope === "MEMBER_OF_MERGED_NODE" ? <p className="se-merged-note">合并疾病节点成员映射：该 logit 属于整个节点，不能作为此子病种的特异预测。</p> : <span className="se-mapping-scope">{hypothesis.mapping_scope === "EXACT_SINGLE_NODE" ? "精确单疾病节点映射" : `映射范围：${hypothesis.mapping_scope || "未提供"}`}</span>}
              <details className="se-node-id"><summary>TxGNN 原始疾病节点 ID</summary><code>{hypothesis.txgnn_disease_id || "未提供"}</code></details>
            </div>)}
          {Array.isArray(row.ot_top_diseases) && row.ot_top_diseases.length > 0 && <details className="se-node-id"><summary>独立查看靶点 Open Targets 关联前 3 项</summary>{row.ot_top_diseases.map((d: Evidence) => <p key={d.disease_id}>{d.disease_name} · {d.disease_id} · {fmt(d.overall_score)}</p>)}</details>}
        </div><aside className="se-spr-relation"><div className="se-subheading"><strong>EC-KG 既有来源检查</strong></div><dl className="se-spr-counts"><div><dt>关联记录</dt><dd>{fmt(row.ec_relation_rows)}</dd></div><div><dt>非文本且无映射歧义断言</dt><dd>{fmt(row.non_text_unambiguous_assertion_rows)}</dd></div></dl>
          <p className="se-review-meaning">{row.review_status === "REVIEW_EXISTING_SOURCE_ASSERTIONS_NOT_AUTOMATIC_BINDING_CONFIRMATION" ? "已检出需复核的既有来源断言。确认谓词、测量端点和实体身份后，才能判断与候选结合关系是否一致。" : row.review_status === "TEXT_OR_MAPPING_REVIEW_ONLY" ? "当前仅有文本或映射待审线索，不能作为确认结合证据。" : row.review_status === "NO_ECKG_RELATION_MATCH" ? "本快照未匹配到该药物—靶点来源关系；不代表关系不存在或已经完成查新。" : row.review_status || "当前未提供关系审查状态。"}</p>
          {row.predicates && <div className="se-spr-source"><span>来源谓词</span><code>{row.predicates}</code></div>}{row.primary_sources && <div className="se-spr-source"><span>原始来源</span><code>{row.primary_sources}</code></div>}
          <span className="se-direction-note">{truth(row.treatment_direction_established) ? "治疗方向状态请核对原始附录" : "治疗干预方向尚未确立"}</span>
        </aside></div>
        <details className="se-spr-original"><summary>查看原配对标识与附录字段 <ChevronDown size={15} /></summary><Fields row={row} keys={["ligand_inchikey", "target_chembl_id", "selection_role", "txgnn_available", "ot_status", "top50_ot_overlap", "treatment_direction_established", "ec_relation_rows", "non_text_unambiguous_assertion_rows", "review_status", "source_paths"]} /></details>
      </article>;
    })}</div>
  </section>;
}

export default function SupplementalEvidence({ entity, mode = "all" }: { entity: Entity; mode?: "all" | "spr" }) {
  return mode === "spr" ? <SprReview entity={entity} /> : <SourceRelations key={`${entity.kind}-${entity.id}`} entity={entity} />;
}
