import { LoadingState } from "./LoadingState";
import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Database,
  ExternalLink,
  LoaderCircle,
  Search,
  X,
} from "lucide-react";
import { api, Entity, Evidence, fmt, label } from "./types";

type Result = {
  items: Evidence[];
  total: number;
  all_total: number;
  page: number;
  page_size: number;
};
export default function EvidenceBrowser({
  entity,
  section,
}: {
  entity: Entity;
  section: string;
}) {
  const [data, setData] = useState<Result | null>(null);
  const [page, setPage] = useState(1);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const control = new AbortController();
    setBusy(true);
    setError("");
    const t = setTimeout(
      () =>
        api<Result>(
          `/api/evidence?kind=${entity.kind}&id=${encodeURIComponent(entity.id)}&section=${section}&page=${page}&page_size=15&search=${encodeURIComponent(query)}`,
          control.signal,
        )
          .then((r) => {
            setData(r);
            setError("");
          })
          .catch((e) => {
            if (e.name !== "AbortError") setError(e.message);
          })
          .finally(() => {
            if (!control.signal.aborted) setBusy(false);
          }),
      160,
    );
    return () => {
      clearTimeout(t);
      control.abort();
    };
  }, [entity.id, entity.kind, section, page, query, revision]);
  return (
    <div className="evidence-browser" aria-busy={busy}>
      <div className="filter-search">
        <Search size={15} />
        <input
          aria-label={`搜索${section}证据`}
          placeholder="搜索名称、疾病编号或证据内容…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setPage(1);
          }}
        />
        {query && (
          <button
            className="icon-button"
            aria-label="清空证据筛选"
            onClick={() => {
              setQuery("");
              setPage(1);
            }}
          >
            <X size={16} />
          </button>
        )}
        {busy && data && (
          <LoaderCircle size={17} className="spin" aria-label="正在更新证据" />
        )}
      </div>
      {busy && !data ? (
        <LoadingState text="正在加载证据…" />
      ) : error ? (
        <div className="error-panel" role="alert">{error}<button className="button secondary" onClick={() => setRevision(v => v + 1)}>重新加载证据</button></div>
      ) : !data?.items.length ? (
        <div className="empty">
          <Database size={25} />
          <strong>
            {query ? "没有匹配的记录" : "当前本地数据没有匹配证据"}
          </strong>
          <span>缺失不表示不存在这种关联。</span>
        </div>
      ) : (
        <div className="browser-rows">
          {data.items.map((row, i) => {
            const score = row.score ?? row.association_score;
            const name = label(row);
            return (
              <div className="browser-row" key={`${row.id}-${i}`}>
                <div className="browser-row-title">
                  <span className="browser-index">
                    {String((data.page - 1) * 15 + i + 1).padStart(2, "0")}
                  </span>
                  {section === "known_targets" &&
                  (row.explorable ?? row.in_catalog) ? (
                    <a
                      className="text-button"
                      href={`#/target/${encodeURIComponent(row.id)}/overview`}
                    >
                      {name}
                      <ArrowUpRight size={13} />
                    </a>
                  ) : (
                    <strong>{name}</strong>
                  )}
                  {score != null && (
                    <span className="browser-score">{fmt(score)}</span>
                  )}
                  {row.url && /^https?:\/\//.test(row.url) && (
                    <a
                      className="source-link"
                      href={row.url}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={`查看${name}来源`}
                    >
                      <ExternalLink size={13} />
                    </a>
                  )}
                </div>
                <div className="browser-meta">
                  {row.id && <code>{row.id}</code>}
                  {row.phase != null && (
                    <span
                      className={`badge ${row.phase === 4 ? "blue" : "amber"}`}
                    >
                      {row.phase === 4
                        ? "ChEMBL 阶段 4 · 已批准"
                        : `临床研究 · 阶段 ${row.phase}`}
                    </span>
                  )}
                  {row.source && <span>{row.source}</span>}
                  {row.category && <span>{row.category}</span>}
                  {row.relation && <span>{row.relation}</span>}
                </div>
                {(row.mechanism_of_action ||
                  row.description ||
                  row.evidence ||
                  row.action_type) && (
                  <p>
                    {[
                      row.action_type,
                      row.mechanism_of_action ||
                        row.description ||
                        row.evidence,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                )}
                {row.mapping_note && <p>{row.mapping_note}</p>}
              </div>
            );
          })}
        </div>
      )}
      {data && (
        <div className="pagination">
          <span>
            {query ? `${data.total} 个匹配 / ` : ""}共{" "}
            {data.all_total.toLocaleString()} 条
          </span>
          <div>
            <button
              aria-label="上一页证据"
              disabled={page <= 1 || busy}
              onClick={() => setPage((p) => p - 1)}
            >
              <ChevronLeft size={15} />
            </button>
            <span>
              {data.page} / {Math.max(1, Math.ceil(data.total / 15))}
            </span>
            <button
              aria-label="下一页证据"
              disabled={page >= Math.ceil(data.total / 15) || busy}
              onClick={() => setPage((p) => p + 1)}
            >
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const propertyNames: Record<string, string> = {
  smiles: "规范 SMILES",
  molecular_formula: "分子式",
  molecular_weight: "分子量",
  parent_molecular_formula: "原上市药物分子式",
  parent_molecular_weight: "原上市药物分子量",
  approval_year: "获批年份",
  brand_names: "商品名",
  routes: "给药途径",
  mechanism_of_action: "已知作用机制",
  action_types: "作用方向",
  fda_application_numbers: "FDA 申请编号",
  metadata_scope: "信息归属",
  species_role: "模型分子身份",
  species_name: "模型分子名称",
  family: "靶点家族",
  subfamily: "靶点亚类",
  sequence_length: "蛋白长度",
  organism: "物种",
  sequence: "蛋白序列",
  af_mean_plddt: "AlphaFold 平均 pLDDT",
  pocket_mean_plddt: "口袋平均 pLDDT",
  assay_lane: "实验分路",
  approved_name: "蛋白标准名称",
  subcellular_locations: "亚细胞定位",
  tractability: "成药性注释",
  go: "Gene Ontology",
  synonyms: "别名",
  hallmarks: "功能标志",
  chemical_probes: "化学探针",
  tissue_expression: "组织表达",
  go_annotations: "Gene Ontology 注释",
  function_descriptions: "蛋白功能",
  target_classes: "靶点分类",
  safety_liabilities: "已知安全性注释",
  genetic_constraint: "遗传约束",
  genomic_location: "基因组位置",
  canonical_transcript: "标准转录本",
  hbd: "氢键供体",
  hba: "氢键受体",
  tpsa: "极性表面积",
  rotatable_bonds: "可旋转键",
  logp: "LogP",
};
export function Value({ value }: { value: any }) {
  if (value == null || value === "") return <span>—</span>;
  if (typeof value !== "object") return <span>{fmt(value)}</span>;
  if (Array.isArray(value)) {
    if (!value.length) return <span>本地无记录</span>;
    return (
      <div className="property-entries">
        {value.map((v, i) => (
          <div key={i}>
            <Value value={v} />
          </div>
        ))}
      </div>
    );
  }
  return (
    <dl className="object-properties">
      {Object.entries(value).map(([k, v]) => (
        <div key={k}>
          <dt>{propertyNames[k] || k}</dt>
          <dd>
            <Value value={v} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
export function Metadata({ entity }: { entity: Entity }) {
  const [query, setQuery] = useState("");
  const fields = Object.entries(entity.properties).filter(([key]) =>
    `${key} ${propertyNames[key] || ""}`
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  );
  return (
    <section className="panel metadata-panel">
      <div className="section-title">
        <div>
          <span className="eyebrow">COMPLETE ENTITY RECORD</span>
          <h2>完整实体档案</h2>
        </div>
        <span className="badge neutral">原始注释可展开</span>
      </div>
      {entity.kind === "target" &&
        entity.properties.tissue_expression?.length > 0 && (
          <ExpressionChart rows={entity.properties.tissue_expression} />
        )}
      <div className="full-identifiers">
        {Object.entries(entity.identifiers).map(([key, value]) => (
          <span key={key}>
            <small>{key.replace(/_/g, " ")}</small>
            <code>{Array.isArray(value) ? value.join("; ") : fmt(value)}</code>
          </span>
        ))}
      </div>
      {entity.description && (
        <p className="metadata-description">{entity.description}</p>
      )}
      <div className="metadata-filter">
        <div className="filter-search">
          <Search size={17} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="检索档案字段"
            placeholder="查找字段，如 GO、序列、组织表达…"
          />
          {query && (
            <button
              className="icon-button"
              aria-label="清空档案筛选"
              onClick={() => setQuery("")}
            >
              <X size={16} />
            </button>
          )}
        </div>
        <span>{fields.length} 个字段</span>
      </div>
      {!fields.length && (
        <div className="empty">
          <Database size={23} />
          <strong>没有匹配的档案字段</strong>
          <span>可尝试中文名称或原始英文字段名。</span>
        </div>
      )}
      {fields.map(([key, value]) => (
        <details
          className="property-detail"
          key={`${key}-${Boolean(query)}`}
          open={query.trim() ? true : undefined}
        >
          <summary>
            <span>{propertyNames[key] || key.replace(/_/g, " ")}</span>
            <small>
              {Array.isArray(value)
                ? `${value.length} 条注释`
                : typeof value === "object"
                  ? "结构化注释"
                  : String(value ?? "—").slice(0, 100)}
            </small>
            <ChevronRight size={14} />
          </summary>
          <div className="property-value">
            <Value value={value} />
          </div>
        </details>
      ))}
    </section>
  );
}

function ExpressionChart({ rows }: { rows: Evidence[] }) {
  const sources = [...new Set(rows.map((r) => r.source))];
  const [source, setSource] = useState(sources[0]);
  const [query, setQuery] = useState("");
  const candidates = rows.filter(
    (r) => r.source === source && typeof r.value === "number",
  );
  const unit = candidates[0]?.unit || "";
  const maximum = Math.max(0.001, ...candidates.map((r) => r.value));
  const displayed = candidates
    .filter((r) => r.name.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => b.value - a.value)
    .slice(0, 12);
  return (
    <div className="expression-chart">
      <div className="section-title">
        <div>
          <span className="eyebrow">TISSUE EXPRESSION</span>
          <h2>组织表达谱</h2>
        </div>
        <span className="badge neutral">
          {unit} · {candidates.length} 个组织
        </span>
      </div>
      <div className="expression-controls">
        <select
          aria-label="组织表达来源"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        >
          {sources.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <div className="filter-search">
          <Search size={14} />
          <input
            aria-label="筛选组织"
            placeholder="筛选组织名称…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      </div>
      <div className="expression-bars">
        {displayed.map((r, i) => (
          <div key={r.id + i} title={`${r.name}: ${r.value} ${r.unit}`}>
            <span>{r.name}</span>
            <div>
              <i style={{ width: `${(r.value / maximum) * 100}%` }} />
            </div>
            <strong>{fmt(r.value, 2)}</strong>
          </div>
        ))}
      </div>
      <p>
        显示当前来源中表达量最高的 12 个匹配组织。GTEx TPM 与 HPA nTPM
        分别展示；完整记录可在下方档案中展开。
      </p>
    </div>
  );
}
