import { useEffect, useRef, useState } from 'react';
import { UploadCloud, Download, CheckCircle2, FileSpreadsheet, ArrowRight, RefreshCw } from 'lucide-react';
import './spr-results.css';
import SPRInput, { type Entry } from './SPRInput';

type Row = Record<string, string | number | boolean | null>;
type Preview = { entry_method?: string; token: string | null; batch: string; valid_count: number; error_count: number; skipped_count: number; errors: {line:number;message:string}[]; items: Row[] };
const results: Record<string,string> = {detected:'检出响应',not_detected:'未检出响应',inconclusive:'无法判定'};
const quality: Record<string,string> = {pass:'通过',fail:'未通过',review:'待确认'};
const value = (v: Row[string]) => v === null || v === undefined || v === '' ? '—' : String(v);
async function request<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, body === undefined ? undefined : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
export default function SPRResults() {
  const [batch,setBatch] = useState('');
  const [file,setFile] = useState<File | null>(null);
  const [preview,setPreview] = useState<Preview | null>(null);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const [saved,setSaved] = useState('');
  const [drag,setDrag] = useState(false);
  const [search,setSearch] = useState('');
  const [page,setPage] = useState(1);
  const [revision,setRevision] = useState(0);
  const [history,setHistory] = useState<{total:number;items:Row[]}>({total:0,items:[]});
  const [loading,setLoading] = useState(true);
  const [historyError,setHistoryError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const [expanded,setExpanded] = useState('');
  const [previewPage,setPreviewPage] = useState(1);
  useEffect(() => {
    let active = true;
    setLoading(true); setHistoryError('');
    const timer = setTimeout(() => request<{total:number;items:Row[]}>(`/api/spr-results?search=${encodeURIComponent(search)}&page=${page}&page_size=20`).then(data => {if(active)setHistory(data);}).catch(e => {if(active)setHistoryError(e.message);}).finally(() => {if(active)setLoading(false);}), 200);
    return () => {active=false;clearTimeout(timer);};
  },[search,page,revision]);
  function choose(next: File | null) {
    if (busy) return;
    setPreview(null);setSaved('');setError('');setPreviewPage(1);
    if(next && (!next.name.toLowerCase().endsWith('.csv') || next.size > 2*1024*1024)) {setError('请选择最大 2 MB 的 CSV 文件。Excel 请另存为 UTF-8 CSV。');setFile(null);return;}
    setFile(next);
  }
  async function validate() {
    if (!file || !batch.trim()) return;
    setBusy(true);setError('');setSaved('');setPreview(null);setPreviewPage(1);
    try {
      const content = new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());
      setPreview(await request<Preview>('/api/spr-results/preview',{batch:batch.trim(),filename:file.name,content}));
    } catch(e) {setError(e instanceof TypeError ? '文件无法按 UTF-8 读取或网络连接失败，请核对编码与连接后重试。' : (e as Error).message);}
    finally {setBusy(false);}
  }
  function invalidate() { setPreview(null); setSaved(''); setError(''); }
  async function validateOnline(rows: Entry[]) {
    setBusy(true); setError(''); setSaved(''); setPreview(null); setPreviewPage(1);
    try { setPreview(await request<Preview>('/api/spr-results/preview', {batch:batch.trim(),rows})); }
    catch(e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function commit() {
    if(!preview?.token) return;
    setBusy(true);setError('');
    try {
      const response = await request<{count:number}>('/api/spr-results/commit',{token:preview.token});
      setSaved(`批次 ${preview.batch} 的 ${response.count} 条结果已保存，审核状态：待审核。`);
      setPreview(null);setFile(null);if(input.current)input.current.value='';setRevision(v=>v+1);setPage(1);setSearch('');
    }catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  return <div className="spr-results-workspace">
    <header className="sru-hero"><div><span className="sru-kicker">EXPERIMENT RESULTS / SPR</span><h2>让实验结果回到研究配对</h2><p>在线填写或 CSV 批量导入，按实验配对归档每一次测量。</p></div></header>
    <ol className="sru-steps" aria-label="结果上传流程"><li className={!preview && !saved ? 'current':''}><span>01</span>填写实验结果</li><li className={preview ? 'current':''}><span>02</span>校验与预览</li><li className={saved ? 'current':''}><span>03</span>确认入库</li></ol>
    <SPRInput batch={batch} onBatch={v=>{setBatch(v);invalidate();}} busy={busy} onChange={invalidate} onValidate={validateOnline} saved={saved}>
    <div className="sru-upload-grid"><section className="sru-card"><h3><UploadCloud size={20}/>上传结果文件</h3><label className="sru-label" htmlFor="spr-batch">实验批次 <span>同批次的样本编号须唯一</span></label><input id="spr-batch" className="sru-input" value={batch} maxLength={120} disabled={busy} onChange={e=>{setBatch(e.target.value);setPreview(null);setSaved('');}} placeholder="例如：SPR-20260910-AR-01"/>
      <div className={`sru-drop ${drag ? 'dragging':''}`} onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);choose(e.dataTransfer.files[0] || null);}}><FileSpreadsheet size={34}/><strong>{file ? file.name : '拖入 CSV，或选择文件'}</strong><span>{file ? `${(file.size/1024).toFixed(1)} KB · UTF-8 CSV` : '支持 UTF-8 CSV · 最大 2 MB · 最多 2,000 行'}</span><input ref={input} id="spr-result-file" type="file" accept=".csv,text/csv" disabled={busy} onChange={e=>choose(e.target.files?.[0] || null)}/><label htmlFor="spr-result-file">选择 CSV 文件</label></div>
      <button className="sru-primary" disabled={!file || !batch.trim() || busy} onClick={validate}>{busy ? '正在处理…' : '校验并预览'}<ArrowRight size={17}/></button>
    </section><aside className="sru-card sru-guide"><span className="sru-kicker">IMPORT GUIDE</span><h3>先核对，再入库</h3><p>模板按上方选择的分组、排序和靶点筛选生成。默认仅包含基线 384 项，连续模板序号与原候选编号分别显示。仅填写已完成实验的行，空白行会自动跳过。</p><dl><dt>必填</dt><dd>样本编号、实验日期、响应结果、质控结果；保留模板中的实验编号与配对编号。</dd><dt>结果 / 质控</dt><dd>检出响应 / 未检出响应 / 无法判定<br/>通过 / 未通过 / 待确认</dd><dt>亲和力与动力学</dt><dd>KD 与 KD_unit 配套填写，支持 M、mM、uM、nM、pM；ka、kd、Rmax、χ² 可选。未检出不填 KD。</dd><dt>重复实验</dt><dd>每次重复使用不同 sample_id。已入库样本不会被覆盖，修订请使用新批次并在 notes 引用原记录。</dd></dl><p className="sru-note">这里只接收结果表。原始传感曲线、仪器工程文件与拟合报告请另行保留；检出响应不等于特异性结合，上传不代表审核通过。</p></aside></div>
    </SPRInput>
    {error && <div className="sru-error" role="alert">{error}</div>}{saved && <div className="sru-success" role="status"><CheckCircle2 size={20}/>{saved}</div>}
    {preview && <section className="sru-card" aria-label="上传校验预览"><div className="sru-section-heading"><div><h3>校验预览 · {preview.batch}</h3><p role="status">{preview.valid_count} 行通过 · {preview.error_count} 行错误 · {preview.skipped_count} 行空白跳过</p></div><button className="sru-primary" disabled={busy || !preview.token || preview.error_count>0} onClick={commit}>确认导入 {preview.valid_count} 条</button></div>
      {preview.errors.length>0 && <div className="sru-error" role="alert"><strong>请修正后重新上传；本次尚未写入任何结果。</strong><ul>{preview.errors.slice(0,50).map(e=><li key={e.line}>第 {preview.entry_method==='online'?e.line-1:e.line} {preview.entry_method==='online'?'条':'行'}：{e.message}</li>)}</ul>{preview.errors.length>50 && <p>仅展示前 50 个错误，共 {preview.error_count} 个。</p>}</div>}
      <div className="sru-table-scroll"><table><thead><tr><th>来源行 / 样本</th><th>实验配对</th><th>实测响应</th><th>KD（原值）</th><th>KD / nM</th><th>质控</th></tr></thead><tbody>{preview.items.slice((previewPage-1)*20,previewPage*20).map(row=><tr key={String(row.sample_id)}><td>{row.entry_method==='online'?Number(row.source_line)-1:value(row.source_line)} · {value(row.sample_id)}</td><td>{value(row.experiment_id)}<small>{value(row.drug_name)} → {value(row.target_name)}</small></td><td>{results[String(row.result)]}</td><td>{value(row.KD)} {value(row.KD_unit)==='—'?'':value(row.KD_unit)}</td><td>{value(row.KD_nM)}</td><td>{quality[String(row.qc)]}</td></tr>)}</tbody></table></div>
      <div className="sru-pagination"><span>预览有效行 · 确认后保存全部 {preview.valid_count} 条 · 30 分钟内有效</span><button disabled={previewPage<=1} onClick={()=>setPreviewPage(p=>p-1)}>上一页</button><span>{previewPage} / {Math.max(1,Math.ceil(preview.valid_count/20))}</span><button disabled={previewPage*20>=preview.valid_count} onClick={()=>setPreviewPage(p=>p+1)}>下一页</button></div>
    </section>}
    <section className="sru-card" aria-label="已上传 SPR 结果"><div className="sru-section-heading"><div><span className="sru-kicker">OBSERVATION REGISTER</span><h3>已上传结果 <span className="sru-count">{history.total}</span></h3><p>保留原始单位与提交来源；全部为待审核的实验观察记录。</p></div><a className="sru-export" href={`/api/spr-results.csv?search=${encodeURIComponent(search)}`}><Download size={16}/>导出筛选结果</a></div>
      <div className="sru-history-tools"><input className="sru-input" aria-label="搜索已上传结果" placeholder="搜索批次、样本、药物、靶点或实验编号…" value={search} onChange={e=>{setSearch(e.target.value);setPage(1);}}/><button aria-label="刷新结果" onClick={()=>setRevision(v=>v+1)}><RefreshCw size={18}/></button></div>
      {loading ? <p role="status">正在读取结果…</p> : historyError ? <p className="sru-error" role="alert">{historyError}</p> : !history.items.length ? <div className="sru-empty"><FileSpreadsheet size={30}/><strong>{search ? '没有匹配的结果' : '尚未上传实验结果'}</strong><p>{search ? '调整关键词或清除搜索。' : '下载模板并填写实验结果，完成校验后即可入库。'}</p></div> : <div className="sru-table-scroll"><table><thead><tr><th>批次 / 样本</th><th>药物 → 靶点</th><th>实测响应</th><th>KD / nM</th><th>质控 / 审核</th><th>实验日期 / 详情</th></tr></thead><tbody>{history.items.map(row=>{const key=`${row.upload_id}-${row.sample_id}`;return <tr key={key}><td>{value(row.batch)}<small>{value(row.sample_id)}</small></td><td><a href={`#/drug/${row.drug_id}/experiments`}>{value(row.drug_name)}</a> → <a href={`#/target/${row.target_id}/experiments`}>{value(row.target_name)}</a><small>{value(row.experiment_id)} · {row.is_control ? '参考对照':'候选'}</small></td><td>{results[String(row.result)]}</td><td>{value(row.KD_nM)}<small>原值 {value(row.KD)} {value(row.KD_unit)==='—'?'':value(row.KD_unit)}</small></td><td>{quality[String(row.qc)]}<small>待审核</small></td><td>{value(row.experiment_date)}<button className="sru-detail-button" aria-expanded={expanded===key} onClick={()=>setExpanded(expanded===key?'':key)}>查看记录</button>{expanded===key && <dl className="sru-record"><dt>精确配对</dt><dd>{value(row.pair_id)}</dd><dt>构建 / 最高浓度 µM</dt><dd>{value(row.construct)} / {value(row.max_concentration_uM)}</dd><dt>ka / M⁻¹s⁻¹ · kd / s⁻¹</dt><dd>{value(row.ka_M_inv_s_inv)} · {value(row.kd_s_inv)}</dd><dt>Rmax / RU · χ²</dt><dd>{value(row.Rmax_RU)} · {value(row.chi2)}</dd><dt>备注</dt><dd>{value(row.notes)}</dd><dt>提交来源</dt><dd>{value(row.uploaded_by)} · {value(row.uploaded_at)}<br/>{row.entry_method==='online'?'在线填写':value(row.filename)} · 第 {row.entry_method==='online'?Number(row.source_line)-1:value(row.source_line)} {row.entry_method==='online'?'条':'行'}</dd></dl>}</td></tr>;})}</tbody></table></div>}
      <div className="sru-pagination"><span>共 {history.total} 条</span><button disabled={page<=1||loading} onClick={()=>setPage(p=>p-1)}>上一页</button><span>{page} / {Math.max(1,Math.ceil(history.total/20))}</span><button disabled={page*20>=history.total||loading} onClick={()=>setPage(p=>p+1)}>下一页</button></div>
    </section>
  </div>;
}
