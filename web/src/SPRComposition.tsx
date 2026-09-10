import { useState } from "react";
import data from "./generated/spr384Composition.json";

export default function SPRComposition() {
  const [section, setSection] = useState(0);
  const selected = data.sections[section];
  return <section className="panel browse-intro" aria-label="当前384内部配比">
    <span className="eyebrow">384 候选内部配比 · 2026.09.10 统计快照</span>
    <h2>排名利用 46.6% · 相对较低排名探索 53.4%</h2>
    <p>384 个候选均位于各药物的前 20 / 384 名。以下为设计分布，尚未实验验证。</p>
    <label>查看配比维度 <select aria-label="查看384配比维度" value={section} onChange={e => setSection(Number(e.target.value))}>{data.sections.map((s,i) => <option key={s.title} value={i}>{s.title}</option>)}</select></label>
    <div style={{display:"grid",gap:12,marginTop:20}}>{selected.rows.map((r,i) => <div key={r.label}>
      <div style={{display:"flex",justifyContent:"space-between",gap:16}}><span>{r.label}</span><strong>{r.count} 对 · {r.percent.toFixed(1)}%</strong></div>
      <div style={{height:10,background:"#e8edf5",borderRadius:5,marginTop:6}}><div style={{width:`${r.percent}%`,height:"100%",background:["#4263eb","#748ffc","#d69e36","#8d99ae"][i%4],borderRadius:5}} /></div>
    </div>)}</div>
    <details style={{marginTop:20}}><summary>排名 × 实验建议 × 跨领域：查看交叉分布</summary>
      <div style={{overflowX:"auto",marginTop:12}}><table style={{width:"100%",textAlign:"left",borderSpacing:12}}><thead><tr>{["排名/384","数量","占比","仍可探索","先解决","降低投入","跨领域"].map(s => <th key={s}>{s}</th>)}</tr></thead><tbody>{data.rank_bands.map(r => <tr key={r.label}><td>{r.label}</td><td>{r.count}</td><td>{r.percent.toFixed(1)}%</td><td>{r.EXPLORABLE}</td><td>{r.FIRST_RESOLVE}</td><td>{r.EVIDENCE_DEPRIORITIZE}</td><td>{r.cross_area}</td></tr>)}</tbody></table></div>
      <p>后三类实验建议相加等于该排名组数量；跨领域是其中另一个维度，不额外相加。前10中“仍可探索”共100对，可在下方联合筛选查看。</p>
    </details>
    <details className="spr-method-notes"><summary>统计口径与实验边界</summary><p>{data.scope}。每个维度以384为分母，维度之间不可相加。</p><p>{data.rank_definition}</p>
    <p>全部推荐共同疾病满足TxGNN前50且OT≥0.3，但有额外证据与高排名是不同维度。“仍可探索”也不等于已确认结合。当前记录中384条均未获实验室／供应商确认，实际体系准备需另核。</p></details>
  </section>;
}
