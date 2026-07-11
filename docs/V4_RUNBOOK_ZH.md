# FDA 老药新靶点 v4 运行手册

所有命令在仓库根目录执行。任何一步失败后不得继续使用旧同名中间文件冒充本轮结果。

## 1. 冻结实体范围

```bash
python scripts/freeze_project_scope_v4.py
python scripts/build_universe_scope_audit_v4.py
python scripts/audit_project_target_sequence_integrity_v4.py
```

合同：750 个药物记录（723 个唯一模型配体结构）、463 条靶点序列、347,250 条 ID 审计 pair、334,749 个唯一结构物理 pair；drug/target manifest SHA-256 必须与 `configs/current_pipeline_v4.yaml` 一致。Top3000 中任何 ConPLEx 序列与结构模板不一致的 pair 必须隔离，不能进入 final1000/384。

## 2. 重建完整空间与 Top3000

```bash
python scripts/build_full_project_universe_v4.py

python scripts/annotate_sequence_homology_risk.py \
  --candidates outputs/current_production_package_v2/full_untruncated_universe_v4/pre_boltz_top3000_v4.csv \
  --known-controls outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_union_v4.csv \
  --sequences outputs/full_conplex_active_moiety_v4/protein_sequence_representatives.csv \
  --output outputs/current_production_package_v2/full_untruncated_universe_v4/pre_boltz_top3000_v4_homology_audited.csv

python scripts/annotate_compound_assay_liability.py \
  --input outputs/current_production_package_v2/full_untruncated_universe_v4/pre_boltz_top3000_v4_homology_audited.csv \
  --output outputs/current_production_package_v2/full_untruncated_universe_v4/pre_boltz_top3000_v4_fully_audited.csv
```

合同：Top3000 无 active-moiety × target 重复；旧 Top300 gate 为 false；旧 Boltz reuse 为 false。

## 3. 生成签名 Boltz 输入

```bash
python scripts/build_boltz2_complex_input_package.py \
  --root . \
  --source outputs/current_production_package_v2/full_untruncated_universe_v4/pre_boltz_top3000_v4_fully_audited.csv \
  --out-dir outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_input_package_v4_signed \
  --top-n 3000 --smoke-n 6 --use-template --use-pocket-constraint \
  --pocket-max-contacts 32 --pocket-max-distance 6.0
```

合同：3,000/3,000 输入具备 YAML、模型配体、蛋白序列、模板 PDB、口袋和完整输入 SHA-256。

## 4. 正式 Boltz-2 运行

```bash
python scripts/run_boltz2_batched_queue.py \
  --input-manifest outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_input_package_v4_signed/boltz2_input_manifest.csv \
  --input-dir outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_input_package_v4_signed/inputs \
  --out-dir outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_run_v4_seeded \
  --top-n 3000 --batch-size 5 --gpus 0,1 --min-free-gb 12 \
  --recycling-steps 3 --sampling-steps 50 --diffusion-samples 2 \
  --sampling-steps-affinity 50 --diffusion-samples-affinity 2 \
  --seed-base 20260710 --seed-scheme batch_index_offset
```

合同：600 个 batch 全部 `success`；provenance 3,000/3,000；confidence 与 affinity 文件逐项齐全；运行参数、seed、checkpoint 和输入签名一致。

若批运行因显存峰值留下缺失条目，先按原批次 seed 单条恢复。该步骤不降低模型参数，也不把计算失败解释为阴性：

```bash
python scripts/recover_boltz2_missing_rows_v4.py \
  --manifest outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_input_package_v4_signed/boltz2_input_manifest.csv \
  --run-dir outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_run_v4_seeded \
  --gpus 0,1 --recycling-steps 3 --sampling-steps 50 --diffusion-samples 2 \
  --sampling-steps-affinity 50 --diffusion-samples-affinity 2 \
  --num-workers 0 --tmp-dir .tmp/bv4
```

恢复后仍须满足 600/600 batch `success`、3,000/3,000 provenance 完整；恢复记录保存在 `recovery_rows.csv` 和 `recovery_summary.json`。

```bash
python scripts/rebuild_boltz_output_provenance_v4.py \
  --manifest outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_input_package_v4_signed/boltz2_input_manifest.csv \
  --run-dir outputs/current_production_package_v2/full_untruncated_universe_v4/boltz_full_run_v4_seeded \
  --expected-rows 3000 --source-label v4_formal_seeded
```

该步只重建并核对 confidence、affinity、model0 CIF、model1 CIF 的逐文件 SHA-256，不重跑模型。正式终结会逐行复算并比对。

## 5. 正式终结

```bash
python scripts/finalize_full_universe_v4.py
```

合同：Boltz 3,000/3,000、局部口袋姿势审计 3,000/3,000；输出 final1000=1,000、审阅池=512、物理初始提名=384。序列/模板不一致、已知对、同家族风险、严重化合物风险和 ion channel 均不进入审阅池。

## 6. Boltz 已知阳性校准

```bash
python scripts/build_known_control_boltz_calibration_v4.py

python scripts/build_boltz2_complex_input_package.py \
  --root . \
  --source outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_v4.csv \
  --out-dir outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_input_v4_sequence_matched_signed \
  --top-n 96 --smoke-n 6 --use-template --use-pocket-constraint \
  --pocket-max-contacts 32 --pocket-max-distance 6.0

python scripts/run_boltz2_batched_queue.py \
  --input-manifest outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_input_v4_sequence_matched_signed/boltz2_input_manifest.csv \
  --input-dir outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_input_v4_sequence_matched_signed/inputs \
  --out-dir outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_run_v4_sequence_matched_seeded \
  --top-n 96 --batch-size 4 --gpus 0,1 --min-free-gb 12 \
  --recycling-steps 3 --sampling-steps 50 --diffusion-samples 2 \
  --sampling-steps-affinity 50 --diffusion-samples-affinity 2 \
  --seed-base 20260710 --seed-scheme batch_index_offset
```

```bash
python scripts/rebuild_boltz_output_provenance_v4.py \
  --manifest outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_input_v4_sequence_matched_signed/boltz2_input_manifest.csv \
  --run-dir outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_run_v4_sequence_matched_seeded \
  --expected-rows 96 --source-label known_control_v4

python scripts/finalize_known_control_boltz96_v4.py

python scripts/audit_boltz_known_positive_calibration_v4.py \
  --known outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_refined_pose_audited_v4.csv \
  --discovery outputs/current_production_package_v2/formal_full_universe_v4/refined_top3000_v4_complete.csv \
  --output outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_calibration_v4.csv
```

该对照已排除序列/模板不一致靶点，不含真实阴性，不报告 AUROC/准确率。

## 7. Open Targets 26.06 疾病补全

```bash
python scripts/fetch_final1000_opentargets_full_diseases.py \
  --final1000 outputs/current_production_package_v2/full_untruncated_universe_v4/pre_boltz_top3000_v4_fully_audited.csv \
  --out-dir outputs/current_production_package_v2/full_untruncated_universe_v4/opentargets_top3000_target_completion_v4 \
  --page-size 500 --max-pages 0 --top-n 15
```

该层不参与物理排序。当前合同为 Open Targets 数据 26.06、API 26.6.3、215/215 靶点成功。

## 8. final384 精确证据审计

```bash
FORMAL=outputs/current_production_package_v2/formal_full_universe_v4

python scripts/audit_final_candidates_chembl_activity.py \
  --candidates "$FORMAL/agent_review_pool_v4_complete.csv" \
  --output "$FORMAL/agent_review_pool_chembl_activity_audited.csv" \
  --cache "$FORMAL/chembl_activity_cache.json"

python scripts/audit_final_candidates_pubmed.py \
  --candidates "$FORMAL/agent_review_pool_v4_complete.csv" \
  --out-dir "$FORMAL/agent_review_pool_pubmed_audit"
```

ChEMBL 区分 binding assay 与非 binding activity；PubMed 自动结果仅为名称/基因共现筛查，PMID 必须由智能体人工核实。

## 9. 逐条智能体审阅

```bash
python scripts/prepare_final384_agent_review_batches.py \
  --candidates "$FORMAL/agent_review_pool_v4_complete.csv" \
  --opentargets outputs/current_production_package_v2/full_untruncated_universe_v4/opentargets_top3000_target_completion_v4/refined_final_1000_candidates_with_full_opentargets_diseases.csv \
  --chembl "$FORMAL/agent_review_pool_chembl_activity_audited.csv" \
  --pubmed "$FORMAL/agent_review_pool_pubmed_audit/formal_pair_pubmed_audit.csv" \
  --out-dir "$FORMAL/agent_review" --batches 16 \
  --expected-rows 512 --rank-column review_pool_rank --output-prefix review512

python scripts/merge_final384_agent_reviews.py \
  --input-all "$FORMAL/agent_review/review512_agent_review_input_all.csv" \
  --reviews-dir "$FORMAL/agent_review" \
  --output "$FORMAL/agent_review_pool_reviewed.csv"

python scripts/select_reviewed_final384_v4.py \
  --review-pool "$FORMAL/agent_review_pool_v4_complete.csv" \
  --reviewed "$FORMAL/agent_review_pool_reviewed.csv" \
  --output "$FORMAL/final384_reviewed_selected_v4_complete.csv"
```

合同：512/512 覆盖；pair_id 无重复；所有结论、文献类别、疾病、暴露、活性物种、机制、实验、风险、数据库失败补查状态、置信度和来源非空。D、contradictory、未解决数据库查询失败或需要活性代谢物重跑的条目不得进入最终 384；回填不能放宽药物/靶点/骨架/family 硬上限。

## 10. 最终交付

```bash
python scripts/build_final_v4_delivery_package.py \
  --formal-dir "$FORMAL" \
  --selected384 "$FORMAL/final384_reviewed_selected_v4_complete.csv" \
  --opentargets-final1000 outputs/current_production_package_v2/full_untruncated_universe_v4/opentargets_top3000_target_completion_v4/refined_final_1000_candidates_with_full_opentargets_diseases.csv \
  --funnel-summary outputs/current_production_package_v2/full_untruncated_universe_v4_active_collapsed_sensitivity/full_untruncated_universe_v4_active_collapsed_sensitivity_summary.json \
  --known-summary outputs/current_production_package_v2/full_untruncated_universe_v4/full_untruncated_universe_v4_summary.json \
  --boltz-known-summary outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_calibration_v4.summary.json \
  --rank-sensitivity-summary outputs/current_production_package_v2/full_untruncated_universe_v4_active_collapsed_sensitivity/active_collapsed_rank_sensitivity_v4.summary.json \
  --chembl-cache "$FORMAL/chembl_activity_cache.json" \
  --pubmed-cache "$FORMAL/agent_review_pool_pubmed_audit/pubmed_pair_audit_cache.json" \
  --out-dir outputs/current_production_package_v2/final_delivery_v4
```

最终目录必须包含完整/中文 CSV、Excel、主报告 PDF、384 条详细证据卡 PDF、实验矩阵和带 SHA-256 的交付 manifest。

```bash
python scripts/audit_final_delivery_v4.py
```

只有生成 `FINAL_DELIVERY_AUDIT_V4.json` 且状态为 `formal_delivery_audit_passed`，本轮才能标记完成。
