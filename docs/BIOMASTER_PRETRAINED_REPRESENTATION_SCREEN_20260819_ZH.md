# BioMaster 分子/蛋白预训练表征筛选

更新时间：2026-08-19

## 结论

本轮实际接入并测试了两个与当前 champion 互补的冻结表征：

- 分子：MoLFormer-XL pooled 768 维，作为 `Morgan2048` 后的独立 gated residual；
- 蛋白：ESM-C 600M pooled 1152 维，作为 `ProtBERT1024 + 原有 ESM2-650M` 之后的第二个独立 gated residual。

当前决策：

1. **ESM-C 进入候选队列，暂不替换 champion。** 它在同 seed、完整预算的 S5 old-drug entity-cold 上有稳定正向信号，在 S3 也有正向点估计，但 S3 target-cluster bootstrap 仍跨 0。
2. **MoLFormer 暂不晋级。** 短屏的 S3 点估计为正，但 S5 明显下降；它不能作为默认分子 residual。
3. **MoLFormer + ESM-C 不叠加。** 四格短屏没有协同，S5 反而低于单独 ESM-C。
4. 当前默认模型仍保持原 pooled-ESM2 ODTI V2；不自动 promotion。

## 表征覆盖与质量

| 表征 | 对齐实体 | 输出 | 质量门 |
|---|---:|---:|---|
| MoLFormer-XL | 62,477 drugs；61,985 available | 62,477 × 768 float32 | PASS；491 个超出 202-token envelope 的分子显式 quarantine，未截断 |
| ESM-C 600M | 428 targets | 428 × 1152 float32 | PASS；313,718 residues、442 windows、0 unknown token |

ESM-C 长蛋白采用 2,046 residue 窗口、256 overlap，先做 overlap-stitch，再做 residue-uniform whole-protein mean；没有把长蛋白静默截断。

## 四格短屏

短屏为同一 seed、3 epoch、S3/S5 paired exploratory screen：

| protocol | MoLFormer | ESM-C | 两者同时 |
|---|---:|---:|---:|
| S3 micro-AUPRC Δ | +0.0061 | −0.0043 | −0.0001 |
| S5 micro-AUPRC Δ | −0.0320 | +0.0512 | −0.0036 |

短屏的作用是筛选方向，不是重写正式 champion 结果。

## ESM-C 正式复核

使用相同的 20260816、20260817 seeds、完整 40 epoch/patience 8，并与同 seed 的 E0 预测做 pair-aligned 聚合：

| protocol | ESM-C | matching E0 | Δ |
|---|---:|---:|---:|
| S3 micro-AUPRC | 0.63637 | 0.62473 | +0.01164 |
| S5 micro-AUPRC | 0.56906 | 0.53964 | +0.02942 |

500 次 target/scaffold cluster bootstrap：

- S3：scaffold-cluster AUPRC CI `[+0.0012, +0.0220]`，target-cluster CI `[-0.0038, +0.0278]`；
- S5：target-cluster AUPRC CI `[+0.0024, +0.0554]`，scaffold-cluster CI `[+0.0008, +0.0547]`。

因此 ESM-C 是**值得进入 5-seed formal promotion suite 的候选**，但尚未满足 S3/S5 全套、多 seed、source-heldout 和 prospective gate。

## 主要产物

- MoLFormer 特征包：`outputs/biomaster_odti_pretrained_features_v1/molformer_xl_both_10pct/`
- ESM-C 特征包：`outputs/biomaster_odti_pretrained_features_v1/esmc_600m/`
- 四格短屏：`outputs/biomaster_odti_pretrained_residual_screen_v1/`
- ESM-C S3 正式复核：`outputs/biomaster_odti_esmc_formal_s3_20260819/`
- ESM-C S5 正式复核：`outputs/biomaster_odti_esmc_formal_s5_20260819/`
- 配对审计：`outputs/biomaster_odti_pretrained_formal_audit_v1/PRETRAINED_RESIDUAL_FORMAL_AUDIT_V1.json`

## 下一步

只推进 ESM-C 单独分支到 S2/S3/S5 五 seeds，并加入 source-heldout external 与 calibration 检查；不再继续调 MoLFormer gate 或两者堆叠，除非新的实体冷 benchmark 明确显示分子侧收益。
