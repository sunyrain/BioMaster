# Tracked result snapshots

The current wet-lab handoff is `spr384_final_experiment_table_20260910/`: 384 candidate pairs plus 112 separately counted reference controls. The reviewed CSV/JSON records, control revisions, and compact quality audits are versioned. Experimental observations in `biomaster_explorer/spr_results/`, authentication files, databases, checkpoints, and large feature/score files remain local.

`outputs/.gitignore` is the explicit allowlist for new snapshots. Directory exceptions alone must not expose all descendants: that previously exposed per-system caches and a 417 MiB checkpoint. Add individual reviewed files to the allowlist; do not force-add whole output directories.

The 2026-09-10 cleanup is complete. `disk_reclamation_execution_20260910/RESULT.json` and `SERVICE_VERIFICATION.json` record the result. Historical manifests describe what existed when an experiment ran; retired intermediates may need rebuilding. The large per-file deletion journal remains local.

`biomaster_best_model_20260906/retargetmap_selected_v1/` is the selected standalone catalog model: one checkpoint, required inputs, model card and inference CLI. `DELIVERED_MODEL.json` records the validated archive; `docs/BIOMASTER_SELECTED_MODEL_20260906_ZH.md` explains all 78 development runs, fixed temporal/KIRHub regressions and direction-specific tradeoffs. The <=2025 full-fit weights have no independent test score; reported temporal regressions use separate <=2022 refits.

`outputs/` is a local runtime directory and is ignored by default. Git only retains compact CSV/JSON/Markdown artifacts that are needed to audit reported counts, metrics, hashes and claim boundaries.

`biomaster_unified_interaction_20260906/` contains the shared global/atom-residue/pocket-geometry research model, five first-round ablations and chronological refit/regression artifacts. `ROUND_STATUS.json` records actual completion; compact provenance is tracked, while atom/residue banks, LMDBs and checkpoints remain local. Single-seed first-round results do not promote a production model.

`biomaster_odti_v4_20260905/` holds the completed R1 development experiment: nine newly trained models, two historical V3 checkpoints evaluated on the same panel, two simple baselines, full feature caches and predictions. Its compact summaries are retained for audit. This is not the complete V4 geometry architecture or a production promotion; see `docs/BIOMASTER_ODTI_V4_R1_TRAIN_TEST_20260905_ZH.md`.

The single current contract is `configs/biomaster_current_contract_v1.json`. Its
audited drug-centric reporting chain is represented by:

- `biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json`
- `biomaster_comprehensive_consensus_720x384_v5/CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json`
- `biomaster_bidirectional_v6_stage_a_dense/ENSEMBLE_STAGE_A_SUMMARY_V6.json`
- `biomaster_bidirectional_v6_full_fit/seed_*/FULL_FIT_SUMMARY_V6.json`
- `biomaster_bidirectional_v6_720x384/BIDIRECTIONAL_V6_FULL_FIT_720X384_SUMMARY.json`
- `old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json`
- `target_discovery_scope_ch37_v3/TARGET_DISCOVERY_SCOPE_SUMMARY_V3.json`
- `evidence_routing_compute_execution_20260808_v1/leakage_safe_ranker_v10/` compact model-selection and external-evaluation summaries

The internal directory names above are provenance identifiers, not parallel
production models. The current roles are: one general drug-to-target system,
one FULL_FIT directional scoring component, one reverse-query auxiliary head,
and one separate KIRHub kinase-functional specialist. All other output
directories are exploratory or historical unless promoted by the current
contract.

Not tracked:

- model checkpoints and optimizer state;
- full pair-score matrices and bootstrap rows;
- feature arrays, database extracts and API caches;
- docking, structure-prediction and molecular-dynamics run directories.

Wet-lab preparation artifacts are written under `retargetmap_experiment_20260901/`:

- `RETARGETMAP_EXPERIMENT_PRELIMINARY_POOL_V1.csv.gz` is the preliminary, not-yet-procured candidate pool.
- `RETARGETMAP_EXPERIMENT_PRELIMINARY_POOL_SUMMARY_V1.json` records the frozen scope, counts, input hashes, model roles and claim boundary.
- `RETARGETMAP_TARGET_ASSAY_FEASIBILITY_ROSTER_V1.csv` is the 268-target lab handoff table for assay availability, reference-control, orthogonal-method, cost and lead-time review.

Summary files may contain paths to untracked artifacts together with their SHA-256 hashes. This is intentional: the hash records the audited local input/output without publishing the large artifact itself.
