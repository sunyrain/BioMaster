# Tracked result snapshots

`outputs/` is a local runtime directory and is ignored by default. Git only retains compact CSV/JSON/Markdown artifacts that are needed to audit reported counts, metrics, hashes and claim boundaries.

The current drug-centric reporting chain is represented by:

- `biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json`
- `biomaster_comprehensive_consensus_720x384_v5/CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json`
- `biomaster_bidirectional_v6_stage_a_dense/ENSEMBLE_STAGE_A_SUMMARY_V6.json`
- `biomaster_bidirectional_v6_full_fit/seed_*/FULL_FIT_SUMMARY_V6.json`
- `biomaster_bidirectional_v6_720x384/BIDIRECTIONAL_V6_FULL_FIT_720X384_SUMMARY.json`
- `evidence_routing_compute_execution_20260808_v1/leakage_safe_ranker_v10/` compact model-selection and external-evaluation summaries

Not tracked:

- model checkpoints and optimizer state;
- full pair-score matrices and bootstrap rows;
- feature arrays, database extracts and API caches;
- docking, structure-prediction and molecular-dynamics run directories.

Summary files may contain paths to untracked artifacts together with their SHA-256 hashes. This is intentional: the hash records the audited local input/output without publishing the large artifact itself.
