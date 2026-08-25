# BioMaster Experiment Closure Tracker（历史快照）

Generated UTC: 2026-06-05T07:21:57Z

This file records a historical experiment-closure snapshot generated on June 5, 2026. It is **not** the current runtime status: the current workspace no longer contains the `outputs/report_scale` full-DiffDock artifacts referenced below. Do not report the old queue as running, completed, or resumed without rebuilding a manifest, job index, and frozen execution protocol.

For the current ODTI model/data program, the authoritative machine-readable summaries are:

- `outputs/biomaster_odti_live_status_v1/BIOMASTER_ODTI_LIVE_STATUS_V1.json`
- `outputs/biomaster_odti_model_data_readiness_v1/BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.json`
- `outputs/biomaster_odti_external_entity_cold_landscape_v1/EXTERNAL_ENTITY_COLD_LANDSCAPE_V1.json`

The historical tracker references are retained below for provenance only:

- `outputs/sota_validation/sota_compute_closure_summary.json`
- `outputs/sota_validation/external_dependency_audit/sota_external_dependency_audit.json`
- `outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json`
- `outputs/sota_validation/sota_artifact_manifest.json`

## Historical Conclusion（2026-06-05）

At the time of generation, the core P0 data and validation layers were ready and the full DiffDock expansion was reported as running. That statement is historical and has not been revalidated in the current workspace.

## Current Workspace Reconciliation

The current audit reports `NOT_INITIALIZED_IN_CURRENT_WORKSPACE` for the old full-DiffDock branch: no `outputs/report_scale` directory, job index, score directory, or current queue process is present. The old experiment-closure audit therefore remains `not_complete` and must not be combined with the current ODTI E0 results.

The current ODTI execution queue is documented in:

- `docs/BIOMASTER_ACTIVE_EXECUTION_QUEUE_20260817_ZH.md`
- `outputs/biomaster_odti_model_data_readiness_v1/BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.md`

## Running Required Computation

### Full DiffDock expansion

- Screen session: `biomaster_diffdock_full_resume`
- Log: `logs/diffdock_full_resume_20260605_021454.log`
- Job index: `outputs/report_scale/diffdock_full_run/diffdock_full_job_index.csv`
- Completed jobs: 2437 / 3653
- Completed job percentage: 66.7123%
- Scored rows: 609250 / 913170
- Scored row percentage: 66.7181%
- Remaining rows by completed score files: 303920
- Completed rank1 outputs in scored jobs: 472491
- Technical missing outputs in scored jobs: 136759
- Active jobs: 2437, 2438, 2439, 2440
- Busy GPUs: 4 / 4
- Estimated remaining time: 67.1958 hours
- Estimated finish UTC: 2026-06-08T02:33:41Z

Interpretation: the queue is actively producing rank1 SDF files. Current active chunks are normal in-flight jobs, not stale locks.

Latest ligand-level technical failure audit:

- Scored ligands: 594
- Scored rows: 601500
- Completed rank1 outputs: 465657
- Technical missing outputs in scored jobs: 135843
- Missing percentage in scored jobs: 22.5840%
- Rescue-recommended ligands: 64
- Rescue-recommended rows: 64837
- Zero-completed chunks: 205
- Mask-rotate zero-completed chunks: 205

Interpretation: these missing rows are DiffDock technical output failures, not biological negative calls. The multi-ligand rescue watcher is configured to rerun this failure audit and rebuild the aggregate rescue queue after the main queue and the single-ligand rescue queue complete.

### Automatic rescue watchers

- Single-ligand rescue watcher: `biomaster_diffdock_ligand_rescue_after_full`
- Multi-ligand rescue watcher: `biomaster_diffdock_multi_ligand_rescue_after_single`
- Single-ligand watcher log: `logs/diffdock_ligand_rescue_after_full.log`
- Multi-ligand watcher log: `logs/diffdock_multi_ligand_rescue_after_single.log`

Interpretation: both watchers are active and waiting for the main full DiffDock queue to complete and GPUs to become idle. They were restarted with `--min-free-gb 2`, which matches the local 32 GB GPU configuration.

## Required After Full DiffDock Completes

1. Merge the full DiffDock score files and completed rank1 outputs.
2. Run the prepared ligand rescue queue for `CHEMBL3039504`.
3. Rebuild full-docking TopK tables after rescue completion.
4. Re-run structure-facing audits that depend on full DiffDock coverage:
   - pose readability and local geometry QC
   - pocket pLDDT confidence audit
   - residue-contact interpretability
   - final structure-adjusted candidate ranking
5. Refresh the integrated expert review panel and final shortlist.
6. Regenerate the computation closure summary and artifact manifest.
7. Update the website/report artifacts after the computation layer is closed.

## Prepared Required Rescue Queue

- Ligand: `CHEMBL3039504`
- Reason: original multi-fragment or salt-form SDF causes clustered DiffDock technical failures.
- Rescue input: largest organic parent SDF
- Queued rows: 998
- Jobs: 16
- Parent SDF: `data/processed/ligands_sdf_chembl_parent/CHEMBL3039504_parent.sdf`
- Queue index: `outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent/diffdock_ligand_rescue_job_index.csv`

This should run only after the main full DiffDock queue has completed or a GPU is verified idle.

An active watcher is now waiting to start this queue after the main full DiffDock queue completes and GPUs are idle.

## Prepared Aggregate Multi-Ligand Rescue Queue

The currently prepared aggregate queue is an interim prebuilt queue:

- Candidate ligands: 48
- Queued ligands: 48
- Queued rows: 47871
- Jobs: 768
- Queue index: `outputs/report_scale/diffdock_full_multi_ligand_rescue/aggregate/diffdock_multi_ligand_rescue_job_index.csv`

This queue is not treated as final. The watcher `biomaster_diffdock_multi_ligand_rescue_after_single` will rerun `build_diffdock_full_ligand_failure_audit.py` and rebuild the aggregate queue after the main queue and `CHEMBL3039504` rescue complete. The latest audit currently recommends 64 ligands and 64837 rows, so the final aggregate queue may expand relative to this interim queue.

## Wet-Lab Focusing State

The current experiment-planning package is already narrowed enough for expert discussion and procurement checks while full DiffDock continues in the background:

- Candidate rows: 3921
- Experiment-ready rows: 885
- Review-ready rows: 763
- Experiment-or-review-ready rows: 1648
- Balanced shortlist: 300
- Expert-review panel: 50
- Wave1 panel: 24
- Purchase and assay queue: 94
- First experiment panel: 12
- Core first batch: 6
- Procurement/platform checklist: 12

Current first-batch use should still be treated as planning support, not final efficacy evidence. Before spending wet-lab budget, the 12 procurement rows require manual confirmation of compound availability, salt/form, purity, storage, solubility, clinically achievable concentration, assay availability, model expression, counterscreens, contraindications, and drug-drug interaction risk.

## Completed Core Evidence Layers

- ConPLex affinity screening and candidate ranking
- FDA known-target benchmark and enrichment audit
- Multi-disease Open Targets evidence integration
- TxGNN drug-disease and disease-direction evidence
- Pathway and disease-context evidence
- Rule-based ADMET and label-risk audit
- ML ADMET endpoint audit using local TDC-trained QSAR models
- ChEMBL target druggability and clinical-stage audit
- KG explainability paths
- Tissue-context audit from HPA and GTEx
- DepMap oncology dependency scoring
- LINCS/CMap disease-signature reversal scoring
- Network medicine / PPI proximity on the expanded STRING/HuRI graph
- Chemotype diversity audit
- Local supervised independent DTI corroboration over final candidates
- DiffDock disease-direction structural docking and multiple rescue rounds
- Pose quality, pocket confidence, and pose interpretability audits for completed candidates
- Standard PoseBusters/ProLIF checks
- AutoDock Vina, smina, and GNINA CNN rescoring layers
- Boltz-2 second-model Top50 validation and high-sampling finalist validation
- Final multi-evidence prioritization, ablation, novelty, specificity, calibration, and validation-panel design
- Integrated Top50 expert review panel and Wave1 24-candidate panel
- Computation artifact manifest

### Expanded network-medicine layer

- Edge table: `data/processed/huri_string_expanded_edges.csv`
- Metadata: `data/processed/huri_string_expanded_edges.metadata.json`
- Graph size: 14074 nodes and 137275 edges
- Candidate rows audited: 23744
- Final-priority rows audited: 3921
- Candidate-row PPI coverage: 93.8637%
- Unique target-protein coverage: 429 / 467 (91.8630%)
- SOTA-network positive final rows: 3342 / 3921 (85.2334%)
- SOTA-network coverage-gap rows: 262

## Completed P0 Downloads

- GNINA binary and runtime
- LINCS/CMap Level 5 compound signatures and metadata
- Direction-level disease signatures
- GTEx tissue expression
- HPA tissue expression
- DepMap/CCLE oncology dependency context
- GOA human annotations
- UniProt-to-Reactome mappings
- FDA approved small-molecule table with structures
- ChEMBL druggable proteome table
- Expanded PPI/network medicine source data for dependency-readiness purposes:
  - STRING human protein info: `data/external/string/9606.protein.info.v12.0.txt.gz`
  - STRING human physical links: `data/external/string/9606.protein.physical.links.v12.0.txt.gz`
  - HuRI interactome: `data/external/huri/HuRI.tsv`
  - Expanded processed edge table: `data/processed/huri_string_expanded_edges.csv`

## Optional SOTA Extensions Not Yet Closed

These are not blockers for the current multi-evidence expert-review package. They can be used to strengthen SOTA comparison if more time is available.

### P1: formal pretrained DTI models

- DrugBAN formal inference
  - Code is present under `third_party/DrugBAN`.
  - Missing runtime dependencies: `dgl`, `dgllife`.
  - Existing demo checkpoints are not yet validated as a formal benchmark checkpoint for this project.
  - Recommended treatment: build a separate environment or container, then run a small reproducibility check before integrating outputs.

- DeepDTA formal inference
  - Code is present under `third_party/DeepDTA`.
  - Missing legacy TensorFlow/Keras runtime and validated pretrained checkpoint.
  - Recommended treatment: isolate in a separate legacy environment; do not install into the active DiffDock environment.

- GraphDTA optional inference
  - Code and validated checkpoints are absent.
  - Recommended treatment: add only if a third independent DTI baseline is required.

### P2: optional second complex model

- Chai-1 or AF3-style rerun
  - Missing `chai_lab` environment and model downloads.
  - Boltz-2 validation is already complete, so this is an optional corroboration layer rather than a blocker.

### P2: expanded PPI / network medicine data

- Dependency-readiness status: ready.
- Evidence: STRING human protein info, STRING physical links, HuRI TSV, and expanded processed PPI edge table are available locally.
- Background supplement: STRING human aliases are still being downloaded safely through `biomaster_ppi_string_download`; the incomplete alias file remains under `.part` until `gzip -t` passes and the file is atomically renamed.
- This is no longer a computation blocker: the expanded STRING/HuRI graph has already been used for the network-proximity audit and SOTA-network prioritization.

## Operational Commands

Check full DiffDock progress:

```bash
python scripts/summarize_diffdock_full_progress.py --run-dir outputs/report_scale/diffdock_full_run
```

Check GPU occupancy:

```bash
nvidia-smi
```

Check background sessions:

```bash
screen -ls
```

Refresh closure summaries:

```bash
python scripts/audit_sota_external_dependencies.py --root .
python scripts/build_sota_compute_closure_summary.py --root .
python scripts/build_sota_artifact_manifest.py --root .
```

## Completion Definition

The experiment should be considered computationally closed only after:

1. Full DiffDock reaches 3653 / 3653 jobs.
2. The `CHEMBL3039504` ligand rescue queue has either completed or been formally documented as a stable technical failure.
3. Full DiffDock merge and final TopK docking tables are regenerated.
4. Structure, pose, pocket-confidence, and interpretability audits are rerun on the final merged state.
5. Integrated final candidate ranking and expert-review panels are regenerated.
6. Closure summary and artifact manifest are updated after all final outputs exist.
7. The website/report reflects the final computation state.
