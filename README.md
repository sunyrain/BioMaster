# BioMaster

BioMaster is a reproducible, physics-first FDA small-molecule target-repurposing workflow. The current production line uses ChEMBL mechanism-of-action anchors, Open Targets tractability annotations, ConPLEx sequence/SMILES prioritisation, AlphaFold/P2Rank/PUResNet pocket priors, and refined Boltz-2 protein-ligand follow-up.

## Current Production Line

The formal local funnel is:

```text
915 FDA structure entries × 891 unique ChEMBL-MoA target sequences = 815,265 raw pairs
└── 750 direct-action therapeutic drugs × 463 non-GPCR target-engagement sequences
    = 347,250 ID-level audit pairs
    = 334,749 unique model-ligand x target physical pairs
    ├── 491 ChEMBL/FDA known controls (calibration only; excluded from discovery)
    └── physics-first diverse shortlist: 3,000
        ├── ConPLEx full-space score and bidirectional rank
        ├── AlphaFold/P2Rank/PUResNet target pocket prior
        └── refined Boltz-2 pair follow-up
            ├── formal candidate package: 1,000 pair hypotheses
            └── stronger wet-lab nomination queue: 384 pair hypotheses
```

The 384-row output is a nomination queue, not an executable four-plate map. Multiple target-specific assays are represented, so dose, replicates, positive/negative controls, and counterscreens must be defined before physical plate layout.

The previous 106,561-row table was derived after a per-drug Top300 truncation. It is retained for provenance but is not the formal universe and is not used to claim recall. The v4 production path retains 347,250 ID-level rows for audit, ranks 334,749 structure-collapsed physical pairs, and applies no Top300 hard gate.

Build the scope audit with:

```bash
python scripts/build_universe_scope_audit_v4.py
```

See `docs/PRODUCTION_PIPELINE_V4_ZH.md` for the frozen scientific and output contracts.

Formal selection uses versioned 750-drug and 463-target entity manifests, one shared full-space ConPLEx calibration scale, Open Targets target classification/tractability, active-moiety × target deduplication, sequence-homology extension risk, RDKit assay-liability flags, continuous Boltz outputs, and two-sample conditional pose stability. Open Targets disease evidence, TxGNN, STRING, and expression signatures are auxiliary interpretation fields and do not determine the physics-first rank.

Large datasets, model weights, third-party repositories, and generated structure outputs are intentionally excluded from git.

## Historical Demo Scope

The small package demo and early scale scripts below reflect the original general workflow. They are retained for regression testing and examples; they are not the current formal candidate-selection entrypoint.

1. Build and normalize an FDA-approved small-molecule library.
2. Build a 1000-protein human target library with AlphaFold receptor paths.
3. Score all 915,000 drug-target pairs with ConPLex.
4. Build structure-ready and DiffDock-ready manifests.
5. Rank all pairs with Open Targets, STRING v12.0 API, and TxGNN disease evidence.
6. Structurally enhance the Stage 5 Top1000 candidates with DiffDock.

The repository contains the pipeline code and small examples. The full local data products are not committed because they include multi-GB datasets, model weights, third-party code, and generated docking outputs.

## Repository Layout

```text
biomaster/      Core Python package and demo CLI.
scripts/        Scale-run utilities for ChEMBL/PubChem, AlphaFold, Open Targets, STRING, TxGNN, and DiffDock.
examples/       Small CSV examples for the demo pipeline.
tests/          Unit tests for the core pipeline.
docs/           Release notes, data-access notes, and current result summaries.
```

## Install

```bash
python -m pip install -e .
python -m pip install pytest
```

Some scale scripts require additional packages or external tools depending on the step:

- `requests` and `openpyxl` for API and workbook processing.
- `pyyaml` for DiffDock job configuration.
- RDKit for molecule handling in enrichment and docking preparation.
- Local clones or installs of ConPLex, DiffDock, and TxGNN for model execution.

## Quick Demo

The demo uses only small bundled examples:

```bash
python -m biomaster.cli run-demo --out outputs/demo --offline
```

Expected demo outputs:

```text
outputs/demo/drug_library.csv
outputs/demo/protein_library.csv
outputs/demo/diffdock_manifest.csv
outputs/demo/stage4_affinity_candidates.csv
outputs/demo/stage5_disease_ranked_candidates.csv
```

## Historical Scale Workflow

The production-scale run in this workspace used the scripts below:

```bash
python scripts/enrich_drug_structures_pubchem_chembl.py
python scripts/build_alphafold_receptor_manifest.py
python scripts/download_opentargets_filtered.py
python scripts/download_string_filtered.py
python scripts/rerank_stage5_open_targets_string.py
python scripts/run_txgnn_cancer_inference.py
python scripts/merge_stage5_with_txgnn.py
python scripts/build_diffdock_ready_manifest.py
python scripts/build_stage5_top_diffdock_manifest.py --top-n 1000
python scripts/prepare_diffdock_full_run.py
python scripts/run_diffdock_full_queue.py
python scripts/merge_stage6_top_diffdock.py
python scripts/export_stage6_report_artifacts.py
python scripts/write_biomaster_paper_zh_cn.py
```

The exact scale-run command arguments depend on local data paths and model installs. See `docs/DATA_ACCESS.md` and `docs/RESULTS_STATUS_ZH.md` for the current local run status and data provenance.

## What Is Not Tracked

The following are intentionally ignored:

- FDA workbook and other raw source datasets.
- AlphaFold human proteome tarball.
- ChEMBL/PubChem-derived SDF library.
- Open Targets, STRING, and TxGNN processed evidence tables.
- ConPLex, DiffDock, TxGNN, DrugBAN, and DeepDTA third-party repositories or weights.
- Full-scale manifests, score tables, logs, structure predictions, and model outputs.

This keeps the GitHub repository lightweight and avoids publishing data or weights that should remain local or be obtained from their original sources.

## Current Result Summary

Current audited counts are generated from CSV/JSON artifacts rather than copied into reports:

- 915 FDA structure entries and 892 ChEMBL-MoA genes represented by 891 unique sequences.
- 815,265 full ConPLEx predictions; numerical regression against the prior subset is effectively exact.
- 750 drug records (723 unique model-ligand structures) × 463 targets = 347,250 ID-level / 334,749 physical non-GPCR target-engagement pairs; `BCL2L10` was restored after independent P2Rank/PUResNet pocket consensus.
- 491 in-scope known ID-pair controls, or 473 unique active-moiety × target controls. With average ranks for tied ConPLEx scores, ID-pair Recall@100 is 59.27% and Recall@300 is 83.10%; active-moiety-collapsed values are 59.20% and 83.30%. This is calibration, not temporal generalization.
- The legacy Top300 plus absolute-score gate has been removed from v4. Structure and direct-small-molecule tractability retain 427/491 known controls (86.97%) before discovery exclusion.

## Tests

```bash
pytest -q
```

Current local test status: `39 passed`.

## Release Boundary

This private repository is intended for collaborative code review and reproducibility planning. It should not be treated as a public data release. Recreate large results locally from source databases and model downloads.
