# BioMaster

BioMaster is a reproducible drug-repositioning screening workflow for FDA-approved small molecules, human protein targets, disease evidence integration, and structural docking follow-up.

The current working project has completed the full Step 1-5 screening layer locally and uses the GitHub repository as a code and documentation release. Large datasets, model weights, third-party repositories, and generated docking outputs are intentionally excluded from git.

## Current Scientific Scope

The local run focuses on cancer-related repositioning hypotheses:

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

## Scale Workflow

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
- Full 915k manifests, score tables, logs, docking outputs, and model outputs.

This keeps the GitHub repository lightweight and avoids publishing data or weights that should remain local or be obtained from their original sources.

## Current Result Summary

The local BioMaster run has completed Step 1-5 main screening and Top1000 structural enhancement:

- 915 FDA-approved small molecules.
- 1000 human proteins.
- 915,000 ConPLex-scored drug-target pairs.
- 915,000 Stage 5 disease-ranked pairs.
- 913,170 DiffDock-ready pairs.
- Top1000 Stage 6 consensus candidates, with 940/1000 DiffDock outputs completed.

Full DiffDock is a long-running structural enhancement task and is not required for Step 1-5 completion.

## Tests

```bash
pytest -q
```

Current local test status before release: `8 passed`.

## Release Boundary

This private repository is intended for collaborative code review and reproducibility planning. It should not be treated as a public data release. Recreate large results locally from source databases and model downloads.
