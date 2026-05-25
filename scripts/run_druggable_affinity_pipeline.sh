#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-outputs/druggable_proteome}"
DEVICE="${DEVICE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TOP_N="${TOP_N:-10000}"
PREDICTIONS="${OUT_DIR}/conplex_predictions_druggable_unique_sequences.tsv"

export PYTHONPATH="third_party/ConPLex:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}/conplex_cache"

python -m conplex_dti predict \
  --data-file "${OUT_DIR}/conplex_pairs_druggable_unique_sequences.tsv" \
  --model-path third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt \
  --outfile "${PREDICTIONS}" \
  --data-cache-dir "${OUT_DIR}/conplex_cache" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}"

python scripts/expand_druggable_conplex_predictions.py \
  --predictions "${PREDICTIONS}" \
  --proteins "${OUT_DIR}/protein_library_druggable_chembl.csv" \
  --drugs data/processed/drug_library_pubchem_chembl_mapped.csv \
  --out-affinity "${OUT_DIR}/conplex_affinity_scores_druggable.csv" \
  --out-top "${OUT_DIR}/stage4_affinity_candidates_druggable_top${TOP_N}.csv" \
  --out-diffdock-seed "${OUT_DIR}/top${TOP_N}_druggable_diffdock_seed_manifest.csv" \
  --metadata "${OUT_DIR}/druggable_conplex_expansion.metadata.json" \
  --top-n "${TOP_N}"
