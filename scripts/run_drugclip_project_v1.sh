#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/third_party/sota_dti_2026/Drug-The-Whole-Genome"
UNICORE="$ROOT/third_party/sota_dti_2026/Uni-Core"
INPUT_DIR="$ROOT/outputs/affinity_first_remote_discovery_v1/drugclip_inputs_v1"
OUTPUT_DIR="$ROOT/outputs/affinity_first_remote_discovery_v1/drugclip_inference_v1"
WEIGHT_DIR="$REPO/data/model_weights/6_folds"

for fold in 0 1 2 3 4 5; do
  test -s "$WEIGHT_DIR/fold_${fold}.pt"
done
test -s "$INPUT_DIR/project723_ligands.lmdb"
test -s "$INPUT_DIR/project308_strict_pockets.lmdb"

mkdir -p "$OUTPUT_DIR/mol_embeddings"
ln -sfn "$INPUT_DIR/project308_strict_pockets.lmdb" "$INPUT_DIR/pocket.lmdb"
export PYTHONPATH="$UNICORE:$REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
CUDA_VISIBLE_DEVICES=0 python ./unimol/encode_mols.py \
  --user-dir ./unimol \
  ./dict \
  --valid-subset test \
  --num-workers 0 \
  --ddp-backend c10d \
  --distributed-world-size 1 \
  --batch-size 64 \
  --task drugclip \
  --loss in_batch_softmax \
  --arch drugclip \
  --max-pocket-atoms 256 \
  --fp16 \
  --seed 1 \
  --mol-path "$INPUT_DIR/project723_ligands.lmdb" \
  --save-dir "$OUTPUT_DIR/mol_embeddings" \
  --write-npy \
  >"$OUTPUT_DIR/encode_molecules.log" 2>&1 &
MOL_PID=$!

CUDA_VISIBLE_DEVICES=1 python ./unimol/encode_pockets.py \
  --user-dir ./unimol \
  ./dict \
  --valid-subset test \
  --num-workers 0 \
  --ddp-backend c10d \
  --distributed-world-size 1 \
  --batch-size 4 \
  --task drugclip \
  --loss in_batch_softmax \
  --arch drugclip \
  --max-pocket-atoms 256 \
  --fp16 \
  --seed 1 \
  --pocket-dir "$INPUT_DIR" \
  >"$OUTPUT_DIR/encode_pockets.log" 2>&1 &
POCKET_PID=$!

status=0
wait "$MOL_PID" || status=1
wait "$POCKET_PID" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "DrugCLIP encoding failed; inspect $OUTPUT_DIR/*.log" >&2
  exit 1
fi

cd "$ROOT"
python scripts/merge_drugclip_project_scores_v1.py
