#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/recovered_dta_720x46_v1"
INPUT="$OUT/inputs"
DRUGCLIP_REPO="$ROOT/third_party/sota_dti_2026/Drug-The-Whole-Genome"
UNICORE="$ROOT/third_party/sota_dti_2026/Uni-Core"
WEIGHTS="$DRUGCLIP_REPO/data/model_weights/6_folds"
STRICT_MOL_REPS="$ROOT/outputs/strict_dta_720x338_v1/drugclip/mol_embeddings/mol_reps0None.npy"

for fold in 0 1 2 3 4 5; do
  test -s "$WEIGHTS/fold_${fold}.pt"
done
test -s "$INPUT/CONPLEX_720_X_46_INPUT.tsv"
test -s "$INPUT/strict720_ligands.lmdb"
test -s "$INPUT/recovered44_predicted_pockets.lmdb"
test -s "$STRICT_MOL_REPS"

mkdir -p "$OUT/conplex_cache" "$OUT/drugclip/mol_embeddings"
ln -sfn "$INPUT/recovered44_predicted_pockets.lmdb" "$INPUT/pocket.lmdb"
ln -sfn "$STRICT_MOL_REPS" "$OUT/drugclip/mol_embeddings/mol_reps0None.npy"

cd "$ROOT"
export PYTHONPATH="$ROOT/third_party/ConPLex:$ROOT/.local_deps/conplex:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 "$ROOT/.venvs/conplex/bin/python" -m conplex_dti predict \
  --data-file "$INPUT/CONPLEX_720_X_46_INPUT.tsv" \
  --model-path "$ROOT/third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt" \
  --outfile "$OUT/CONPLEX_720_X_46_PREDICTIONS.tsv" \
  --data-cache-dir "$OUT/conplex_cache" \
  --device 0 \
  --batch-size 2048 \
  >"$OUT/conplex.log" 2>&1 &
CONPLEX_PID=$!

export PYTHONPATH="$UNICORE:$DRUGCLIP_REPO:${PYTHONPATH:-}"
cd "$DRUGCLIP_REPO"
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
  --pocket-dir "$INPUT" \
  >"$OUT/drugclip/encode_pockets.log" 2>&1 &
POCKET_PID=$!

status=0
wait "$CONPLEX_PID" || status=1
wait "$POCKET_PID" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "Recovered DTA inference failed; inspect $OUT/conplex.log and $OUT/drugclip/encode_pockets.log" >&2
  exit 1
fi

cd "$ROOT"
python scripts/merge_recovered_dta_720x46_v1.py
