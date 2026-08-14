#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/BioMaster"
PYTHON="$ROOT/.conda_envs/boltz2/bin/python"
CALIB="$ROOT/outputs/boltz2_calibration_338_v1"
DISC="$ROOT/outputs/boltz2_discovery_conditional_v1"
STATE="$CALIB/BOLTZ2_CONDITIONAL_PIPELINE_STATE_V1.json"
LOG="$CALIB/automatic_pipeline.log"

cd "$ROOT"
mkdir -p "$CALIB" "$DISC" "$ROOT/.tmp/boltz2_calibration_338_v1" "$ROOT/.tmp/boltz2_discovery_conditional_v1"

write_state() {
  local stage="$1"
  local status="$2"
  local detail="${3:-}"
  STAGE="$stage" STATUS="$status" DETAIL="$detail" STATE="$STATE" "$PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATE"])
previous = {}
if path.exists():
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        previous = {}
history = previous.get("history", [])
entry = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "stage": os.environ["STAGE"],
    "status": os.environ["STATUS"],
    "detail": os.environ.get("DETAIL", ""),
}
history.append(entry)
path.write_text(
    json.dumps({**entry, "history": history}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

on_error() {
  local code=$?
  write_state "${CURRENT_STAGE:-unknown}" "FAILED" "exit_code=$code"
  exit "$code"
}
trap on_error ERR

run_boltz() {
  local manifest="$1"
  local input_dir="$2"
  local out_dir="$3"
  local batch_size="$4"
  local tmp_dir="$5"
  "$PYTHON" scripts/run_boltz2_batched_queue.py \
    --input-manifest "$manifest" \
    --input-dir "$input_dir" \
    --out-dir "$out_dir" \
    --top-n 0 \
    --batch-size "$batch_size" \
    --gpus 0,1,2,3 \
    --allow-kernels \
    --recycling-steps 1 \
    --sampling-steps 50 \
    --diffusion-samples 1 \
    --sampling-steps-affinity 50 \
    --diffusion-samples-affinity 1 \
    --num-workers 4 \
    --preprocessing-threads 8 \
    --tmp-dir "$tmp_dir" \
    --min-free-gb 40
}

recover_missing() {
  local manifest="$1"
  local input_dir="$2"
  local run_dir="$3"
  local missing_manifest="$4"
  local recovery_out="$5"
  local tmp_dir="$6"
  "$PYTHON" scripts/build_missing_boltz2_manifest_v1.py \
    --manifest "$manifest" \
    --run-dir "$run_dir" \
    --output "$missing_manifest" \
    --diffusion-samples 1
  local missing_rows
  missing_rows=$("$PYTHON" -c "import pandas as pd; print(len(pd.read_csv(r'$missing_manifest')))" )
  if [[ "$missing_rows" -gt 0 ]]; then
    run_boltz "$missing_manifest" "$input_dir" "$recovery_out" 1 "$tmp_dir"
  fi
}

CURRENT_STAGE="WAIT_FOR_GNINA"
write_state "$CURRENT_STAGE" "WAITING" "Boltz starts after the running GNINA discovery pipeline releases all GPUs"
while pgrep -f "scripts/auto_run_gnina_discovery_7511_v1.sh" >/dev/null \
   || pgrep -f "outputs/gnina_discovery_7511_v1/execution_inputs/GNINA_DISCOVERY_7511_JOBS_V1.csv" >/dev/null \
   || pgrep -f "tools/gnina/gnina.1.3.2" >/dev/null; do
  sleep 120
done
write_state "$CURRENT_STAGE" "COMPLETE" "GNINA processes no longer active"

CURRENT_STAGE="PREPARE_BOLTZ_CALIBRATION"
write_state "$CURRENT_STAGE" "RUNNING"
"$PYTHON" scripts/prepare_boltz2_calibration_338_v1.py
write_state "$CURRENT_STAGE" "COMPLETE" "16992 frozen control calculations prepared"

CURRENT_STAGE="RUN_BOLTZ_CALIBRATION"
write_state "$CURRENT_STAGE" "RUNNING" "4 GPUs; official Boltz 2.2.1 weights; 50/50 screening protocol"
run_boltz \
  outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_calibration_338_v1/inputs \
  outputs/boltz2_calibration_338_v1/formal_screen_run \
  100 \
  .tmp/boltz2_calibration_338_v1
# A second pass resumes only batches that did not produce a complete signed result set.
run_boltz \
  outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_calibration_338_v1/inputs \
  outputs/boltz2_calibration_338_v1/formal_screen_run \
  100 \
  .tmp/boltz2_calibration_338_v1
recover_missing \
  outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_calibration_338_v1/inputs \
  outputs/boltz2_calibration_338_v1/formal_screen_run \
  outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_MISSING_AFTER_BATCH_RETRY_V1.csv \
  outputs/boltz2_calibration_338_v1/formal_screen_run/recovery_single_pair \
  .tmp/boltz2_calibration_338_v1_recovery
write_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="EVALUATE_BOLTZ_CALIBRATION"
write_state "$CURRENT_STAGE" "RUNNING"
"$PYTHON" scripts/evaluate_boltz2_calibration_338_v1.py \
  --run-dir outputs/boltz2_calibration_338_v1/formal_screen_run
write_state "$CURRENT_STAGE" "COMPLETE" "Gate-A and Gate-B target qualifications written"

CURRENT_STAGE="PREPARE_CONDITIONAL_DISCOVERY"
write_state "$CURRENT_STAGE" "RUNNING"
"$PYTHON" scripts/prepare_boltz2_discovery_after_calibration_v1.py
write_state "$CURRENT_STAGE" "COMPLETE" "candidate count is data-dependent on Boltz target qualification"

CURRENT_STAGE="RUN_CONDITIONAL_DISCOVERY"
write_state "$CURRENT_STAGE" "RUNNING" "target-qualified local/remote lanes plus restricted information-only controls"
run_boltz \
  outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_discovery_conditional_v1/inputs \
  outputs/boltz2_discovery_conditional_v1/screen_run \
  100 \
  .tmp/boltz2_discovery_conditional_v1
run_boltz \
  outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_discovery_conditional_v1/inputs \
  outputs/boltz2_discovery_conditional_v1/screen_run \
  100 \
  .tmp/boltz2_discovery_conditional_v1
recover_missing \
  outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_discovery_conditional_v1/inputs \
  outputs/boltz2_discovery_conditional_v1/screen_run \
  outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_MISSING_AFTER_BATCH_RETRY_V1.csv \
  outputs/boltz2_discovery_conditional_v1/screen_run/recovery_single_pair \
  .tmp/boltz2_discovery_conditional_v1_recovery
write_state "$CURRENT_STAGE" "COMPLETE"

CURRENT_STAGE="EVALUATE_CONDITIONAL_DISCOVERY"
write_state "$CURRENT_STAGE" "RUNNING"
"$PYTHON" scripts/evaluate_boltz2_discovery_conditional_v1.py
write_state "$CURRENT_STAGE" "COMPLETE" "target-calibrated candidate evidence written"

CURRENT_STAGE="PIPELINE_COMPLETE"
write_state "$CURRENT_STAGE" "COMPLETE" "Boltz-2 calibration and conditional discovery finished"
printf 'Boltz-2 conditional pipeline complete: %s\n' "$(date -u +%FT%TZ)" >> "$LOG"
