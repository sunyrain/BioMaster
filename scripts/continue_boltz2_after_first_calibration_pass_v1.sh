#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/BioMaster"
PYTHON="$ROOT/.conda_envs/boltz2/bin/python"
CALIB="$ROOT/outputs/boltz2_calibration_338_v1"
REPAIR="$ROOT/outputs/boltz2_calibration_chain_repair_v1"
DISC="$ROOT/outputs/boltz2_discovery_conditional_v1"
STATE="$CALIB/BOLTZ2_CONDITIONAL_PIPELINE_STATE_V1.json"
FIRST_PASS_PID="${FIRST_PASS_PID:-61300}"
OLD_DRIVER_PID="${OLD_DRIVER_PID:-50711}"

cd "$ROOT"

write_state() {
  local stage="$1" status="$2" detail="${3:-}"
  STAGE="$stage" STATUS="$status" DETAIL="$detail" STATE="$STATE" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
p=Path(os.environ['STATE']); old={}
if p.exists():
    try: old=json.loads(p.read_text(encoding='utf-8'))
    except Exception: old={}
h=old.get('history',[])
e={'utc':datetime.now(timezone.utc).isoformat(),'stage':os.environ['STAGE'],'status':os.environ['STATUS'],'detail':os.environ.get('DETAIL','')}
h.append(e); p.write_text(json.dumps({**e,'history':h},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
}

run_boltz() {
  "$PYTHON" scripts/run_boltz2_batched_queue.py \
    --input-manifest "$1" --input-dir "$2" --out-dir "$3" --top-n 0 \
    --batch-size "$4" --gpus 0,1,2,3 --allow-kernels \
    --recycling-steps 1 --sampling-steps 50 --diffusion-samples 1 \
    --sampling-steps-affinity 50 --diffusion-samples-affinity 1 \
    --num-workers 4 --preprocessing-threads 8 --tmp-dir "$5" --min-free-gb 40
}

write_state "WAIT_FIRST_CALIBRATION_PASS" "WAITING" "handoff after the original 16992-row first pass"
while [[ -r "/proc/$FIRST_PASS_PID/stat" ]]; do
  process_state=$(awk '{print $3}' "/proc/$FIRST_PASS_PID/stat" 2>/dev/null || true)
  [[ "$process_state" == "Z" ]] && break
  sleep 60
done
kill -KILL "$OLD_DRIVER_PID" 2>/dev/null || true
sleep 2

write_state "REPAIR_CALIBRATION_CHAIN_MAPPING" "RUNNING" "regenerate mmCIF label_asym_id mappings and backfill affected targets"
"$PYTHON" scripts/prepare_boltz2_calibration_338_v1.py --output-dir "$REPAIR"
"$PYTHON" scripts/build_boltz2_chain_repair_manifest_v1.py \
  --manifest "$REPAIR/BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv" \
  --target-audit "$REPAIR/BOLTZ2_CALIBRATION_TARGET_INPUT_AUDIT_V1.csv" \
  --output "$REPAIR/BOLTZ2_CHAIN_MAPPING_REPAIR_MANIFEST_V1.csv"
run_boltz \
  outputs/boltz2_calibration_chain_repair_v1/BOLTZ2_CHAIN_MAPPING_REPAIR_MANIFEST_V1.csv \
  outputs/boltz2_calibration_chain_repair_v1/inputs \
  outputs/boltz2_calibration_338_v1/formal_screen_run/chain_mapping_repair_v1 \
  64 .tmp/boltz2_calibration_chain_repair_v1
"$PYTHON" scripts/build_missing_boltz2_manifest_v1.py \
  --manifest outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv \
  --run-dir outputs/boltz2_calibration_338_v1/formal_screen_run \
  --output outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_TECHNICAL_INCOMPLETE_FINAL_V1.csv
write_state "REPAIR_CALIBRATION_CHAIN_MAPPING" "COMPLETE"

write_state "EVALUATE_BOLTZ_CALIBRATION" "RUNNING"
"$PYTHON" scripts/evaluate_boltz2_calibration_338_v1.py \
  --run-dir outputs/boltz2_calibration_338_v1/formal_screen_run
write_state "EVALUATE_BOLTZ_CALIBRATION" "COMPLETE"

write_state "PREPARE_CONDITIONAL_DISCOVERY" "RUNNING"
"$PYTHON" scripts/prepare_boltz2_discovery_after_calibration_v1.py
write_state "PREPARE_CONDITIONAL_DISCOVERY" "COMPLETE"

write_state "RUN_CONDITIONAL_DISCOVERY" "RUNNING"
run_boltz \
  outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_INPUT_MANIFEST_V1.csv \
  outputs/boltz2_discovery_conditional_v1/inputs \
  outputs/boltz2_discovery_conditional_v1/screen_run \
  100 .tmp/boltz2_discovery_conditional_v1
"$PYTHON" scripts/build_missing_boltz2_manifest_v1.py \
  --manifest outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_INPUT_MANIFEST_V1.csv \
  --run-dir outputs/boltz2_discovery_conditional_v1/screen_run \
  --output outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_RETRY_MANIFEST_V1.csv
retry_rows=$("$PYTHON" -c "import pandas as pd; print(len(pd.read_csv('outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_RETRY_MANIFEST_V1.csv')))" )
if [[ "$retry_rows" -gt 0 ]]; then
  run_boltz \
    outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_RETRY_MANIFEST_V1.csv \
    outputs/boltz2_discovery_conditional_v1/inputs \
    outputs/boltz2_discovery_conditional_v1/screen_run/retry_v1 \
    64 .tmp/boltz2_discovery_conditional_retry_v1
fi
"$PYTHON" scripts/build_missing_boltz2_manifest_v1.py \
  --manifest outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_INPUT_MANIFEST_V1.csv \
  --run-dir outputs/boltz2_discovery_conditional_v1/screen_run \
  --output outputs/boltz2_discovery_conditional_v1/BOLTZ2_CONDITIONAL_DISCOVERY_TECHNICAL_INCOMPLETE_V1.csv
write_state "RUN_CONDITIONAL_DISCOVERY" "COMPLETE"

write_state "EVALUATE_CONDITIONAL_DISCOVERY" "RUNNING"
"$PYTHON" scripts/evaluate_boltz2_discovery_conditional_v1.py
write_state "PIPELINE_COMPLETE" "COMPLETE" "calibration, conditional discovery, and target-internal evidence tiers complete"
