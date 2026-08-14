#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/gnina_discovery_7511_v1"
STATE="$OUT/GNINA_DISCOVERY_7511_AUTOMATIC_STATE_V1.json"
JOBS="$OUT/execution_inputs/GNINA_DISCOVERY_7511_JOBS_V1.csv"

write_state() {
  local stage="$1"
  local status="$2"
  local message="$3"
  python - "$STATE" "$stage" "$status" "$message" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "stage": sys.argv[2], "status": sys.argv[3], "message": sys.argv[4],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  local exit_code=$?
  write_state "FAILED" "BLOCKED" "Discovery pipeline stopped with exit code ${exit_code}; inspect automatic.log."
  exit "$exit_code"
}
trap on_error ERR

write_state "PREPARATION" "RUNNING" "Waiting for the 7,511-pair conformer and job preparation."
while pgrep -f '[p]repare_gnina_discovery_7511_v1.py' >/dev/null; do sleep 30; done

python - <<'PY'
import json
path = "outputs/gnina_discovery_7511_v1/execution_inputs/GNINA_DISCOVERY_7511_PREPARATION_SUMMARY_V1.json"
s = json.load(open(path))
assert s["requested_pairs"] == 7511, s
assert s["ready_targets"] == 69 and s["blocked_targets"] == 0, s
assert s["receptor_hash_mismatches"] == 0, s
assert s["failed_pairs"] == 0, s
PY

write_state "DISCOVERY_DOCKING" "RUNNING" "Running GNINA for 7,511 unreported FDA-target pairs on four GPUs."
python scripts/run_gnina_calibration_338_v1.py \
  --jobs "$JOBS" \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 2 \
  --cpus-per-job 8 \
  --run-label discovery7511

run_summary="$OUT/execution_inputs/GNINA_CALIBRATION_RUN_SUMMARY_DISCOVERY7511_V1.json"
if ! python -c 'import json,sys; s=json.load(open(sys.argv[1])); raise SystemExit(0 if s["failed_targets"] == 0 else 1)' "$run_summary"; then
  write_state "DISCOVERY_RETRY" "RUNNING" "Retrying incomplete target jobs with resume semantics."
  python scripts/run_gnina_calibration_338_v1.py \
    --jobs "$JOBS" \
    --gpus 0,1,2,3 \
    --jobs-per-gpu 2 \
    --cpus-per-job 8 \
    --run-label discovery7511_retry1
fi

write_state "TARGET_CALIBRATED_EVALUATION" "RUNNING" "Comparing candidate scores only against the same target's development controls."
python scripts/evaluate_gnina_discovery_7511_v1.py

write_state "DISCOVERY_EVIDENCE_READY" "COMPLETED" "All 7,511 pairs have target-calibrated GNINA evidence or an explicit out-of-domain label."
trap - ERR
echo "GNINA discovery 7,511 automatic pipeline completed."
