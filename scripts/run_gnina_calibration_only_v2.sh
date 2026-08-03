#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/BioMaster"
BASE="$ROOT/outputs/affinity_first_remote_discovery_v1"
RUN_DIR="$BASE/production_run_v2"
mkdir -p "$RUN_DIR"
cd "$ROOT"
printf '%s\n' "$$" > "$RUN_DIR/gnina_pipeline.pid"

exec 9>"$RUN_DIR/gnina_pipeline.lock"
if ! flock -n 9; then
  echo "A GNINA calibration pipeline is already running." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
date -u +"%Y-%m-%dT%H:%M:%SZ gnina_calibration_restart_domain_matched" | tee -a "$RUN_DIR/pipeline_status.log"
python scripts/run_gnina_target_calibration_v2.py \
  --controls outputs/affinity_first_remote_discovery_v1/target_docking_calibration_v2/GNINA_TARGET_CALIBRATION_CONTROLS_V2.csv.gz \
  --gpus 0,1 \
  --exhaustiveness 8 \
  --modes 5 \
  --cpus-per-job 6 \
  --prepare-workers 8 \
  --bootstrap-repeats 1000 2>&1 | tee "$RUN_DIR/gnina_calibration.log"

date -u +"%Y-%m-%dT%H:%M:%SZ gnina_calibration_complete" | tee -a "$RUN_DIR/pipeline_status.log"
