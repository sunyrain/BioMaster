#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/BioMaster"
BASE="$ROOT/outputs/affinity_first_remote_discovery_v1"
RUN_DIR="$BASE/production_run_v2"
mkdir -p "$RUN_DIR"
cd "$ROOT"
printf '%s\n' "$$" > "$RUN_DIR/pipeline.pid"

exec 9>"$RUN_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "A structure/GNINA calibration pipeline is already running." >&2
  exit 2
fi

available_kb=$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
if (( available_kb < 20 * 1024 * 1024 )); then
  echo "At least 20 GiB free space is required; available KiB: $available_kb" >&2
  exit 3
fi

export PYTHONUNBUFFERED=1
date -u +"%Y-%m-%dT%H:%M:%SZ structure_validation_start" | tee "$RUN_DIR/pipeline_status.log"
python scripts/validate_experimental_holo_structures_v2.py \
  --max-entries-per-target 5 \
  --workers 12 2>&1 | tee "$RUN_DIR/structure_validation.log"

date -u +"%Y-%m-%dT%H:%M:%SZ gnina_calibration_start" | tee -a "$RUN_DIR/pipeline_status.log"
python scripts/build_target_docking_calibration_manifest_v1.py \
  --out-dir outputs/affinity_first_remote_discovery_v1/target_docking_calibration_v2 \
  --output-version V2 \
  --docking-domain-filter \
  --per-class 12 2>&1 | tee "$RUN_DIR/control_manifest_v2.log"

python scripts/run_gnina_target_calibration_v2.py \
  --gpus 0,1 \
  --exhaustiveness 8 \
  --modes 5 \
  --cpus-per-job 6 \
  --prepare-workers 8 \
  --bootstrap-repeats 1000 2>&1 | tee "$RUN_DIR/gnina_calibration.log"

date -u +"%Y-%m-%dT%H:%M:%SZ pipeline_complete" | tee -a "$RUN_DIR/pipeline_status.log"
