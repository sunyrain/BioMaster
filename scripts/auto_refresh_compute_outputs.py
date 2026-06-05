from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def run_command(root: Path, command: list[str]) -> dict[str, Any]:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "ok": False,
            "returncode": 124,
            "elapsedSec": round(time.time() - started, 3),
            "stdout": (exc.stdout or "")[-2000:],
            "stderr": (exc.stderr or "")[-2000:],
            "error": "timeout",
        }
    return {
        "command": command,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsedSec": round(time.time() - started, 3),
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
        "error": "" if result.returncode == 0 else "nonzero_exit",
    }


def count_vina_audits(root: Path) -> int:
    paths = list(
        root.glob("outputs/sota_validation/vina_consensus_rescoring_final3921/vina_consensus_candidate_audit.csv")
    )
    paths.extend(
        root.glob("outputs/sota_validation/vina_consensus_rescoring_final3921_shards/shard_*/vina_consensus_candidate_audit.csv")
    )
    return sum(1 for path in paths if path.exists() and path.stat().st_size > 0)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def all_done(root: Path) -> bool:
    diffdock = read_json(root / "outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json")
    vina = read_json(root / "outputs/sota_validation/vina_consensus_rescoring_final3921_merged/vina_consensus_summary.json")
    diffdock_done = bool(diffdock) and int(diffdock.get("completedJobs") or 0) >= int(diffdock.get("totalJobs") or 1)
    vina_done = bool(vina) and int(vina.get("missingFinalRows") or 0) == 0
    return diffdock_done and vina_done


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodically refresh active BioMaster compute summaries.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--manifest-every", type=int, default=3, help="Run artifact manifest every N loops.")
    parser.add_argument("--max-loops", type=int, default=0, help="0 means run until DiffDock and Vina merged outputs complete.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    status_path = root / "logs/auto_refresh_compute_outputs.status.json"
    loop_index = 0
    while True:
        loop_index += 1
        steps: list[dict[str, Any]] = []
        steps.append(run_command(root, ["python", "scripts/summarize_diffdock_full_progress.py"]))

        audit_count = count_vina_audits(root)
        if audit_count:
            steps.append(run_command(root, ["python", "scripts/merge_vina_consensus_shards.py", "--root", "."]))

        steps.append(run_command(root, ["python", "scripts/build_sota_compute_closure_summary.py", "--root", "."]))
        if args.manifest_every > 0 and loop_index % args.manifest_every == 0:
            steps.append(run_command(root, ["python", "scripts/build_sota_artifact_manifest.py", "--root", "."]))

        payload = {
            "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loop": loop_index,
            "intervalSec": args.interval_sec,
            "vinaAuditCount": audit_count,
            "steps": steps,
            "allDone": all_done(root),
        }
        write_status(status_path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

        if payload["allDone"]:
            break
        if args.max_loops and loop_index >= args.max_loops:
            break
        time.sleep(max(30, args.interval_sec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
