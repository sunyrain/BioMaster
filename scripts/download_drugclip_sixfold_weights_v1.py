#!/usr/bin/env python3
"""Extract only the official DrugCLIP six-fold checkpoints from the remote ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/bgao95/DrugCLIP_data/resolve/main/model_weights.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "third_party" / "sota_dti_2026" / "Drug-The-Whole-Genome" / "data" / "model_weights" / "6_folds"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with RemoteZip(URL) as archive:
        names = {entry.filename: entry for entry in archive.infolist()}
        for fold in range(6):
            member = f"model_weights/6_folds/fold_{fold}.pt"
            info = names[member]
            output = output_dir / f"fold_{fold}.pt"
            status = "reused"
            if not output.exists() or output.stat().st_size != info.file_size:
                temporary = output.with_suffix(".pt.part")
                temporary.unlink(missing_ok=True)
                with archive.open(member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"Incomplete extraction for {member}")
                temporary.replace(output)
                status = "downloaded"
            records.append(
                {
                    "fold": fold,
                    "path": str(output.relative_to(ROOT)),
                    "bytes": output.stat().st_size,
                    "sha256": sha256(output),
                    "status": status,
                }
            )
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": URL,
        "license": "CC BY-NC 4.0 for model weights and generated outputs",
        "folds": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    summary_path = output_dir / "DRUGCLIP_SIXFOLD_WEIGHTS_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
