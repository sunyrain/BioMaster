from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    stage5_path = resolve(root, args.stage5)
    diffdock_manifest_path = resolve(root, args.diffdock_manifest)
    out_manifest = resolve(root, args.out_manifest)
    skipped_out = resolve(root, args.skipped_out)

    diffdock_by_pair = {row["pair_id"]: row for row in read_csv(diffdock_manifest_path)}
    stage5_rows = read_csv(stage5_path)

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in stage5_rows:
        if len(selected) + len(skipped) >= args.top_n:
            break
        pair_id = row["pair_id"]
        if pair_id in seen:
            continue
        seen.add(pair_id)
        manifest_row = diffdock_by_pair.get(pair_id)
        if not manifest_row:
            skipped.append({**row, "skip_reason": "pair_missing_from_diffdock_manifest"})
            continue
        if manifest_row.get("diffdock_ready") not in {"true", "True", "1", "yes"}:
            skipped.append({**row, "skip_reason": "diffdock_not_ready"})
            continue

        merged = dict(manifest_row)
        for key, value in row.items():
            merged[f"stage5_{key}"] = value
        merged["top_n_source"] = f"stage5_top_{args.top_n}"
        selected.append(merged)

    if selected:
        write_csv(out_manifest, list(selected[0].keys()), selected)
    else:
        write_csv(out_manifest, ["pair_id"], [])
    if skipped:
        write_csv(skipped_out, list(skipped[0].keys()), skipped)
    else:
        write_csv(skipped_out, ["pair_id", "skip_reason"], [])

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage5": str(stage5_path),
        "diffdock_manifest": str(diffdock_manifest_path),
        "out_manifest": str(out_manifest),
        "skipped_out": str(skipped_out),
        "requested_top_n": args.top_n,
        "selected_diffdock_ready": len(selected),
        "skipped": len(skipped),
        "unique_drugs": len({row["drug_id"] for row in selected}),
        "unique_proteins": len({row["protein_id"] for row in selected}),
    }
    out_manifest.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a DiffDock-ready manifest from Stage 5 top-ranked pairs.")
    parser.add_argument("--stage5", default="outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--diffdock-manifest", default="outputs/report_scale/manifest_915k_diffdock_ready.csv")
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--out-manifest", default=None)
    parser.add_argument("--skipped-out", default=None)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.out_manifest is None:
        args.out_manifest = f"outputs/report_scale/stage5_top{args.top_n}_diffdock_ready_manifest.csv"
    if args.skipped_out is None:
        args.skipped_out = f"outputs/report_scale/stage5_top{args.top_n}_diffdock_skipped.csv"
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
