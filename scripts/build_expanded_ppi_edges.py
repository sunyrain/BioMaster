from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_COLUMNS = [
    "stringId_A",
    "stringId_B",
    "preferredName_A",
    "preferredName_B",
    "score",
    "nscore",
    "fscore",
    "pscore",
    "ascore",
    "escore",
    "dscore",
    "tscore",
    "source_query",
    "filter_rule",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def read_string_info(path: Path) -> dict[str, str]:
    info = pd.read_csv(path, sep="\t", compression="infer", dtype=str).fillna("")
    required = {"#string_protein_id", "preferred_name"}
    missing = required.difference(info.columns)
    if missing:
        raise ValueError(f"STRING info missing columns: {sorted(missing)}")
    mapping = {}
    for _, row in info.iterrows():
        string_id = str(row["#string_protein_id"]).strip()
        symbol = norm_symbol(row["preferred_name"])
        if string_id and symbol:
            mapping[string_id] = symbol
    return mapping


def read_gtex_ensg_to_symbol(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", compression="infer", skiprows=2, dtype=str, usecols=["Name", "Description"])
    frame = frame.fillna("")
    mapping = {}
    for _, row in frame.iterrows():
        ensg = str(row["Name"]).split(".", 1)[0].strip()
        symbol = norm_symbol(row["Description"])
        if ensg and symbol:
            mapping[ensg] = symbol
    return mapping


def string_edges(path: Path, id_to_symbol: dict[str, str], min_score: int) -> pd.DataFrame:
    edges = pd.read_csv(path, sep=r"\s+", compression="infer", dtype={"protein1": str, "protein2": str, "combined_score": int})
    required = {"protein1", "protein2", "combined_score"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"STRING physical links missing columns: {sorted(missing)}")
    edges = edges[pd.to_numeric(edges["combined_score"], errors="coerce").fillna(0).astype(int) >= min_score].copy()
    edges["preferredName_A"] = edges["protein1"].map(id_to_symbol).map(norm_symbol)
    edges["preferredName_B"] = edges["protein2"].map(id_to_symbol).map(norm_symbol)
    edges = edges[(edges["preferredName_A"] != "") & (edges["preferredName_B"] != "")]
    edges = edges[edges["preferredName_A"] != edges["preferredName_B"]].copy()
    if edges.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.DataFrame(
        {
            "stringId_A": edges["protein1"],
            "stringId_B": edges["protein2"],
            "preferredName_A": edges["preferredName_A"],
            "preferredName_B": edges["preferredName_B"],
            "score": pd.to_numeric(edges["combined_score"], errors="coerce").fillna(0).astype(float) / 1000.0,
            "nscore": "",
            "fscore": "",
            "pscore": "",
            "ascore": "",
            "escore": "",
            "dscore": "",
            "tscore": "",
            "source_query": "string_physical_links_v12",
            "filter_rule": f"string_physical_combined_score_gte_{min_score}",
        }
    )
    return out[OUTPUT_COLUMNS]


def huri_edges(path: Path, ensg_to_symbol: dict[str, str], score: float) -> pd.DataFrame:
    huri = pd.read_csv(path, sep="\t", header=None, names=["ensg_a", "ensg_b"], dtype=str).fillna("")
    huri["preferredName_A"] = huri["ensg_a"].map(ensg_to_symbol).map(norm_symbol)
    huri["preferredName_B"] = huri["ensg_b"].map(ensg_to_symbol).map(norm_symbol)
    huri = huri[(huri["preferredName_A"] != "") & (huri["preferredName_B"] != "")]
    huri = huri[huri["preferredName_A"] != huri["preferredName_B"]].copy()
    if huri.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.DataFrame(
        {
            "stringId_A": huri["ensg_a"],
            "stringId_B": huri["ensg_b"],
            "preferredName_A": huri["preferredName_A"],
            "preferredName_B": huri["preferredName_B"],
            "score": float(score),
            "nscore": "",
            "fscore": "",
            "pscore": "",
            "ascore": "",
            "escore": "",
            "dscore": "",
            "tscore": "",
            "source_query": "huri_binary_interactome",
            "filter_rule": f"huri_binary_score_{score:g}",
        }
    )
    return out[OUTPUT_COLUMNS]


def merge_edges(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined["_pair"] = [
        canonical_pair(norm_symbol(a), norm_symbol(b))
        for a, b in zip(combined["preferredName_A"], combined["preferredName_B"], strict=False)
    ]
    combined["_score_numeric"] = pd.to_numeric(combined["score"], errors="coerce").fillna(0.0)
    combined = combined.sort_values(["_pair", "_score_numeric"], ascending=[True, False])
    combined = combined.drop_duplicates("_pair", keep="first").copy()
    combined[["preferredName_A", "preferredName_B"]] = pd.DataFrame(combined["_pair"].tolist(), index=combined.index)
    return combined.drop(columns=["_pair", "_score_numeric"])[OUTPUT_COLUMNS].sort_values(
        ["preferredName_A", "preferredName_B"]
    )


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Expanded PPI Edge Table",
            "",
            f"Generated: {summary['createdUtc']}",
            "",
            "## Scope",
            "",
            f"- Output rows: {summary['outputRows']}",
            f"- Unique nodes: {summary['uniqueNodes']}",
            f"- STRING rows retained: {summary['stringRowsRetained']}",
            f"- HuRI rows retained: {summary['huriRowsRetained']}",
            f"- Source counts: {summary['sourceCounts']}",
            "",
            "## Inputs",
            "",
            f"- STRING protein info: `{summary['inputs']['stringInfo']}`",
            f"- STRING physical links: `{summary['inputs']['stringPhysicalLinks']}`",
            f"- HuRI: `{summary['inputs']['huri']}`",
            f"- GTEx gene median TPM: `{summary['inputs']['gtex']}`",
            "",
            "## Interpretation",
            "",
            "- STRING physical links are filtered by combined score before graph construction.",
            "- HuRI ENSG IDs are mapped to gene symbols through the GTEx gene table.",
            "- Duplicate gene-symbol pairs are collapsed, keeping the higher source score.",
            "- This file is intended as an expanded network-medicine input, not as direct drug-target evidence.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build expanded STRING + HuRI PPI edges for network medicine audits.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--string-info", default="data/external/string/9606.protein.info.v12.0.txt.gz")
    parser.add_argument("--string-physical", default="data/external/string/9606.protein.physical.links.v12.0.txt.gz")
    parser.add_argument("--huri", default="data/external/huri/HuRI.tsv")
    parser.add_argument("--gtex", default="data/external/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz")
    parser.add_argument("--min-string-score", type=int, default=700)
    parser.add_argument("--huri-score", type=float, default=0.7)
    parser.add_argument("--out", default="data/processed/huri_string_expanded_edges.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    string_info_path = root / args.string_info
    string_physical_path = root / args.string_physical
    huri_path = root / args.huri
    gtex_path = root / args.gtex
    out_path = root / args.out
    for path in [string_info_path, string_physical_path, huri_path, gtex_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    id_to_symbol = read_string_info(string_info_path)
    ensg_to_symbol = read_gtex_ensg_to_symbol(gtex_path)
    string_frame = string_edges(string_physical_path, id_to_symbol, args.min_string_score)
    huri_frame = huri_edges(huri_path, ensg_to_symbol, args.huri_score)
    merged = merge_edges([string_frame, huri_frame])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output": str(out_path),
        "outputRows": int(len(merged)),
        "uniqueNodes": int(pd.unique(pd.concat([merged["preferredName_A"], merged["preferredName_B"]], ignore_index=True)).size)
        if not merged.empty
        else 0,
        "sourceCounts": dict(Counter(merged["source_query"].astype(str))) if not merged.empty else {},
        "stringRowsRetained": int(len(string_frame)),
        "huriRowsRetained": int(len(huri_frame)),
        "stringIdToSymbolRows": int(len(id_to_symbol)),
        "gtexEnsgToSymbolRows": int(len(ensg_to_symbol)),
        "parameters": {
            "minStringScore": args.min_string_score,
            "huriScore": args.huri_score,
        },
        "inputs": {
            "stringInfo": str(string_info_path),
            "stringPhysicalLinks": str(string_physical_path),
            "huri": str(huri_path),
            "gtex": str(gtex_path),
        },
        "outputs": {
            "csv": str(out_path),
            "json": str(out_path.with_suffix(".metadata.json")),
            "markdown": str(out_path.with_suffix(".md")),
        },
    }
    write_json(out_path.with_suffix(".metadata.json"), summary)
    out_path.with_suffix(".md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
