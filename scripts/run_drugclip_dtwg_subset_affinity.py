#!/usr/bin/env python3
"""Run a DrugCLIP DTWG-covered subset audit for BioMaster drugs.

This intentionally does not claim a 5306-protein full DrugCLIP run. DrugCLIP is
pocket-based, so this script uses official DTWG pre-encoded AlphaFold pocket
embeddings and evaluates only BioMaster proteins covered by those embeddings.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DRUGCLIP_ROOT = ROOT / "third_party/sota_dti_2026/Drug-The-Whole-Genome"
UNICORE_ROOT = ROOT / "third_party/sota_dti_2026/Uni-Core"
sys.path.insert(0, str(DRUGCLIP_ROOT))
sys.path.insert(0, str(UNICORE_ROOT))

from unicore.data import Dictionary  # noqa: E402
from unimol.models.drugclip import BindingAffinityModel  # noqa: E402
from unimol.tasks.drugclip import DrugCLIP  # noqa: E402


def pct(num: float, den: float) -> float:
    return round((100.0 * num / den), 4) if den else 0.0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, np.ndarray)) else False:
        return ""
    return value


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def extract_checkpoint_folds(model_zip: Path, out_dir: Path, folds: int = 6) -> list[Path]:
    ckpt_dir = out_dir / "model_weights" / "6_folds"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    expected = [ckpt_dir / f"fold_{i}.pt" for i in range(folds)]
    if all(path.exists() and path.stat().st_size > 1_000_000 for path in expected):
        return expected
    with zipfile.ZipFile(model_zip) as zf:
        for i in range(folds):
            member = f"model_weights/6_folds/fold_{i}.pt"
            target = ckpt_dir / f"fold_{i}.pt"
            if target.exists() and target.stat().st_size > 1_000_000:
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
    return expected


def molecule_record(smiles: str, seed: int) -> tuple[dict[str, Any] | None, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    status = "etkdg_3d"
    code = AllChem.EmbedMolecule(mol, params)
    if code == 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                status = "etkdg_3d_mmff"
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                status = "etkdg_3d_uff"
        except Exception:
            status = "etkdg_3d_unoptimized"
    else:
        try:
            AllChem.Compute2DCoords(mol)
            status = "rdkit_2d_fallback"
        except Exception:
            return None, "conformer_failed"
    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return {"atoms": atoms, "coordinates": [coords], "smi": smiles, "smiles": smiles}, status


def build_molecule_lmdb(drug_csv: Path, lmdb_path: Path, meta_path: Path, force: bool, seed: int) -> pd.DataFrame:
    if lmdb_path.exists() and meta_path.exists() and not force:
        return pd.read_csv(meta_path)
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    if lmdb_path.exists():
        lmdb_path.unlink()
    drugs = pd.read_csv(drug_csv).fillna("")
    records: list[dict[str, Any]] = []
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=4 * 1024**3)
    with env.begin(write=True) as txn:
        write_idx = 0
        for source_idx, row in tqdm(list(drugs.iterrows()), desc="building mol lmdb"):
            smiles = clean_text(row.get("canonical_smiles")) or clean_text(row.get("isomeric_smiles"))
            record, status = molecule_record(smiles, seed + int(source_idx))
            ok = record is not None
            if ok:
                txn.put(str(write_idx).encode("ascii"), pickle.dumps(record))
            records.append(
                {
                    "molIndex": write_idx if ok else "",
                    "sourceIndex": int(source_idx),
                    "drugId": clean_text(row.get("drug_id")),
                    "drugName": clean_text(row.get("drug_name")),
                    "chemblId": clean_text(row.get("chembl_id")),
                    "smiles": smiles,
                    "molInputOk": bool(ok),
                    "molInputStatus": status,
                }
            )
            if ok:
                write_idx += 1
    env.close()
    meta = pd.DataFrame(records)
    meta.to_csv(meta_path, index=False)
    return meta


def load_task_and_model(device: torch.device) -> tuple[DrugCLIP, BindingAffinityModel]:
    args = argparse.Namespace(seed=1, max_seq_len=512, num_workers=0)
    mol_dict = Dictionary.load(str(DRUGCLIP_ROOT / "dict/dict_mol.txt"))
    pocket_dict = Dictionary.load(str(DRUGCLIP_ROOT / "dict/dict_pkt.txt"))
    mol_dict.add_symbol("[MASK]", is_special=True)
    pocket_dict.add_symbol("[MASK]", is_special=True)
    task = DrugCLIP(args, mol_dict, pocket_dict)
    model = BindingAffinityModel(args, mol_dict, pocket_dict)
    model.to(device)
    model.eval()
    return task, model


def encode_molecules(
    lmdb_path: Path,
    meta: pd.DataFrame,
    ckpts: list[Path],
    out_path: Path,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if out_path.exists() and not force:
        return np.load(out_path)
    ok_meta = meta[meta["molInputOk"].astype(bool)].copy()
    if len(ok_meta) != len(meta):
        raise RuntimeError(f"DrugCLIP molecule input has failures: {len(meta) - len(ok_meta)}")
    if not torch.cuda.is_available():
        raise RuntimeError("DrugCLIP model construction requires CUDA in this codebase.")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    task, model = load_task_and_model(device)
    dataset = task.load_mols_dataset_dtwg(str(lmdb_path), "atoms", "coordinates", dataset_type=1)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collater, num_workers=0, shuffle=False)
    all_folds: list[np.ndarray] = []
    for fold_idx, ckpt in enumerate(ckpts):
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing:
            raise RuntimeError(f"Missing checkpoint keys in fold {fold_idx}: {missing[:20]}")
        if len(unexpected) > 32:
            raise RuntimeError(f"Unexpected checkpoint keys look abnormal in fold {fold_idx}: {unexpected[:20]}")
        model.eval()
        reps: list[np.ndarray] = []
        with torch.no_grad():
            for sample in tqdm(loader, desc=f"encoding mol fold {fold_idx}"):
                net = sample["net_input"]
                dist = net["mol_src_distance"].to(device, non_blocking=True)
                et = net["mol_src_edge_type"].to(device, non_blocking=True)
                st = net["mol_src_tokens"].to(device, non_blocking=True)
                padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                graph_attn_bias = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(mol_x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
                mol_encoder_rep = mol_outputs[0][:, 0, :]
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                reps.append(mol_emb.detach().cpu().numpy().astype(np.float32))
        all_folds.append(np.concatenate(reps, axis=0))
        del state
        torch.cuda.empty_cache()
        gc.collect()
    arr = np.stack(all_folds, axis=1).astype(np.float32)
    np.save(out_path, arr)
    return arr


def parse_dtwg_uniprot(name: Any) -> str:
    match = re.search(r"AF-([A-Z0-9]+)-F1", str(name))
    return match.group(1) if match else ""


def build_drugclip_scores(
    mol_embs: np.ndarray,
    meta: pd.DataFrame,
    protein_csv: Path,
    dtwg_names_path: Path,
    dtwg_emb_path: Path,
    out_dir: Path,
) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
    proteins = pd.read_csv(protein_csv, usecols=["protein_id"])
    protein_universe = set(proteins["protein_id"].astype(str))
    names = np.load(dtwg_names_path, allow_pickle=True)
    parsed = np.asarray([parse_dtwg_uniprot(x) for x in names], dtype=object)
    mask = np.asarray([x in protein_universe for x in parsed], dtype=bool)
    covered_names = names[mask]
    covered_proteins_for_pockets = parsed[mask]
    raw_embs = np.load(dtwg_emb_path, mmap_mode="r")
    pocket_embs = np.asarray(raw_embs[mask], dtype=np.float32).reshape((-1, 6, 128))
    protein_order = [pid for pid in proteins["protein_id"].astype(str).tolist() if pid in set(covered_proteins_for_pockets)]
    by_protein: dict[str, list[int]] = defaultdict(list)
    for idx, pid in enumerate(covered_proteins_for_pockets):
        by_protein[str(pid)].append(idx)
    scores = np.zeros((pocket_embs.shape[0], mol_embs.shape[0]), dtype=np.float32)
    for fold_idx in range(mol_embs.shape[1]):
        scores += pocket_embs[:, fold_idx, :] @ mol_embs[:, fold_idx, :].T
    scores /= float(mol_embs.shape[1])
    medians = np.median(scores, axis=1, keepdims=True)
    mads = np.median(np.abs(scores - medians), axis=1, keepdims=True)
    zscores = 0.6745 * (scores - medians) / (mads + 1e-6)
    protein_scores = np.full((len(protein_order), mol_embs.shape[0]), -np.inf, dtype=np.float32)
    for pidx, pid in enumerate(protein_order):
        protein_scores[pidx, :] = zscores[by_protein[pid], :].max(axis=0)
    ok_meta = meta[meta["molInputOk"].astype(bool)].copy()
    drug_ids = ok_meta["drugId"].astype(str).tolist()
    pocket_meta = pd.DataFrame(
        {
            "pocketIndex": np.where(mask)[0],
            "proteinId": covered_proteins_for_pockets,
            "dtwgName": [str(x) for x in covered_names],
        }
    )
    pocket_meta.to_csv(out_dir / "drugclip_dtwg_covered_pockets.csv", index=False)
    protein_pocket_counts = pocket_meta.groupby("proteinId").size().reset_index(name="pocketCount")
    protein_pocket_counts.to_csv(out_dir / "drugclip_dtwg_covered_proteins.csv", index=False)
    drug_major = protein_scores.T
    long_df = pd.DataFrame(
        {
            "drugId": np.repeat(drug_ids, len(protein_order)),
            "proteinId": np.tile(protein_order, len(drug_ids)),
            "drugclipScore": drug_major.reshape(-1),
        }
    )
    long_df["pairId"] = long_df["drugId"] + "__" + long_df["proteinId"]
    long_df[["pairId", "drugId", "proteinId", "drugclipScore"]].to_csv(
        out_dir / "drugclip_dtwg_subset_affinity_scores.csv.gz", index=False
    )
    return drug_major, drug_ids, protein_order, pocket_meta


def rank_matrix(score_matrix: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(score_matrix, dtype=np.int32)
    for i in range(score_matrix.shape[0]):
        order = np.argsort(-score_matrix[i], kind="mergesort")
        ranks[i, order] = np.arange(1, score_matrix.shape[1] + 1, dtype=np.int32)
    return ranks


def pair_ranks(
    label: str,
    score_matrix: np.ndarray,
    drug_ids: list[str],
    protein_ids: list[str],
    known_csv: Path,
    out_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    known = pd.read_csv(known_csv).fillna("")
    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
    protein_to_idx = {p: i for i, p in enumerate(protein_ids)}
    ranks = rank_matrix(score_matrix)
    records = []
    for _, row in known.iterrows():
        drug = str(row.get("drugId", ""))
        protein = str(row.get("proteinId", ""))
        if drug not in drug_to_idx or protein not in protein_to_idx:
            continue
        di = drug_to_idx[drug]
        pi = protein_to_idx[protein]
        records.append(
            {
                "model": label,
                "pairId": f"{drug}__{protein}",
                "drugId": drug,
                "drugName": row.get("drugName", ""),
                "proteinId": protein,
                "geneName": row.get("geneName", ""),
                "targetName": row.get("targetName", ""),
                "rank": int(ranks[di, pi]),
                "score": float(score_matrix[di, pi]),
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["rank", "drugId", "proteinId"])
    df.to_csv(out_path, index=False)
    cutoffs = [1, 3, 5, 10, 20, 50, 100]
    summary: dict[str, Any] = {
        "model": label,
        "candidateTargetUniverse": len(protein_ids),
        "knownPairsEvaluated": int(len(df)),
        "drugsWithKnownPairs": int(df["drugId"].nunique()) if not df.empty else 0,
        "medianRank": float(df["rank"].median()) if not df.empty else None,
        "meanRank": float(df["rank"].mean()) if not df.empty else None,
    }
    for cutoff in cutoffs:
        summary[f"microRecall@{cutoff}"] = pct(int((df["rank"] <= cutoff).sum()) if not df.empty else 0, len(df))
    if not df.empty:
        per_drug = df.assign(hit100=df["rank"] <= 100).groupby("drugId")["hit100"].mean()
        summary["macroDrugRecall@100"] = round(100.0 * float(per_drug.mean()), 4)
    else:
        summary["macroDrugRecall@100"] = 0.0
    return df, summary


def load_conplex_subset(conplex_csv: Path, drug_ids: list[str], protein_ids: list[str]) -> np.ndarray:
    drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
    protein_to_idx = {p: i for i, p in enumerate(protein_ids)}
    matrix = np.full((len(drug_ids), len(protein_ids)), -np.inf, dtype=np.float32)
    drug_set = set(drug_ids)
    protein_set = set(protein_ids)
    for chunk in tqdm(pd.read_csv(conplex_csv, usecols=["pair_id", "affinity_score"], chunksize=500_000), desc="loading ConPLex subset"):
        parsed = [str(pid).rsplit("__", 1) for pid in chunk["pair_id"].astype(str)]
        chunk["drugId"] = [x[0] if len(x) == 2 else "" for x in parsed]
        chunk["proteinId"] = [x[1] if len(x) == 2 else "" for x in parsed]
        sub = chunk[chunk["drugId"].isin(drug_set) & chunk["proteinId"].isin(protein_set)]
        for row in sub.itertuples(index=False):
            matrix[drug_to_idx[row.drugId], protein_to_idx[row.proteinId]] = float(row.affinity_score)
    missing = int(np.isneginf(matrix).sum())
    if missing:
        raise RuntimeError(f"ConPLex subset matrix has {missing} missing cells.")
    return matrix


def summarize_full_conplex_baseline(known_csv: Path) -> dict[str, Any]:
    known = pd.read_csv(known_csv)
    rank_col = "exactMinRank"
    summary: dict[str, Any] = {
        "model": "ConPLex_full_5306_baseline",
        "candidateTargetUniverse": 5306,
        "knownPairsEvaluated": int(len(known)),
        "drugsWithKnownPairs": int(known["drugId"].nunique()),
        "medianRank": float(known[rank_col].median()),
        "meanRank": float(known[rank_col].mean()),
    }
    for cutoff in [1, 3, 5, 10, 20, 50, 100, 500, 1000]:
        summary[f"microRecall@{cutoff}"] = pct(int((known[rank_col] <= cutoff).sum()), len(known))
    return summary


def write_report(out_dir: Path, summary: dict[str, Any], model_summaries: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    rows = []
    for item in model_summaries:
        rows.append(
            "| {model} | {univ} | {pairs} | {median} | {r10} | {r50} | {r100} |".format(
                model=item["model"],
                univ=item["candidateTargetUniverse"],
                pairs=item["knownPairsEvaluated"],
                median=fmt(item.get("medianRank")),
                r10=fmt(item.get("microRecall@10")),
                r50=fmt(item.get("microRecall@50")),
                r100=fmt(item.get("microRecall@100")),
            )
        )
    lines = [
        "# DrugCLIP DTWG 覆盖子集审计",
        "",
        f"生成时间: {summary['createdUtc']}",
        "",
        "## 关键结论",
        "",
        f"- DrugCLIP-DTWG 可覆盖 BioMaster 蛋白: {summary['coveredProteins']}/{summary['biomasterProteins']} ({summary['coveredProteinPct']}%).",
        f"- 可覆盖 known-target 阳性 pair: {summary['coveredKnownPairs']}/{summary['knownPairsTotal']} ({summary['coveredKnownPairPct']}%).",
        "- 下面的 DrugCLIP 指标只在覆盖子集上计算，不能等同于 5306 蛋白全量重排。",
        "",
        "## 指标对比",
        "",
        "| Model | Target universe | Known pairs | Median rank | Recall@10 | Recall@50 | Recall@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 输出文件",
        "",
        "- `drugclip_dtwg_subset_affinity_scores.csv.gz`: DrugCLIP 覆盖子集打分矩阵。",
        "- `drugclip_known_pair_ranks.csv`: DrugCLIP 覆盖子集 known-target rank。",
        "- `conplex_dtwg_subset_known_pair_ranks.csv`: ConPLex 在同一覆盖子集上的 known-target rank。",
        "- `drugclip_dtwg_covered_pockets.csv`: DTWG pocket 到 BioMaster protein 的覆盖明细。",
        "- `summary.json`: 机器可读摘要。",
        "",
        "## 方法备注",
        "",
        "DrugCLIP 使用官方 6-fold 权重重新编码 915 个 BioMaster/FDA 药物。蛋白侧使用官方 DTWG AlphaFold pocket embeddings；同一蛋白多个 pocket 取最大 z-score。z-score 按每个 pocket 在 915 个药物上的 median/MAD 计算。",
    ]
    (out_dir / "DRUGCLIP_DTWG_SUBSET_AUDIT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_zip = (ROOT / args.model_zip).resolve()
    ckpts = extract_checkpoint_folds(model_zip, out_dir, folds=6)
    lmdb_path = out_dir / "biomaster_915_drugclip_mols.lmdb"
    mol_meta_path = out_dir / "biomaster_915_drugclip_mol_manifest.csv"
    mol_meta = build_molecule_lmdb((ROOT / args.drug_csv).resolve(), lmdb_path, mol_meta_path, args.force, args.seed)
    mol_emb_path = out_dir / "biomaster_915_drugclip_mol_embs_6fold.npy"
    mol_embs = encode_molecules(lmdb_path, mol_meta, ckpts, mol_emb_path, args.batch_size, args.force)
    drugclip_matrix, drug_ids, covered_proteins, pocket_meta = build_drugclip_scores(
        mol_embs,
        mol_meta,
        (ROOT / args.protein_csv).resolve(),
        (ROOT / args.dtwg_names).resolve(),
        (ROOT / args.dtwg_embeddings).resolve(),
        out_dir,
    )
    drugclip_rank_df, drugclip_summary = pair_ranks(
        "DrugCLIP_DTWG_subset",
        drugclip_matrix,
        drug_ids,
        covered_proteins,
        (ROOT / args.known_csv).resolve(),
        out_dir / "drugclip_known_pair_ranks.csv",
    )
    conplex_matrix = load_conplex_subset((ROOT / args.conplex_csv).resolve(), drug_ids, covered_proteins)
    np.save(out_dir / "conplex_dtwg_subset_scores.npy", conplex_matrix)
    _, conplex_subset_summary = pair_ranks(
        "ConPLex_DTWG_subset",
        conplex_matrix,
        drug_ids,
        covered_proteins,
        (ROOT / args.known_csv).resolve(),
        out_dir / "conplex_dtwg_subset_known_pair_ranks.csv",
    )
    conplex_full_summary = summarize_full_conplex_baseline((ROOT / args.known_csv).resolve())
    known = pd.read_csv((ROOT / args.known_csv).resolve())
    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outDir": str(out_dir),
        "biomasterDrugs": int(len(drug_ids)),
        "biomasterProteins": int(pd.read_csv((ROOT / args.protein_csv).resolve(), usecols=["protein_id"])["protein_id"].nunique()),
        "coveredProteins": int(len(covered_proteins)),
        "coveredProteinPct": pct(len(covered_proteins), pd.read_csv((ROOT / args.protein_csv).resolve(), usecols=["protein_id"])["protein_id"].nunique()),
        "coveredPockets": int(len(pocket_meta)),
        "knownPairsTotal": int(len(known)),
        "coveredKnownPairs": int(len(drugclip_rank_df)),
        "coveredKnownPairPct": pct(len(drugclip_rank_df), len(known)),
        "modelSummaries": [conplex_full_summary, conplex_subset_summary, drugclip_summary],
        "notes": [
            "DrugCLIP is pocket-based; this run uses official DTWG pre-encoded AlphaFold pocket embeddings.",
            "The covered-subset metrics are not directly comparable to a 5306-protein full-universe Recall@100.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, summary, summary["modelSummaries"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-csv", default="outputs/fda_step1/drug_library.csv")
    parser.add_argument("--protein-csv", default="outputs/druggable_proteome/protein_library_druggable_chembl.csv")
    parser.add_argument("--known-csv", default="outputs/sota_validation/known_target_per_drug_recall/known_target_per_drug_pair_ranks.csv")
    parser.add_argument("--conplex-csv", default="outputs/druggable_proteome/conplex_affinity_scores_druggable.csv")
    parser.add_argument("--model-zip", default="third_party/sota_dti_2026/Drug-The-Whole-Genome/data_downloads/model_weights.zip")
    parser.add_argument("--dtwg-names", default="third_party/sota_dti_2026/Drug-The-Whole-Genome/data_downloads/benchmark_throughput/dtwg_af_names_.npy")
    parser.add_argument("--dtwg-embeddings", default="third_party/sota_dti_2026/Drug-The-Whole-Genome/data_downloads/benchmark_throughput/dtwg_af_embeddings.npy")
    parser.add_argument("--out-dir", default="outputs/sota_validation/drugclip_biomaster_dtwg_subset")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
