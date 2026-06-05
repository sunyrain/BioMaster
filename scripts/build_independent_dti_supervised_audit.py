from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_GROUPS = {
    "hydrophobic": set("AILMFWV"),
    "aromatic": set("FWYH"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQCY"),
    "small": set("AGS"),
    "proline": set("P"),
    "cysteine": set("C"),
}


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def pct(numerator: float | int, denominator: float | int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.array(values, dtype=float), q))


def safe_metric(y_true: list[int], y_score: list[float], metric: str) -> float | None:
    if len(set(y_true)) < 2:
        return None
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_score))
    if metric == "ap":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(metric)


def clean_id(value: Any) -> str:
    return str(value or "").strip()


def base_accession(value: Any) -> str:
    return clean_id(value).split("-")[0].split(".")[0]


def valid_mol(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def load_drug_smiles(path: Path) -> dict[str, str]:
    table = pd.read_excel(path).fillna("")
    smiles: dict[str, str] = {}
    for _, row in table.iterrows():
        drug_id = clean_id(row.get("ChEMBL ID"))
        smi = clean_id(row.get("SMILES"))
        if not drug_id or drug_id in smiles or valid_mol(smi) is None:
            continue
        smiles[drug_id] = smi
    return smiles


def load_protein_sequences(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, low_memory=False).fillna("")
    sequences: dict[str, str] = {}
    for _, row in table.iterrows():
        accession = base_accession(row.get("protein_id"))
        seq = clean_id(row.get("sequence")).upper()
        if accession and seq and accession not in sequences:
            sequences[accession] = seq
    return sequences


def pair_key(drug_id: Any, protein_id: Any) -> str:
    return f"{clean_id(drug_id)}__{base_accession(protein_id)}"


def load_positive_pairs(path: Path, drug_smiles: dict[str, str], protein_sequences: dict[str, str]) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False).fillna("")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in table.iterrows():
        drug_id = clean_id(row.get("drugIdBase"))
        protein_id = base_accession(row.get("targetAccessionBase") or row.get("targetAccession"))
        key = pair_key(drug_id, protein_id)
        if key in seen or drug_id not in drug_smiles or protein_id not in protein_sequences:
            continue
        rows.append(
            {
                "pairId": key,
                "drugId": drug_id,
                "protein": protein_id,
                "label": 1,
                "sampleSource": "fda_label_positive",
                "drug": row.get("fdaDrugName", ""),
                "target": row.get("targetGenes", ""),
                "targetName": row.get("targetPrefName", ""),
                "actionType": row.get("actionType", ""),
                "mechanismOfAction": row.get("mechanismOfAction", ""),
                "therapeuticArea": row.get("therapeuticArea", ""),
                "approvalYear": row.get("approvalYear", ""),
            }
        )
        seen.add(key)
    return pd.DataFrame(rows)


def build_negative_pairs(
    positives: pd.DataFrame,
    candidate_pair_ids: set[str],
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    positive_pairs = set(positives["pairId"].astype(str))
    drugs = sorted(positives["drugId"].astype(str).unique())
    proteins = sorted(positives["protein"].astype(str).unique())
    target_count = max(len(positives) * negative_ratio, len(positives))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = target_count * 80
    while len(rows) < target_count and attempts < max_attempts:
        attempts += 1
        if attempts % 2:
            anchor = positives.iloc[rng.randrange(len(positives))]
            drug_id = str(anchor["drugId"])
            protein_id = rng.choice(proteins)
        else:
            anchor = positives.iloc[rng.randrange(len(positives))]
            drug_id = rng.choice(drugs)
            protein_id = str(anchor["protein"])
        key = pair_key(drug_id, protein_id)
        if key in positive_pairs or key in candidate_pair_ids or key in seen:
            continue
        rows.append(
            {
                "pairId": key,
                "drugId": drug_id,
                "protein": protein_id,
                "label": 0,
                "sampleSource": "sampled_unlabeled_negative",
            }
        )
        seen.add(key)
    return pd.DataFrame(rows)


def drug_features(smiles: str, fp_bits: int) -> tuple[np.ndarray, bool]:
    mol = valid_mol(smiles)
    if mol is None:
        return np.zeros(fp_bits + 8, dtype=np.float32), False
    bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_bits)
    fp = np.zeros((fp_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, fp)
    desc = np.array(
        [
            Descriptors.MolWt(mol) / 700.0,
            Crippen.MolLogP(mol) / 8.0,
            rdMolDescriptors.CalcTPSA(mol) / 200.0,
            Lipinski.NumHDonors(mol) / 8.0,
            Lipinski.NumHAcceptors(mol) / 16.0,
            Lipinski.NumRotatableBonds(mol) / 16.0,
            QED.qed(mol),
            mol.GetNumHeavyAtoms() / 80.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([fp, np.clip(desc, -2.0, 2.0)]), True


def kmer_index(kmer: str, size: int) -> int:
    digest = hashlib.blake2b(kmer.encode("ascii", errors="ignore"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % size


def protein_features(sequence: str, kmer_bits: int) -> tuple[np.ndarray, bool]:
    seq = "".join(ch for ch in str(sequence or "").upper() if ch in AA)
    if not seq:
        return np.zeros(len(AA) + len(AA_GROUPS) + kmer_bits + 3, dtype=np.float32), False
    length = len(seq)
    aa_counts = Counter(seq)
    comp = np.array([aa_counts.get(aa, 0) / length for aa in AA], dtype=np.float32)
    groups = np.array(
        [sum(aa_counts.get(aa, 0) for aa in members) / length for members in AA_GROUPS.values()],
        dtype=np.float32,
    )
    kmers = np.zeros(kmer_bits, dtype=np.float32)
    if length >= 3:
        denom = length - 2
        for idx in range(denom):
            kmers[kmer_index(seq[idx : idx + 3], kmer_bits)] += 1.0
        kmers /= max(float(denom), 1.0)
    length_features = np.array(
        [
            math.log1p(length) / math.log1p(2500),
            min(length, 2500) / 2500.0,
            1.0 if length >= 1000 else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([comp, groups, kmers, length_features]), True


def build_feature_matrix(
    pairs: pd.DataFrame,
    drug_smiles: dict[str, str],
    protein_sequences: dict[str, str],
    fp_bits: int,
    kmer_bits: int,
) -> tuple[np.ndarray, list[bool], list[bool]]:
    drug_cache: dict[str, tuple[np.ndarray, bool]] = {}
    protein_cache: dict[str, tuple[np.ndarray, bool]] = {}
    features: list[np.ndarray] = []
    drug_ok: list[bool] = []
    protein_ok: list[bool] = []
    for _, row in pairs.iterrows():
        drug_id = clean_id(row.get("drugId"))
        protein_id = base_accession(row.get("protein"))
        if drug_id not in drug_cache:
            drug_cache[drug_id] = drug_features(drug_smiles.get(drug_id, ""), fp_bits)
        if protein_id not in protein_cache:
            protein_cache[protein_id] = protein_features(protein_sequences.get(protein_id, ""), kmer_bits)
        dfeat, dok = drug_cache[drug_id]
        pfeat, pok = protein_cache[protein_id]
        interaction = np.array(
            [
                dfeat[fp_bits + 0] * pfeat[len(AA) + 0],
                dfeat[fp_bits + 1] * pfeat[len(AA) + 1],
                dfeat[fp_bits + 2] * pfeat[len(AA) + 4],
                dfeat[fp_bits + 6] * pfeat[-3],
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([dfeat, pfeat, interaction]))
        drug_ok.append(dok)
        protein_ok.append(pok)
    return np.vstack(features).astype(np.float32), drug_ok, protein_ok


def split_metrics(
    name: str,
    pairs: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    split_kind: str,
    seed: int,
    model_args: dict[str, Any],
) -> dict[str, Any]:
    if len(set(y.tolist())) < 2:
        return {"split": name, "status": "not_run_single_class"}
    if split_kind == "stratified":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y))
    elif split_kind == "drug":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups=pairs["drugId"].astype(str).values))
    elif split_kind == "target":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups=pairs["protein"].astype(str).values))
    else:
        raise ValueError(split_kind)
    if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
        return {"split": name, "status": "not_run_single_class_after_split"}
    model = ExtraTreesClassifier(**model_args)
    model.fit(x[train_idx], y[train_idx])
    scores = model.predict_proba(x[test_idx])[:, 1]
    return {
        "split": name,
        "status": "ok",
        "trainRows": int(len(train_idx)),
        "testRows": int(len(test_idx)),
        "testPositives": int(y[test_idx].sum()),
        "testNegatives": int(len(test_idx) - y[test_idx].sum()),
        "auroc": safe_metric(y[test_idx].tolist(), scores.tolist(), "auroc"),
        "averagePrecision": safe_metric(y[test_idx].tolist(), scores.tolist(), "ap"),
        "positiveScoreMedian": float(statistics.median(scores[y[test_idx] == 1])) if int(y[test_idx].sum()) else None,
        "negativeScoreMedian": float(statistics.median(scores[y[test_idx] == 0])) if int((y[test_idx] == 0).sum()) else None,
    }


def topk_metrics(scored: pd.DataFrame, label_col: str, rank_col: str, cutoffs: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_positive = int(scored[label_col].sum())
    baseline = total_positive / len(scored) if len(scored) else 0.0
    ordered = scored.sort_values(rank_col, ascending=False).reset_index(drop=True)
    for cutoff in cutoffs:
        top = ordered.head(cutoff)
        hits = int(top[label_col].sum())
        precision = hits / len(top) if len(top) else 0.0
        recall = hits / total_positive if total_positive else 0.0
        enrichment = precision / baseline if baseline else None
        rows.append(
            {
                "cutoff": cutoff,
                "rows": int(len(top)),
                "knownHits": hits,
                "precisionPct": round(100.0 * precision, 4),
                "recallPct": round(100.0 * recall, 4),
                "enrichmentVsQueueBaseline": round(enrichment, 6) if enrichment is not None else None,
            }
        )
    return rows


def support_tier(score: float, percentile: float, exact_positive_excluded: bool) -> tuple[str, str]:
    leakage_note = "heldout_exact_fda_positive" if exact_positive_excluded else "no_exact_candidate_pair_in_training"
    if percentile >= 0.90:
        return "A_independent_dti_model_supported", f"top_decile_independent_dti_score; {leakage_note}"
    if percentile >= 0.70:
        return "B_independent_dti_model_supported", f"top_30pct_independent_dti_score; {leakage_note}"
    if percentile >= 0.40:
        return "C_independent_dti_model_review", f"middle_range_independent_dti_score; {leakage_note}"
    return "D_independent_dti_low_model_support", f"low_relative_independent_dti_score; {leakage_note}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Independent DTI Supervised Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This layer trains a local, reproducible, supervised drug-target interaction model from FDA label drug-target positives and sampled unlabeled negatives, then scores the prepared Top1000 SMILES+sequence queue.",
        "",
        "It is an independent DTI evidence layer relative to the ConPLex base ranking, but it is not reported as DrugBAN, DeepDTA, or an external pretrained SOTA checkpoint.",
        "",
        "## Scope",
        "",
        f"- Training positives: {summary['training']['positivePairs']} FDA label drug-UniProt pairs with mapped SMILES and protein sequences.",
        f"- Sampled negatives: {summary['training']['negativePairs']} unlabeled drug-target pairs.",
        f"- Exact candidate positives excluded from training: {summary['training']['excludedCandidatePositivePairs']}.",
        f"- Candidate queue scored: {summary['candidateRows']} rows, {summary['uniqueCandidateDrugs']} drugs, {summary['uniqueCandidateProteins']} proteins.",
        "",
        "## Validation",
        "",
    ]
    for item in summary["validationMetrics"]:
        lines.append(
            f"- {item['split']}: status {item['status']}; AUROC {item.get('auroc')}; AP {item.get('averagePrecision')}; "
            f"test positives/negatives {item.get('testPositives')}/{item.get('testNegatives')}."
        )
    candidate = summary["candidateKnownBenchmark"]
    lines.extend(
        [
            "",
            "## Candidate Benchmark",
            "",
            f"- Top1000 known-pair AUROC/AP: {candidate.get('auroc')}/{candidate.get('averagePrecision')}.",
        ]
    )
    for item in candidate["topK"]:
        lines.append(
            f"- Top{item['cutoff']}: known hits {item['knownHits']}/{item['rows']}; precision {item['precisionPct']}%; recall {item['recallPct']}%; enrichment {item['enrichmentVsQueueBaseline']}x."
        )
    lines.extend(
        [
            "",
            "## Candidate Tiers",
            "",
            f"- Tier counts: {summary['tierCounts']}.",
            f"- A/B supported rows: {summary['abSupportedRows']} ({summary['abSupportedPct']:.2f}%).",
            "",
            "## Top Supported Rows",
            "",
        ]
    )
    for row in summary["topSupportedRows"][:15]:
        lines.append(
            f"- {row['drug']} - {row['target']} ({row['direction']}): score {row['independentDtiScore']:.4f}, percentile {row['independentDtiPercentile']:.4f}, tier {row['independentDtiTier']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This result adds a local independent DTI ranking signal for the Top1000 queue while formal DrugBAN/DeepDTA execution remains dependent on validated checkpoints and graph-learning dependencies.",
            "- Exact FDA-positive candidate pairs were excluded from training to reduce direct pair leakage into the scored candidate queue.",
            "- Sampled unlabeled negatives are not experimentally confirmed non-binders, so the model is best interpreted as a prioritization and corroboration layer rather than a binding assay.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = pd.read_csv(root / args.candidate_queue, low_memory=False).fillna("")
    drug_smiles = load_drug_smiles(root / args.fda_structure_table)
    protein_sequences = load_protein_sequences(root / args.protein_library)
    positives = load_positive_pairs(root / args.fda_expanded_targets, drug_smiles, protein_sequences)

    candidate_pairs = {pair_key(row["drugId"], row["protein"]) for _, row in candidate.iterrows()}
    candidate_positive_pairs = set(positives["pairId"].astype(str)) & candidate_pairs
    training_positives = positives[~positives["pairId"].astype(str).isin(candidate_positive_pairs)].copy()
    negatives = build_negative_pairs(training_positives, candidate_pairs, args.negative_ratio, args.seed)
    training = pd.concat([training_positives, negatives], ignore_index=True).sample(frac=1.0, random_state=args.seed)

    x_train, train_drug_ok, train_protein_ok = build_feature_matrix(
        training, drug_smiles, protein_sequences, args.fp_bits, args.kmer_bits
    )
    y_train = training["label"].astype(int).values

    model_args = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }
    validation_metrics = [
        split_metrics("pair_stratified_holdout", training, x_train, y_train, "stratified", args.seed, model_args),
        split_metrics("drug_group_holdout", training, x_train, y_train, "drug", args.seed + 1, model_args),
        split_metrics("target_group_holdout", training, x_train, y_train, "target", args.seed + 2, model_args),
    ]

    model = ExtraTreesClassifier(**model_args)
    model.fit(x_train, y_train)

    candidate_model_input = candidate.copy()
    candidate_model_input["drugId"] = candidate_model_input["drugId"].astype(str)
    candidate_model_input["protein"] = candidate_model_input["protein"].astype(str).map(base_accession)
    x_candidate, candidate_drug_ok, candidate_protein_ok = build_feature_matrix(
        candidate_model_input, drug_smiles, protein_sequences, args.fp_bits, args.kmer_bits
    )
    scores = model.predict_proba(x_candidate)[:, 1]
    scored = candidate.copy()
    scored["independentDtiScore"] = scores
    scored["independentDtiDrugFeatureReady"] = candidate_drug_ok
    scored["independentDtiProteinFeatureReady"] = candidate_protein_ok
    scored["exactFdaPositiveExcludedFromTraining"] = [
        pair_key(row["drugId"], row["protein"]) in candidate_positive_pairs for _, row in candidate.iterrows()
    ]
    order = pd.Series(scores).rank(method="average", pct=True).values
    scored["independentDtiPercentile"] = order
    tiers = [support_tier(float(score), float(percentile), bool(excluded)) for score, percentile, excluded in zip(scores, order, scored["exactFdaPositiveExcludedFromTraining"])]
    scored["independentDtiTier"] = [tier for tier, _ in tiers]
    scored["independentDtiReason"] = [reason for _, reason in tiers]
    scored["independentDtiRankGlobal"] = scored["independentDtiScore"].rank(method="first", ascending=False).astype(int)
    scored["independentDtiKnownLabel"] = scored["knownDrugTargetPair"].map(truthy).astype(int)
    scored = scored.sort_values("independentDtiRankGlobal")

    direction_rows = []
    for direction, group in scored.groupby("direction", dropna=False):
        group_tier_counts = {str(k): int(v) for k, v in group["independentDtiTier"].value_counts().items()}
        direction_rows.append(
            {
                "direction": direction,
                "rows": int(len(group)),
                "knownRows": int(group["independentDtiKnownLabel"].sum()),
                "abSupportedRows": int(group["independentDtiTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                "medianScore": float(group["independentDtiScore"].median()),
                "tierCounts": json.dumps(group_tier_counts, ensure_ascii=False),
            }
        )
    direction_df = pd.DataFrame(direction_rows).sort_values("direction")

    train_path = out_dir / "independent_dti_supervised_training_pairs.csv"
    candidate_path = out_dir / "independent_dti_supervised_candidate_audit.csv"
    direction_path = out_dir / "independent_dti_supervised_direction_summary.csv"
    top_path = out_dir / "independent_dti_supervised_top_supported.csv"
    training.to_csv(train_path, index=False)
    scored.to_csv(candidate_path, index=False)
    direction_df.to_csv(direction_path, index=False)
    scored.head(args.top_export).to_csv(top_path, index=False)

    y_candidate = scored["independentDtiKnownLabel"].astype(int).tolist()
    score_candidate = scored["independentDtiScore"].astype(float).tolist()
    tier_counts = {str(k): int(v) for k, v in scored["independentDtiTier"].value_counts().items()}
    ab_supported = int(scored["independentDtiTier"].astype(str).str.startswith(("A_", "B_")).sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Local supervised independent DTI model over the prepared Top1000 SMILES+sequence queue.",
        "methodNote": (
            "ExtraTrees classifier trained on FDA label drug-UniProt positives and sampled unlabeled negatives. "
            "Exact FDA-positive candidate pairs are excluded from training before scoring the candidate queue. "
            "This is a local independent DTI corroboration layer, not DrugBAN/DeepDTA pretrained inference and not an experimental binding assay."
        ),
        "training": {
            "positivePairs": int(len(training_positives)),
            "candidateExactPositivePairsExcluded": sorted(candidate_positive_pairs)[:100],
            "excludedCandidatePositivePairs": int(len(candidate_positive_pairs)),
            "negativePairs": int(len(negatives)),
            "trainingRows": int(len(training)),
            "trainingDrugFeatureReadyRows": int(sum(train_drug_ok)),
            "trainingProteinFeatureReadyRows": int(sum(train_protein_ok)),
            "negativeSampling": "same-drug/random-target and same-target/random-drug unlabeled negatives, excluding FDA positives and candidate exact pairs",
        },
        "model": {
            "type": "ExtraTreesClassifier",
            "fpBits": args.fp_bits,
            "proteinKmerBits": args.kmer_bits,
            "nEstimators": args.n_estimators,
            "maxDepth": args.max_depth,
            "minSamplesLeaf": args.min_samples_leaf,
            "randomSeed": args.seed,
        },
        "validationMetrics": validation_metrics,
        "candidateRows": int(len(scored)),
        "scoredRows": int(len(scored)),
        "uniqueCandidateDrugs": int(scored["drugId"].nunique()),
        "uniqueCandidateProteins": int(scored["protein"].nunique()),
        "knownRows": int(scored["independentDtiKnownLabel"].sum()),
        "novelRows": int((scored["independentDtiKnownLabel"] == 0).sum()),
        "medianIndependentDtiScore": float(scored["independentDtiScore"].median()),
        "scoreQuantiles": {
            "q10": quantile(score_candidate, 0.10),
            "q25": quantile(score_candidate, 0.25),
            "q50": quantile(score_candidate, 0.50),
            "q75": quantile(score_candidate, 0.75),
            "q90": quantile(score_candidate, 0.90),
        },
        "tierCounts": tier_counts,
        "abSupportedRows": ab_supported,
        "abSupportedPct": pct(ab_supported, len(scored)),
        "candidateKnownBenchmark": {
            "knownRows": int(scored["independentDtiKnownLabel"].sum()),
            "auroc": safe_metric(y_candidate, score_candidate, "auroc"),
            "averagePrecision": safe_metric(y_candidate, score_candidate, "ap"),
            "topK": topk_metrics(scored, "independentDtiKnownLabel", "independentDtiScore", [50, 100, 300, 500, 1000]),
        },
        "topSupportedRows": scored.head(20)[
            [
                "independentDtiRankGlobal",
                "direction",
                "pairId",
                "drug",
                "target",
                "protein",
                "knownDrugTargetPair",
                "noveltyClass",
                "independentDtiScore",
                "independentDtiPercentile",
                "independentDtiTier",
                "independentDtiReason",
            ]
        ].to_dict(orient="records"),
        "artifacts": {
            "trainingPairsCsv": str(train_path),
            "candidateAuditCsv": str(candidate_path),
            "directionSummaryCsv": str(direction_path),
            "topSupportedCsv": str(top_path),
            "summaryJson": str(out_dir / "independent_dti_supervised_summary.json"),
            "markdown": str(out_dir / "INDEPENDENT_DTI_SUPERVISED_AUDIT.md"),
        },
    }
    write_json(out_dir / "independent_dti_supervised_summary.json", summary)
    (out_dir / "INDEPENDENT_DTI_SUPERVISED_AUDIT.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and audit a local supervised independent DTI model.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/sota_validation/independent_dti_supervised")
    parser.add_argument("--candidate-queue", default="outputs/sota_validation/external_sota_model_inputs/independent_dti_top1000_queue.csv")
    parser.add_argument("--fda-expanded-targets", default="outputs/sota_validation/fda_label_mechanism/fda_label_mechanism_expanded_targets.csv")
    parser.add_argument("--fda-structure-table", default="FDA_approved_small_molecules_2005_2026_with_structures.xlsx")
    parser.add_argument("--protein-library", default="outputs/druggable_proteome/protein_library_druggable_chembl.csv")
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--fp-bits", type=int, default=1024)
    parser.add_argument("--kmer-bits", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--top-export", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary = build(root, args)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
