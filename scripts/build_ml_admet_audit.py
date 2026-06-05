from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


ENDPOINTS: list[dict[str, str]] = [
    {
        "endpoint": "herg",
        "tdcTask": "Tox",
        "endpointClass": "toxicity",
        "positiveMeaning": "hERG inhibition / cardiotoxicity risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "dili",
        "tdcTask": "Tox",
        "endpointClass": "toxicity",
        "positiveMeaning": "drug-induced liver injury risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "ames",
        "tdcTask": "Tox",
        "endpointClass": "toxicity",
        "positiveMeaning": "Ames mutagenicity risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "cyp3a4_veith",
        "tdcTask": "ADME",
        "endpointClass": "metabolism_ddi",
        "positiveMeaning": "CYP3A4 inhibition / drug-interaction risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "cyp2d6_veith",
        "tdcTask": "ADME",
        "endpointClass": "metabolism_ddi",
        "positiveMeaning": "CYP2D6 inhibition / drug-interaction risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "cyp2c9_veith",
        "tdcTask": "ADME",
        "endpointClass": "metabolism_ddi",
        "positiveMeaning": "CYP2C9 inhibition / drug-interaction risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "cyp2c19_veith",
        "tdcTask": "ADME",
        "endpointClass": "metabolism_ddi",
        "positiveMeaning": "CYP2C19 inhibition / drug-interaction risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "cyp1a2_veith",
        "tdcTask": "ADME",
        "endpointClass": "metabolism_ddi",
        "positiveMeaning": "CYP1A2 inhibition / drug-interaction risk",
        "scoreRole": "risk",
    },
    {
        "endpoint": "pgp_broccatelli",
        "tdcTask": "ADME",
        "endpointClass": "transporter",
        "positiveMeaning": "P-glycoprotein interaction signal",
        "scoreRole": "context_risk",
    },
    {
        "endpoint": "bbb_martins",
        "tdcTask": "ADME",
        "endpointClass": "distribution",
        "positiveMeaning": "blood-brain barrier penetration",
        "scoreRole": "context_exposure",
    },
    {
        "endpoint": "hia_hou",
        "tdcTask": "ADME",
        "endpointClass": "absorption",
        "positiveMeaning": "human intestinal absorption",
        "scoreRole": "desirable_exposure",
    },
]

NOVEL_CLASSES = {
    "disease_context_supported_new_pair",
    "model_priority_without_txgnn_kg_path",
}

RDLogger.DisableLog("rdApp.*")


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def pct_str(value: float | int | None) -> str:
    return "NA" if value is None else f"{float(value):.2f}%"


def fmt(value: float | int | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def bool_value(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def mol_from_smiles(smiles: Any) -> Chem.Mol | None:
    text = str(smiles or "").strip()
    if not text or text.lower() == "nan":
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def fingerprint(smiles: Any, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def featurize_series(smiles: pd.Series, n_bits: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    valid_mask: list[bool] = []
    for value in smiles:
        fp = fingerprint(value, n_bits=n_bits)
        if fp is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            features.append(fp)
    if not features:
        return np.empty((0, n_bits), dtype=np.uint8), np.array(valid_mask, dtype=bool)
    return np.vstack(features), np.array(valid_mask, dtype=bool)


def load_tdc_dataset(endpoint: dict[str, str], tdc_path: Path) -> pd.DataFrame:
    if endpoint["tdcTask"] == "ADME":
        from tdc.single_pred import ADME

        loader = ADME(name=endpoint["endpoint"], path=str(tdc_path), print_stats=False)
    else:
        from tdc.single_pred import Tox

        loader = Tox(name=endpoint["endpoint"], path=str(tdc_path), print_stats=False)
    df = loader.get_data()
    required = {"Drug", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{endpoint['endpoint']} missing required TDC columns: {sorted(missing)}")
    return df[["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "label"}).copy()


def train_endpoint(
    endpoint: dict[str, str],
    df: pd.DataFrame,
    random_state: int,
    n_bits: int,
    n_estimators: int,
) -> tuple[ExtraTreesClassifier | None, dict[str, Any]]:
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "smiles"]).copy()
    df["label"] = (df["label"].astype(float) >= 0.5).astype(int)
    x_all, valid_mask = featurize_series(df["smiles"], n_bits=n_bits)
    valid_df = df.loc[valid_mask].reset_index(drop=True)
    invalid_rows = int((~valid_mask).sum())
    if len(valid_df) < 100 or valid_df["label"].nunique() < 2:
        return None, {
            **endpoint,
            "status": "skipped_insufficient_binary_data",
            "rows": int(len(df)),
            "validRows": int(len(valid_df)),
            "invalidSmilesRows": invalid_rows,
        }

    y = valid_df["label"].to_numpy(dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metric = {
        **endpoint,
        "status": "trained",
        "rows": int(len(df)),
        "validRows": int(len(valid_df)),
        "invalidSmilesRows": invalid_rows,
        "positiveRows": int(y.sum()),
        "positivePct": round(pct(int(y.sum()), len(y)), 4),
        "trainRows": int(len(y_train)),
        "testRows": int(len(y_test)),
        "testPositiveRows": int(y_test.sum()),
        "testAuroc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) >= 2 else None,
        "testAveragePrecision": float(average_precision_score(y_test, proba)) if int(y_test.sum()) else None,
        "testBalancedAccuracyAt05": float(balanced_accuracy_score(y_test, pred)),
        "nBits": n_bits,
        "model": "ExtraTreesClassifier",
        "nEstimators": n_estimators,
        "randomState": random_state,
    }
    return model, metric


def probability_tier(value: float | None, role: str) -> str:
    if value is None:
        return "U_unscored"
    if role in {"risk", "context_risk"}:
        if value >= 0.75:
            return "D_high_predicted_risk"
        if value >= 0.55:
            return "C_moderate_predicted_risk"
        if value >= 0.35:
            return "B_low_to_moderate_predicted_risk"
        return "A_low_predicted_risk"
    if role == "desirable_exposure":
        if value >= 0.75:
            return "A_high_predicted_exposure"
        if value >= 0.55:
            return "B_moderate_predicted_exposure"
        if value >= 0.35:
            return "C_low_predicted_exposure"
        return "D_poor_predicted_exposure"
    if value >= 0.75:
        return "C_high_context_exposure"
    if value >= 0.55:
        return "B_moderate_context_exposure"
    return "A_low_context_exposure"


def score_drugs(
    drugs: pd.DataFrame,
    models: dict[str, ExtraTreesClassifier],
    metrics: list[dict[str, Any]],
    n_bits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = drugs.copy()
    smiles_col = "canonical_smiles" if "canonical_smiles" in scored else "SMILES"
    x_drug, valid_mask = featurize_series(scored[smiles_col], n_bits=n_bits)
    scored["mlAdmetSmilesValid"] = valid_mask
    valid_index = np.where(valid_mask)[0]

    endpoint_rows: list[dict[str, Any]] = []
    metric_by_endpoint = {row["endpoint"]: row for row in metrics}
    for endpoint, model in models.items():
        metric = metric_by_endpoint[endpoint]
        proba_col = f"ml_{endpoint}_prob"
        tier_col = f"ml_{endpoint}_tier"
        scored[proba_col] = np.nan
        if len(valid_index):
            scored.loc[valid_index, proba_col] = model.predict_proba(x_drug)[:, 1]
        scored[tier_col] = [
            probability_tier(None if pd.isna(value) else float(value), metric["scoreRole"])
            for value in scored[proba_col]
        ]
        for idx, row in scored.iterrows():
            value = row[proba_col]
            endpoint_rows.append(
                {
                    "drugId": row.get("drug_id", row.get("drugId", "")),
                    "drug": row.get("drug_name", row.get("drug", "")),
                    "chemblId": row.get("chembl_id", row.get("chemblId", "")),
                    "endpoint": endpoint,
                    "tdcTask": metric.get("tdcTask", ""),
                    "endpointClass": metric.get("endpointClass", ""),
                    "scoreRole": metric.get("scoreRole", ""),
                    "positiveMeaning": metric.get("positiveMeaning", ""),
                    "probability": None if pd.isna(value) else round(float(value), 6),
                    "probabilityTier": row[tier_col],
                    "modelTestAuroc": metric.get("testAuroc"),
                    "modelTestAveragePrecision": metric.get("testAveragePrecision"),
                }
            )
    scored = add_composite_scores(scored, metrics)
    return scored, pd.DataFrame(endpoint_rows)


def add_composite_scores(scored: pd.DataFrame, metrics: list[dict[str, Any]]) -> pd.DataFrame:
    risk_endpoints = [row["endpoint"] for row in metrics if row.get("status") == "trained" and row.get("scoreRole") in {"risk", "context_risk"}]
    toxicity_endpoints = [
        row["endpoint"]
        for row in metrics
        if row.get("status") == "trained" and row.get("endpointClass") in {"toxicity", "metabolism_ddi"}
    ]
    desirable_endpoints = [row["endpoint"] for row in metrics if row.get("status") == "trained" and row.get("scoreRole") == "desirable_exposure"]
    exposure_context_endpoints = [
        row["endpoint"]
        for row in metrics
        if row.get("status") == "trained" and row.get("scoreRole") in {"desirable_exposure", "context_exposure"}
    ]

    risk_cols = [f"ml_{endpoint}_prob" for endpoint in risk_endpoints if f"ml_{endpoint}_prob" in scored]
    toxicity_cols = [f"ml_{endpoint}_prob" for endpoint in toxicity_endpoints if f"ml_{endpoint}_prob" in scored]
    desirable_cols = [f"ml_{endpoint}_prob" for endpoint in desirable_endpoints if f"ml_{endpoint}_prob" in scored]
    exposure_cols = [f"ml_{endpoint}_prob" for endpoint in exposure_context_endpoints if f"ml_{endpoint}_prob" in scored]

    scored["mlAdmetRiskMean"] = scored[risk_cols].mean(axis=1, skipna=True) if risk_cols else np.nan
    scored["mlAdmetToxicityRiskMean"] = scored[toxicity_cols].mean(axis=1, skipna=True) if toxicity_cols else np.nan
    scored["mlAdmetRiskMax"] = scored[risk_cols].max(axis=1, skipna=True) if risk_cols else np.nan
    scored["mlAdmetHighRiskEndpointCount"] = (
        (scored[risk_cols] >= 0.75).sum(axis=1).astype(int) if risk_cols else 0
    )
    scored["mlAdmetModerateRiskEndpointCount"] = (
        (scored[risk_cols] >= 0.55).sum(axis=1).astype(int) if risk_cols else 0
    )
    scored["mlAdmetDesirableExposureMean"] = scored[desirable_cols].mean(axis=1, skipna=True) if desirable_cols else np.nan
    scored["mlAdmetExposureContextMean"] = scored[exposure_cols].mean(axis=1, skipna=True) if exposure_cols else np.nan

    safety_score = 100.0 - scored["mlAdmetRiskMean"].fillna(0.5) * 70.0 - scored["mlAdmetHighRiskEndpointCount"].fillna(0) * 4.0
    if desirable_cols:
        safety_score += scored["mlAdmetDesirableExposureMean"].fillna(0.5) * 8.0
    scored["mlAdmetSafetyScore"] = safety_score.clip(0, 100).round(4)
    scored["mlAdmetSafetyTier"] = scored.apply(classify_drug_tier, axis=1)
    scored["mlAdmetRiskFlags"] = scored.apply(lambda row: risk_flags(row, metrics), axis=1)
    return scored


def classify_drug_tier(row: pd.Series) -> str:
    if not bool_value(row.get("mlAdmetSmilesValid")):
        return "U_unscored_invalid_smiles"
    high = int(row.get("mlAdmetHighRiskEndpointCount") or 0)
    moderate = int(row.get("mlAdmetModerateRiskEndpointCount") or 0)
    score = float(row.get("mlAdmetSafetyScore") or 0.0)
    if high >= 3 or score < 45:
        return "D_high_ml_admet_risk"
    if high >= 1 or moderate >= 4 or score < 60:
        return "C_ml_admet_review"
    if moderate >= 1 or score < 75:
        return "B_manageable_ml_admet_signal"
    return "A_low_ml_admet_risk"


def risk_flags(row: pd.Series, metrics: list[dict[str, Any]]) -> str:
    flags: list[str] = []
    for metric in metrics:
        if metric.get("status") != "trained" or metric.get("scoreRole") not in {"risk", "context_risk"}:
            continue
        endpoint = metric["endpoint"]
        value = row.get(f"ml_{endpoint}_prob")
        if pd.isna(value):
            continue
        if float(value) >= 0.75:
            flags.append(f"high_{endpoint}")
        elif float(value) >= 0.55:
            flags.append(f"moderate_{endpoint}")
    return "; ".join(flags) if flags else "none"


def join_candidates(candidates: pd.DataFrame, drug_scores: pd.DataFrame) -> pd.DataFrame:
    drug_cols = [
        col
        for col in drug_scores.columns
        if col.startswith("ml_")
        or col.startswith("mlAdmet")
        or col in {"drug_id", "drug_name", "chembl_id", "canonical_smiles"}
    ]
    joined = candidates.merge(
        drug_scores[drug_cols],
        left_on="drugId",
        right_on="drug_id",
        how="left",
        suffixes=("", "_mlDrug"),
    )
    joined["mlAdmetCandidateSafetyScore"] = pd.to_numeric(joined["mlAdmetSafetyScore"], errors="coerce").fillna(50.0)
    base_col = "sotaContextScore" if "sotaContextScore" in joined else "sotaNetworkScore"
    risk_penalty = ((100.0 - joined["mlAdmetCandidateSafetyScore"]) / 100.0 * 3.0).clip(0, 6)
    joined["sotaMlAdmetScore"] = (pd.to_numeric(joined[base_col], errors="coerce").fillna(0.0) - risk_penalty).clip(0, 100).round(4)
    joined["sotaMlAdmetTier"] = joined.apply(classify_candidate_tier, axis=1)
    joined["sotaMlAdmetAction"] = joined.apply(classify_candidate_action, axis=1)
    joined = joined.sort_values(["sotaMlAdmetScore", base_col], ascending=[False, False]).reset_index(drop=True).copy()
    joined["sotaMlAdmetRankGlobal"] = range(1, len(joined) + 1)
    joined["sotaMlAdmetRankWithinDirection"] = (
        joined.groupby("direction")["sotaMlAdmetScore"].rank(method="first", ascending=False).astype(int)
    )
    front = [
        "sotaMlAdmetRankGlobal",
        "sotaMlAdmetRankWithinDirection",
        "sotaMlAdmetScore",
        "sotaMlAdmetTier",
        "sotaMlAdmetAction",
        "mlAdmetCandidateSafetyScore",
        "mlAdmetSafetyTier",
        "mlAdmetRiskMean",
        "mlAdmetRiskMax",
        "mlAdmetHighRiskEndpointCount",
        "mlAdmetModerateRiskEndpointCount",
        "mlAdmetRiskFlags",
    ]
    ordered = front + [col for col in joined.columns if col not in front]
    return joined[ordered]


def classify_candidate_tier(row: pd.Series) -> str:
    base_tier = str(row.get("sotaContextTier") or row.get("sotaNetworkTier") or "")
    safety_tier = str(row.get("mlAdmetSafetyTier") or "")
    score = float(row.get("sotaMlAdmetScore") or 0.0)
    if safety_tier.startswith("D_") or str(row.get("sotaContextAction") or row.get("sotaNetworkAction") or "") in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return "D_safety_or_resolution_review"
    if score >= 90 and base_tier.startswith(("A_", "B_")) and safety_tier.startswith(("A_", "B_")):
        return "A_ml_admet_supported_priority"
    if score >= 80 and safety_tier.startswith(("A_", "B_", "C_")):
        return "B_ml_admet_review_priority"
    if score >= 65:
        return "C_context_or_secondary_review"
    return "D_low_priority_or_sparse_support"


def classify_candidate_action(row: pd.Series) -> str:
    safety_tier = str(row.get("mlAdmetSafetyTier") or "")
    novelty = str(row.get("noveltyClass") or "")
    strict_novel = bool_value(row.get("strictNovelPairFlag")) or novelty in NOVEL_CLASSES
    if safety_tier.startswith("D_"):
        return "ml_admet_high_risk_review"
    if safety_tier.startswith("C_"):
        return "ml_admet_moderate_risk_review"
    if strict_novel and safety_tier.startswith(("A_", "B_")):
        return "novel_ml_admet_supported_review"
    return "ml_admet_supported_review"


def topk_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positives_total = int(pd.to_numeric(candidates.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum())
    base_rate = positives_total / len(candidates) if len(candidates) else 0.0
    for cutoff in [20, 50, 100, 200, 500, 1000, 2000]:
        if cutoff > len(candidates):
            continue
        top = candidates.head(cutoff)
        hits = int(pd.to_numeric(top.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum())
        expected = cutoff * base_rate
        rows.append(
            {
                "groupType": "all",
                "groupValue": "all",
                "cutoff": cutoff,
                "knownDrugTargetRows": hits,
                "recallKnownDrugTargetPct": round(pct(hits, positives_total), 4),
                "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
                "lowMlAdmetRiskRows": int(top["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                "highMlAdmetRiskRows": int(top["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum()),
                "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top else 0,
                "tierCounts": dict(Counter(top["sotaMlAdmetTier"].astype(str))),
                "mlAdmetSafetyTierCounts": dict(Counter(top["mlAdmetSafetyTier"].astype(str))),
                "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else int(top["drug"].nunique()),
                "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else int(top["target"].nunique()),
            }
        )
    for direction, group in candidates.groupby("direction"):
        positives_dir = int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum())
        base_rate_dir = positives_dir / len(group) if len(group) else 0.0
        for cutoff in [20, 50, 100, 200]:
            if cutoff > len(group):
                continue
            top = group.sort_values("sotaMlAdmetScore", ascending=False).head(cutoff)
            hits = int(pd.to_numeric(top.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum())
            expected = cutoff * base_rate_dir
            rows.append(
                {
                    "groupType": "direction",
                    "groupValue": direction,
                    "cutoff": cutoff,
                    "knownDrugTargetRows": hits,
                    "recallKnownDrugTargetPct": round(pct(hits, positives_dir), 4),
                    "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
                    "lowMlAdmetRiskRows": int(top["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                    "highMlAdmetRiskRows": int(top["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum()),
                    "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top else 0,
                    "tierCounts": dict(Counter(top["sotaMlAdmetTier"].astype(str))),
                    "mlAdmetSafetyTierCounts": dict(Counter(top["mlAdmetSafetyTier"].astype(str))),
                    "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else int(top["drug"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else int(top["target"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def direction_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, group in candidates.groupby("direction"):
        top100 = group.sort_values("sotaMlAdmetScore", ascending=False).head(min(100, len(group)))
        rows.append(
            {
                "direction": direction,
                "candidateRows": int(len(group)),
                "uniqueDrugs": int(group["drugId"].nunique()) if "drugId" in group else int(group["drug"].nunique()),
                "medianMlAdmetSafetyScore": round(float(group["mlAdmetCandidateSafetyScore"].median()), 4),
                "lowMlAdmetRiskRows": int(group["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                "lowMlAdmetRiskPct": round(
                    pct(int(group["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum()), len(group)), 4
                ),
                "highMlAdmetRiskRows": int(group["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum()),
                "sotaMlAdmetTierCounts": dict(Counter(group["sotaMlAdmetTier"].astype(str))),
                "mlAdmetSafetyTierCounts": dict(Counter(group["mlAdmetSafetyTier"].astype(str))),
                "top100LowMlAdmetRiskRows": int(top100["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                "top100HighMlAdmetRiskRows": int(top100["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum()),
                "top100KnownRows": int(pd.to_numeric(top100.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                "top100NovelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top100 else 0,
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    drug_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    metrics: list[dict[str, Any]],
    endpoint_long: pd.DataFrame,
) -> dict[str, Any]:
    trained = [row for row in metrics if row.get("status") == "trained"]
    failed = [row for row in metrics if row.get("status") != "trained"]
    top100 = candidates.head(min(100, len(candidates)))
    low_drug = int(drug_scores["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum())
    high_drug = int(drug_scores["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum())
    low_candidates = int(candidates["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum())
    high_candidates = int(candidates["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum())
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Local TDC-trained molecular fingerprint ML ADMET audit for FDA drugs and final candidates.",
        "interpretationNote": "This layer is a lightweight QSAR safety screen trained from public TDC endpoint data. It supports triage and does not replace experimental ADMET, clinical safety review, or label contraindication review.",
        "drugRows": int(len(drug_scores)),
        "candidateRows": int(len(candidates)),
        "endpointRows": int(len(endpoint_long)),
        "trainedEndpointCount": len(trained),
        "failedEndpointCount": len(failed),
        "trainedEndpoints": [row["endpoint"] for row in trained],
        "failedEndpoints": failed,
        "endpointMetricSummary": {
            row["endpoint"]: {
                "testAuroc": row.get("testAuroc"),
                "testAveragePrecision": row.get("testAveragePrecision"),
                "testBalancedAccuracyAt05": row.get("testBalancedAccuracyAt05"),
                "validRows": row.get("validRows"),
                "positivePct": row.get("positivePct"),
            }
            for row in trained
        },
        "drugMlAdmetSafetyTierCounts": dict(Counter(drug_scores["mlAdmetSafetyTier"].astype(str))),
        "candidateMlAdmetSafetyTierCounts": dict(Counter(candidates["mlAdmetSafetyTier"].astype(str))),
        "sotaMlAdmetTierCounts": dict(Counter(candidates["sotaMlAdmetTier"].astype(str))),
        "lowMlAdmetRiskDrugRows": low_drug,
        "lowMlAdmetRiskDrugPct": round(pct(low_drug, len(drug_scores)), 4),
        "highMlAdmetRiskDrugRows": high_drug,
        "highMlAdmetRiskDrugPct": round(pct(high_drug, len(drug_scores)), 4),
        "lowMlAdmetRiskCandidateRows": low_candidates,
        "lowMlAdmetRiskCandidatePct": round(pct(low_candidates, len(candidates)), 4),
        "highMlAdmetRiskCandidateRows": high_candidates,
        "highMlAdmetRiskCandidatePct": round(pct(high_candidates, len(candidates)), 4),
        "medianDrugMlAdmetSafetyScore": round(float(drug_scores["mlAdmetSafetyScore"].median()), 4),
        "medianCandidateMlAdmetSafetyScore": round(float(candidates["mlAdmetCandidateSafetyScore"].median()), 4),
        "top100": {
            "lowMlAdmetRiskRows": int(top100["mlAdmetSafetyTier"].astype(str).str.startswith(("A_", "B_")).sum()),
            "highMlAdmetRiskRows": int(top100["mlAdmetSafetyTier"].astype(str).str.startswith("D_").sum()),
            "knownDrugTargetRows": int(pd.to_numeric(top100.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
            "novelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top100 else 0,
            "tierCounts": dict(Counter(top100["sotaMlAdmetTier"].astype(str))),
            "mlAdmetSafetyTierCounts": dict(Counter(top100["mlAdmetSafetyTier"].astype(str))),
            "uniqueDrugs": int(top100["drugId"].nunique()) if "drugId" in top100 else int(top100["drug"].nunique()),
            "uniqueTargets": int(top100["protein"].nunique()) if "protein" in top100 else int(top100["target"].nunique()),
        },
    }


def markdown(summary: dict[str, Any], metrics: pd.DataFrame, direction_df: pd.DataFrame) -> str:
    lines = [
        "# ML ADMET Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit trains local molecular-fingerprint QSAR models on public TDC ADME/Tox endpoints and scores the FDA drug library plus the final candidate matrix.",
        "",
        "## Summary",
        "",
        f"- Drug rows scored: {summary['drugRows']}",
        f"- Candidate rows scored: {summary['candidateRows']}",
        f"- Trained endpoints: {summary['trainedEndpointCount']} ({', '.join(summary['trainedEndpoints'])})",
        f"- Low or manageable ML ADMET risk in candidates: {summary['lowMlAdmetRiskCandidateRows']} ({pct_str(summary['lowMlAdmetRiskCandidatePct'])})",
        f"- High ML ADMET risk in candidates: {summary['highMlAdmetRiskCandidateRows']} ({pct_str(summary['highMlAdmetRiskCandidatePct'])})",
        f"- Top100 low/manageable risk rows: {summary['top100']['lowMlAdmetRiskRows']}; high-risk rows: {summary['top100']['highMlAdmetRiskRows']}; known rows: {summary['top100']['knownDrugTargetRows']}; novel rows: {summary['top100']['novelRows']}",
        "",
        "## Endpoint Model Metrics",
        "",
        "| Endpoint | Rows | Positive % | AUROC | AP | Balanced accuracy | Role |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics[metrics["status"] == "trained"].sort_values("endpoint").itertuples(index=False):
        lines.append(
            f"| {row.endpoint} | {row.validRows} | {pct_str(row.positivePct)} | {fmt(row.testAuroc)} | "
            f"{fmt(row.testAveragePrecision)} | {fmt(row.testBalancedAccuracyAt05)} | {row.scoreRole} |"
        )
    lines.extend(["", "## Direction Summary", ""])
    lines.extend(
        [
            "| Direction | Candidates | Unique drugs | Median safety score | Low/manageable risk | High risk | Top100 low/manageable |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in direction_df.sort_values("direction").itertuples(index=False):
        lines.append(
            f"| {row.direction} | {row.candidateRows} | {row.uniqueDrugs} | {fmt(row.medianMlAdmetSafetyScore)} | "
            f"{row.lowMlAdmetRiskRows} ({pct_str(row.lowMlAdmetRiskPct)}) | {row.highMlAdmetRiskRows} | "
            f"{row.top100LowMlAdmetRiskRows} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- hERG, DILI, Ames, CYP inhibition, and P-gp interaction are treated as risk or review signals.",
            "- HIA and BBB are pharmacokinetic context signals; they are not automatically treated as toxic effects.",
            "- This is an in silico triage layer. It should be used to prioritize expert review and assay planning, not to make clinical safety claims.",
            "",
            "## Machine-Readable Outputs",
            "",
            "- Drug audit: `outputs/sota_validation/ml_admet/drug_ml_admet_audit.csv`",
            "- Candidate audit: `outputs/sota_validation/ml_admet/candidate_ml_admet_audit.csv`",
            "- Endpoint long table: `outputs/sota_validation/ml_admet/drug_ml_admet_endpoint_scores.csv`",
            "- Integrated final matrix: `outputs/sota_validation/final_prioritization/final_priority_ml_admet_matrix.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local TDC endpoint QSAR models and score ML ADMET risk.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--candidates", default="outputs/sota_validation/final_prioritization/final_priority_sota_context_matrix.csv")
    parser.add_argument("--tdc-path", default="data/external/tdc")
    parser.add_argument("--out-dir", default="outputs/sota_validation/ml_admet")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    drugs = pd.read_csv(root / args.drugs).fillna("")
    candidates = pd.read_csv(root / args.candidates).fillna("")

    metrics: list[dict[str, Any]] = []
    models: dict[str, ExtraTreesClassifier] = {}
    for endpoint in ENDPOINTS:
        try:
            dataset = load_tdc_dataset(endpoint, root / args.tdc_path)
            model, metric = train_endpoint(endpoint, dataset, args.random_state, args.n_bits, args.n_estimators)
            metrics.append(metric)
            if model is not None:
                models[endpoint["endpoint"]] = model
        except Exception as exc:  # noqa: BLE001 - record endpoint-specific feasibility.
            metrics.append(
                {
                    **endpoint,
                    "status": "failed",
                    "failureType": type(exc).__name__,
                    "failureDetail": str(exc)[:800],
                }
            )

    if not models:
        raise RuntimeError("No ML ADMET endpoint models were trained successfully.")

    metrics_df = pd.DataFrame(metrics)
    drug_scores, endpoint_long = score_drugs(drugs, models, metrics, args.n_bits)
    candidate_scores = join_candidates(candidates, drug_scores)
    topk_df = topk_metrics(candidate_scores)
    direction_df = direction_summary(candidate_scores)
    summary = build_summary(drug_scores, candidate_scores, metrics, endpoint_long)

    metrics_df.to_csv(out_dir / "ml_admet_endpoint_model_metrics.csv", index=False)
    drug_scores.to_csv(out_dir / "drug_ml_admet_audit.csv", index=False)
    endpoint_long.to_csv(out_dir / "drug_ml_admet_endpoint_scores.csv", index=False)
    candidate_scores.to_csv(out_dir / "candidate_ml_admet_audit.csv", index=False)
    topk_df.to_csv(out_dir / "ml_admet_topk_metrics.csv", index=False)
    direction_df.to_csv(out_dir / "ml_admet_direction_summary.csv", index=False)
    write_json(out_dir / "ml_admet_summary.json", summary)
    (out_dir / "ML_ADMET_AUDIT.md").write_text(markdown(summary, metrics_df, direction_df), encoding="utf-8")

    candidate_scores.to_csv(final_dir / "final_priority_ml_admet_matrix.csv", index=False)
    candidate_scores.head(300).to_csv(final_dir / "final_priority_ml_admet_top300_expert_shortlist.csv", index=False)
    novel_mask = (
        candidate_scores["noveltyClass"].astype(str).isin(NOVEL_CLASSES)
        if "noveltyClass" in candidate_scores
        else pd.Series(False, index=candidate_scores.index)
    )
    candidate_scores[novel_mask].head(300).to_csv(final_dir / "final_priority_ml_admet_novel_shortlist.csv", index=False)
    candidate_scores[candidate_scores["sotaMlAdmetAction"].isin(["ml_admet_high_risk_review", "ml_admet_moderate_risk_review"])].head(300).to_csv(
        final_dir / "final_priority_ml_admet_risk_review.csv", index=False
    )
    direction_df.to_csv(final_dir / "final_priority_ml_admet_direction_summary.csv", index=False)
    (final_dir / "FINAL_PRIORITY_ML_ADMET_AUDIT.md").write_text(markdown(summary, metrics_df, direction_df), encoding="utf-8")

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
