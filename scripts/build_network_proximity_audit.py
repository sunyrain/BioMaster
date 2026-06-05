from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}

DIRECTION_DISEASE_NAMES = {
    "oncology": ["neoplasm", "cancer"],
    "infectious_disease": ["infectious disease"],
    "cardiovascular": ["cardiovascular disease"],
    "neurology_psychiatry": [
        "nervous system disease",
        "psychiatric disorder",
        "mental or behavioural disorder",
    ],
    "immunology_inflammation": [
        "immune system disease",
        "inflammatory skin disease",
        "inflammatory bowel disease",
    ],
}


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_gene(value: Any) -> str:
    return str(value or "").strip().upper()


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in edges.iterrows():
        a = norm_gene(row.get("preferredName_A"))
        b = norm_gene(row.get("preferredName_B"))
        if not a or not b or a == b:
            continue
        score = number(row.get("score")) or 0.0
        if score > 1.0:
            score = score / 1000.0
        score = max(1e-6, min(1.0, score))
        length = 1.0 / score
        current = graph.get_edge_data(a, b)
        if current is None or score > current.get("score", 0.0):
            graph.add_edge(a, b, score=score, length=length)
    return graph


def multi_source_hop_lengths(graph: nx.Graph, seeds: list[str]) -> dict[str, int]:
    distances: dict[str, int] = {}
    frontier: list[str] = []
    for seed in seeds:
        if seed in graph and seed not in distances:
            distances[seed] = 0
            frontier.append(seed)
    cursor = 0
    while cursor < len(frontier):
        node = frontier[cursor]
        cursor += 1
        next_distance = distances[node] + 1
        for neighbor in graph.neighbors(node):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)
    return distances


def multi_source_weighted_lengths(graph: nx.Graph, seeds: list[str], weight: str = "length") -> dict[str, float]:
    distances: dict[str, float] = {}
    heap: list[tuple[float, str]] = []
    for seed in seeds:
        if seed in graph:
            heapq.heappush(heap, (0.0, seed))
    while heap:
        distance, node = heapq.heappop(heap)
        if node in distances:
            continue
        distances[node] = distance
        for neighbor, attrs in graph[node].items():
            if neighbor in distances:
                continue
            step = number(attrs.get(weight)) or 1.0
            heapq.heappush(heap, (distance + step, neighbor))
    return distances


def direction_ot_rows(opentargets: pd.DataFrame, direction: str) -> pd.DataFrame:
    names = {item.lower() for item in DIRECTION_DISEASE_NAMES.get(direction, [])}
    rows = opentargets[opentargets["disease_name"].str.lower().isin(names)].copy()
    rows["geneNorm"] = rows["gene_name"].map(norm_gene)
    rows["overallScoreNumeric"] = pd.to_numeric(rows["overall_score"], errors="coerce").fillna(0.0)
    rows = rows[rows["geneNorm"].astype(str).str.len() > 0].copy()
    return rows


def build_direct_scores(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    direct: dict[str, dict[str, Any]] = {}
    for gene, group in rows.groupby("geneNorm"):
        best = group.sort_values("overallScoreNumeric", ascending=False).iloc[0]
        direct[gene] = {
            "score": float(best["overallScoreNumeric"]),
            "diseaseName": best.get("disease_name", ""),
            "diseaseId": best.get("disease_id", ""),
            "approvedName": best.get("approved_name", ""),
        }
    return direct


def select_seed_genes(rows: pd.DataFrame, graph_nodes: set[str], max_seed_genes: int, min_score: float) -> list[str]:
    if rows.empty:
        return []
    grouped = (
        rows.groupby("geneNorm", as_index=False)["overallScoreNumeric"]
        .max()
        .sort_values("overallScoreNumeric", ascending=False)
    )
    grouped = grouped[grouped["overallScoreNumeric"] >= min_score].copy()
    if max_seed_genes > 0:
        grouped = grouped.head(max_seed_genes)
    return [gene for gene in grouped["geneNorm"].tolist() if gene in graph_nodes]


def empirical_percentile(distance: float | None, background: list[float]) -> float | None:
    if distance is None or not background:
        return None
    return sum(1 for item in background if item >= distance) / len(background) * 100.0


def z_score_closer(distance: float | None, background: list[float]) -> float | None:
    if distance is None or len(background) < 2:
        return None
    mean = sum(background) / len(background)
    variance = sum((item - mean) ** 2 for item in background) / (len(background) - 1)
    std = math.sqrt(variance)
    if std <= 0:
        return None
    return (mean - distance) / std


def evidence_tier(direct_score: float, target_in_graph: bool, hops: int | None, percentile: float | None) -> tuple[str, str]:
    if direct_score >= 0.5:
        return "A_direct_disease_module", "Open Targets direct disease-module target"
    if direct_score > 0.0:
        return "B_direct_low_score", "Open Targets direct low-score target"
    if not target_in_graph:
        return "U_uncovered_by_string", "Target absent from current STRING subnet"
    if hops == 0:
        return "A_direct_disease_module", "Target is in disease module seed set"
    if hops == 1 or (percentile is not None and percentile >= 90):
        return "B_network_close", "One-hop or top-decile PPI proximity to disease module"
    if hops == 2 or (percentile is not None and percentile >= 70):
        return "C_network_reachable", "Reachable PPI proximity to disease module"
    if hops is not None:
        return "D_network_distant", "Reachable but distant from disease module in current subnet"
    return "U_unreachable_in_string", "Target in STRING subnet but disconnected from disease module"


def build_direction_models(
    graph: nx.Graph,
    opentargets: pd.DataFrame,
    max_seed_genes: int,
    min_seed_score: float,
) -> dict[str, dict[str, Any]]:
    graph_nodes = set(graph.nodes)
    models: dict[str, dict[str, Any]] = {}
    for direction in DIRECTION_LABELS:
        rows = direction_ot_rows(opentargets, direction)
        direct_scores = build_direct_scores(rows)
        seeds = select_seed_genes(rows, graph_nodes, max_seed_genes, min_seed_score)
        if seeds:
            hop_lengths = multi_source_hop_lengths(graph, seeds)
            weighted_lengths = multi_source_weighted_lengths(graph, seeds, weight="length")
        else:
            hop_lengths = {}
            weighted_lengths = {}
        background_hops = [float(value) for node, value in hop_lengths.items() if node not in set(seeds)]
        background_weighted = [float(value) for node, value in weighted_lengths.items() if node not in set(seeds)]
        models[direction] = {
            "otRows": rows,
            "directScores": direct_scores,
            "seedGenesInOpenTargets": int(rows["geneNorm"].nunique()),
            "seedGenesInString": len(seeds),
            "seedGenes": seeds,
            "hopLengths": hop_lengths,
            "weightedLengths": weighted_lengths,
            "backgroundHops": background_hops,
            "backgroundWeighted": background_weighted,
            "reachableStringNodes": len(hop_lengths),
        }
    return models


def target_rows(candidates: pd.DataFrame, models: dict[str, dict[str, Any]], graph: nx.Graph) -> pd.DataFrame:
    graph_nodes = set(graph.nodes)
    unique = candidates[["direction", "target", "protein", "proteinName"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    for _, item in unique.iterrows():
        direction = item.get("direction", "")
        target = norm_gene(item.get("target"))
        model = models.get(direction, {})
        direct_info = (model.get("directScores") or {}).get(target, {})
        direct_score = float(direct_info.get("score") or 0.0)
        target_in_graph = target in graph_nodes
        hops = (model.get("hopLengths") or {}).get(target)
        weighted = (model.get("weightedLengths") or {}).get(target)
        hop_percentile = empirical_percentile(float(hops) if hops is not None else None, model.get("backgroundHops") or [])
        weighted_z = z_score_closer(float(weighted) if weighted is not None else None, model.get("backgroundWeighted") or [])
        tier, reason = evidence_tier(direct_score, target_in_graph, hops, hop_percentile)
        proximity_score = 0.0
        if direct_score > 0:
            proximity_score = max(proximity_score, min(1.0, 0.65 + 0.35 * direct_score))
        if hops is not None:
            proximity_score = max(proximity_score, 1.0 / (1.0 + float(hops)))
        if hop_percentile is not None:
            proximity_score = max(proximity_score, hop_percentile / 100.0)
        rows.append(
            {
                "direction": direction,
                "directionLabelZh": DIRECTION_LABELS.get(direction, direction),
                "target": target,
                "protein": item.get("protein", ""),
                "proteinName": item.get("proteinName", ""),
                "stringCovered": int(target_in_graph),
                "directOpenTargetsScore": round(direct_score, 9),
                "directDiseaseName": direct_info.get("diseaseName", ""),
                "directDiseaseId": direct_info.get("diseaseId", ""),
                "shortestHopToDiseaseModule": "" if hops is None else int(hops),
                "weightedDistanceToDiseaseModule": "" if weighted is None else round(float(weighted), 6),
                "networkProximityPercentile": "" if hop_percentile is None else round(float(hop_percentile), 3),
                "networkProximityZ": "" if weighted_z is None else round(float(weighted_z), 4),
                "networkProximityScore": round(float(proximity_score), 6),
                "networkEvidenceTier": tier,
                "networkEvidenceReason": reason,
                "diseaseModuleSeedGenesInOpenTargets": model.get("seedGenesInOpenTargets", 0),
                "diseaseModuleSeedGenesInString": model.get("seedGenesInString", 0),
                "reachableStringNodesFromModule": model.get("reachableStringNodes", 0),
            }
        )
    return pd.DataFrame(rows)


def candidate_rows(candidates: pd.DataFrame, target_audit: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "direction",
        "target",
        "protein",
        "stringCovered",
        "directOpenTargetsScore",
        "directDiseaseName",
        "shortestHopToDiseaseModule",
        "weightedDistanceToDiseaseModule",
        "networkProximityPercentile",
        "networkProximityZ",
        "networkProximityScore",
        "networkEvidenceTier",
        "networkEvidenceReason",
    ]
    joined = candidates.merge(target_audit[cols], on=["direction", "target", "protein"], how="left")
    return joined


def final_priority_rows(final_priority: pd.DataFrame, target_audit: pd.DataFrame) -> pd.DataFrame:
    if final_priority.empty:
        return pd.DataFrame()
    cols = [
        "direction",
        "target",
        "protein",
        "stringCovered",
        "directOpenTargetsScore",
        "directDiseaseName",
        "shortestHopToDiseaseModule",
        "weightedDistanceToDiseaseModule",
        "networkProximityPercentile",
        "networkProximityZ",
        "networkProximityScore",
        "networkEvidenceTier",
        "networkEvidenceReason",
    ]
    return final_priority.merge(target_audit[cols], on=["direction", "target", "protein"], how="left")


def direction_summary(candidates: pd.DataFrame, target_audit: pd.DataFrame, models: dict[str, dict[str, Any]], graph: nx.Graph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, group in candidates.groupby("direction"):
        target_group = target_audit[target_audit["direction"].eq(direction)]
        covered_targets = int(target_group["stringCovered"].sum()) if "stringCovered" in target_group else 0
        direct_targets = int((pd.to_numeric(target_group["directOpenTargetsScore"], errors="coerce").fillna(0) > 0).sum())
        close_targets = int(target_group["networkEvidenceTier"].isin(["A_direct_disease_module", "B_direct_low_score", "B_network_close"]).sum())
        rows.append(
            {
                "direction": direction,
                "directionLabelZh": DIRECTION_LABELS.get(direction, direction),
                "candidateRows": int(len(group)),
                "uniqueTargets": int(target_group["target"].nunique()),
                "stringCoveredTargets": covered_targets,
                "stringCoveredTargetPct": round(pct(covered_targets, target_group["target"].nunique()), 3),
                "candidateRowsStringCovered": int(group["target"].map(norm_gene).isin(set(graph.nodes)).sum()),
                "candidateRowsStringCoveredPct": round(pct(int(group["target"].map(norm_gene).isin(set(graph.nodes)).sum()), len(group)), 3),
                "directOpenTargetsTargets": direct_targets,
                "networkCloseOrDirectTargets": close_targets,
                "targetTierCounts": dict(Counter(target_group["networkEvidenceTier"])),
                "diseaseModuleSeedGenesInOpenTargets": models[direction]["seedGenesInOpenTargets"],
                "diseaseModuleSeedGenesInString": models[direction]["seedGenesInString"],
                "reachableStringNodesFromModule": models[direction]["reachableStringNodes"],
            }
        )
    return pd.DataFrame(rows)


def summary_payload(
    candidates: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    target_audit: pd.DataFrame,
    direction_audit: pd.DataFrame,
    final_audit: pd.DataFrame,
    graph: nx.Graph,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_coverage = target_audit[["target", "protein", "stringCovered"]].copy()
    target_coverage["stringCovered"] = target_coverage["stringCovered"].fillna(0).astype(int)
    target_coverage = target_coverage.groupby(["target", "protein"], as_index=False)["stringCovered"].max()
    unique_targets_total = int(target_coverage[["target", "protein"]].drop_duplicates().shape[0])
    unique_targets_covered = int(target_coverage["stringCovered"].sum())
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(candidate_audit)),
        "uniqueCandidateTargets": int(candidates["target"].map(norm_gene).nunique()),
        "targetAuditRows": int(len(target_audit)),
        "finalPriorityRows": int(len(final_audit)) if not final_audit.empty else 0,
        "stringGraphNodes": int(graph.number_of_nodes()),
        "stringGraphEdges": int(graph.number_of_edges()),
        "candidateRowsStringCovered": int(candidate_audit["stringCovered"].fillna(0).astype(int).sum()),
        "candidateRowsStringCoveredPct": pct(int(candidate_audit["stringCovered"].fillna(0).astype(int).sum()), len(candidate_audit)),
        "uniqueTargetsStringCovered": unique_targets_covered,
        "uniqueTargetsStringCoveredPct": pct(unique_targets_covered, unique_targets_total),
        "uniqueTargetProteinPairs": unique_targets_total,
        "candidateTierCounts": dict(Counter(candidate_audit["networkEvidenceTier"].fillna("NA"))),
        "targetTierCounts": dict(Counter(target_audit["networkEvidenceTier"].fillna("NA"))),
        "finalPriorityTierCounts": dict(Counter(final_audit["networkEvidenceTier"].fillna("NA"))) if not final_audit.empty else {},
        "directionSummary": direction_audit.to_dict("records"),
        "inputs": {
            "candidates": args.candidates,
            "finalPriority": args.final_priority,
            "opentargets": args.opentargets,
            "string": args.string,
            "maxSeedGenes": args.max_seed_genes,
            "minSeedScore": args.min_seed_score,
        },
        "outputs": {
            "candidateAudit": f"{args.out_dir}/candidate_network_proximity_audit.csv",
            "targetAudit": f"{args.out_dir}/target_network_proximity_audit.csv",
            "directionSummary": f"{args.out_dir}/network_proximity_direction_summary.csv",
            "finalPriorityAudit": f"{args.out_dir}/final_priority_network_proximity_audit.csv",
            "summary": f"{args.out_dir}/network_proximity_summary.json",
            "markdown": f"{args.out_dir}/NETWORK_PROXIMITY_AUDIT.md",
        },
        "methodNote": (
            "Coverage-aware network medicine audit. Disease modules are Open Targets genes for each disease direction; "
            "candidate targets are scored by direct Open Targets disease evidence and shortest-path proximity in the current STRING high-confidence subnet. "
            "Because the local STRING subnet is intentionally filtered, uncovered targets are marked as coverage gaps rather than negative biological evidence."
        ),
    }


def markdown(summary: dict[str, Any], target_audit: pd.DataFrame) -> str:
    lines = [
        "# Network Medicine / PPI Proximity Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit adds an orthogonal network-medicine evidence layer to the ConPLex, Open Targets, TxGNN, DiffDock, and ADMET stack.",
        "",
        "## Scope",
        "",
        f"- Candidate rows audited: {summary['candidateRows']}",
        f"- Unique candidate targets: {summary['uniqueCandidateTargets']}",
        f"- STRING graph: {summary['stringGraphNodes']} nodes, {summary['stringGraphEdges']} edges",
        f"- Candidate-row STRING coverage: {summary['candidateRowsStringCoveredPct']:.2f}%",
        f"- Unique-target STRING coverage: {summary['uniqueTargetsStringCoveredPct']:.2f}%",
        f"- Candidate evidence tiers: {summary['candidateTierCounts']}",
        "",
        "## Direction Summary",
        "",
    ]
    for row in summary["directionSummary"]:
        lines.append(
            f"- {row['directionLabelZh']} ({row['direction']}): candidates={row['candidateRows']}, "
            f"uniqueTargets={row['uniqueTargets']}, STRING-covered targets={row['stringCoveredTargets']} "
            f"({row['stringCoveredTargetPct']:.1f}%), disease seeds in STRING={row['diseaseModuleSeedGenesInString']}, "
            f"tiers={row['targetTierCounts']}"
        )
    lines.extend(
        [
            "",
            "## Top Direct Or Network-Close Targets",
            "",
        ]
    )
    sort_frame = target_audit.copy()
    sort_frame["_score"] = pd.to_numeric(sort_frame["networkProximityScore"], errors="coerce").fillna(0)
    for _, row in sort_frame.sort_values(["direction", "_score"], ascending=[True, False]).groupby("direction").head(8).iterrows():
        lines.append(
            f"- {row['directionLabelZh']} | {row['target']} ({row['protein']}): "
            f"tier={row['networkEvidenceTier']}, directOT={row['directOpenTargetsScore']}, "
            f"hop={row['shortestHopToDiseaseModule']}, percentile={row['networkProximityPercentile']}. "
            f"{row['networkEvidenceReason']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Direct/network-close results are positive orthogonal evidence for expert triage.",
            "- Uncovered targets are not negative results; they indicate the local STRING subnet is too narrow for that target.",
            "- This layer should be expanded with a broader STRING/HuRI/BioGRID interactome before making final network-medicine claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a coverage-aware PPI/network-proximity audit for SOTA validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidates", default="outputs/disease_directions/disease_direction_integrated_candidates.csv")
    parser.add_argument("--final-priority", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--opentargets", default="outputs/sota_validation/opentargets_direction_scores_top_pages.csv")
    parser.add_argument("--string", default="data/processed/string_human_filtered_edges.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/network_proximity")
    parser.add_argument("--max-seed-genes", type=int, default=300)
    parser.add_argument("--min-seed-score", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_frame(root / args.candidates)
    if "target" not in candidates.columns or "direction" not in candidates.columns:
        raise ValueError("Candidate table must contain direction and target columns")
    candidates["target"] = candidates["target"].map(norm_gene)

    final_priority_path = root / args.final_priority
    final_priority = read_frame(final_priority_path) if final_priority_path.exists() else pd.DataFrame()
    if not final_priority.empty and "target" in final_priority.columns:
        final_priority["target"] = final_priority["target"].map(norm_gene)

    opentargets = read_frame(root / args.opentargets)
    edges = read_frame(root / args.string)
    graph = build_graph(edges)
    models = build_direction_models(graph, opentargets, args.max_seed_genes, args.min_seed_score)

    target_audit = target_rows(candidates, models, graph)
    candidate_audit = candidate_rows(candidates, target_audit)
    final_audit = final_priority_rows(final_priority, target_audit) if not final_priority.empty else pd.DataFrame()
    direction_audit = direction_summary(candidates, target_audit, models, graph)
    summary = summary_payload(candidates, candidate_audit, target_audit, direction_audit, final_audit, graph, args)

    target_audit.to_csv(out_dir / "target_network_proximity_audit.csv", index=False)
    candidate_audit.to_csv(out_dir / "candidate_network_proximity_audit.csv", index=False)
    direction_audit.to_csv(out_dir / "network_proximity_direction_summary.csv", index=False)
    if not final_audit.empty:
        final_audit.to_csv(out_dir / "final_priority_network_proximity_audit.csv", index=False)
    write_json(out_dir / "network_proximity_summary.json", summary)
    (out_dir / "NETWORK_PROXIMITY_AUDIT.md").write_text(markdown(summary, target_audit), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
