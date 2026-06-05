from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DIRECTION_KEYWORDS = {
    "oncology": [
        "cancer",
        "carcinoma",
        "tumor",
        "tumour",
        "neoplasm",
        "leukemia",
        "lymphoma",
        "melanoma",
        "glioma",
        "glioblastoma",
        "sarcoma",
        "myeloma",
        "adenocarcinoma",
        "blastoma",
        "malignan",
    ],
    "cardiovascular": [
        "cardiac",
        "heart",
        "cardiomyopathy",
        "myocardial",
        "coronary",
        "atherosclerosis",
        "atherosclerotic",
        "vascular",
        "hypertension",
        "hypertensive",
        "ischemia",
        "ischaemia",
        "infarction",
    ],
    "infectious_disease": [
        "infection",
        "infectious",
        "virus",
        "viral",
        "bacteria",
        "bacterial",
        "tuberculosis",
        "influenza",
        "hiv",
        "hepatitis",
        "sepsis",
        "malaria",
        "parasite",
        "pathogen",
        "pneumonia",
    ],
    "neurology_psychiatry": [
        "alzheimer",
        "parkinson",
        "huntington",
        "epilepsy",
        "seizure",
        "schizophrenia",
        "bipolar",
        "depression",
        "autism",
        "dementia",
        "neuro",
        "brain",
        "amyotrophic",
        "als",
        "stroke",
    ],
    "immunology_inflammation": [
        "inflammatory",
        "inflammation",
        "autoimmune",
        "arthritis",
        "rheumatoid",
        "lupus",
        "asthma",
        "colitis",
        "crohn",
        "psoriasis",
        "allergy",
        "allergic",
        "atopic",
        "dermatitis",
        "immune",
        "sclerosis",
    ],
}


ASPECT_LABELS = {"P": "bp", "F": "mf", "C": "cc"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_gene(value: Any) -> str:
    return str(value or "").strip().upper()


def bounded(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def read_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pairId", "direction", "drug", "target", "protein", "proteinName", "finalPriorityScore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {sorted(missing)}")
    df["targetGeneNorm"] = df["target"].map(normalize_gene)
    return df


def direction_matches(text: str) -> list[str]:
    lower = text.lower()
    matches = []
    for direction, keywords in DIRECTION_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            matches.append(direction)
    return matches


def gene_items(items: Any, max_genes: int) -> list[tuple[str, float]]:
    genes: list[tuple[str, float]] = []
    if not isinstance(items, list):
        return genes
    for item in items[:max_genes]:
        if isinstance(item, (list, tuple)) and item:
            gene = normalize_gene(item[0])
            score = 0.0
            if len(item) > 1:
                try:
                    score = float(item[1])
                except Exception:
                    score = 0.0
            if gene:
                genes.append((gene, score))
        elif isinstance(item, str):
            gene = normalize_gene(item)
            if gene:
                genes.append((gene, 0.0))
    return genes


def build_creeds_direction_sets(creeds_path: Path, out_root: Path, max_genes_per_signature: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    with creeds_path.open("r", encoding="utf-8") as handle:
        signatures = json.load(handle)
    if not isinstance(signatures, list):
        raise ValueError("CREEDS disease signature file is expected to contain a list of signatures.")

    direction_signature_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    per_direction_genes: dict[str, dict[str, Counter[str]]] = {
        direction: {"up": Counter(), "down": Counter()} for direction in DIRECTION_KEYWORDS
    }

    for sig in signatures:
        if not isinstance(sig, dict):
            continue
        name = str(sig.get("disease_name", "") or "")
        cell_type = str(sig.get("cell_type", "") or "")
        text = f"{name} {cell_type}"
        directions = direction_matches(text)
        if not directions:
            continue
        up = gene_items(sig.get("up_genes"), max_genes_per_signature)
        down = gene_items(sig.get("down_genes"), max_genes_per_signature)
        for direction in directions:
            row = {
                "direction": direction,
                "creedsId": sig.get("id", ""),
                "diseaseName": name,
                "cellType": cell_type,
                "organism": sig.get("organism", ""),
                "geoId": sig.get("geo_id", ""),
                "upGeneCount": len(up),
                "downGeneCount": len(down),
            }
            direction_signature_rows.append(row)
            for gene, score in up:
                per_direction_genes[direction]["up"][gene] += 1
                gene_rows.append(
                    {
                        "direction": direction,
                        "gene": gene,
                        "regulation": "up",
                        "creedsId": sig.get("id", ""),
                        "diseaseName": name,
                        "score": score,
                    }
                )
            for gene, score in down:
                per_direction_genes[direction]["down"][gene] += 1
                gene_rows.append(
                    {
                        "direction": direction,
                        "gene": gene,
                        "regulation": "down",
                        "creedsId": sig.get("id", ""),
                        "diseaseName": name,
                        "score": score,
                    }
                )

    signature_df = pd.DataFrame(direction_signature_rows)
    gene_df = pd.DataFrame(gene_rows)

    disease_root = out_root / "data/external/disease_signatures"
    for direction in DIRECTION_KEYWORDS:
        direction_dir = disease_root / direction
        direction_dir.mkdir(parents=True, exist_ok=True)
        if signature_df.empty:
            sig_dir_df = pd.DataFrame()
        else:
            sig_dir_df = signature_df[signature_df["direction"].eq(direction)].sort_values(["diseaseName", "creedsId"])
        sig_dir_df.to_csv(direction_dir / "creeds_direction_signatures.csv", index=False)

        counts = []
        up_counter = per_direction_genes[direction]["up"]
        down_counter = per_direction_genes[direction]["down"]
        all_genes = sorted(set(up_counter) | set(down_counter))
        for gene in all_genes:
            counts.append(
                {
                    "gene": gene,
                    "upSignatureCount": int(up_counter.get(gene, 0)),
                    "downSignatureCount": int(down_counter.get(gene, 0)),
                    "totalSignatureCount": int(up_counter.get(gene, 0) + down_counter.get(gene, 0)),
                }
            )
        counts_df = pd.DataFrame(counts).sort_values(
            ["totalSignatureCount", "upSignatureCount", "gene"],
            ascending=[False, False, True],
        )
        counts_df.to_csv(direction_dir / "creeds_gene_counts.csv", index=False)
        (direction_dir / "creeds_up_genes.txt").write_text(
            "\n".join(gene for gene, _count in up_counter.most_common()) + ("\n" if up_counter else ""),
            encoding="utf-8",
        )
        (direction_dir / "creeds_down_genes.txt").write_text(
            "\n".join(gene for gene, _count in down_counter.most_common()) + ("\n" if down_counter else ""),
            encoding="utf-8",
        )

    return signature_df, gene_df


def parse_reactome(path: Path, target_accessions: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if len(fields) < 6:
                continue
            accession, pathway_id, url, pathway_name, evidence, species = fields[:6]
            if accession not in target_accessions:
                continue
            if species != "Homo sapiens" and not pathway_id.startswith("R-HSA-"):
                continue
            rows.append(
                {
                    "protein": accession,
                    "reactomePathwayId": pathway_id,
                    "reactomePathwayUrl": url,
                    "reactomePathwayName": pathway_name,
                    "reactomeEvidence": evidence,
                }
            )
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(
        columns=["protein", "reactomePathwayId", "reactomePathwayUrl", "reactomePathwayName", "reactomeEvidence"]
    )


def parse_goa(path: Path, target_accessions: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields or fields[0].startswith("!"):
                continue
            if len(fields) < 13:
                continue
            db, accession = fields[0], fields[1]
            if db != "UniProtKB" or accession not in target_accessions:
                continue
            taxon = fields[12]
            if "9606" not in taxon:
                continue
            go_id = fields[4]
            evidence = fields[6]
            aspect = ASPECT_LABELS.get(fields[8], fields[8].lower() or "unknown")
            rows.append(
                {
                    "protein": accession,
                    "geneSymbol": fields[2],
                    "goId": go_id,
                    "goAspect": aspect,
                    "goEvidence": evidence,
                }
            )
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(
        columns=["protein", "geneSymbol", "goId", "goAspect", "goEvidence"]
    )


def join_limited(values: Any, limit: int = 8) -> str:
    vals = [str(v) for v in values if str(v)]
    return "; ".join(vals[:limit])


def build_protein_annotation_summary(candidates: pd.DataFrame, reactome: pd.DataFrame, goa: pd.DataFrame) -> pd.DataFrame:
    proteins = candidates[["protein", "target", "proteinName"]].drop_duplicates("protein").copy()
    if reactome.empty:
        reactome_summary = pd.DataFrame(columns=["protein", "reactomePathwayCount", "reactomeTopPathways", "reactomeTopPathwayIds"])
    else:
        reactome_summary = (
            reactome.groupby("protein")
            .agg(
                reactomePathwayCount=("reactomePathwayId", "nunique"),
                reactomeTopPathways=("reactomePathwayName", lambda x: join_limited(pd.Series(x).dropna().drop_duplicates(), 8)),
                reactomeTopPathwayIds=("reactomePathwayId", lambda x: join_limited(pd.Series(x).dropna().drop_duplicates(), 8)),
            )
            .reset_index()
        )

    if goa.empty:
        go_summary = pd.DataFrame(columns=["protein", "goBpCount", "goMfCount", "goCcCount", "goTopBpIds", "goTopMfIds", "goTopCcIds"])
    else:
        count_table = (
            goa.groupby(["protein", "goAspect"])["goId"]
            .nunique()
            .unstack(fill_value=0)
            .rename(columns={"bp": "goBpCount", "mf": "goMfCount", "cc": "goCcCount"})
            .reset_index()
        )
        for col in ["goBpCount", "goMfCount", "goCcCount"]:
            if col not in count_table:
                count_table[col] = 0
        top_rows = []
        for protein, group in goa.groupby("protein"):
            item = {"protein": protein}
            for aspect, out_col in [("bp", "goTopBpIds"), ("mf", "goTopMfIds"), ("cc", "goTopCcIds")]:
                item[out_col] = join_limited(group[group["goAspect"].eq(aspect)]["goId"].drop_duplicates(), 8)
            top_rows.append(item)
        go_summary = count_table.merge(pd.DataFrame(top_rows), on="protein", how="left")

    out = proteins.merge(reactome_summary, on="protein", how="left").merge(go_summary, on="protein", how="left")
    for col in ["reactomePathwayCount", "goBpCount", "goMfCount", "goCcCount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["reactomeTopPathways", "reactomeTopPathwayIds", "goTopBpIds", "goTopMfIds", "goTopCcIds"]:
        out[col] = out[col].fillna("")
    out["pathwayAnnotationCount"] = out["reactomePathwayCount"] + out["goBpCount"] + out["goMfCount"] + out["goCcCount"]
    return out.sort_values(["pathwayAnnotationCount", "target"], ascending=[False, True])


def build_creeds_gene_summary(signature_df: pd.DataFrame, gene_df: pd.DataFrame) -> pd.DataFrame:
    if gene_df.empty:
        return pd.DataFrame(columns=["direction", "gene", "creedsUpSignatureCount", "creedsDownSignatureCount", "creedsTotalSignatureCount", "creedsMatchedDiseases"])
    counts = (
        gene_df.groupby(["direction", "gene", "regulation"])["creedsId"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={"up": "creedsUpSignatureCount", "down": "creedsDownSignatureCount"})
        .reset_index()
    )
    for col in ["creedsUpSignatureCount", "creedsDownSignatureCount"]:
        if col not in counts:
            counts[col] = 0
    diseases = (
        gene_df.groupby(["direction", "gene"])["diseaseName"]
        .apply(lambda x: join_limited(pd.Series(x).dropna().drop_duplicates(), 8))
        .reset_index(name="creedsMatchedDiseases")
    )
    counts = counts.merge(diseases, on=["direction", "gene"], how="left")
    counts["creedsTotalSignatureCount"] = counts["creedsUpSignatureCount"] + counts["creedsDownSignatureCount"]
    return counts.sort_values(["direction", "creedsTotalSignatureCount", "gene"], ascending=[True, False, True])


def context_tier(score: float, creeds_hit: bool, annotation_count: int) -> str:
    if creeds_hit and annotation_count > 0 and score >= 75:
        return "A_pathway_signature_supported"
    if annotation_count > 0 and score >= 55:
        return "B_pathway_annotated_context"
    if creeds_hit:
        return "B_signature_target_context"
    if annotation_count > 0:
        return "C_pathway_only_context"
    return "D_context_gap"


def build_candidate_audit(candidates: pd.DataFrame, protein_summary: pd.DataFrame, creeds_gene_summary: pd.DataFrame, signature_df: pd.DataFrame) -> pd.DataFrame:
    direction_signature_counts = (
        signature_df.groupby("direction")["creedsId"].nunique().reset_index(name="creedsDirectionSignatureCount")
        if not signature_df.empty
        else pd.DataFrame(columns=["direction", "creedsDirectionSignatureCount"])
    )
    merged = candidates.merge(protein_summary, on=["protein", "target", "proteinName"], how="left")
    merged = merged.merge(
        creeds_gene_summary,
        left_on=["direction", "targetGeneNorm"],
        right_on=["direction", "gene"],
        how="left",
    )
    merged = merged.merge(direction_signature_counts, on="direction", how="left")
    for col in [
        "reactomePathwayCount",
        "goBpCount",
        "goMfCount",
        "goCcCount",
        "pathwayAnnotationCount",
        "creedsUpSignatureCount",
        "creedsDownSignatureCount",
        "creedsTotalSignatureCount",
        "creedsDirectionSignatureCount",
    ]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(0).astype(int)
    for col in [
        "reactomeTopPathways",
        "reactomeTopPathwayIds",
        "goTopBpIds",
        "goTopMfIds",
        "goTopCcIds",
        "creedsMatchedDiseases",
    ]:
        merged[col] = merged.get(col, "").fillna("")
    merged["creedsTargetDirectionHit"] = merged["creedsTotalSignatureCount"].gt(0)
    merged["pathwayAnnotationSupportScore"] = merged["pathwayAnnotationCount"].map(lambda x: 100.0 if x >= 20 else 85.0 if x >= 5 else 65.0 if x >= 1 else 15.0)
    merged["diseaseSignatureSupportScore"] = merged.apply(
        lambda row: 95.0
        if row["creedsTargetDirectionHit"]
        else 55.0
        if row["creedsDirectionSignatureCount"] > 0
        else 15.0,
        axis=1,
    )
    merged["pathwayDiseaseContextScore"] = (
        0.45 * merged["pathwayAnnotationSupportScore"] + 0.55 * merged["diseaseSignatureSupportScore"]
    ).map(lambda x: round(bounded(float(x)), 4))
    merged["pathwayDiseaseContextTier"] = merged.apply(
        lambda row: context_tier(
            float(row["pathwayDiseaseContextScore"]),
            bool(row["creedsTargetDirectionHit"]),
            int(row["pathwayAnnotationCount"]),
        ),
        axis=1,
    )
    selected_cols = [
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "finalPriorityScore",
        "finalPriorityTier",
        "reviewTrack",
        "noveltyClass",
        "reactomePathwayCount",
        "reactomeTopPathways",
        "goBpCount",
        "goMfCount",
        "goCcCount",
        "goTopBpIds",
        "goTopMfIds",
        "goTopCcIds",
        "creedsDirectionSignatureCount",
        "creedsTargetDirectionHit",
        "creedsUpSignatureCount",
        "creedsDownSignatureCount",
        "creedsTotalSignatureCount",
        "creedsMatchedDiseases",
        "pathwayAnnotationSupportScore",
        "diseaseSignatureSupportScore",
        "pathwayDiseaseContextScore",
        "pathwayDiseaseContextTier",
    ]
    available_cols = [col for col in selected_cols if col in merged.columns]
    return merged.sort_values(["finalPriorityScore", "pathwayDiseaseContextScore"], ascending=[False, False])[available_cols]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Pathway and Disease-Signature Context Audit",
        "",
        f"Created UTC: {summary['createdUtc']}",
        "",
        "## Inputs",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Unique targets: {summary['uniqueTargets']}",
        f"- Reactome annotated targets: {summary['reactomeAnnotatedTargets']}",
        f"- GO annotated targets: {summary['goAnnotatedTargets']}",
        f"- CREEDS signatures assigned to directions: {summary['creedsDirectionSignatureRows']}",
        "",
        "## Candidate Context",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidates with Reactome annotation | {summary['candidateReactomeAnnotatedRows']} |",
        f"| Candidates with GO annotation | {summary['candidateGoAnnotatedRows']} |",
        f"| Candidates with CREEDS target-direction hit | {summary['candidateCreedsTargetHitRows']} |",
        f"| Top100 candidates with CREEDS target-direction hit | {summary['top100CreedsTargetHitRows']} |",
        "",
        "## Tier Counts",
        "",
    ]
    for tier, count in summary["tierCounts"].items():
        lines.append(f"- {tier}: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Reactome and GO are used as target function/pathway annotations. CREEDS is used as a disease-expression context layer.",
            "These fields support expert interpretation and candidate triage; they are not treated as direct evidence of therapeutic efficacy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pathway and disease-signature context audit for final BioMaster candidates.")
    parser.add_argument("--candidates", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv"))
    parser.add_argument("--reactome", type=Path, default=Path("data/external/pathways/UniProt2Reactome_All_Levels.txt"))
    parser.add_argument("--goa", type=Path, default=Path("data/external/pathways/goa_human.gaf.gz"))
    parser.add_argument("--creeds", type=Path, default=Path("data/external/disease_signatures/creeds/disease_signatures-v1.0.json"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/sota_validation/pathway_disease_context"))
    parser.add_argument("--max-genes-per-signature", type=int, default=300)
    args = parser.parse_args()

    root = Path.cwd()
    args.outdir.mkdir(parents=True, exist_ok=True)
    final_out = root / "outputs/sota_validation/final_prioritization"
    final_out.mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(args.candidates)
    target_accessions = set(candidates["protein"].dropna().astype(str))

    signature_df, gene_df = build_creeds_direction_sets(args.creeds, root, args.max_genes_per_signature)
    reactome = parse_reactome(args.reactome, target_accessions)
    goa = parse_goa(args.goa, target_accessions)
    protein_summary = build_protein_annotation_summary(candidates, reactome, goa)
    creeds_gene_summary = build_creeds_gene_summary(signature_df, gene_df)
    candidate_audit = build_candidate_audit(candidates, protein_summary, creeds_gene_summary, signature_df)

    signature_df.to_csv(args.outdir / "creeds_direction_signature_summary.csv", index=False)
    creeds_gene_summary.to_csv(args.outdir / "creeds_direction_gene_summary.csv", index=False)
    reactome.to_csv(args.outdir / "reactome_target_pathway_annotations.csv", index=False)
    goa.to_csv(args.outdir / "goa_target_annotations.csv", index=False)
    protein_summary.to_csv(args.outdir / "protein_pathway_annotation_summary.csv", index=False)
    candidate_audit.to_csv(args.outdir / "candidate_pathway_disease_context_audit.csv", index=False)
    candidate_audit.to_csv(final_out / "final_priority_pathway_disease_context_augmented_table.csv", index=False)
    candidate_audit.head(300).to_csv(final_out / "final_priority_pathway_disease_context_top300_expert_shortlist.csv", index=False)

    reactome_annotated = set(reactome["protein"]) if not reactome.empty else set()
    go_annotated = set(goa["protein"]) if not goa.empty else set()
    summary = {
        "createdUtc": utc_now(),
        "inputs": {
            "candidates": str(args.candidates),
            "reactome": str(args.reactome),
            "goa": str(args.goa),
            "creeds": str(args.creeds),
        },
        "candidateRows": int(len(candidates)),
        "uniqueTargets": int(candidates["protein"].nunique()),
        "reactomeAnnotationRows": int(len(reactome)),
        "goAnnotationRows": int(len(goa)),
        "reactomeAnnotatedTargets": int(len(reactome_annotated)),
        "goAnnotatedTargets": int(len(go_annotated)),
        "creedsDirectionSignatureRows": int(len(signature_df)),
        "creedsDirectionGeneRows": int(len(gene_df)),
        "candidateReactomeAnnotatedRows": int(candidate_audit["reactomePathwayCount"].gt(0).sum()),
        "candidateGoAnnotatedRows": int((candidate_audit["goBpCount"] + candidate_audit["goMfCount"] + candidate_audit["goCcCount"]).gt(0).sum()),
        "candidateCreedsTargetHitRows": int(candidate_audit["creedsTargetDirectionHit"].astype(bool).sum()),
        "top100CreedsTargetHitRows": int(candidate_audit.head(100)["creedsTargetDirectionHit"].astype(bool).sum()),
        "tierCounts": candidate_audit["pathwayDiseaseContextTier"].value_counts().to_dict(),
        "directionSignatureCounts": signature_df["direction"].value_counts().to_dict() if not signature_df.empty else {},
        "outputs": {
            "candidateAudit": str(args.outdir / "candidate_pathway_disease_context_audit.csv"),
            "proteinSummary": str(args.outdir / "protein_pathway_annotation_summary.csv"),
            "creedsSignatureSummary": str(args.outdir / "creeds_direction_signature_summary.csv"),
            "creedsGeneSummary": str(args.outdir / "creeds_direction_gene_summary.csv"),
            "finalAugmentedTable": str(final_out / "final_priority_pathway_disease_context_augmented_table.csv"),
            "finalTop300": str(final_out / "final_priority_pathway_disease_context_top300_expert_shortlist.csv"),
        },
        "methodNote": "Reactome/GO are target pathway annotations; CREEDS is mapped by disease-name keyword to project directions and used only as disease-expression context.",
    }
    (args.outdir / "pathway_disease_context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.outdir / "PATHWAY_DISEASE_CONTEXT_AUDIT.md", summary)
    write_markdown(final_out / "FINAL_PRIORITY_PATHWAY_DISEASE_CONTEXT_AUDIT.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
