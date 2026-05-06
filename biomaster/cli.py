from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    build_drug_library,
    build_protein_library,
    convert_conplex_predictions,
    make_screening_manifest,
    make_conplex_input,
    merge_affinity_scores,
    rank_disease_relevance,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BioMaster first-five-step screening pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    drugs = subparsers.add_parser("build-drugs", help="Build the small-molecule library")
    drugs.add_argument("--seed-csv", dest="seed_table", help="Seed table path (.csv, .tsv, .xlsx, or .xlsm)")
    drugs.add_argument("--seed-table", dest="seed_table", help=argparse.SUPPRESS)
    drugs.add_argument("--out", required=True)
    drugs.add_argument("--sdf-dir")
    drugs.add_argument("--year-min", type=int)
    drugs.add_argument("--limit", type=int)
    drugs.add_argument("--no-pubchem", action="store_true")
    drugs.add_argument("--strict-external", action="store_true")

    proteins = subparsers.add_parser("build-proteins", help="Build the protein target library")
    proteins.add_argument("--out", required=True)
    proteins.add_argument("--source-csv")
    proteins.add_argument("--limit", type=int)
    proteins.add_argument("--alphafold", action="store_true")
    proteins.add_argument("--no-uniprot", action="store_true")
    proteins.add_argument("--strict-external", action="store_true")

    manifest = subparsers.add_parser("make-screening-manifest", help="Create DiffDock screening pairs")
    manifest.add_argument("--drugs", required=True)
    manifest.add_argument("--proteins", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--output-prefix", default="runs/diffdock")
    manifest.add_argument("--limit", type=int)

    conplex_input = subparsers.add_parser("make-conplex-input", help="Create ConPLex prediction input TSV")
    conplex_input.add_argument("--manifest", required=True)
    conplex_input.add_argument("--out", required=True)
    conplex_input.add_argument("--limit", type=int)
    conplex_input.add_argument("--keep-missing", action="store_true")

    conplex_predictions = subparsers.add_parser(
        "convert-conplex-predictions",
        help="Convert ConPLex prediction TSV to affinity_scores.csv format",
    )
    conplex_predictions.add_argument("--predictions", required=True)
    conplex_predictions.add_argument("--out", required=True)
    conplex_predictions.add_argument("--model", default="ConPLex")

    affinity = subparsers.add_parser("merge-affinity", help="Merge DiffDock and affinity model scores")
    affinity.add_argument("--manifest", required=True)
    affinity.add_argument("--docking-scores", required=True)
    affinity.add_argument("--affinity-scores", required=True)
    affinity.add_argument("--out", required=True)
    affinity.add_argument("--top-n", type=int)

    disease = subparsers.add_parser("rank-disease", help="Rank candidates by disease relevance")
    disease.add_argument("--candidates", required=True)
    disease.add_argument("--string", required=True)
    disease.add_argument("--disgenet", required=True)
    disease.add_argument("--out", required=True)
    disease.add_argument("--disease-id")
    disease.add_argument("--top-n", type=int, default=100)

    demo = subparsers.add_parser("run-demo", help="Run the bundled first-five-step demo")
    demo.add_argument("--out", default="outputs/demo")
    demo.add_argument("--offline", action="store_true", help="Use only seed fields; skip external API lookups")

    args = parser.parse_args(argv)

    if args.command == "build-drugs":
        if not args.seed_table:
            parser.error("build-drugs requires --seed-csv or --seed-table")
        rows = build_drug_library(
            seed_csv=args.seed_table,
            out_csv=args.out,
            sdf_dir=args.sdf_dir,
            year_min=args.year_min,
            fetch_pubchem=not args.no_pubchem,
            limit=args.limit,
            strict_external=args.strict_external,
        )
        print(f"wrote {len(rows)} drugs to {args.out}")
        return 0

    if args.command == "build-proteins":
        rows = build_protein_library(
            out_csv=args.out,
            source_csv=args.source_csv,
            limit=args.limit,
            fetch_uniprot=not args.no_uniprot,
            include_alphafold=args.alphafold,
            strict_external=args.strict_external,
        )
        print(f"wrote {len(rows)} proteins to {args.out}")
        return 0

    if args.command == "make-screening-manifest":
        rows = make_screening_manifest(
            drugs_csv=args.drugs,
            proteins_csv=args.proteins,
            out_csv=args.out,
            output_prefix=args.output_prefix,
            limit=args.limit,
        )
        print(f"wrote {len(rows)} pairs to {args.out}")
        return 0

    if args.command == "make-conplex-input":
        rows = make_conplex_input(
            manifest_csv=args.manifest,
            out_tsv=args.out,
            limit=args.limit,
            skip_missing=not args.keep_missing,
        )
        print(f"wrote {len(rows)} ConPLex input pairs to {args.out}")
        return 0

    if args.command == "convert-conplex-predictions":
        rows = convert_conplex_predictions(
            predictions_tsv=args.predictions,
            out_csv=args.out,
            model=args.model,
        )
        print(f"wrote {len(rows)} ConPLex affinity scores to {args.out}")
        return 0

    if args.command == "merge-affinity":
        rows = merge_affinity_scores(
            manifest_csv=args.manifest,
            docking_scores_csv=args.docking_scores,
            affinity_scores_csv=args.affinity_scores,
            out_csv=args.out,
            top_n=args.top_n,
        )
        print(f"wrote {len(rows)} stage-4 candidates to {args.out}")
        return 0

    if args.command == "rank-disease":
        rows = rank_disease_relevance(
            candidates_csv=args.candidates,
            string_csv=args.string,
            disgenet_csv=args.disgenet,
            out_csv=args.out,
            disease_id=args.disease_id,
            top_n=args.top_n,
        )
        print(f"wrote {len(rows)} stage-5 ranked candidates to {args.out}")
        return 0

    if args.command == "run-demo":
        run_demo(Path(args.out), offline=args.offline)
        return 0

    return 2


def run_demo(out_dir: Path, offline: bool = False) -> None:
    examples = Path("examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    drug_library = out_dir / "drug_library.csv"
    protein_library = out_dir / "protein_library.csv"
    manifest = out_dir / "diffdock_manifest.csv"
    stage4 = out_dir / "stage4_affinity_candidates.csv"
    stage5 = out_dir / "stage5_disease_ranked_candidates.csv"

    build_drug_library(
        seed_csv=examples / "drug_seed.csv",
        out_csv=drug_library,
        sdf_dir=None if offline else out_dir / "structures" / "drugs",
        year_min=2016,
        fetch_pubchem=not offline,
    )
    build_protein_library(
        out_csv=protein_library,
        source_csv=examples / "protein_seed.csv",
        fetch_uniprot=not offline,
        include_alphafold=not offline,
    )
    make_screening_manifest(
        drugs_csv=drug_library,
        proteins_csv=protein_library,
        out_csv=manifest,
        output_prefix=str(out_dir / "diffdock_runs"),
    )
    merge_affinity_scores(
        manifest_csv=manifest,
        docking_scores_csv=examples / "docking_scores.csv",
        affinity_scores_csv=examples / "affinity_scores.csv",
        out_csv=stage4,
        top_n=1000,
    )
    rank_disease_relevance(
        candidates_csv=stage4,
        string_csv=examples / "string_edges.csv",
        disgenet_csv=examples / "disgenet_gene_disease.csv",
        disease_id="C0006826",
        out_csv=stage5,
        top_n=100,
    )
    print(f"demo complete: {stage5}")


if __name__ == "__main__":
    raise SystemExit(main())
