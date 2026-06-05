from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0"
    return f"{numerator / denominator * 100:.1f}"


def candidate_rows(rows: list[dict[str, str]], limit: int) -> str:
    lines: list[str] = []
    for row in rows[:limit]:
        lines.append(
            " & ".join(
                [
                    tex_escape(row.get("stage6_unique_rank", "")),
                    tex_escape(row.get("drug_name", ""))[:46],
                    tex_escape(row.get("gene_name", "")),
                    tex_escape(row.get("protein_id", "")),
                    tex_escape(row.get("affinity_score", "")),
                    tex_escape(row.get("diffdock_confidence", "") or "NA"),
                    tex_escape(row.get("stage6_consensus_score", "")),
                    tex_escape(row.get("structural_status", "")),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> Path:
    unique_csv = Path(args.unique_csv)
    top_csv = Path(args.top_csv)
    prep_metadata = load_json(Path(args.prep_metadata))
    merge_metadata = load_json(Path(args.merge_metadata))
    unique_rows = read_csv(unique_csv)
    top_rows = read_csv(top_csv)
    status_counts = Counter(row.get("structural_status", "") for row in unique_rows)
    receptor_counts = Counter(row.get("diffdock_receptor_status", "") for row in unique_rows)

    completed = status_counts.get("completed", 0)
    missing = len(unique_rows) - completed
    top_completed = sum(1 for row in top_rows if row.get("structural_status") == "completed")

    report = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{BioMaster Druggable Proteome Screening Report}}
\author{{Computational Drug-Target Prioritization Workflow}}
\date{{Updated from local run artifacts: {tex_escape(merge_metadata.get("created_utc", ""))}}}

\begin{{document}}
\maketitle

\section*{{Executive Summary}}
This report summarizes a computational screen of FDA-approved small molecules against the ChEMBL druggable proteome. The workflow first predicts drug-protein affinity with ConPLex, then performs structure-level prioritization with AlphaFold receptor preparation and DiffDock docking for high-priority representatives.

The outputs are intended for candidate prioritization and expert review. They are not claims of clinical efficacy or experimentally confirmed binding.

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
Usable FDA-approved small molecules & {prep_metadata["drug_rows_usable"]} \\
Valid ChEMBL druggable protein records & {prep_metadata["protein_rows_input_valid"]} \\
Unique protein sequences & {prep_metadata["unique_sequences"]} \\
Full ConPLex affinity matrix & {prep_metadata["expanded_protein_drug_pairs"]} \\
Top affinity candidates & {len(top_rows)} \\
DiffDock representative complexes & {len(unique_rows)} \\
Completed DiffDock representatives & {completed} ({fmt_pct(completed, len(unique_rows))}\%) \\
Missing or failed DiffDock representatives & {missing} ({fmt_pct(missing, len(unique_rows))}\%) \\
Top10000 rows with mapped completed structure & {top_completed} ({fmt_pct(top_completed, len(top_rows))}\%) \\
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{Workflow}}
\begin{{enumerate}}
\item Standardize the FDA-approved small-molecule library and ChEMBL druggable protein table.
\item Collapse identical protein sequences to avoid redundant ConPLex inference.
\item Predict affinity for 815,265 unique drug-sequence pairs and expand scores back to 4,854,990 drug-target pairs.
\item Select the Top10000 affinity candidates.
\item Collapse Top10000 candidates by drug and protein sequence, yielding 1,872 representative docking jobs.
\item Prepare AlphaFold receptor structures and run DiffDock to generate rank-1 ligand poses and confidence scores.
\item Merge DiffDock results back to both the representative table and the full Top10000 candidate table.
\end{{enumerate}}

\begin{{figure}}[h]
\centering
\includegraphics[width=\textwidth]{{docs/assets/biomaster-main-figure.png}}
\caption{{BioMaster screening concept: small-molecule library, druggable protein targets, AI-DTI affinity scoring, structure-level docking, and prioritized candidate outputs.}}
\end{{figure}}

\section*{{Model Roles}}
\textbf{{ConPLex}} is used for broad drug-target affinity prioritization from chemical and protein-sequence representations. Its score is a ranking signal, not a measured binding constant.

\textbf{{AlphaFold}} provides receptor structures for selected target representatives. Long receptors are cropped only when required for DiffDock feasibility.

\textbf{{DiffDock}} generates rank-1 ligand poses and confidence scores for high-priority representatives. The confidence score is used as structure-level supporting evidence.

\section*{{Representative Results}}
\scriptsize
\begin{{longtable}}{{r p{{0.27\linewidth}} l l r r r l}}
\toprule
Rank & Drug & Gene & Protein & Affinity & Docking & Consensus & Status \\
\midrule
\endfirsthead
\toprule
Rank & Drug & Gene & Protein & Affinity & Docking & Consensus & Status \\
\midrule
\endhead
{candidate_rows(unique_rows, args.top_n)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Docking Audit}}
\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Status & Count \\
\midrule
Completed & {completed} \\
Missing output & {missing} \\
\bottomrule
\end{{tabular}}
\hspace{{0.5in}}
\begin{{tabular}}{{lr}}
\toprule
Receptor preparation status & Count \\
\midrule
"""
    for label, count in receptor_counts.most_common():
        report += f"{tex_escape(label)} & {count} \\\\\n"
    report += r"""\bottomrule
\end{tabular}
\end{center}

\section*{Interpretation}
Open Targets is not required for the current primary screen. It is a target-disease association resource and should be reintroduced when the analysis is focused on a specific disease, cancer type, mutation, or molecular subtype. In the current druggable-proteome workflow, the first objective is to identify high-affinity drug-target candidates within a pharmacologically relevant target space.

\section*{Recommended Next Step}
The next practical step is to review the structure-enhanced Top100 candidates, remove pairs with weak biological rationale or problematic docking output, and select 20--50 candidates for deeper expert review. A smaller 5--20 candidate subset can then be chosen for literature review, orthogonal docking, or experimental validation.

\end{document}
"""

    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(report.strip() + "\n", encoding="utf-8")
    print(json.dumps({"out_tex": str(out_tex), "rows": len(unique_rows), "completed": completed}, ensure_ascii=False))
    return out_tex


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an English LaTeX report for the druggable-proteome run.")
    parser.add_argument(
        "--unique-csv",
        default="outputs/druggable_proteome/stage6_druggable_top_unique_diffdock_consensus.csv",
    )
    parser.add_argument(
        "--top-csv",
        default="outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.csv",
    )
    parser.add_argument(
        "--prep-metadata",
        default="outputs/druggable_proteome/druggable_proteome_conplex_prep.metadata.json",
    )
    parser.add_argument(
        "--merge-metadata",
        default="outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.metadata.json",
    )
    parser.add_argument("--out-tex", default="outputs/druggable_proteome/biomaster_druggable_external_report.tex")
    parser.add_argument("--top-n", type=int, default=18)
    args = parser.parse_args()
    build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
