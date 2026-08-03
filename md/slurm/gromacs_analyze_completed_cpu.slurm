#!/usr/bin/env bash
#SBATCH --job-name=biomaster_md_analysis
#SBATCH --partition=batch1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=analysis-%A_%a.out
#SBATCH --error=analysis-%A_%a.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load gcc-9.3.0 intel2020u2

GMX=/public/software/gromacs-2021.6/bin/gmx_mpi
MANIFEST=${MANIFEST:?Submit with --export=ALL,MANIFEST=/absolute/path/to/manifest.tsv}
SUMMARIZER=${SUMMARIZER:?Submit with --export=ALL,SUMMARIZER=/absolute/path/to/summarize_gromacs_screen.py}
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [[ -z "$line" ]]; then
  echo "No manifest row for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r global_index pair_id workdir context_tier <<< "$line"
cd "$workdir"
if ! grep -q '^COMPLETED' md_status.tsv; then
  printf 'SKIPPED_NOT_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" > analysis_status.tsv
  exit 0
fi

mkdir -p analysis
printf 'RUNNING\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$HOSTNAME" > analysis_status.tsv
trap 'printf "FAILED\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$HOSTNAME" > analysis_status.tsv' ERR

# Use static selections so every case has the same named groups.
"$GMX" select -s production.tpr -on analysis/groups.ndx -select \
  '"System" all; "Protein" group "Protein"; "Backbone" group "Backbone"; "ProteinHeavy" group "Protein" and not name "H*"; "LigandHeavy" resname LIG and not name "H*"' \
  > analysis/select.log 2>&1

# Reconstruct molecules and place the protein at the box center. This prevents
# periodic-image jumps from being mistaken for ligand escape.
printf 'Protein\nSystem\n' | "$GMX" trjconv \
  -s production.tpr -f production.xtc -n analysis/groups.ndx \
  -o analysis/production_centered.xtc -pbc mol -center -ur compact \
  > analysis/trjconv.log 2>&1
printf 'System\n' | "$GMX" trjconv \
  -s production.tpr -f analysis/production_centered.xtc -n analysis/groups.ndx \
  -o analysis/reference_centered.gro -dump 0 \
  > analysis/reference.log 2>&1

# Both RMSD series use the protein backbone for least-squares alignment.
printf 'Backbone\nLigandHeavy\n' | "$GMX" rms \
  -s analysis/reference_centered.gro -f analysis/production_centered.xtc -n analysis/groups.ndx \
  -o analysis/ligand_rmsd_nm.xvg -fit rot+trans -nomw -tu ps \
  > analysis/ligand_rmsd.log 2>&1
printf 'Backbone\nBackbone\n' | "$GMX" rms \
  -s analysis/reference_centered.gro -f analysis/production_centered.xtc -n analysis/groups.ndx \
  -o analysis/backbone_rmsd_nm.xvg -fit rot+trans -nomw -tu ps \
  > analysis/backbone_rmsd.log 2>&1

# Minimum heavy-atom distance and heavy-atom pair contact count (0.4 nm).
printf 'LigandHeavy\nProteinHeavy\n' | "$GMX" mindist \
  -s production.tpr -f analysis/production_centered.xtc -n analysis/groups.ndx \
  -od analysis/min_distance_nm.xvg -on analysis/contact_atom_pairs.xvg \
  -d 0.4 -group -tu ps \
  > analysis/mindist.log 2>&1

printf 'Temperature\nPressure\nDensity\n0\n' | "$GMX" energy \
  -f production.edr -o analysis/thermodynamics.xvg \
  > analysis/energy.log 2>&1

python3 "$SUMMARIZER" \
  --analysis-dir analysis \
  --global-index "$global_index" \
  --pair-id "$pair_id" \
  --context-tier "$context_tier" \
  --output analysis/summary.json

rm -f analysis/production_centered.xtc
sha256sum analysis/*.xvg analysis/summary.json > analysis/sha256.txt
printf 'COMPLETED\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$HOSTNAME" > analysis_status.tsv
