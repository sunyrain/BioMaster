#!/usr/bin/env bash
#SBATCH --job-name=biomaster_md
#SBATCH --partition=batch1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load gcc-9.3.0 intel2020u2

GMX=/public/software/gromacs-2021.6/bin/gmx_mpi
TPR=${TPR:?Submit with --export=ALL,TPR=/absolute/path/run.tpr}
DEFFNM=${DEFFNM:-production}

mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -s "$TPR" \
  -deffnm "$DEFFNM" \
  -ntomp "$SLURM_CPUS_PER_TASK" \
  -pin on \
  -pme auto \
  -cpt 15 \
  -maxh 23.5
