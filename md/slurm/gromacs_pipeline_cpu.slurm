#!/usr/bin/env bash
#SBATCH --job-name=biomaster_md5ns
#SBATCH --partition=batch1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load gcc-9.3.0 intel2020u2

GMX=/public/software/gromacs-2021.6/bin/gmx_mpi
WORKDIR=${WORKDIR:?Submit with --export=ALL,WORKDIR=/absolute/path,SEED=integer}
SEED=${SEED:?Submit with --export=ALL,WORKDIR=/absolute/path,SEED=integer}
cd "$WORKDIR"

sed "s/__SEED__/${SEED}/g" nvt.mdp.template > nvt.mdp

"$GMX" grompp -f em.mdp -c complex.gro -p complex.top -o em.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm em -ntomp "$SLURM_CPUS_PER_TASK" -pin on

"$GMX" grompp -f nvt.mdp -c em.gro -p complex.top -o nvt.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm nvt -ntomp "$SLURM_CPUS_PER_TASK" -pin on

"$GMX" grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p complex.top -o npt.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm npt -ntomp "$SLURM_CPUS_PER_TASK" -pin on

"$GMX" grompp -f production_5ns.mdp -c npt.gro -t npt.cpt -p complex.top -o production.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm production -ntomp "$SLURM_CPUS_PER_TASK" -pin on -cpt 15

sha256sum complex.top complex.gro production.tpr production.gro production.xtc production.edr > output_sha256.txt
