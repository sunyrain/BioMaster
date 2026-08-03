#!/usr/bin/env bash
#SBATCH --job-name=biomaster_md_screen
#SBATCH --partition=batch1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load gcc-9.3.0 intel2020u2

GMX=/public/software/gromacs-2021.6/bin/gmx_mpi
MANIFEST=${MANIFEST:?Submit with --export=ALL,MANIFEST=/absolute/path/to/manifest.tsv}
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [[ -z "$line" ]]; then
  echo "No manifest row for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r md_rank pair_id workdir seed <<< "$line"
cd "$workdir"

printf 'RUNNING\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$HOSTNAME" > md_status.tsv
trap 'printf "FAILED\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$HOSTNAME" > md_status.tsv' ERR

sed "s/__SEED__/${seed}/g" nvt.mdp.template > nvt.mdp
"$GMX" grompp -f em.mdp -c complex.gro -p complex.top -o em.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm em -ntomp "$SLURM_CPUS_PER_TASK" -pin on

"$GMX" grompp -f nvt.mdp -c em.gro -p complex.top -o nvt.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm nvt -ntomp "$SLURM_CPUS_PER_TASK" -pin on

"$GMX" grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p complex.top -o npt.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm npt -ntomp "$SLURM_CPUS_PER_TASK" -pin on

"$GMX" grompp -f production_5ns.mdp -c npt.gro -t npt.cpt -p complex.top \
  -o production.tpr -maxwarn 0
mpirun -bootstrap slurm -np "$SLURM_NTASKS" "$GMX" mdrun \
  -deffnm production -ntomp "$SLURM_CPUS_PER_TASK" -pin on -cpt 15

sha256sum complex.top complex.gro production.tpr production.gro \
  production.xtc production.edr > output_sha256.txt
printf 'COMPLETED\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$HOSTNAME" > md_status.tsv
