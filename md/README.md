# BioMaster MD workspace

This directory contains tracked MD runners and scheduler templates only. Runtime
trajectories and force-field outputs belong under `md_runs/`, which is ignored.

Local environment:

```bash
conda activate /root/autodl-tmp/BioMaster/.conda_envs/md_openmm
python md/run_openmm_amber.py --help
```

The runner accepts a pre-audited, solvated AMBER `prmtop`/`inpcrd` pair. It does
not infer biological assemblies, cofactors, protonation states, membranes, or
ligand charge states. Those decisions must be completed before production MD.

Map a Boltz conditional pose to an experimental receptor before parameterization:

```bash
python md/prepare_experimental_pose.py \
  --experimental-pdb receptor_holo.pdb \
  --experimental-chain A \
  --boltz-cif candidate_model_0.cif \
  --ligand-smiles 'CC...' \
  --pair-id CHEMBL_ID_TARGET_SEQUENCE \
  --output-dir md_runs/example/prep
```

Summarize ligand stability after a short run:

```bash
python md/analyze_ligand_stability.py \
  --prmtop md_runs/example/prep/complex.prmtop \
  --inpcrd md_runs/example/prep/complex.inpcrd \
  --trajectory md_runs/example/run/trajectory.dcd \
  --equilibration-frames 20 \
  --output md_runs/example/run/stability_summary.json
```

The pose mapper preserves the audited SMILES graph and uses only the Boltz
coordinates. It does not decide protonation, biological assembly, cofactors,
metals, or membrane composition. Those remain explicit preparation gates.

Remote CPU submission:

```bash
sbatch --export=ALL,TPR=/absolute/path/production.tpr,DEFFNM=production \
  md/slurm/gromacs_cpu.slurm
```
