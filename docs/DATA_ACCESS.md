# Data Access Notes

This repository does not publish raw data, generated full-scale outputs, model weights, or third-party repositories. Recreate them locally from the original sources.

## Local Data Used In The Current Run

The current BioMaster workspace used these local source files:

- FDA small-molecule workbook: `/root/autodl-tmp/BioMaster/FDA_approved_small_molecules_2005_2026_with_structures.xls`
- AlphaFold human proteome v6 tarball: `/root/autodl-tmp/BioMaster/UP000005640_9606_HUMAN_v6.tar`
- TxGNN Explorer archive: `/root/autodl-tmp/BioMaster/TxGNNExplorer_v2.zip`
- TxGNN graph files: `/root/autodl-tmp/BioMaster/kg.csv`, `/root/autodl-tmp/BioMaster/edges.csv`, `/root/autodl-tmp/BioMaster/nodes.tab`
- Downloaded model archives and weights under `/root/autodl-tmp/BioMaster/downloads/`

These paths are intentionally git-ignored.

## Public/External Sources

### ChEMBL and PubChem

Used for molecular structures and identifiers. The processed local file is:

```text
data/processed/drug_library_pubchem_chembl_mapped.csv
```

Current local coverage:

- 915/915 ChEMBL molfile-derived SDF paths.
- 915/915 InChIKey.
- 915/915 molecular formula.
- 913/915 PubChem CID.
- 0/915 DrugBank ID.

DrugBank identifiers are not complete in the current run and require a separate licensed/compliant source.

### AlphaFold

Used from AlphaFold human proteome v6:

```text
UP000005640_9606_HUMAN_v6.tar
```

Processed local outputs:

```text
data/processed/protein_library_1000_alphafold_paths.csv
outputs/report_scale/manifest_915k_structure_ready.csv
outputs/report_scale/manifest_915k_diffdock_ready.csv
```

The two `outputs/report_scale` paths belong to the historical full-DiffDock branch and are
not present in the current workspace snapshot. They are reproducibility targets, not evidence
that the old queue is currently running or complete. The active ODTI program uses its own frozen
feature stores and status artifacts under `outputs/biomaster_odti_*`.

Current local coverage:

- 1000 proteins.
- 998 proteins with PDB/CIF paths.
- Missing proteins: `P49895`, `P49908`.
- 913,170 DiffDock-ready pairs.

### Open Targets

Used through the Open Targets Platform GraphQL API:

```text
https://api.platform.opentargets.org/api/v4/graphql
```

Processed local output:

```text
data/processed/opentargets_target_disease_scores.csv
```

The current disease focus is cancer:

```text
MONDO_0004992
```

### STRING

Used through STRING v12.0 API rather than full offline STRING human downloads:

```text
https://version-12-0.string-db.org/api/tsv/network
https://version-12-0.string-db.org/api/tsv/interaction_partners
```

Processed local output:

```text
data/processed/string_human_filtered_edges.csv
```

Current filter:

- Species: 9606
- Required score: 700
- Disease seed genes: `TP53`, `EGFR`, `BCL2`, `JAK1`
- Output edges: 2547

This is a disease-related high-confidence subnetwork. It is sufficient for the current Stage 5 run; complete STRING offline files are optional future reproducibility enhancements.

### TxGNN

Processed local output:

```text
data/processed/txgnn_drug_disease_scores.csv
```

Current local mapping:

- 915 FDA drugs.
- 738 mapped to TxGNN drug nodes.
- 177 unmapped.
- Disease node: cancer / `MONDO_0004992`.

Mapping is exact after conservative name normalization. No fuzzy matching is used.

## Not Included In Git

The following are excluded by `.gitignore`:

- `data/`
- `downloads/`
- `outputs/`
- `third_party/`
- root-level raw graph dumps and archives
- model weights and docking outputs

This is intentional. Keep the GitHub repository as code plus documentation, and distribute large artifacts through controlled storage if needed.
