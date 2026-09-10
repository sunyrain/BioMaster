# ReTargetMap catalog model

SMOKE TEST ONLY. These weights have not been selected.

One neural checkpoint ranks 720 drugs and 384 targets. This directory includes the selected network, cached inputs and catalog mappings. It runs independently of the research repository and does not download encoder weights.

Install `requirements.txt`, then run:

```sh
python infer.py --drug AAOVKJBEBIDNHE-UHFFFAOYSA-N --top-k 20
python infer.py --target Q16236 --top-k 20
```

The Python API is `CatalogRanker(directory)` from `retargetmap`. `score_pairs` returns both directional scores; `rank_targets` and `rank_drugs` rank catalog IDs or an explicit candidate subset. Ranking output includes the actual candidate count and readable gene/drug names. Unknown IDs, duplicate candidates and invalid arguments are rejected.

Default inference is FP32 on CPU; `--device cuda` enables GPU inference. Scores are logits, not binding probabilities or experimentally measured affinities. Default target ranking uses the 384-target core; the 745-target registry is not one rank denominator. The current bundle supports the listed catalog, not arbitrary new SMILES or protein sequences.

Global drug representation: `drugclip_morgan`. Local interaction refinement: `False`. Training supervision cutoff: `2018`. Hashes are checked on load. Optional geometry can be absent in a local bundle; geometric messages are then disabled. Public encoder chronology and relation overlap are not certified. Provenance and the upstream terms for the representations actually used are recorded in `provenance.json`.
