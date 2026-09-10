# ReTargetMap catalog model

Status: SMOKE TEST ONLY — this is not a selected model.

This directory is self-contained. It ranks exactly 720 packaged drugs and 384 packaged targets using one neural checkpoint and cached label-free encoder features. No training scripts, support stores, internet download or optimizer state are needed at inference.

Install `requirements.txt`, then run:

```sh
python infer.py --drug AAOVKJBEBIDNHE-UHFFFAOYSA-N --top-k 20
python infer.py --target Q16236 --top-k 20
```

The Python interface is `from retargetmap import CatalogRanker`; initialize it with this directory. `score_pairs` returns both directional logits. `rank_targets` and `rank_drugs` accept catalog ID candidate lists and report the actual candidate count. Drugs use full InChIKeys; targets use UniProt accessions. Unknown IDs and duplicate ranking candidates are rejected. Scores are not calibrated probabilities or measured affinities. The 745-target registry is not a common ranking denominator.

Default inference uses FP32 and CPU; `--device cuda` is optional. Hashes are verified on load. Optional receptor geometry can be absent; the loader then disables geometric messages. The public DrugCLIP source/weights used for cached representations have research/noncommercial license constraints; see provenance.json. Downstream date cutoffs do not certify public pretraining chronology or absence of overlap.
