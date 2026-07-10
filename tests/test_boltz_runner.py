from pathlib import Path
from argparse import Namespace
import hashlib

import pandas as pd
import pytest

from scripts.run_boltz2_batched_queue import completed_prediction_stems, prepare_batches, read_json, write_json


def write_complete_prediction(path: Path, stem: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"confidence_{stem}_model_0.json").write_text(
        '{"confidence_score":0.8,"ligand_iptm":0.7,"complex_iplddt":0.6}'
    )
    (path / f"affinity_{stem}.json").write_text('{"affinity_probability_binary":0.75}')
    (path / f"{stem}_model_0.cif").write_text("model0")
    (path / f"{stem}_model_1.cif").write_text("model1")


def test_completed_prediction_stems_requires_valid_json_and_two_cifs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    first = predictions / "PAIR_A"
    second = predictions / "PAIR_B"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    write_complete_prediction(first, "PAIR_A")
    (second / "confidence_PAIR_B_model_0.json").write_text("{}")

    assert completed_prediction_stems(tmp_path) == {"PAIR_A"}


def test_read_json_returns_persisted_status(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text('{"status": "success"}', encoding="utf-8")

    assert read_json(path) == {"status": "success"}


def test_prepare_batches_only_resumes_matching_signed_results(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    yaml_path = input_dir / "001_PAIR.yaml"
    yaml_path.write_text("version: 1\n", encoding="utf-8")
    yaml_sha = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "externalQueueRank": [1],
            "pairId": ["PAIR"],
            "yamlFile": [yaml_path.name],
            "yamlSha256": [yaml_sha],
            "inputSignatureSha256": ["input-signature"],
        }
    ).to_csv(manifest, index=False)
    args = Namespace(
        input_manifest=str(manifest.relative_to(tmp_path)),
        input_dir=str(input_dir.relative_to(tmp_path)),
        out_dir="run",
        top_n=1,
        batch_size=1,
        seed_base=100,
        force=False,
    )
    first = prepare_batches(tmp_path, args, "run-signature")
    batch = first[0]
    prediction = Path(batch["runDir"]) / "predictions" / "001_PAIR"
    prediction.mkdir(parents=True)
    write_complete_prediction(prediction, "001_PAIR")
    write_json(
        Path(batch["statusPath"]),
        {"status": "success", "batchInputSignature": batch["batchInputSignature"]},
    )
    Path(batch["provenancePath"]).parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "pairId": ["PAIR"],
            "batchInputSignature": [batch["batchInputSignature"]],
            "resultCompletedVerified": [True],
        }
    ).to_csv(batch["provenancePath"], index=False)

    resumed = prepare_batches(tmp_path, args, "run-signature")
    assert resumed[0]["skip"] is True

    with pytest.raises(RuntimeError, match="different input signature"):
        prepare_batches(tmp_path, args, "different-run-signature")
