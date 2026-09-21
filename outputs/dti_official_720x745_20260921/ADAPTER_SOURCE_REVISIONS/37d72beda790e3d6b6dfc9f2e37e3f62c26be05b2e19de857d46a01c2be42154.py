#!/usr/bin/env python3
"""Resumable MAMMAL BindingDB pKd inference using the author's task class."""
import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, dump, directory, read_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    name = 'MAMMAL_pKd'
    start = time.time()
    status(name, 'LOADING_NATIVE_MODEL')
    sys.path.insert(0, str(ROOT / '.external/MAMMAL'))
    from mammal.model import Mammal
    from mammal.examples.dti_bindingdb_kd.task import DtiBindingdbKdTask as Task
    from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
    checkpoint = ROOT / 'data/research/dti_official_weights_20260921/MAMMAL'
    model = Mammal.from_pretrained(str(checkpoint), strict=True, local_files_only=True).eval().cuda()
    tokenizer = ModularTokenizerOp.from_pretrained(str(checkpoint / 'tokenizer'))
    torch.set_num_threads(3)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    drugs, targets = inputs()

    def predict(rows):
        samples = [Task.data_preprocessing(sample_dict={'target_seq': t.sequence, 'drug_seq': d.smiles,
                   'data.sample_id': f'{d.drug_id}__{t.target_id}'},
                   tokenizer_op=tokenizer, target_sequence_key='target_seq', drug_sequence_key='drug_seq',
                   norm_y_mean=None, norm_y_std=None, device='cuda') for d, t in rows]
        with torch.inference_mode():
            batch = model.forward_encoder_only(samples)
            batch = Task.process_model_output(batch, norm_y_mean=5.79384684128215, norm_y_std=1.33808027428196)
            values = batch['model.out.dti_bindingdb_kd'].detach().cpu().numpy()
        assert np.isfinite(values).all()
        return values

    sample = list(zip(list(drugs.itertuples())[:4], list(targets.itertuples())[:4]))
    single = np.array([predict([r])[0] for r in sample])
    batched = predict(sample)
    delta = float(np.max(np.abs(single-batched)))
    assert delta < 2e-4, delta
    dump(directory(name) / 'REPLAY_CHECK.json', dict(status='PASS', native_single_vs_batch_max_abs_delta=delta,
                                                   native_example_scores=single.tolist()))
    contract(name, checkpoint_sha256=sha(checkpoint/'model.safetensors'), adapter_sha256=sha(__file__),
             native_source_revision='ebca44f32db6faf85101de87dfde32034fe4e701', precision='float32_cuda_tf32_off',
             head='native scalar head at first token, inverse author BindingDB pKd normalization',
             preprocessing='Author DtiBindingdbKdTask defaults: protein1250, drug256, encoder1512 tokens')
    if args.smoke:
        status(name, 'SMOKE_PASS', elapsed_seconds=time.time()-start)
        return
    frame = read_scores(name)
    generated = 0
    inference_start = time.time()
    failures = []
    for target in targets.itertuples():
        done = set(frame.loc[frame.target_id.eq(target.target_id), 'drug_id'])
        pending = [d for d in drugs.itertuples() if d.drug_id not in done]
        if not pending:
            continue
        rows = []
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset+args.batch_size]
            try:
                values = predict([(drug, target) for drug in batch])
                rows.extend(dict(drug_id=d.drug_id, target_id=target.target_id, score=float(v)) for d, v in zip(batch, values))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for drug in batch:
                    value = predict([(drug, target)])[0]
                    rows.append(dict(drug_id=drug.drug_id, target_id=target.target_id, score=float(value)))
            generated += len(batch)
            if generated % 64 == 0:
                elapsed = time.time()-inference_start
                status(name, 'INFERENCE', scored_pairs=len(frame)+len(rows), computed_this_run=generated, target=target.gene,
                       elapsed_seconds=time.time()-start, inference_pairs_per_second=generated/elapsed,
                       eta_seconds=(536400-len(frame)-len(rows))*elapsed/generated)
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        save_scores(name, frame)
        print(target.gene, len(frame), flush=True)
    status(name, 'COMPLETE', scored_pairs=len(frame), elapsed_seconds=time.time()-start)


if __name__ == '__main__':
    main()
