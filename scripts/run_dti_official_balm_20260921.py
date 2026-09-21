#!/usr/bin/env python3
"""Native BALM checkpoint and author-config length eligibility, no fitting."""
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, dump, directory


def main():
    start = time.time()
    name = 'BALM'
    status(name, 'LOADING_NATIVE_MODEL')
    torch.set_num_threads(3)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(ROOT / '.external/BALM'))
    from balm.models import BALM
    from balm.configs import Configs
    import balm.models.utils as native
    cfg_path = ROOT / '.external/BALM/default_configs/balm_peft.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    enc = ROOT / 'data/research/dti_native_encoders_20260921'
    cfg['model_configs']['protein_model_name_or_path'] = str(enc / 'esm2_t30_150M_UR50D')
    cfg['model_configs']['drug_model_name_or_path'] = str(enc / 'ChemBERTa-77M-MTR')
    configs = Configs(**cfg)
    checkpoint = ROOT / 'data/research/dti_official_weights_20260921/BALM/pytorch_model.bin'
    native.hf_hub_download = lambda **_: str(checkpoint)
    model = BALM(configs.model_configs)
    # The author loader maps historical PEFT parameter names then merges adapters.
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    mapped = dict(ck)
    for key, value in ck.items():
        for layer in ('query', 'key', 'value'):
            if f'attention.self.{layer}.' in key and '.base_layer.' not in key:
                mapped[key.replace(f'attention.self.{layer}.', f'attention.self.{layer}.base_layer.')] = value
    missing = [key for key in model.state_dict() if key not in mapped]
    assert not missing, missing
    model = native.load_trained_model(model, configs.model_configs, is_training=False).cuda().eval()
    pt = AutoTokenizer.from_pretrained(cfg['model_configs']['protein_model_name_or_path'], local_files_only=True)
    dt = AutoTokenizer.from_pretrained(cfg['model_configs']['drug_model_name_or_path'], local_files_only=True)
    drugs, targets = inputs()
    eligible_d, eligible_t, failures = [], [], []

    def encode(frame, key, text_key, tokenizer, encoder, projection, limit, side):
        vectors = []
        for i, row in enumerate(frame.itertuples()):
            tokens = tokenizer(getattr(row, text_key), return_tensors='pt')
            length = int(tokens.input_ids.shape[1])
            if length > limit:
                failures.append(dict(entity_type=side, entity_id=getattr(row, key), token_length=length,
                                     reason=f'AUTHOR_TRAINER_LENGTH_FILTER_GT_{limit}'))
                continue
            with torch.inference_mode():
                vector = projection(encoder(**{k: v.cuda() for k, v in tokens.items() if k in ('input_ids', 'attention_mask')})['pooler_output'])
            vectors.append(vector.cpu())
            (eligible_d if side == 'drug' else eligible_t).append(row)
            if i % 50 == 0:
                status(name, 'ENCODING_' + side.upper(), encoded=i, total=len(frame), elapsed_seconds=time.time()-start)
        return torch.cat(vectors)

    dp = encode(drugs, 'drug_id', 'smiles', dt, model.drug_model, model.drug_projection, 512, 'drug')
    tp = encode(targets, 'target_id', 'sequence', pt, model.protein_model, model.protein_projection, 1024, 'target')
    assert not model.relu_before_cosine
    cosine = torch.stack([torch.nn.functional.cosine_similarity(dp, row.expand_as(dp)) for row in tp], dim=1)
    scores = model.cosine_similarity_to_pkd(cosine, pkd_upper_bound=10., pkd_lower_bound=1.999999995657055).numpy()
    checks = []
    for di, ti in [(0, 0), (len(eligible_d)//2, len(eligible_t)//2), (-1, -1)]:
        p = pt(eligible_t[ti].sequence, return_tensors='pt').to('cuda')
        d = dt(eligible_d[di].smiles, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            value = model(dict(protein_input_ids=p.input_ids, protein_attention_mask=p.attention_mask,
                               drug_input_ids=d.input_ids, drug_attention_mask=d.attention_mask))['cosine_similarity']
            pkd = model.cosine_similarity_to_pkd(value, pkd_upper_bound=10., pkd_lower_bound=1.999999995657055).item()
        delta = abs(pkd-float(scores[di, ti]))
        assert delta < 2e-5, delta
        checks.append(delta)
    frame = pd.DataFrame(dict(drug_id=np.repeat([d.drug_id for d in eligible_d], len(eligible_t)),
                              target_id=np.tile([t.target_id for t in eligible_t], len(eligible_d)), score=scores.reshape(-1)))
    contract(name, checkpoint_sha256=sha(checkpoint), adapter_sha256=sha(__file__), native_source_revision='c0a4ecb828c44ba1633e06e885da12c5a6ababd6',
             config_sha256=sha(cfg_path), head='cosine transformed to author pKd scale [2,10]', precision='float32_cuda_tf32_off',
             preprocessing='Native tokenizers; author trainer filters >1024 protein / >512 drug tokens, no invented truncation',
             length_policy_source='.external/BALM/balm/trainer.py:193', actual_drugs=len(eligible_d), actual_targets=len(eligible_t))
    dump(directory(name) / 'REPLAY_CHECK.json', dict(status='PASS', native_forward_abs_deltas=checks, missing_checkpoint_keys=missing))
    dump(directory(name) / 'INPUT_FAILURES.json', failures)
    save_scores(name, frame)
    status(name, 'COMPLETE' if len(frame)==536400 else 'COMPLETE_WITH_NATIVE_LENGTH_FILTER', scored_pairs=len(frame),
           missing_pairs=536400-len(frame), elapsed_seconds=time.time()-start)
    print(name, len(frame), 'scores', len(failures), 'excluded entities', flush=True)


if __name__ == '__main__':
    main()
