#!/usr/bin/env python3
"""Expand previously qualified native CPU adapters, without changing the website."""
import argparse
import hashlib
import tarfile
import time
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, historical, dump, directory, read_scores


def fetch_dtbind(targets):
    from fetch_frontier_dti_assets_20260916 import RangeStream
    dest = ROOT / 'outputs/frontier_dti_20260916/dtbind/protein_graph'
    wanted = set(targets.uniprot_id)
    url = 'https://zenodo.org/records/17283638/files/DTBind_datasets.tar.gz?download=1'
    rows = []
    entered = False
    with RangeStream(url, workers=2, cache=ROOT / '.cache/frontier_dti/dtbind_archive_ranges') as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as arc:
            for member in arc:
                if '/occurrence/protein_graph/' in member.name and member.isfile():
                    entered = True
                    accession = member.name.rsplit('/', 1)[-1].removesuffix('.pt')
                    if accession not in wanted:
                        continue
                    data = arc.extractfile(member).read()
                    path = dest / (accession + '.pt')
                    digest = hashlib.sha256(data).hexdigest()
                    if path.exists():
                        assert sha(path) == digest, accession
                    else:
                        path.write_bytes(data)
                    rows.append(dict(uniprot_id=accession, archive_member=member.name, sha256=digest))
                elif entered and '/occurrence/protein_graph' not in member.name:
                    break
    dump(directory('DTBind_occurrence') / 'GRAPH_AUDIT_745.json', dict(source=url, graphs=rows,
         requested=745, absent=sorted(wanted - {r['uniprot_id'] for r in rows})))


def probe_features(targets):
    from transformers import AutoModel, AutoTokenizer
    cache = ROOT / 'outputs/frontier_dti_20260916/probematch/features'
    missing = [t for t in targets.itertuples() if not (cache / f'protein_{t.target_id}.npy').exists()]
    if not missing:
        return
    path = ROOT / '.cache/frontier_dti/prot_bert_bfd'
    tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=False, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval().cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    for i, target in enumerate(missing):
        status('ProbeMatchDTI', 'ENCODING_NATIVE_PROTEIN_FEATURES', encoded=i, to_encode=len(missing), elapsed_seconds=time.time()-started)
        tokens = tokenizer.batch_encode_plus([' '.join(target.sequence[:5000])], add_special_tokens=True, padding=True, return_tensors='pt')
        with torch.inference_mode():
            result = model(**{k: v.cuda() for k, v in tokens.items() if k in ('input_ids', 'attention_mask')}).last_hidden_state[0, :1200].float().cpu().numpy()
        assert np.isfinite(result).all()
        np.save(cache / f'protein_{target.target_id}.npy', result)
    dump(directory('ProbeMatchDTI') / 'FEATURE_BUILD.json', dict(new_proteins=len(missing),
         encoder='Rostlab/prot_bert_bfd', precision='float32_cuda_tf32_off', input_truncation_residues=5000, retained_tokens=1200))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['ProbeMatchDTI', 'DTBind_occurrence'])
    parser.add_argument('--fetch-graphs', action='store_true')
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    name = args.model
    start = time.time()
    drugs, targets = inputs()
    torch.set_num_threads(args.threads)
    RDLogger.DisableLog('rdApp.warning')
    status(name, 'PREPARING_NATIVE_INPUTS')
    if args.fetch_graphs:
        fetch_dtbind(targets)
    if args.features_only:
        probe_features(targets)
        status(name, 'FEATURES_READY')
        return
    import run_catalog_dti_20260916 as native
    native.status = lambda _, **kw: status(name, kw.pop('state', 'PREPARING'), **kw)
    adapter = (native.Probe if name == 'ProbeMatchDTI' else native.DTBind)(drugs, targets, args.threads)
    checks = []
    old = historical(name)
    for target in targets.itertuples():
        subset = old.loc[old.target_id.eq(target.target_id)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        drug = next(d for d in drugs.itertuples() if d.drug_id == row.drug_id)
        adapter.prepare_target(target)
        value = adapter.score(target, drug)
        delta = abs(value - row.score)
        assert delta < 1e-6, (target.target_id, delta)
        checks.append(dict(target_id=target.target_id, drug_id=drug.drug_id, delta=delta))
        if len(checks) == 5:
            break
    dump(directory(name) / 'REPLAY_CHECK.json', dict(status='PASS', pairs=checks))
    ckpt = ROOT / ('.external/ProbeMatchDTI/model/All_Model' if name == 'ProbeMatchDTI' else '.external/DTBind/models/occurrence_model.pth')
    contract(name, checkpoint_sha256=sha(ckpt), adapter_sha256=sha(__file__),
             native_adapter_sha256=sha(ROOT / 'scripts/run_catalog_dti_20260916.py'), precision='float32_cpu',
             head='output6_softmax_class1' if name == 'ProbeMatchDTI' else 'occurrence_probability',
             historical_reuse='exact frozen drug and protein identities, replay checked',
             random_seed='sha256(pair_id) first8 hex' if name == 'ProbeMatchDTI' else 20260916)
    frame = read_scores(name)
    failures = []
    generated = 0
    inference_start = time.time()
    for target in targets.itertuples():
        done = set(frame.loc[frame.target_id.eq(target.target_id), 'drug_id'])
        if len(done) == len(drugs):
            continue
        try:
            adapter.prepare_target(target)
        except Exception as exc:
            failures.append(dict(target_id=target.target_id, uniprot_id=target.uniprot_id,
                                 reason=f'{type(exc).__name__}: {exc}', affected_pairs=len(drugs)-len(done)))
            continue
        batch = []
        for drug in drugs.itertuples():
            if drug.drug_id in done:
                continue
            try:
                score = adapter.score(target, drug)
                assert np.isfinite(score)
                batch.append(dict(drug_id=drug.drug_id, target_id=target.target_id, score=score))
            except Exception as exc:
                failures.append(dict(target_id=target.target_id, drug_id=drug.drug_id, reason=f'{type(exc).__name__}: {exc}', affected_pairs=1))
            generated += 1
            if generated % 50 == 0:
                elapsed = time.time()-inference_start
                status(name, 'INFERENCE', scored_pairs=len(frame)+len(batch), computed_this_run=generated,
                       target=target.gene, elapsed_seconds=time.time()-start, inference_pairs_per_second=generated/elapsed,
                       eta_upper_seconds=(536400-len(frame)-len(batch))*elapsed/generated)
        if batch:
            frame = pd.concat([frame, pd.DataFrame(batch)], ignore_index=True)
            save_scores(name, frame)
        dump(directory(name) / 'INPUT_FAILURES.json', failures)
        print(target.gene, 'scored', len(frame), flush=True)
    save_scores(name, frame)
    dump(directory(name) / 'INPUT_FAILURES.json', failures)
    status(name, 'COMPLETE' if len(frame)==536400 else 'COMPLETE_WITH_MISSING_INPUTS',
           scored_pairs=len(frame), missing_pairs=536400-len(frame), elapsed_seconds=time.time()-start)


if __name__ == '__main__':
    main()
