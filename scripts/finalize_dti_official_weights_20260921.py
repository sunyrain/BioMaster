#!/usr/bin/env python3
"""Wait for the active downloader, resume failures, then finalize collection reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_official_weights_download_20260921'


def state(status, **fields):
    path = OUT / 'FINALIZATION_STATUS.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields), indent=2)+'\n')
    tmp.replace(path)


def active(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'scripts/download_dti_official_weights_20260921.py' in command
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    while active(args.wait_pid):
        state('WAITING_FOR_ACTIVE_DOWNLOAD', downloader_pid=args.wait_pid)
        time.sleep(30)
    for attempt in range(4):
        summary_path = OUT/'DOWNLOAD_SUMMARY.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary.get('status') == 'PASS':
            break
        if attempt == 3:
            state('NEEDS_REVIEW', reason='Downloader still incomplete after three resumptions', last_summary=summary)
            return 1
        state('RESUMING_DOWNLOAD', attempt=attempt+1)
        with (OUT/'RUN.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(ROOT/'scripts/download_dti_official_weights_20260921.py'), '--workers', '10'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    state('FINALIZING_REPORTS')
    manifest = json.loads((OUT/'DOWNLOAD_MANIFEST.json').read_text())
    verified = json.loads((OUT/'VERIFIED_FILES.json').read_text())
    expected = {r['destination']:r for r in manifest['files']}
    done = {r['destination']:r for r in verified}
    assert len(done) == len(verified) == len(expected)
    assert done.keys() == expected.keys()
    for name, row in done.items():
        original = expected[name]
        assert row['status'] == 'VERIFIED'
        assert (ROOT/name).stat().st_size == original['bytes'] == row['bytes']
        for field in ['sha256', 'md5', 'git_blob']:
            if original.get('expected_'+field):
                assert original['expected_'+field] == row[field]
    subprocess.run([sys.executable, str(ROOT/'scripts/report_dti_official_weights_20260921.py')], cwd=ROOT, check=True)
    collection = json.loads((OUT/'COLLECTION_SUMMARY.json').read_text())
    assert collection['status'] == 'COMPLETE_HASH_VERIFIED'
    assert len(collection['graphban_archives']) == 3
    assert all(row['author_md5_matched'] and row['zip_directory_readable'] for row in collection['graphban_archives'])
    state('COMPLETE_HASH_VERIFIED', new_files=len(done), reused_files=collection['reused_files'],
        new_bytes=sum(r['bytes'] for r in done.values()),
        manifest_sha256=hashlib.sha256((OUT/'DOWNLOAD_MANIFEST.json').read_bytes()).hexdigest(),
        report='docs/BIOMASTER_DTI_OFFICIAL_WEIGHTS_DOWNLOAD_20260921_ZH.md',
        native_forward_qualification=False, new_training_started=False)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        state('NEEDS_REVIEW', error_type=type(error).__name__)
        raise
