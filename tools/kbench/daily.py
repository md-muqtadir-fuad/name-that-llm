"""Daily Kaggle Benchmarks generation (run once per day after the quota resets).
  1) re-run the models that had failures in the previous batch (same prompt window; dedupe happens at cleaning)
  2) run the next batch on the 10 priority models
Writes raw outputs to research/ext_data/kbench/<batch>/, appends paper/data_log.csv, keeps state in state.json.
Usage (repo root):  .venv\\Scripts\\python.exe tools\\kbench\\daily.py [--n 70] [--dry]"""
import argparse
import datetime as dt
import glob
import json
import os
import pathlib
import re
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK = ROOT / 'tools/kbench/ntl_gen_task.py'
STATE = ROOT / 'tools/kbench/state.json'
KAGGLE = str(ROOT / '.venv/Scripts/kaggle.exe')
DATASET = 'seyitkaangunes/ntl-extra-api-passages'
# 2026-10-18 decision: gemini-3.1-pro-preview and gpt-6-astra dropped (expensive, easy classes)
PRIORITY = ['gpt-5.5-2026-04-23', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gemini-3.6-flash', 'gemini-3.8-flash',
            'claude-opus-4-6-default', 'claude-opus-4-8-default', 'claude-opus-5-default', 'claude-sonnet-5-default']
SLUG_OF = {'openai/gpt-5.5-2026-04-23': 'gpt-5.5-2026-04-23', 'openai/gpt-5.6-luna': 'gpt-5.6-luna',
           'openai/gpt-5.6-sol': 'gpt-5.6-sol', 'openai/gpt-5.6-terra': 'gpt-5.6-terra', 'openai/gpt-6-astra': 'gpt-6-astra',
           'google/gemini-3.6-flash': 'gemini-3.6-flash', 'google/gemini-3.8-flash': 'gemini-3.8-flash',
           'google/gemini-3.1-pro-preview': 'gemini-3.1-pro-preview',
           'anthropic/claude-opus-4-6@default': 'claude-opus-4-6-default', 'anthropic/claude-opus-4-8@default': 'claude-opus-4-8-default',
           'anthropic/claude-opus-5@default': 'claude-opus-5-default', 'anthropic/claude-sonnet-5@default': 'claude-sonnet-5-default'}
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')


def sh(args, timeout=None):
    print('$', ' '.join(args[1:]), flush=True)
    r = subprocess.run(args, capture_output=True, text=True, env=ENV, timeout=timeout)
    out = (r.stdout or '') + (r.stderr or '')
    print('\n'.join(out.strip().splitlines()[-6:]), flush=True)
    return out


def set_task(start, n, batch):
    s = TASK.read_text(encoding='utf-8')
    s = re.sub(r'^START, N = .*$', f'START, N = {start}, {n}          # set by daily.py', s, flags=re.M)
    s = re.sub(r"^BATCH = .*$", f"BATCH = '{batch}'", s, flags=re.M)
    TASK.write_text(s, encoding='utf-8', newline='\n')


def run_batch(start, n, batch, models, dry):
    set_task(start, n, batch)
    if dry:
        print(f'[dry] would run {batch} [{start}, {start + n}) on {models}')
        return None
    sh([KAGGLE, 'b', 't', 'push', 'ntl-gen', '-f', str(TASK), '-d', DATASET], timeout=900)
    for _ in range(90):   # the push triggers a default run; wait until the task is runnable
        if 'Status:   Completed' in sh([KAGGLE, 'b', 't', 'status', 'ntl-gen'], timeout=120):
            break
        import time
        time.sleep(20)
    args = [KAGGLE, 'b', 't', 'run', 'ntl-gen']
    for m in models:
        args += ['-m', m]
    sh(args + ['--wait', '10000'], timeout=11000)
    out = ROOT / f'research/ext_data/kbench/{batch}'
    sh([KAGGLE, 'b', 't', 'download', 'ntl-gen', '-o', str(out), '-f'], timeout=900)
    fs = glob.glob(str(out / f'**/ntl_gen_{batch}_*.csv'), recursive=True)
    if not fs:
        print('no output files'); return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d['ok'] = d.error.fillna('').astype(str).eq('') & d.response.fillna('').astype(str).str.len().gt(0)
    stats = d.groupby('llm').ok.agg(['sum', 'size'])
    print(stats.to_string())
    failed = [SLUG_OF.get(l, l) for l, r in stats.iterrows() if r['sum'] < r['size'] and SLUG_OF.get(l, l) in models]
    with open(ROOT / 'paper/data_log.csv', 'a', encoding='utf-8', newline='') as f:
        f.write(f"{dt.date.today().isoformat()},kb_{batch},ntl-gen daily.py,Kaggle Benchmarks,{len(models)} labels,"
                f"window {start}-{start + n - 1},{int(stats['sum'].sum())}/{int(stats['size'].sum())} ok,about N words,"
                f"downloaded,research/ext_data/kbench/{batch},models with failures: {' '.join(failed) or 'none'}\n")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=70)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    st = json.loads(STATE.read_text()) if STATE.exists() else {'next_start': 123, 'batch_no': 3, 'pending_retry': None}
    pr = st.get('pending_retry')
    if pr and pr.get('models'):
        f = run_batch(pr['start'], pr['n'], pr['batch'] + 'r', pr['models'], a.dry)
        if not a.dry:
            st['pending_retry'] = None
    batch = f"b{st['batch_no']:03d}"
    failed = run_batch(st['next_start'], a.n, batch, PRIORITY, a.dry)
    if not a.dry:
        st['pending_retry'] = {'batch': batch, 'start': st['next_start'], 'n': a.n, 'models': failed or []}
        st['next_start'] += a.n
        st['batch_no'] += 1
        STATE.write_text(json.dumps(st, indent=2))
    print('state:', json.dumps(st))


if __name__ == '__main__':
    sys.exit(main())
