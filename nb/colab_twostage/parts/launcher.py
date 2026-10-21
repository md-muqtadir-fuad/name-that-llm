# ====================================================================== launcher (single GPU)
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

T0 = time.time()
DRY = ARGS.dry   # planning only, no GPU work (local check)
WORK = Path('dry_out_twostage') if DRY else Path(os.environ.get('NTL_WORK', '/content/ntl_twostage'))
os.environ['NTL_WORK'] = str(WORK)
OUT = WORK / 'lora_twostage'
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get('NTL_DATA', 'data'))


def say(msg):
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


ast.parse(WORKER_SRC, filename='worker.py')
train_csv = DATA / 'train.csv'
train = pd.read_csv(train_csv, dtype=str)
assert train.id.is_unique and train.notna().all().all()
LABELS = np.array(sorted(train.model.unique()))
assert len(LABELS) == 18
CLASS_LAB = np.array([next(i for i, lab in enumerate(LABS) if m.startswith(lab)) for m in LABELS])
n_orig = len(train)
host_fold = np.full(n_orig, -1)
for k, (_, va) in enumerate(GroupKFold(n_splits=HP['n_folds']).split(np.zeros(n_orig), train.model, train.prompt_id)):
    host_fold[va] = k
pd.DataFrame({'id': train.id, 'prompt_id': train.prompt_id, 'model': train.model, 'fold': host_fold}).to_csv(
    OUT / 'folds.csv', index=False)

# ---- public pool (stage 1). Exact-text overlap with host train is dropped (leakage guard).
pool_path = Path(os.environ.get('NTL_POOL', str(DATA / 'pool_v1.parquet')))
if DRY and not pool_path.exists():
    pool_path = Path('research/ext_data/twostage/pool_v1.parquet')
pool = pd.read_parquet(pool_path)
pool = pool[pool['model'].isin(LABELS)].copy()
norm = lambda s: s.astype(str).str.lower().str.split().str.join(' ')
n_raw = len(pool)
pool = pool[~norm(pool.text).isin(set(norm(train.text)))]
pool = pd.DataFrame({'id': 'pool_' + pool['id'].astype(str), 'prompt_id': 'pool_' + pool['prompt_id'].astype(str),
                     'text': pool['text'].astype(str), 'model': pool['model'].astype(str)})
assert pool.id.is_unique and not set(pool.id) & set(train.id)
train = pd.concat([train[['id', 'prompt_id', 'text', 'model']], pool], ignore_index=True)
train_csv = WORK / 'train_plus_pool.csv'
train.to_csv(train_csv, index=False)
y = np.searchsorted(LABELS, train.model.to_numpy(dtype=str))
groups = train.prompt_id.to_numpy(dtype=str)
char_len = train.text.str.len().to_numpy()
n = len(train)
say(f'host {n_orig} rows; pool {n - n_orig} rows ({n_raw - (n - n_orig)} dropped as host duplicates), '
    f'{len(set(groups[n_orig:]))} prompts, from {pool_path}; per label {np.bincount(y[n_orig:], minlength=18).tolist()}')


def size_rule(remaining, rng):
    if remaining <= 3:
        return remaining
    if remaining == 4:
        return 2
    return int(rng.integers(2, 4))


def plan_job(fit, seed, n_epochs):
    """Per epoch: same-prompt micro-batches of size 2-3 covering every fitting row once, length-grouped updates of
    about eff_batch rows, update order shuffled (as in the Gemma trainer)."""
    rng = np.random.default_rng(seed)
    epochs = []
    for _ in range(n_epochs):
        mbs = []
        for g in sorted(set(groups[fit])):
            members = rng.permutation(fit[groups[fit] == g])
            i = 0
            while i < len(members):
                s = size_rule(len(members) - i, rng)
                mbs.append(members[i:i + s].tolist())
                i += s
        mbs = [mbs[j] for j in rng.permutation(len(mbs))]
        win = HP['length_window']
        mbs = [mb for s in range(0, len(mbs), win) for mb in sorted(mbs[s:s + win], key=lambda mb: char_len[mb].max())]
        updates, cur, cnt = [], [], 0
        for mb in mbs:
            cur.append(mb)
            cnt += len(mb)
            if cnt >= HP['eff_batch']:
                updates.append(cur)
                cur, cnt = [], 0
        if cur:
            if updates and cnt < HP['eff_batch'] // 2:
                updates[-1].extend(cur)
            else:
                updates.append(cur)
        updates = [updates[j] for j in rng.permutation(len(updates))]
        flat = [r for u in updates for mb in u for r in mb]
        assert sorted(flat) == sorted(fit.tolist()), 'every fitting row exactly once per epoch'
        assert all(len(set(groups[mb])) == 1 for u in updates for mb in u)
        epochs.append(updates)
    return epochs


def take(rows, n_prompts, per_prompt):
    keep = sorted(set(groups[rows]))[:n_prompts]
    return np.concatenate([rows[groups[rows] == g][:per_prompt] for g in keep])


host = np.arange(n_orig)
pool_rows = np.arange(n_orig, n)
pool_prompts = np.array(sorted(set(groups[pool_rows])))
parts = np.array_split(pool_prompts[np.random.default_rng(HP['seed']).permutation(len(pool_prompts))], HP['stage1_parts'])
full_jobs, spec = {}, {}
prev = None
for i, pp in enumerate(parts):
    nm = 'stage1' + 'abc'[i]
    full_jobs[nm] = (pool_rows[np.isin(groups[pool_rows], pp)], np.array([], dtype=int))
    spec[nm] = dict(init_from=prev, n_epochs=HP['stage1_epochs'], lr_mult=1.0)
    prev = nm
for k in range(3):
    full_jobs[f'fold{k}'] = (host[host_fold != k], host[host_fold == k])
full_jobs['final'] = (host, np.array([], dtype=int))
for nm in ('fold0', 'fold1', 'fold2', 'final'):
    spec[nm] = dict(init_from=prev, n_epochs=HP['stage2_epochs'], lr_mult=HP['stage2_lr_mult'])
job_seed = {name: HP['seed'] + i for i, name in enumerate(full_jobs)}
jobs = SMOKE_JOBS if SMOKE else JOBS
if ARGS.jobs:
    jobs = [j for j in jobs if j in ARGS.jobs.split(',')]
if SMOKE:
    spec['fold0'] = dict(spec['fold0'], init_from='stage1a')
plan, full_counts = {}, {}
for name, (fit, valid) in full_jobs.items():
    full_plan = plan_job(fit, job_seed[name], spec[name]['n_epochs'])
    full_counts[name] = dict(fit=int(len(fit)), valid=int(len(valid)), updates=sum(len(e) for e in full_plan),
                             train_rows=int(spec[name]['n_epochs'] * len(fit)))
    if name not in jobs:
        continue
    if SMOKE:
        fit = take(fit, *SMOKE_FIT)
        valid = take(valid, *SMOKE_VALID) if len(valid) else valid
        job_plan = plan_job(fit, job_seed[name], 1)
    else:
        job_plan = full_plan
    plan[name] = dict(fit=fit.tolist(), valid=valid.tolist(), seed=job_seed[name], **spec[name],
                      epochs=[[[[int(r) for r in mb] for mb in u] for u in ep] for ep in job_plan])
    say(f"plan {name}: fit {len(fit)} valid {len(valid)} updates {sum(len(e) for e in job_plan)} "
        f"init_from {spec[name]['init_from']} lr_mult {spec[name]['lr_mult']}")
say(f'full-run update counts {json.dumps({k: v["updates"] for k, v in full_counts.items()})}')

model_dir = None if DRY else Path(os.environ['NTL_MODEL_DIR'])
train_sha = hashlib.sha256(Path(train_csv).read_bytes()).hexdigest()
run = dict(smoke=SMOKE, backbone=BACKBONE, backbone_spec=BACKBONES[BACKBONE], hp=HP, labels=LABELS.tolist(),
           labs=LABS, class_lab=CLASS_LAB.tolist(), worker_jobs={'0': jobs}, full_counts=full_counts,
           job_seed=job_seed, train_csv=str(train_csv), train_sha256=train_sha,
           model_dir=str(model_dir) if model_dir else None,
           fold_method='sklearn GroupKFold(n_splits=5).split(X, y, groups=prompt_id) on host rows, default ordering')
(OUT / 'run_config.json').write_text(json.dumps(run, indent=2))
(OUT / 'plan.json').write_text(json.dumps(plan))
(WORK / 'worker.py').write_text(WORKER_SRC)
if DRY:
    say('dry run: plan written, worker not launched')
    sys.exit(0)

import torch  # noqa: E402
assert torch.cuda.device_count() >= 1, 'no GPU on this runtime'
del torch
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
proc = subprocess.Popen([sys.executable, '-u', str(WORK / 'worker.py'), '0'], env=env, cwd=str(WORK),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open(WORK / 'worker0.log', 'a') as handle:
    for line in proc.stdout:
        handle.write(line)
        handle.flush()
        if line.startswith('W0|') or 'Error' in line or 'Traceback' in line:
            print(line.rstrip()[:2000], flush=True)
code = proc.wait()
summary = dict(smoke=SMOKE, backbone=BACKBONE, exit_code=code, jobs={})
for name in plan:
    path = OUT / name / 'training_report.json'
    if path.exists():
        r = json.loads(path.read_text())
        summary['jobs'][name] = {k: r.get(k) for k in ('fit_rows', 'valid_rows', 'total_updates', 'pack', 'train_seconds',
                                                       'median_sec_per_row', 'peak_train', 'nonfinite_packs',
                                                       'skipped_updates', 'heldout_accuracy', 'oom_events')}
        summary['jobs'][name]['complete_json'] = (OUT / name / 'COMPLETE.json').exists()
per_row = [v['median_sec_per_row'] for v in summary['jobs'].values() if v.get('median_sec_per_row')]
if per_row:
    sec = float(np.mean(per_row))
    summary['full_run_estimate_hours'] = {k: round(v['train_rows'] * sec / 3600, 2) for k, v in full_counts.items()}
(OUT / f"summary_{'smoke' if SMOKE else 'run'}.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str), flush=True)
ok = code == 0 and all(summary['jobs'].get(nm, {}).get('complete_json') for nm in plan)
say('ALL JOBS COMPLETE' if ok else 'SOME JOBS FAILED')
if not ok:
    raise SystemExit(1)
