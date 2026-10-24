"""E049: which feature source earns its weight, per model and per family? Exact Shapley values of the 7 sources of the
strong OOF blend (research/ext_oof/q35heads, manifest weights; log-probability blend), on overall balanced accuracy,
per-label recall, family accuracy and within-family accuracy. Averaged over the bundle's 3 scenarios; stability = sign
of the overall Shapley value on each of the 5 prompt-grouped folds. CPU only, a few seconds.
Run from the repo root: python experiments/importance/source_shapley.py"""
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'paper')
from log import LOG, log_row  # noqa: E402

B = Path('research/ext_oof/q35heads')
OUT = Path('experiments/importance/out')
FIG = Path('research/importance')
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

z = np.load(B / 'oof_predictions.npz', allow_pickle=True)   # our own downloaded OOF bundle
man = json.loads((B / 'bundle_manifest.json').read_text())
classes = z['classes'].astype(str)
names = z['component_names'].astype(str).tolist()
train = pd.read_csv('data/train.csv', dtype=str).set_index('id').loc[z['ids'].astype(str)]
y = np.searchsorted(classes, train.model.to_numpy())
assert (classes[y] == train.model.to_numpy()).all()
fam = np.array([c.split('-')[0] for c in classes])          # claude, gemini, glm, gpt, kimi, qwen3.8 -> qwen3.8
fam = np.where(np.char.startswith(fam, 'qwen'), 'qwen', fam)
fold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k

W = man['weights']
GROUPS = ['svm', 'style', 'deberta', 'roberta', 'qwen8', 'qwen14', 'q35']
gof = lambda c: next(g for g in GROUPS if c == g or c.startswith(g + '_') or c.startswith(g))
assert all(gof(c) in GROUPS for c in W)
LOGP = {}   # scenario -> group -> weighted log-prob sum
for s in range(z['components'].shape[1]):
    LOGP[s] = {g: sum(W[c] * z['components'][names.index(c), s] for c in W if gof(c) == g)   # components are log-probs
               for g in GROUPS}


def recall_per_label(pred, rows):
    return np.array([(pred[rows][y[rows] == j] == j).mean() for j in range(18)])


def metrics(score, rows):
    """overall bacc, per-label recall, family bacc, within-family bacc (argmax restricted to the true family)."""
    if score is None:   # empty coalition: chance
        rec = np.full(18, 1 / 18)
        fams = {f: 1 / (fam == f).sum() for f in np.unique(fam)}
        return dict(bacc=1 / 18, **{f'rec:{c}': r for c, r in zip(classes, rec)},
                    **{f'fam:{f}': 1 / 6 for f in fams}, **{f'within:{f}': v for f, v in fams.items() if (fam == f).sum() > 1})
    pred = score.argmax(1)
    rec = recall_per_label(pred, rows)
    out = dict(bacc=rec.mean(), **{f'rec:{c}': r for c, r in zip(classes, rec)})
    pf = fam[pred]
    for f in np.unique(fam):
        lab = np.flatnonzero(fam == f)
        out[f'fam:{f}'] = np.mean([(pf[rows][y[rows] == j] == f).mean() for j in lab])
        if len(lab) > 1:
            sub = lab[score[:, lab].argmax(1)]
            out[f'within:{f}'] = np.mean([(sub[rows][y[rows] == j] == j).mean() for j in lab])
    return out


def shapley(s, rows):
    n = len(GROUPS)
    val = {}
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            sc = sum(LOGP[s][GROUPS[i]] for i in S) if S else None
            val[S] = metrics(sc, rows)
    keys = list(val[()].keys())
    phi = {g: dict.fromkeys(keys, 0.0) for g in GROUPS}
    for i, g in enumerate(GROUPS):
        rest = [j for j in range(n) if j != i]
        for r in range(n):
            wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(rest, r):
                T = tuple(sorted(S + (i,)))
                for k in keys:
                    phi[g][k] += wgt * (val[T][k] - val[S][k])
    full = val[tuple(range(n))]
    loo = {g: {k: full[k] - val[tuple(j for j in range(n) if j != i)][k] for k in keys} for i, g in enumerate(GROUPS)}
    return phi, loo, full


allrows = np.arange(len(y))
rows_out, fold_rows = [], []
for s in LOGP:
    phi, loo, full = shapley(s, allrows)
    for g in GROUPS:
        for k in phi[g]:
            rows_out.append(dict(scenario=s, group=g, metric=k, shapley=phi[g][k], loo=loo[g][k], full=full[k]))
    for k in range(5):
        pk, _, fk = shapley(s, np.flatnonzero(fold == k))
        for g in GROUPS:
            fold_rows.append(dict(scenario=s, fold=k, group=g, shapley_bacc=pk[g]['bacc'], full_bacc=fk['bacc']))
df = pd.DataFrame(rows_out)
df.to_csv(OUT / 'shapley_all.csv', index=False)
fd = pd.DataFrame(fold_rows)
fd.to_csv(OUT / 'shapley_bacc_per_fold.csv', index=False)
avg = df.groupby(['group', 'metric'])[['shapley', 'loo', 'full']].mean().reset_index()
avg.to_csv(OUT / 'shapley_mean_over_scenarios.csv', index=False)
fpos = fd.groupby(['group', 'fold']).shapley_bacc.mean().gt(0).groupby('group').sum()

# summary tables
piv = lambda pre: avg[avg.metric.str.startswith(pre)].pivot(index='group', columns='metric', values='shapley').loc[GROUPS]
lab = piv('rec:')
lab.columns = [c[4:] for c in lab.columns]
lab = lab[classes]
lab.to_csv(OUT / 'shapley_per_label.csv')
famt = pd.concat([piv('fam:'), piv('within:')], axis=1)
famt.to_csv(OUT / 'shapley_per_family.csv')
ov = avg[avg.metric == 'bacc'].set_index('group').loc[GROUPS]
full_bacc = float(ov.full.iloc[0])

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(13, 4.2))
v = np.abs(lab.to_numpy()).max()
im = ax.imshow(lab.to_numpy(), cmap='RdBu', vmin=-v, vmax=v, aspect='auto')
ax.set_xticks(range(18), lab.columns, rotation=60, ha='right', fontsize=8)
ax.set_yticks(range(len(GROUPS)), GROUPS)
for i in range(len(GROUPS)):
    for j in range(18):
        ax.text(j, i, f'{lab.iat[i, j] * 100:.0f}', ha='center', va='center', fontsize=6.5)
ax.set_title(f'Shapley contribution to per-label recall (points), strong OOF blend, bacc {full_bacc:.4f}')
fig.colorbar(im, ax=ax, fraction=0.02)
fig.tight_layout()
fig.savefig(FIG / 'shapley_per_label.png', dpi=130)

md = [f'# E049 Source Shapley (strong OOF blend, bacc {full_bacc:.4f}, mean of 3 scenarios)', '',
      '| source | Shapley bacc | leave-one-out | folds with Shapley > 0 |', '|---|---|---|---|']
md += [f'| {g} | {ov.shapley[g]:+.4f} | {ov.loo[g]:+.4f} | {int(fpos[g])}/5 |' for g in GROUPS]
md += ['', '## Shapley per family (fam = right family; within = right model inside the true family)', '',
       '| source | ' + ' | '.join(famt.columns) + ' |', '|---' * (len(famt.columns) + 1) + '|']
md += [f'| {g} | ' + ' | '.join(f'{x:+.3f}' for x in famt.loc[g]) + ' |' for g in GROUPS]
md += ['', '## Top source per label (Shapley on recall)', '']
md += [f'- {c}: ' + ', '.join(f'{g} {lab.at[g, c]:+.3f}' for g in lab[c].sort_values(ascending=False).index[:3]) for c in classes]
(FIG / 'SHAPLEY.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print('\n'.join(md))

rows = [r for r in csv.DictReader(open(LOG, encoding='utf-8')) if not r['id'].startswith('E049')]
with open(LOG, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for g in GROUPS:
    log_row(id=f'E049_{g}', category='analysis', metric='shapley_bacc_oof',
            description=f'Exact Shapley value of source {g} in the strong OOF blend (7 sources, manifest weights, log-prob blend)',
            config='research/ext_oof/q35heads; mean of 3 scenarios', baseline=0, value=f'{ov.shapley[g]:.4f}',
            folds_pos=f'{int(fpos[g])}/5', files='experiments/importance/out/*.csv; research/importance/*',
            notes=f'leave-one-out {ov.loo[g]:+.4f}; full blend bacc {full_bacc:.4f}')
