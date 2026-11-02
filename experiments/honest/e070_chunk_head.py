"""E070: does a centred head trained on random 2-3 answer chunks beat one trained on whole prompts (18 answers), when the
test only has 2-3 answers per prompt? (Hypothesis from the second agent report: Gemma's twin works on test-like groups
because our trainer centres on 2-3 chunks, twostage_train.py line 276.)
Data: our Gemma-4-12B two-stage OOF features z (30,720 = 8 pooled layers x 3,840) of our GroupKFold folds 0-2; each
fold's rows were featurized by the model that did not train on them. Inside each fold, 5 prompt-grouped splits: a new
linear head (multinomial LR on standardized centred z) is fit on 4/5 of the prompts and scored on the rest.
Train-centring: 'whole prompt' (each row minus its prompt mean) vs 'chunks 2-3' (5 random chunkings, stacked).
Test-centring: random 2-3 groups (test-like, 5 draws) and whole prompt (oracle). Reference: the trained twin head.
Run from the repo root: python experiments/honest/e070_chunk_head.py"""
import glob
import sys

import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

C = 0.002


def chunks(pid, rng):
    g = np.full(len(pid), -1); nxt = 0
    for p in np.unique(pid):
        ii = rng.permutation(np.flatnonzero(pid == p)); i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4)); g[ii[i:i + n]] = nxt; nxt += 1; i += n
    return g


def centre(Z, g):
    out = Z.copy()
    for k in np.unique(g):
        ii = g == k; out[ii] -= out[ii].mean(0)
    return out


res = []
for f in sorted(glob.glob('research/kernels_2026-10-26/gemma_s2/ts/lora_twostage/fold[0-9]')):
    v = np.load(f'{f}/val_outputs.npz', allow_pickle=True)   # our own kernel output
    H = load_file(f'{f}/heads.safetensors')
    Z, y, pid = v['z'].astype(np.float32), v['y'], v['prompt_id']
    ups = np.unique(pid)
    rng = np.random.default_rng(0)
    sp = {q: i % 5 for i, q in enumerate(rng.permutation(ups))}
    sfold = np.array([sp[q] for q in pid])
    for s in range(5):
        tr, te = sfold != s, sfold == s
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
        Zs = (Z - mu) / sd
        heads = {}
        heads['whole prompt'] = LogisticRegression(C=C, max_iter=3000).fit(centre(Zs[tr], pid[tr]), y[tr])
        Xc = np.concatenate([centre(Zs[tr], chunks(pid[tr], np.random.default_rng(100 + r))) for r in range(5)])
        heads['chunks 2-3'] = LogisticRegression(C=C, max_iter=3000).fit(Xc, np.tile(y[tr], 5))
        for d in range(5):
            gt = chunks(pid[te], np.random.default_rng(1000 + 10 * s + d))
            Zt = centre(Zs[te], gt)
            for name, h in heads.items():
                res.append(dict(fold=f[-5:], split=s, draw=d, test='groups 2-3', head=name, bacc=bas(y[te], h.predict(Zt))))
            ref = centre(Z[te], gt) @ H['centered.weight'].astype(np.float32).T + H['centered.bias']
            res.append(dict(fold=f[-5:], split=s, draw=d, test='groups 2-3', head='trained twin head (ref)', bacc=bas(y[te], ref.argmax(1))))
        Zo = centre(Zs[te], pid[te])
        for name, h in heads.items():
            res.append(dict(fold=f[-5:], split=s, draw=0, test='whole prompt (oracle)', head=name, bacc=bas(y[te], h.predict(Zo))))
    print(f'{f[-5:]} done', flush=True)
R = pd.DataFrame(res)
S = R.groupby(['test', 'head']).bacc.agg(['mean', 'std', 'count'])
print(S.round(4).to_string())
P = R[R.test == 'groups 2-3'].pivot_table(index=['fold', 'split', 'draw'], columns='head', values='bacc')
dlt = P['chunks 2-3'] - P['whole prompt']
cell = dlt.groupby(level=[0, 1]).mean()
print(f"\nchunks 2-3 minus whole prompt, test groups 2-3: {dlt.mean():+.4f} (se over 15 fold-splits {cell.std() / np.sqrt(len(cell)):.4f}, "
      f"better in {(cell > 0).sum()}/{len(cell)} fold-splits)")
R.to_csv('experiments/honest/out/e070_chunk_head.csv', index=False)
log_row(id='E070', category='centring', metric='bacc_centred_head_groups2-3',
        description='centred head trained on random 2-3 answer chunks vs on whole prompts, scored on test-like 2-3 groups (Gemma two-stage OOF features, frozen)',
        config=f'our folds 0-2, 5 prompt-grouped splits each, multinomial LR C={C} on standardized centred z (30,720); chunk training = 5 random chunkings stacked; 5 test draws',
        baseline=f"{P['whole prompt'].mean():.4f}", value=f"{P['chunks 2-3'].mean():.4f}", folds_pos=f'{(cell > 0).sum()}/{len(cell)} fold-splits',
        files='experiments/honest/out/e070_chunk_head.csv',
        notes=f"diff {dlt.mean():+.4f} (se {cell.std() / np.sqrt(len(cell)):.4f}); trained twin head ref {P['trained twin head (ref)'].mean():.4f}; "
              f"oracle whole-prompt test: whole {R[(R.test != 'groups 2-3') & (R['head'] == 'whole prompt')].bacc.mean():.4f}, "
              f"chunks {R[(R.test != 'groups 2-3') & (R['head'] == 'chunks 2-3')].bacc.mean():.4f}")
