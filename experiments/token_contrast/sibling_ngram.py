"""E055: token-level signatures between confusable models (watermark-style 'token contrast' test).
For each model pair, take only prompts that BOTH models answered (pool_v4 + train), so topic cannot separate them.
Tokenise with the Qwen3 tokenizer (local copy), then a hashed logistic regression on token n-grams of order 1, 1-2 and
1-3 predicts which of the two wrote each passage; 5-fold GroupKFold by prompt, balanced accuracy.
Null: labels swapped at random within each prompt (keeps the paired structure, destroys the label-text link).
Reading: gains of 1-2 / 1-3 over unigrams = context-dependent token preferences (style or a watermark-like bias);
a pair at the null level has no token-level signature in this data. CPU only.
Run from the repo root: python experiments/token_contrast/sibling_ngram.py"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

sys.path.insert(0, 'paper')
from log import LOG, log_row  # noqa: E402

OUT = Path('experiments/token_contrast/out')
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = [('gemini-3.7-flash', 'gemini-3.8-flash'), ('glm-5.3', 'glm-5.3-flash'), ('gemini-3.6-flash', 'gemini-3.7-flash'),
         ('gpt-5.6-sol', 'gpt-5.6-terra'), ('gpt-5.6-luna', 'gpt-5.6-sol'), ('claude-opus-4-8', 'claude-sonnet-5'),
         ('claude-fable-5', 'glm-5.3'), ('gpt-5.5', 'gpt-5.6-terra'), ('gemini-3.6-flash', 'gemini-3.8-flash'),
         ('gpt-6-astra', 'claude-opus-5')]          # last = easy cross-family control
MAX_PROMPTS = 2500
T0 = time.time()
tok = AutoTokenizer.from_pretrained('research/ext_data/hf/Qwen3-0.6B-Base')
pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet', columns=['prompt_id', 'text', 'model'])
host = pd.read_csv('data/train.csv', dtype=str)[['prompt_id', 'text', 'model']]
df = pd.concat([pool.assign(prompt_id='pool_' + pool.prompt_id.astype(str)), host.assign(prompt_id='host_' + host.prompt_id)],
               ignore_index=True)
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])


def score(docs, y, groups, order):
    hv = HashingVectorizer(analyzer=lambda d: [' '.join(d[i:i + n]) for n in range(1, order + 1) for i in range(len(d) - n + 1)],
                           n_features=2 ** 21, alternate_sign=False, norm=None)
    X = TfidfTransformer(sublinear_tf=True).fit_transform(hv.transform(docs))
    pred = np.empty(len(y), dtype=int)
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = LogisticRegression(C=2.0, max_iter=2000, solver='liblinear').fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return bacc(y, pred)


rows, rng = [], np.random.default_rng(0)
for a, b in PAIRS:
    d = df[df.model.isin([a, b])]
    both = d.groupby('prompt_id').model.nunique()
    keep = both[both == 2].index.to_numpy()
    keep = rng.permutation(keep)[:MAX_PROMPTS]
    d = d[d.prompt_id.isin(keep)].drop_duplicates(['prompt_id', 'model']).sort_values(['prompt_id', 'model'])
    docs = [list(map(str, ids)) for ids in tok(d.text.tolist(), add_special_tokens=False)['input_ids']]
    y = (d.model.to_numpy() == b).astype(int)
    g = d.prompt_id.to_numpy()
    res = dict(pair=f'{a} vs {b}', prompts=len(keep), passages=len(d))
    for order in (1, 2, 3):
        res[f'acc_{order}gram'] = score(docs, y, g, order)
    nulls = []
    for _ in range(2):   # within-prompt label swap
        flip = dict(zip(keep, rng.integers(0, 2, len(keep))))
        yn = np.where(np.array([flip[p] for p in g]) == 1, 1 - y, y)
        nulls.append(score(docs, yn, g, 2))
    res['null_mean'], res['null_max'] = float(np.mean(nulls)), float(np.max(nulls))
    rows.append(res)
    print(f"[{time.time() - T0:5.0f}s] {res['pair']:42s} prompts {len(keep):5d} | uni {res['acc_1gram']:.3f} "
          f"bi {res['acc_2gram']:.3f} tri {res['acc_3gram']:.3f} | null {res['null_mean']:.3f}", flush=True)
out = pd.DataFrame(rows)
out['gain_bi_over_uni'] = out.acc_2gram - out.acc_1gram
out['gain_tri_over_bi'] = out.acc_3gram - out.acc_2gram
out.to_csv(OUT / 'sibling_ngram.csv', index=False)

keep_rows = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E055')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep_rows[0].keys()))
    wr.writeheader()
    wr.writerows(keep_rows)
for r in out.itertuples():
    log_row(id=f"E055_{r.pair.replace(' vs ', '_vs_')}", category='analysis', metric='pair_bacc_groupcv',
            description=f'Token n-gram signature test (paired prompts, Qwen3 tokens): {r.pair}',
            config=f'hashed LR, orders 1/1-2/1-3, GroupKFold(5) by prompt, <= {MAX_PROMPTS} paired prompts',
            baseline=f'{r.null_mean:.4f}', value=f'{r.acc_3gram:.4f}', files='experiments/token_contrast/out/sibling_ngram.csv',
            notes=f'uni {r.acc_1gram:.3f}, bi {r.acc_2gram:.3f}, tri {r.acc_3gram:.3f}; null max {r.null_max:.3f}; '
                  f'{r.prompts} prompts')
