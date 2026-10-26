"""E061: dedicated attention-BiLSTM for gemini-3.7-flash vs gemini-3.8-flash (the most confused pair; H1: near-identical).
Data: prompts answered by BOTH models (pool_v4 + host train), so topic cannot separate them.
A) GroupKFold(5) by prompt over pool+host pairs: single-passage balanced accuracy and PAIRED accuracy (given both answers
   to a prompt, is the 3.8 answer the one scored higher?).  B) train on pool pairs only, test on the 150 host prompts
   (the real distribution), compared with the blend's pairwise accuracy there (~0.60, H2).  Null: labels swapped within
   prompt. Architecture = E050 word BiLSTM (emb 128, BiLSTM 128, attention + max pooling), binary head. CPU only.
Run from the repo root: python experiments/pair_lstm/gem37_38.py"""
import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch import nn

sys.path.insert(0, 'paper')
try:   # repo logging helper; absent on Kaggle (results are logged locally from the output CSV)
    from log import LOG, log_row  # noqa: E402
except ImportError:
    LOG = None

A, B = 'gemini-3.7-flash', 'gemini-3.8-flash'
OUT = Path('experiments/pair_lstm/out')
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(os.cpu_count())
T0 = time.time()
say = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
TOK = re.compile(r"\w+|[^\w\s]")
EPOCHS, LR, BS, MAXLEN, VOCAB = 6, 1.5e-3, 32, 256, 20000

pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet', columns=['prompt_id', 'text', 'model'])
host = pd.read_csv('data/train.csv', dtype=str)[['prompt_id', 'text', 'model']]
df = pd.concat([pool.assign(prompt_id='pool_' + pool.prompt_id.astype(str), part='pool'),
                host.assign(prompt_id='host_' + host.prompt_id, part='host')], ignore_index=True)
df = df[df.model.isin([A, B])]
both = df.groupby('prompt_id').model.nunique()
df = df[df.prompt_id.isin(both[both == 2].index)].drop_duplicates(['prompt_id', 'model']).reset_index(drop=True)
df['y'] = (df.model == B).astype(int)
say(f"paired prompts: pool {df[df.part == 'pool'].prompt_id.nunique()}, host {df[df.part == 'host'].prompt_id.nunique()}")


class AttnBiLSTM(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.att = nn.Linear(256, 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        lens = (x > 0).sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=x.shape[1])
        m = (x > 0).unsqueeze(2)
        a = torch.softmax(self.att(h).masked_fill(~m, -1e4), 1)
        mx = h.masked_fill(~m, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([(a * h).sum(1), mx], 1))).squeeze(1)


def encode(texts, vocab):
    return [np.array([vocab.get(w, 1) for w in TOK.findall(t)[:MAXLEN]] or [1], dtype=np.int64) for t in texts]


def batches(seqs, idx, rng=None):
    idx = np.array(idx)
    idx = rng.permutation(idx) if rng is not None else idx
    for s in range(0, len(idx), BS):
        b = idx[s:s + BS]
        x = np.zeros((len(b), max(len(seqs[i]) for i in b)), dtype=np.int64)
        for r, i in enumerate(b):
            x[r, :len(seqs[i])] = seqs[i]
        yield b, torch.from_numpy(x)


def train_predict(tr, te, y):
    vocab_cnt = Counter(w for t in df.text.iloc[tr] for w in TOK.findall(t))
    vocab = {w: i + 2 for i, (w, c) in enumerate(vocab_cnt.most_common(VOCAB)) if c >= 2}
    seqs = encode(df.text.tolist(), vocab)
    torch.manual_seed(0)
    model = AttnBiLSTM(len(vocab) + 2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for _ in range(EPOCHS):
        model.train()
        for b, x in batches(seqs, tr, rng):
            loss = F.binary_cross_entropy_with_logits(model(x), torch.tensor(y[b], dtype=torch.float32))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    s = np.zeros(len(te))
    pos = {i: r for r, i in enumerate(te)}
    with torch.no_grad():
        for b, x in batches(seqs, te):
            s[[pos[i] for i in b]] = model(x).numpy()
    return s


def metrics(rows, score, y):
    pred = (score > np.median(score)).astype(int)   # threshold at the median: classes are balanced by construction
    single = np.mean([(pred[y[rows] == k] == k).mean() for k in (0, 1)])
    d = pd.DataFrame({'p': df.prompt_id.to_numpy()[rows], 'y': y[rows], 's': score})
    w = d.pivot_table(index='p', columns='y', values='s')
    paired = float((w[1] > w[0]).mean())
    return single, paired


res = []
y_true = df.y.to_numpy()
rng0 = np.random.default_rng(1)
flip = dict(zip(df.prompt_id.unique(), rng0.integers(0, 2, df.prompt_id.nunique())))
y_null = np.where(df.prompt_id.map(flip).to_numpy() == 1, 1 - y_true, y_true)
for name, yy in (('real', y_true), ('null', y_null)):
    oof = np.zeros(len(df))
    for k, (tr, te) in enumerate(GroupKFold(5).split(df, yy, df.prompt_id)):
        oof[te] = train_predict(tr, te, yy)
        say(f'{name} fold {k} done')
    single, paired = metrics(np.arange(len(df)), oof, yy)
    res.append(dict(test=f'A_groupcv_{name}', single_bacc=single, paired_acc=paired, n=len(df)))
    say(f'A {name}: single {single:.3f} paired {paired:.3f}')
tr = np.flatnonzero(df.part == 'pool')
te = np.flatnonzero(df.part == 'host')
single, paired = metrics(te, train_predict(tr, te, y_true), y_true)
res.append(dict(test='B_pool_to_host', single_bacc=single, paired_acc=paired, n=len(te)))
say(f'B pool->host: single {single:.3f} paired {paired:.3f} (host rows {len(te)})')
pd.DataFrame(res).to_csv(OUT / 'gem37_38.csv', index=False)
if LOG is None:
    sys.exit(0)

keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E061')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
    wr.writeheader()
    wr.writerows(keep)
r = {x['test']: x for x in res}
log_row(id='E061', category='member', metric='pair_bacc', description='Dedicated attention-BiLSTM for gemini-3.7-flash vs 3.8-flash on paired prompts (pool_v4 + host)',
        config=f'word BiLSTM 128 + attention/max pool, {EPOCHS} ep, lr {LR}, vocab {VOCAB}, maxlen {MAXLEN}',
        baseline=f"{r['A_groupcv_null']['single_bacc']:.4f}", value=f"{r['A_groupcv_real']['single_bacc']:.4f}",
        files='experiments/pair_lstm/out/gem37_38.csv',
        notes=f"paired acc {r['A_groupcv_real']['paired_acc']:.3f} (null {r['A_groupcv_null']['paired_acc']:.3f}); "
              f"pool->host single {r['B_pool_to_host']['single_bacc']:.3f} paired {r['B_pool_to_host']['paired_acc']:.3f}; "
              'blend pairwise on host ~0.60 (H2)')
