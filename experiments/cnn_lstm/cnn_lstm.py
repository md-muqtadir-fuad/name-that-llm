"""E050: light CPU networks instead of LLMs. Byte-level CNN and word-level BiLSTM, two-stage like the public LoRAs:
stage 1 on the public pool only (nishantdahal/ntl-handwriting-blend pool_v4.parquet, 75,783 passages, CC-BY-SA-4.0;
exact host duplicates dropped), shared by all folds because it holds no host rows; stage 2 fine-tunes on each fold's
host training rows (GroupKFold(5) by prompt_id, default order). Writes OOF log-probabilities.
Usage (repo root): python experiments/cnn_lstm/cnn_lstm.py cnn|lstm   (env SMOKE=1: tiny run to check the code)
                   python experiments/cnn_lstm/cnn_lstm.py eval        (standalone + added to the strong OOF blend; logs E050)"""
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

ARCH = sys.argv[1]
SMOKE = os.environ.get('SMOKE') == '1'
OUT = Path('experiments/cnn_lstm/out' + ('_smoke' if SMOKE else ''))
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(os.cpu_count())
T0 = time.time()
say = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)

train = pd.read_csv('data/train.csv', dtype=str)
LABELS = np.array(sorted(train.model.unique()))
y = np.searchsorted(LABELS, train.model.to_numpy())
fold = np.full(len(train), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])

HP = dict(cnn=dict(max_len=1024, s1_epochs=3, s1_lr=2e-3, s2_epochs=4, s2_lr=5e-4, bs=64),
          lstm=dict(max_len=256, s1_epochs=2, s1_lr=2e-3, s2_epochs=4, s2_lr=5e-4, bs=64, vocab=30000))


class CharCNN(nn.Module):
    """UTF-8 bytes -> 3 conv layers -> masked max + mean pooling."""
    def __init__(self, k=18):
        super().__init__()
        self.emb = nn.Embedding(257, 32, padding_idx=0)
        self.c1 = nn.Conv1d(32, 128, 5, padding=2)
        self.c2 = nn.Conv1d(128, 128, 5, padding=2)
        self.c3 = nn.Conv1d(128, 192, 3, padding=1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(384, k)

    def forward(self, x):
        m = (x > 0).float().unsqueeze(1)
        h = F.max_pool1d(F.relu(self.c1(self.emb(x).transpose(1, 2))), 2)
        m = F.max_pool1d(m, 2)
        h = F.max_pool1d(F.relu(self.c2(h)), 2)
        m = F.max_pool1d(m, 2)
        h = F.relu(self.c3(h))
        mx = (h - (1 - m) * 1e4).max(2).values
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1)
        return self.fc(self.drop(torch.cat([mx, mean], 1)))


class BiLSTM(nn.Module):
    """Case-kept word and punctuation tokens -> BiLSTM -> attention + max pooling."""
    def __init__(self, n_vocab, k=18):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.att = nn.Linear(256, 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(512, k)

    def forward(self, x):
        lens = (x > 0).sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=x.shape[1])
        m = (x > 0).unsqueeze(2)
        a = torch.softmax(self.att(h).masked_fill(~m, -1e4), 1)
        mx = h.masked_fill(~m, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([(a * h).sum(1), mx], 1)))


TOK = re.compile(r"\w+|[^\w\s]")


def encode(texts, vocab=None):
    if ARCH == 'cnn':
        return [np.frombuffer(t.encode('utf-8')[:HP['cnn']['max_len']], dtype=np.uint8).astype(np.int64) + 1 for t in texts]
    return [np.array([vocab.get(w, 1) for w in TOK.findall(t)[:HP['lstm']['max_len']]] or [1], dtype=np.int64) for t in texts]


def batches(seqs, idx, bs, rng=None):
    """Length-bucketed batches (shuffled when rng is given) padded to the batch maximum."""
    idx = np.array(idx)
    if rng is not None:
        idx = rng.permutation(idx)
        chunks = [sorted(idx[s:s + bs * 50], key=lambda i: len(seqs[i])) for s in range(0, len(idx), bs * 50)]
        bl = [c[s:s + bs] for c in chunks for s in range(0, len(c), bs)]
        bl = [bl[j] for j in rng.permutation(len(bl))]
    else:
        idx = sorted(idx, key=lambda i: len(seqs[i]))
        bl = [idx[s:s + bs] for s in range(0, len(idx), bs)]
    for b in bl:
        L = max(len(seqs[i]) for i in b)
        x = np.zeros((len(b), L), dtype=np.int64)
        for r, i in enumerate(b):
            x[r, :len(seqs[i])] = seqs[i]
        yield np.array(b), torch.from_numpy(x)


def fit(model, seqs, labels, idx, epochs, lr, bs, cw, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(idx) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    w = torch.tensor(cw, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b, x in batches(seqs, idx, bs, rng):
            loss = F.cross_entropy(model(x), torch.from_numpy(labels[b]), weight=w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(b), n + len(b)
        say(f'  epoch {ep + 1}/{epochs} loss {tot / n:.4f}')


@torch.no_grad()
def predict(model, seqs, idx):
    model.eval()
    out = np.zeros((len(idx), 18), dtype=np.float32)
    pos = {i: r for r, i in enumerate(idx)}
    for b, x in batches(seqs, idx, 128):
        out[[pos[i] for i in b]] = F.log_softmax(model(x), 1).numpy()
    return out


def train_arch():
    hp = HP[ARCH]
    pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet')
    norm = lambda s: s.astype(str).str.lower().str.split().str.join(' ')
    pool = pool[pool.model.isin(LABELS) & ~norm(pool.text).isin(set(norm(train.text)))].reset_index(drop=True)
    if SMOKE:
        pool = pool.sample(1500, random_state=0).reset_index(drop=True)
        hp = dict(hp, s1_epochs=1, s2_epochs=1)
    yp = np.searchsorted(LABELS, pool.model.to_numpy())
    say(f'{ARCH}: pool {len(pool)} rows, host {len(train)} rows, hp {hp}')
    vocab = None
    if ARCH == 'lstm':   # vocabulary from the pool only (no host text, so no validation-fold leakage)
        cnt = Counter(w for t in pool.text for w in TOK.findall(t))
        vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(hp['vocab'])) if c >= 3}
    seqs = encode(pool.text.tolist() + train.text.tolist(), vocab)
    n_pool = len(pool)
    labels = np.concatenate([yp, y])
    host = np.arange(n_pool, n_pool + len(train))
    make = lambda: CharCNN() if ARCH == 'cnn' else BiLSTM(len(vocab) + 2)
    torch.manual_seed(0)
    model = make()
    say(f'stage 1: {sum(p.numel() for p in model.parameters()):,} parameters')
    cnt = np.bincount(yp, minlength=18).astype(float)
    fit(model, seqs, labels, np.arange(n_pool), hp['s1_epochs'], hp['s1_lr'], hp['bs'], (cnt / cnt.mean()) ** -0.5, 1)
    s1_state = {k: v.clone() for k, v in model.state_dict().items()}
    s1 = predict(model, seqs, host)
    say(f'stage 1 only (pool), host bacc {bacc(y, s1.argmax(1)):.4f}')
    oof = np.full((len(train), 18), np.nan, dtype=np.float32)
    for k in ([0] if SMOKE else range(5)):
        model.load_state_dict(s1_state)
        tr, va = host[fold != k], host[fold == k]
        fit(model, seqs, labels, tr, hp['s2_epochs'], hp['s2_lr'], 32, np.ones(18), 10 + k)
        oof[fold == k] = predict(model, seqs, va)
        say(f'fold {k}: bacc {bacc(y[fold == k], oof[fold == k].argmax(1)):.4f}')
    ok = np.isfinite(oof).all(1)
    say(f'{ARCH} OOF bacc {bacc(y[ok], oof[ok].argmax(1)):.4f}')
    np.savez(OUT / f'oof_{ARCH}.npz', ids=train.id.to_numpy(dtype=str), oof=oof, stage1=s1, labels=LABELS,
             seconds=time.time() - T0)


def evaluate():
    z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
    assert (z['classes'].astype(str) == LABELS).all()
    pos = pd.Series(np.arange(len(train)), index=train.id).loc[z['ids'].astype(str)].to_numpy()
    ref = np.full_like(z['reference'], np.nan)
    ref[:, pos] = z['reference']                       # (3 scenarios, rows in train order, 18) log-probs
    members = {a: np.load(OUT / f'oof_{a}.npz')['oof'] for a in ('cnn', 'lstm') if (OUT / f'oof_{a}.npz').exists()}
    if len(members) == 2:
        members['cnn+lstm'] = (members['cnn'] + members['lstm']) / 2
    GRID = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows, res = [], []
    for name, m in members.items():
        stage1 = (np.load(OUT / f'oof_{name}.npz')['stage1'] if name in ('cnn', 'lstm') else None)
        per_fold = []
        for s in range(ref.shape[0]):
            score = {w: (ref[s] + w * m).argmax(1) for w in GRID}
            pred = np.empty(len(y), dtype=int)
            for k in range(5):   # weight chosen on the other 4 folds, applied to fold k
                o = fold != k
                wk = max(GRID, key=lambda w: bacc(y[o], score[w][o]))
                pred[fold == k] = score[wk][fold == k]
                per_fold.append(dict(scenario=s, fold=k, w=wk, base=bacc(y[fold == k], score[0][fold == k]),
                                     value=bacc(y[fold == k], pred[fold == k])))
            res.append(dict(member=name, scenario=s, base=bacc(y, score[0]), value=bacc(y, pred),
                            best_fixed_w=max(GRID, key=lambda w: bacc(y, score[w]))))
        pf = pd.DataFrame(per_fold)
        pf['member'] = name
        rows.append(pf)
        r = pd.DataFrame([x for x in res if x.get('member') == name])
        fpos = int((pf.groupby('fold').value.mean() - pf.groupby('fold').base.mean()).gt(0).sum())
        alone = bacc(y, m.argmax(1))
        s1 = f'; stage-1-only host bacc {bacc(y, stage1.argmax(1)):.4f}' if stage1 is not None else ''
        print(f'{name}: standalone OOF {alone:.4f}{s1}; blend {r.base.mean():.4f} -> {r.value.mean():.4f} '
              f'({r.value.mean() - r.base.mean():+.4f}, {fpos}/5 folds, chosen w {sorted(set(pf.w))})')
        res_row = dict(id=f'E050_{name}', category='member', metric='oof_bacc_strong_blend_lofo_w',
                       description=f'Light CPU network ({name}) two-stage: pool_v4 stage 1, host stage 2; added to strong OOF '
                                   'blend with weight chosen leave-one-fold-out',
                       config=str({a: HP[a] for a in HP if a in name}), baseline=f'{r.base.mean():.4f}',
                       value=f'{r.value.mean():.4f}', folds_pos=f'{fpos}/5',
                       files='experiments/cnn_lstm/out/*', notes=f'standalone OOF {alone:.4f}{s1}')
        res_row['delta'] = f'{r.value.mean() - r.base.mean():+.4f}'
        res.append(dict(log=res_row))
    pd.concat(rows).to_csv(OUT / 'blend_per_fold.csv', index=False)
    pd.DataFrame([x for x in res if 'log' not in x]).to_csv(OUT / 'blend_summary.csv', index=False)
    sys.path.insert(0, 'paper')
    from log import LOG, log_row
    keep = [r for r in csv.DictReader(open(LOG, encoding='utf-8')) if not r['id'].startswith('E050')]
    with open(LOG, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)
    for x in res:
        if 'log' in x:
            log_row(**x['log'])


if __name__ == '__main__':
    evaluate() if ARCH == 'eval' else train_arch()
