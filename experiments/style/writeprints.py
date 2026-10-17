"""Writeprints-style dense stylometric feature set (Abbasi & Chen 2008, compact version) + LR / LightGBM.

Lexical: char/word counts, letter (26) / digit (10) / special-char (~40) frequencies, upper-case ratio, word-length
distribution (1..20), vocabulary richness (TTR, hapax, dis-legomena, Yule's K, Simpson's D, Honore's R).
Syntactic: 300+ function-word frequencies, punctuation frequencies, spaCy fine POS tag (~50) frequencies and the
top POS-tag bigrams. Structural: sentence count / length stats, sentence-initial word class.
  python writeprints.py <split>
"""
import json
import re
import string
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from common import CACHE, K, N, TEXTS, evaluate, get_folds, load_bases, log_result, save_oof, y

FW = sorted(ENGLISH_STOP_WORDS)
SPECIAL = list('~@#$%^&*-_=+><[]{}/\\|') + ['—', '–', '‘', '’', '“', '”', '…', 'é']
PUNCT = list(',.;:!?\'"()')
WRE = re.compile(r"[A-Za-z’']+")
SRE = re.compile(r'(?<=[.!?])\s+')


def wp(t, tags):
    n = max(len(t), 1)
    words = WRE.findall(t)
    nw = max(len(words), 1)
    lw = [w.lower() for w in words]
    f = [np.log(n), np.log(nw), sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
         sum(c.isspace() for c in t) / n]
    low = t.lower()
    f += [low.count(ch) / n for ch in string.ascii_lowercase]
    f += [t.count(ch) / n for ch in string.digits]
    f += [t.count(ch) / n for ch in SPECIAL + PUNCT]
    wl = Counter(min(len(w), 20) for w in words)
    f += [wl.get(i, 0) / nw for i in range(1, 21)]
    cnt = Counter(lw)
    V1 = sum(1 for v in cnt.values() if v == 1)
    V2 = sum(1 for v in cnt.values() if v == 2)
    V = max(len(cnt), 1)
    s2 = sum(v * v for v in cnt.values())
    f += [V / nw, V1 / nw, V2 / nw, 1e4 * (s2 - nw) / (nw * nw), s2 / (nw * nw),
          100 * np.log(nw) / max(1 - V1 / V, 1e-3)]
    f += [cnt.get(w, 0) / nw for w in FW]
    sents = [s for s in SRE.split(t.strip()) if s]
    sl = np.array([len(WRE.findall(s)) for s in sents]) if sents else np.array([0])
    f += [len(sents), sl.mean(), sl.std(), sl.min(), sl.max(), (sl > 30).mean(), (sl < 12).mean()]
    first = [s.split()[0].lower() if s.split() else '' for s in sents]
    f += [np.mean([w in ENGLISH_STOP_WORDS for w in first]) if first else 0]
    tg = tags.split()
    nt = max(len(tg), 1)
    return f, Counter(tg), Counter(zip(tg, tg[1:])), nt


def build():
    tags = json.loads((CACHE / 'pos_tag.json').read_text(encoding='utf-8'))
    rows = [wp(t, g) for t, g in zip(TEXTS, tags)]
    tagset = sorted(set(k for r in rows for k in r[1]))
    bi = Counter()
    for r in rows:
        bi.update(r[2])
    bigr = [k for k, _ in bi.most_common(150)]
    X = np.array([r[0] + [r[1].get(k, 0) / r[3] for k in tagset] + [r[2].get(k, 0) / r[3] for k in bigr]
                  for r in rows], dtype=np.float64)
    return np.nan_to_num(X)


if __name__ == '__main__':
    split = int(sys.argv[1])
    bases = load_bases()
    folds = get_folds(split)
    X = build()
    print('writeprints features', X.shape, flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for C in (0.01, 0.03, 0.1):
        S = np.zeros((N, K))
        for a, b in folds:
            sc = StandardScaler().fit(X[a])
            m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[a]), y[a])
            S[b] = m.predict_log_proba(sc.transform(X[b]))
        name = f'writeprints_lr_C{C}'
        save_oof(name, split, S)
        r = evaluate(S, f'{name} [s{split}]', bases, folds)
        r['cand'] = name
        log_result(r, split)
    import lightgbm as lgb
    S = np.zeros((N, K))
    for a, b in folds:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                               random_state=0, verbose=-1, n_jobs=2).fit(X[a], y[a])
        S[b] = np.log(np.clip(m.predict_proba(X[b]), 1e-9, 1))
    name = 'writeprints_lgb'
    save_oof(name, split, S)
    r = evaluate(S, f'{name} [s{split}]', bases, folds)
    r['cand'] = name
    log_result(r, split)
