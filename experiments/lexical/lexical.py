"""Per-model lexical/stylometric profiles, with focus on separating sibling variants.
Topic is controlled by design: every prompt is answered by (almost) every model."""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = sys.argv[1] if len(sys.argv) > 1 else 'research/lexical_properties.md'
tr = pd.read_csv('data/train.csv')
FAM = {'OpenAI': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],
       'Google': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
       'Anthropic': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],
       'ZAI': ['glm-5.3', 'glm-5.3-flash'],
       'Confusable: GLM vs claude-opus-5': ['glm-5.3', 'glm-5.3-flash', 'claude-opus-5'],
       'Confusable: kimi vs gpt-5.5': ['kimi-k3', 'gpt-5.5']}

from stylo_feats import feats, WORD  # noqa: E402

F = pd.DataFrame([feats(t) for t in tr.text])
F['model'] = tr.model.values
groups = tr.prompt_id.values
models = sorted(tr.model.unique())
prof = F.groupby('model').mean()
lines = ['# Lexical and stylometric properties per model variant', '',
         'Train set only (2,699 passages, 150 prompts). Each prompt is answered by every model, so differences are not topic effects.',
         'Rates: /100w = per 100 words; 0/1 features are the share of passages containing the item.', '']

# 1) full profile table (transposed: features x models), grouped by family
order = sum([FAM[k] for k in ['OpenAI', 'Google', 'Anthropic', 'ZAI']], []) + ['qwen3.8-max', 'kimi-k3']
short = {m: m.replace('gemini-', 'gem-').replace('claude-', 'cl-').replace('-preview', '') for m in order}
lines += ['## 1. Full profile (per-model means)', '',
          '| feature | ' + ' | '.join(short[m] for m in order) + ' |', '|---|' + '---|' * len(order)]
for col in prof.columns:
    v = prof.loc[order, col].values
    hi, lo = v.max(), v.min()
    cells = [(f'**{x:.2f}**' if x == hi else f'_{x:.2f}_' if x == lo else f'{x:.2f}') for x in v]
    lines.append(f'| {col} | ' + ' | '.join(cells) + ' |')
lines.append('')
lines.append('Bold = highest model, italic = lowest.')
lines.append('')


# 2) within-family discriminative features: paired by prompt -> per-feature eta^2 (between-model variance share after
# removing the prompt mean), plus grouped-CV accuracy of a stylometric-only classifier within the family
def within_family(members):
    sub = F[F.model.isin(members)].copy()
    g = groups[sub.index]
    X = sub.drop(columns='model')
    Xc = X - X.groupby(g).transform('mean')   # remove prompt (topic/length-request) effect
    res = []
    for col in X.columns:
        tot = (Xc[col] ** 2).sum()
        if tot <= 0:
            continue
        between = sum(len(Xc[sub.model == m]) * Xc.loc[sub.model == m, col].mean() ** 2 for m in members)
        res.append((col, between / tot))
    res.sort(key=lambda r: -r[1])
    yv = sub.model.values
    gkf = GroupKFold(5)
    pred = np.empty(len(sub), dtype=object)
    predc = np.empty(len(sub), dtype=object)
    for a, b in gkf.split(X, yv, g):
        for XX, P in ((X, pred), (Xc, predc)):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
            clf.fit(XX.iloc[a], yv[a])
            P[b] = clf.predict(XX.iloc[b])
    return res, balanced_accuracy_score(yv, pred), balanced_accuracy_score(yv, predc), sub, Xc


lines += ['## 2. What separates sibling variants', '',
          'eta² = share of within-prompt variance explained by which model wrote the passage (prompt mean removed). '
          'Classifier = logistic regression on these ~55 hand features only, 5-fold GroupKFold by prompt, balanced accuracy '
          '(chance = 1/k). "prompt-centred" = features minus their prompt mean, the oracle upper bound if test prompts were known.', '']
for fam, members in FAM.items():
    res, ba, bac, sub, Xc = within_family(members)
    lines += [f'### {fam} ({len(members)} variants, chance {1 / len(members):.2f})', '',
              f'Hand-feature classifier: **{ba:.3f}** balanced accuracy (prompt-centred: {bac:.3f}).', '',
              '| feature | eta² | ' + ' | '.join(members) + ' |', '|---|---|' + '---|' * len(members)]
    for col, e in res[:12]:
        lines.append(f'| {col} | {e:.3f} | ' + ' | '.join(f'{prof.loc[m, col]:.2f}' for m in members) + ' |')
    lines.append('')


# 3) distinctive words/bigrams per model vs its siblings: log-odds ratio with informative Dirichlet prior (Monroe et al.)
def tokens(t):
    w = [x.lower() for x in WORD.findall(t)]
    return w + [a + ' ' + b for a, b in zip(w, w[1:])]


cnt = {m: Counter() for m in models}
docfreq = {m: Counter() for m in models}
for t, m in zip(tr.text, tr.model):
    tk = tokens(t)
    cnt[m].update(tk)
    docfreq[m].update(set(tk))
prior = Counter()
for m in models:
    prior.update(cnt[m])
a0 = sum(prior.values())


def log_odds(m, others):
    ci = cnt[m]
    cj = Counter()
    for o in others:
        cj.update(cnt[o])
    ni, nj = sum(ci.values()), sum(cj.values())
    out = []
    for wd, aw in prior.items():
        if docfreq[m][wd] < 5:   # must appear in >= 5 of the model's passages (avoid prompt-specific words)
            continue
        yi, yj = ci[wd], cj[wd]
        d = np.log((yi + aw) / (ni + a0 - yi - aw)) - np.log((yj + aw) / (nj + a0 - yj - aw))
        z = d / np.sqrt(1 / (yi + aw) + 1 / (yj + aw))
        out.append((z, wd, docfreq[m][wd], sum(docfreq[o][wd] for o in others) / len(others)))
    out.sort(reverse=True)
    return out


lines += ['## 3. Signature words and bigrams', '',
          'Log-odds ratio with informative Dirichlet prior (z-score), model vs its siblings (vs all other models for '
          'single-model labs). Only items in at least 5 of the model\'s 150 passages. Format: item (docs with it in this '
          'model / mean docs in siblings).', '']
for m in order:
    fam = next((k for k in ['OpenAI', 'Google', 'Anthropic', 'ZAI'] if m in FAM[k]), None)
    others = [o for o in FAM[fam] if o != m] if fam else [o for o in models if o != m]
    top = log_odds(m, others)[:18]
    lines.append(f'- **{m}** vs {"siblings" if fam else "all others"}: ' +
                 ', '.join(f'"{wd}" ({df}/{od:.0f})' for z, wd, df, od in top))
lines.append('')

# 4) opening words
lines += ['## 4. Most common first two words per model', '']
for m in order:
    o = Counter(' '.join(t.split()[:2]).lower() for t in tr.text[tr.model == m]).most_common(5)
    lines.append(f'- **{m}**: ' + ', '.join(f'"{k}" ({v})' for k, v in o))
lines.append('')
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote', OUT)
