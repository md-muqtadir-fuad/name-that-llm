"""Builds nb/eda/ntl-eda.ipynb + kernel-metadata.json (private Kaggle CPU notebook).
Part A: EDA of the competition data (computed in the notebook). Part B: community results (numbers embedded from
research/community_2026-11-01.md, research/leader_update_2026-10-24.md and our E065 honest re-scoring).
Run from the repo root:  python nb/eda/build.py [--test]   (--test executes every code cell locally first)"""
import json
import sys
import time
from pathlib import Path

import nbformat as nbf

D = Path('nb/eda')
C = []
md = lambda s: C.append(('md', s.strip('\n')))
code = lambda s: C.append(('code', s.strip('\n')))

md(r'''
# Name That LLM: data EDA and what the public results teach

**Part A** looks at the data: how the 18 authors differ in length, typography and style, how strongly topic dominates
style, which models are near-twins, and where a prompt-grouped baseline goes wrong.
**Part B** reads the public results: how the public notebooks climbed from 0.54 to 0.664, how noisy one leaderboard
reading is, which single changes failed, why prompt-recovery tricks lose on the board, and an honest re-scoring of the
public 0.664 stack's members.

Everything in Part A runs on CPU in a few minutes. Part B uses numbers reported by the notebook authors (marked *claim*)
and our own measurements (marked *ours*).
''')
code(r'''
import glob, os, re, time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
T0 = time.time()
cand = [p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p] or ['data/train.csv']
DATA = os.path.dirname(cand[0])
train = pd.read_csv(os.path.join(DATA, 'train.csv'))
test = pd.read_csv(os.path.join(DATA, 'test.csv'))

def fam(m):
    for p, f in (('claude', 'Anthropic'), ('gemini', 'Google'), ('gpt', 'OpenAI'), ('glm', 'Zhipu'), ('kimi', 'Moonshot')):
        if m.startswith(p): return f
    return 'Alibaba'
FAM_COL = {'Anthropic': '#d97757', 'Google': '#4285f4', 'OpenAI': '#10a37f', 'Zhipu': '#8e44ad', 'Moonshot': '#2c3e50', 'Alibaba': '#e6a700'}
LABELS = sorted(train.model.unique(), key=lambda m: (fam(m), m))
L2I = {m: i for i, m in enumerate(LABELS)}
train['family'] = train.model.map(fam)
train['words'] = train.text.str.split().str.len()
y = train.model.map(L2I).to_numpy()
# the 6 labels Kaggle Benchmarks cannot generate (alekseyosipenko): no first-party extra data for them
NO_BENCH = {'claude-fable-5', 'gemini-3.7-flash', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3', 'qwen3.8-max'}
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
def fam_legend(ax, **kw):
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAM_COL.items()], frameon=False, **kw)
print(f'train {train.shape}, test {test.shape} (visible rows only; the hidden test has ~4,000 rows)')
''')

md(r'''
## A1. Structure

150 prompts, each answered once by (almost) every one of the 18 models, so a prompt is a perfect "same topic, different
author" block. The visible test has only 3 rows; the hidden test (~4,000 rows, ~40% public) is not structured like this:
the evidence in Part B says it has only a few passages per prompt.
''')
code(r'''
per_prompt = train.groupby('prompt_id').model.nunique()
fam_n = train.groupby('family').model.nunique()
print('passages per prompt:', per_prompt.value_counts().to_dict())
missing = [(p, m) for p in train.prompt_id.unique() for m in LABELS if not ((train.prompt_id == p) & (train.model == m)).any()]
print('missing (prompt, model) pairs:', missing)
print('models per family:', fam_n.to_dict())
print('words per passage: median', int(train.words.median()), '| 5-95%', train.words.quantile([.05, .95]).astype(int).tolist())
''')

md(r'''
## A2. Length carries almost no author signal

Passages range from ~40 to ~230 words, yet an additive model `log(words) ~ prompt + model` explains only a few percent
of that spread: the length was evidently set per request (each answer got its own target), not by the author or the
topic. Normalising inside each prompt (z-score over its 18 answers) shows the small author habits that remain
(gemini-3.8-flash a bit short, Claude opus-4-8 / opus-5 a bit long), all within about a quarter of a standard deviation.
Length is a weak feature; the real signal is in how the text is written.
''')
code(r'''
lw = np.log(train.words)
z = (lw - train.groupby('prompt_id').words.transform(lambda s: np.log(s).mean())) / train.groupby('prompt_id').words.transform(lambda s: np.log(s).std())
train['len_z'] = z
def r2(groups):
    pred = lw.groupby([train[g] for g in groups]).transform('mean') if len(groups) == 1 else None
    return 1 - ((lw - pred) ** 2).sum() / ((lw - lw.mean()) ** 2).sum()
r_p, r_m = r2(['prompt_id']), r2(['model'])
# additive fit by alternating means
P = np.zeros(len(lw)); M = np.zeros(len(lw)); mu = lw.mean()
for _ in range(20):
    P = (lw - mu - M).groupby(train.prompt_id).transform('mean').to_numpy()
    M = (lw - mu - P).groupby(train.model).transform('mean').to_numpy()
r_both = 1 - ((lw - mu - P - M) ** 2).sum() / ((lw - mu) ** 2).sum()
print(f'R^2 of log length: prompt alone {r_p:.2f} | model alone {r_m:.2f} | prompt + model {r_both:.2f}')
order = train.groupby('model').words.median().sort_values().index
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [3, 2]})
bp = ax[0].boxplot([train.words[train.model == m] for m in order], vert=False, patch_artist=True, showfliers=False, widths=.6)
for b, m in zip(bp['boxes'], order): b.set_facecolor(FAM_COL[fam(m)]); b.set_alpha(.75)
ax[0].set_yticks(range(1, 19), order); ax[0].set_xlabel('words per passage'); ax[0].set_title('Raw length')
mz = train.groupby('model').len_z.mean().sort_values()
ax[1].barh(mz.index, mz.values, color=[FAM_COL[fam(m)] for m in mz.index])
ax[1].axvline(0, color='k', lw=.6); ax[1].set_title('Length relative to the same prompt (mean z)')
fam_legend(ax[1], loc='lower right', fontsize=7)
plt.tight_layout(); plt.show()
''')

md(r'''
## A3. Typography fingerprints

Share of each model's passages that contain a mark at least once. These marks are the cheapest strong signal in the
competition (the public "Who am I?" typography likelihood moved the board from 0.651 to 0.654, *claim*). Families
cluster: Claude and GLM write closed em dashes (x—y) and straight apostrophes and never curly ones; GPT uses curly
apostrophes and quotes; Gemini almost never uses an em dash and uses fewer colons. Spaced em dashes are an
opus-4-6 / opus-5 habit.
''')
code(r'''
MARKS = {
    'spaced em dash " — "': r' — ', 'closed em dash "x—y"': r'\S—\S', 'en dash "–"': r'–',
    'curly apostrophe ’': r'’', "straight apostrophe '": r"\w'\w", 'curly quotes “”': r'[“”]', 'straight quotes "': r'"',
    'semicolon ;': r';', 'colon :': r':', 'parentheses (': r'\(', 'ellipsis …/...': r'…|\.\.\.', 'exclamation !': r'!',
    'question ?': r'\?', 'line break': r'\n', 'markdown **': r'\*\*'}
H = pd.DataFrame({k: train.text.str.contains(v, regex=True) for k, v in MARKS.items()}).groupby(train.model).mean().loc[LABELS]
fig, ax = plt.subplots(figsize=(13, 6.2))
im = ax.imshow(H.values, cmap='magma_r', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(H.shape[1]), H.columns, rotation=40, ha='right'); ax.set_yticks(range(18), LABELS)
for t, m in zip(ax.get_yticklabels(), LABELS): t.set_color(FAM_COL[fam(m)])
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]; ax.text(j, i, f'{v:.2f}'.lstrip('0') if v < 1 else '1', ha='center', va='center', fontsize=6.5, color='white' if v > .55 else 'black')
plt.colorbar(im, fraction=.025, label='share of passages'); ax.set_title('Typography: share of passages containing the mark (rows coloured by family)')
plt.tight_layout(); plt.show()
spread = (H.max() - H.min()).sort_values(ascending=False)
print('marks with the widest spread across models:', spread.head(6).round(2).to_dict())
''')

md(r'''
## A4. Style beyond punctuation

Per-passage style statistics, averaged per model and shown as z-scores across the 18 models.
''')
code(r'''
WORD = re.compile(r"[A-Za-z’']+")
def style(t):
    w = WORD.findall(t); n = max(len(w), 1); lw_ = [x.lower() for x in w]
    sents = [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s]
    return dict(sent_len=n / max(len(sents), 1), ttr_80=len(set(lw_[:80])) / max(min(n, 80), 1),
                word_len=np.mean([len(x) for x in w]) if w else 0, commas_100w=100 * t.count(',') / n,
                first_person_100w=100 * sum(x in {'i', 'me', 'my', 'we', 'our', 'us'} for x in lw_) / n,
                second_person_100w=100 * sum(x in {'you', 'your', 'yours'} for x in lw_) / n,
                contractions_100w=100 * len(re.findall(r"\w[’']\w", t)) / n,
                long_words_share=sum(len(x) >= 9 for x in w) / n)
S = pd.DataFrame([style(t) for t in train.text]); S['model'] = train.model.values
SM = S.groupby('model').mean().loc[LABELS]
Z = (SM - SM.mean()) / SM.std()
fig, ax = plt.subplots(figsize=(9.5, 6.2))
im = ax.imshow(Z.values, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_xticks(range(Z.shape[1]), Z.columns, rotation=35, ha='right'); ax.set_yticks(range(18), LABELS)
for t, m in zip(ax.get_yticklabels(), LABELS): t.set_color(FAM_COL[fam(m)])
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]): ax.text(j, i, f'{SM.values[i, j]:.2g}', ha='center', va='center', fontsize=6.5)
plt.colorbar(im, fraction=.03, label='z-score across models (cell text = raw mean)'); ax.set_title('Style statistics per model')
plt.tight_layout(); plt.show()
''')

md(r'''
## A5. Topic beats author, until the prompt is removed

For each passage, find its nearest neighbour (cosine on word TF-IDF). Without help, the neighbour is almost always
another answer to the **same prompt** (over 99%), and almost never the same model. Subtracting each prompt's mean
vector (possible only when the other answers to the prompt are known) breaks most of the topic match and lifts
same-model neighbours to about twice chance, no more. This is the "prompt confound": author style is a faint signal
under a loud topic signal, and on the test the other answers to a prompt are mostly absent.
''')
code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
def nn_stats(X, name):
    X = normalize(X); Sim = X @ X.T; Sim = Sim.toarray() if hasattr(Sim, 'toarray') else Sim
    np.fill_diagonal(Sim, -9); nn = Sim.argmax(1)
    return dict(view=name, same_prompt=(train.prompt_id.values[nn] == train.prompt_id.values).mean(),
                same_model=(y[nn] == y).mean(), same_family=(train.family.values[nn] == train.family.values).mean())
def centre(X):
    X = X.copy()
    for p, ii in train.groupby('prompt_id').indices.items(): X[ii] -= X[ii].mean(0)
    return X
Xw = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=20000).fit_transform(train.text)
Xc = TfidfVectorizer(sublinear_tf=True, min_df=3, analyzer='char_wb', ngram_range=(2, 5), max_features=20000).fit_transform(train.text)
R = pd.DataFrame([nn_stats(Xw, 'word TF-IDF'), nn_stats(centre(normalize(Xw).toarray()), 'word TF-IDF, prompt-centred'),
                  nn_stats(Xc, 'char 2-5 TF-IDF'), nn_stats(centre(normalize(Xc).toarray()), 'char 2-5, prompt-centred')]).set_index('view')
print(R.round(3).to_string())
ax = R.plot.barh(figsize=(9, 3.2), color=['#999999', '#e45756', '#4c78a8']); ax.set_xlabel('share of passages whose nearest neighbour is ...')
ax.axvline(1 / 18, ls=':', color='k', lw=.8); ax.text(1 / 18, 3.45, ' chance (model)', fontsize=7)
ax.legend(frameon=False, fontsize=8, loc='lower right'); ax.invert_yaxis(); plt.tight_layout(); plt.show()
''')

md(r'''
## A6. Who sounds like whom

Model centroids of prompt-centred character n-gram profiles, compared by cosine. The tree recovers the labs. The
tightest cluster by far is Gemini: the 3.6, 3.7 and 3.8 flash generations (and 3.1-pro next to them) write almost the
same way.
''')
code(r'''
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
Xcc = centre(normalize(Xc).toarray())
Cn = normalize(np.stack([Xcc[y == i].mean(0) for i in range(18)]))
Sim = Cn @ Cn.T
Lk = linkage(squareform(1 - Sim, checks=False), 'average')
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1.25, 1]})
dn = dendrogram(Lk, labels=LABELS, orientation='left', ax=ax[1], color_threshold=0, above_threshold_color='#555555')
for t in ax[1].get_yticklabels(): t.set_color(FAM_COL[fam(t.get_text())])
ax[1].set_title('Average-linkage tree on 1 - cosine'); o = dn['leaves'][::-1]
im = ax[0].imshow(Sim[np.ix_(o, o)], cmap='viridis', vmin=Sim[~np.eye(18, dtype=bool)].min(), vmax=Sim[~np.eye(18, dtype=bool)].max())
ax[0].set_xticks(range(18), [LABELS[i] for i in o], rotation=90); ax[0].set_yticks(range(18), [LABELS[i] for i in o])
for tl in list(ax[0].get_xticklabels()) + list(ax[0].get_yticklabels()): tl.set_color(FAM_COL[fam(tl.get_text())])
plt.colorbar(im, ax=ax[0], fraction=.04, label='cosine of prompt-centred centroids'); ax[0].set_title('Model similarity (diagonal off-scale)')
plt.tight_layout(); plt.show()
off = pd.Series({(LABELS[i], LABELS[j]): Sim[i, j] for i in range(18) for j in range(i + 1, 18)}).sort_values(ascending=False)
print('closest pairs:', [(a, b, round(v, 3)) for (a, b), v in off.head(6).items()])
''')

md(r'''
## A7. A prompt-grouped baseline and where it fails

Character + word TF-IDF with a multinomial logistic regression, 5-fold `GroupKFold` by prompt (a prompt is never in
train and validation at once). Scored with the competition metric, balanced accuracy.
''')
code(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import scipy.sparse as sp
folds = list(GroupKFold(5).split(train, y, train.prompt_id))
oof = np.zeros((len(y), 18))
for tr, va in folds:
    v1 = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, lowercase=False).fit(train.text.values[tr])
    v2 = TfidfVectorizer(sublinear_tf=True, token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=2, max_features=30000, lowercase=False).fit(train.text.values[tr])
    X = lambda ii: sp.hstack([v1.transform(train.text.values[ii]), v2.transform(train.text.values[ii])]).tocsr()
    oof[va] = LogisticRegression(C=20, max_iter=3000).fit(X(tr), y[tr]).predict_proba(X(va))
pred = oof.argmax(1)
ba = balanced_accuracy_score(y, pred)
fam_arr = np.array([fam(m) for m in LABELS])
wrong = pred != y; within = wrong & (fam_arr[pred] == fam_arr[y])
print(f'OOF balanced accuracy {ba:.4f} | errors: {wrong.mean():.1%} of passages, of which {within.sum() / wrong.sum():.0%} '
      f'stay inside the right family')
rec = pd.Series([(pred[y == i] == i).mean() for i in range(18)], index=LABELS)
print(f"mean recall: 12 Benchmarks-served labels {rec[[m for m in LABELS if m not in NO_BENCH]].mean():.3f} | "
      f"6 not served {rec[[m for m in LABELS if m in NO_BENCH]].mean():.3f}")
CM = confusion_matrix(y, pred, normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={'width_ratios': [1.3, 1]})
im = ax[0].imshow(CM, cmap='Blues', vmin=0, vmax=CM.max())
ax[0].set_xticks(range(18), LABELS, rotation=90); ax[0].set_yticks(range(18), LABELS)
for tl in list(ax[0].get_xticklabels()) + list(ax[0].get_yticklabels()): tl.set_color(FAM_COL[fam(tl.get_text())])
b = np.r_[0, np.cumsum(pd.Series([fam(m) for m in LABELS]).value_counts(sort=False).reindex(dict.fromkeys(fam(m) for m in LABELS)).values)] - .5
for v in b: ax[0].axhline(v, color='k', lw=.5); ax[0].axvline(v, color='k', lw=.5)
for i in range(18):
    for j in range(18):
        if CM[i, j] >= .08: ax[0].text(j, i, f'{CM[i, j]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=6, color='white' if CM[i, j] > .4 else 'k')
ax[0].set_xlabel('predicted'); ax[0].set_ylabel('true'); ax[0].set_title('Row-normalised confusion (family blocks outlined)')
rs = rec.sort_values()
ax[1].barh(rs.index, rs.values, color=[FAM_COL[fam(m)] for m in rs.index], hatch=None)
for i, m in enumerate(rs.index):
    if m in NO_BENCH: ax[1].text(rs[m] + .01, i, 'no Benchmarks data', va='center', fontsize=6.5)
ax[1].axvline(ba, color='k', ls='--', lw=.8); ax[1].set_title('Recall per model (dashed = balanced accuracy)')
plt.tight_layout(); plt.show()
''')

md(r'''
## A8. How much does knowing the other answers to a prompt help?

The same baseline, but each held-out prompt is cut into random groups of *k* passages whose mean is subtracted before
the readout (a model trained on prompt-centred features). With all 18 answers this is the "oracle" that makes
prompt-recovery tricks look great in CV. With the 2-3 answers per prompt the test seems to have, centring is no better
than reading each passage alone, and with 2 it is worse (the pair mean still carries the partner's style).
Features: 384-dim SVD of the TF-IDF (unsupervised, fit once on all train text).
''')
code(r'''
from sklearn.decomposition import TruncatedSVD
Xall = sp.hstack([TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, lowercase=False).fit_transform(train.text),
                  TfidfVectorizer(sublinear_tf=True, token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=2, max_features=30000, lowercase=False).fit_transform(train.text)]).tocsr()
E = TruncatedSVD(384, random_state=0).fit_transform(Xall); E /= np.linalg.norm(E, axis=1, keepdims=True)
Ec = E.copy()
for p, ii in train.groupby('prompt_id').indices.items(): Ec[ii] -= Ec[ii].mean(0)
rng = np.random.default_rng(0); res = {}
for k in (1, 2, 3, 6, 18):
    pr = np.zeros(len(y), int)
    for tr, va in folds:
        if k == 1:
            pr[va] = LogisticRegression(C=2, max_iter=3000).fit(E[tr], y[tr]).predict(E[va]); continue
        clf = LogisticRegression(C=2, max_iter=3000).fit(Ec[tr], y[tr])
        Ev = E[va].copy()
        for p in np.unique(train.prompt_id.values[va]):
            ii = va[train.prompt_id.values[va] == p]; ii = ii[rng.permutation(len(ii))]
            for g in range(0, len(ii), k):
                gg = np.searchsorted(va, ii[g:g + k]); Ev[gg] -= Ev[gg].mean(0) if len(gg) > 1 else 0
        pr[va] = clf.predict(Ev)
    res[k] = balanced_accuracy_score(y, pr)
res = pd.Series(res)
print('balanced accuracy by known passages per prompt:', res.round(4).to_dict())
ax = res.plot(marker='o', figsize=(6, 3.2), color='#4c78a8'); ax.set_xscale('log'); ax.set_xticks(res.index, res.index)
ax.set_xlabel('passages per prompt known together (k)'); ax.set_ylabel('balanced accuracy'); ax.set_title('Prompt centring needs company')
plt.tight_layout(); plt.show()
''')

md(r'''
## A9. Sibling pairs: separable or not?

For each pair, a binary logistic regression on character TF-IDF, prompt-grouped CV (*single*: one passage at a time).
*Paired*: both answers to the same prompt are seen together and the classifier only has to say which is which
(difference of the two vectors), the best case for any "relative" trick. Cross-family pairs are references.
''')
code(r'''
PAIRS = [('gemini-3.7-flash', 'gemini-3.8-flash'), ('glm-5.3', 'glm-5.3-flash'), ('gpt-5.6-sol', 'gpt-5.6-terra'),
         ('gpt-5.6-luna', 'gpt-5.6-terra'), ('gemini-3.6-flash', 'gemini-3.8-flash'), ('claude-opus-4-8', 'claude-sonnet-5'),
         ('claude-opus-4-6', 'claude-opus-4-8'), ('claude-fable-5', 'glm-5.3'), ('claude-opus-5', 'gpt-5.5'), ('gemini-3.1-pro-preview', 'kimi-k3')]
Xn = normalize(Xc).toarray(); rows = []
for a, b in PAIRS:
    ia, ib = L2I[a], L2I[b]; m = np.isin(y, [ia, ib]); idx = np.flatnonzero(m)
    yy = (y[idx] == ib).astype(int); g = train.prompt_id.values[idx]; ps = np.zeros(len(idx))
    for tr, va in GroupKFold(5).split(idx, yy, g):
        ps[va] = LogisticRegression(C=4, max_iter=3000).fit(Xn[idx[tr]], yy[tr]).predict(Xn[idx[va]])
    single = (ps == yy).mean()
    both = [p for p in np.unique(g) if (g == p).sum() == 2]
    D_ = np.stack([Xn[idx[(g == p) & (yy == 1)][0]] - Xn[idx[(g == p) & (yy == 0)][0]] for p in both])
    Dx = np.vstack([D_, -D_]); Dy = np.r_[np.ones(len(D_)), np.zeros(len(D_))]; Dg = np.r_[both, both]; pp = np.zeros(len(Dy))
    for tr, va in GroupKFold(5).split(Dx, Dy, Dg):
        pp[va] = LogisticRegression(C=4, max_iter=3000).fit(Dx[tr], Dy[tr]).predict(Dx[va])
    rows.append(dict(pair=f'{a} vs {b}', family='same' if fam(a) == fam(b) else 'cross', single=single, paired=(pp == Dy).mean()))
PR = pd.DataFrame(rows).set_index('pair')
print(PR.round(3).to_string())
ax = PR[['single', 'paired']].plot.barh(figsize=(9, 4.6), color=['#9ecae1', '#3182bd']); ax.axvline(.5, color='k', lw=.8, ls=':')
ax.set_xlim(.4, 1); ax.invert_yaxis(); ax.set_xlabel('accuracy (0.5 = chance)'); ax.legend(frameon=False, loc='lower right')
for t in ax.get_yticklabels(): t.set_fontweight('bold' if PR.loc[t.get_text(), 'family'] == 'cross' else 'normal')
ax.set_title('Pairwise separability (bold = cross-family reference)'); plt.tight_layout(); plt.show()
print(f'Part A done in {time.time() - T0:.0f}s')
''')

md(r'''
**Reading.** Cross-lab references separate at 0.89-0.96 from a single passage. Inside a lab it ranges from chance
(gemini-3.7 vs 3.8-flash, glm-5.3 vs flash, gpt-5.6-sol vs terra) to 0.73 (Claude opus-4-6 vs opus-4-8). Seeing both
answers to the same prompt helps the pairs that have a real difference (Claude, gemini-3.6 vs 3.8, gpt-luna vs terra)
and does nothing for the chance-level pairs. claude-fable-5 vs glm-5.3 is the odd one: two labs, yet as hard as the
hardest siblings, which matches their shared closed em dashes, straight apostrophes and colons in A3.
''')

md(r'''
# Part B: what the public results teach

Numbers below are the authors' own public-leaderboard claims unless marked *ours*. Sources: the notebooks' markdown,
pulled 2026-10-24 to 2026-11-01.

## B1. How the public notebooks climbed
''')
code(r'''
LINEAGE = [  # (public LB claim, notebook, what it added)
    (0.541, 'lightningv08 LoRA + adaptive centring', 'Qwen3-8B LoRA, frozen probes, TF-IDF SVM, adaptive centring'),
    (0.549, 'Lucas Boesen LoRA x AC', 'LoRA24 + adaptive centring'),
    (0.552, 'Lucas v16 + Claude gate', 'seed-7 LoRA, token attention, within-Claude gate'),
    (0.559, 'stylegram', 'family gates on stylegram SVC'),
    (0.592, 'nishantdahal two-stage', 'pre-train on public pool, then host folds'),
    (0.620, 'Nishant part two', 'pool 35k -> 84k (+0.04 member OOF), Qwen3-14B + Qwen2.5-14B + DeBERTa'),
    (0.636, 'fleongg / lightningv08 27B', 'complementary stack, 10% ATLAS, Sinkhorn; Qwen3.8-27B member'),
    (0.639, 'kirilltym teach-more', 'Fourier/PCA/kNN helper readouts, synthetic set'),
    (0.651, 'dhruvtibarewal nlm-v27', 'LoRA24 v5 fold adapters: Qwen3-14B, Qwen3-8B, Llama-3.1-8B'),
    (0.654, 'Roman Tamrazov "Who am I?"', '37-mark typography likelihood'),
    (0.660, 'mercurydust V31', 'family gates in the complementary stack, damped Sinkhorn'),
    (0.663, 'omerfarukmerey ATLAS 20%', 'ATLAS 10% -> 20%, hard deadline, dual-GPU 8B'),
    (0.664, 'judith007 v2 / zbigniewkirchner', 'Sinkhorn 0.30 / SVC on the 198k open pool'),
]
LB = pd.DataFrame(LINEAGE, columns=['lb', 'notebook', 'added'])
fig, ax = plt.subplots(figsize=(13, 5))
ax.step(range(len(LB)), LB.lb, where='mid', color='#4c78a8'); ax.scatter(range(len(LB)), LB.lb, color='#4c78a8', zorder=3)
for i, r in LB.iterrows():
    ax.annotate(f"{r.lb:.3f}  {r.notebook}", (i, r.lb), xytext=(6, -3 if i % 2 else 7), textcoords='offset points', fontsize=7, rotation=0)
for v, t, c in ((0.703, 'LB #1, 2026-10-25', '#e45756'), (0.737, 'LB #1, 2026-11-01', '#b22222'), (0.66, 'gold cut-off ~0.66 (2026-10-25)', '#e6a700')):
    ax.axhline(v, ls='--', lw=.8, color=c); ax.text(len(LB) - .5, v + .002, t, ha='right', fontsize=7, color=c)
ax.set_xticks([]); ax.set_ylabel('public LB (claim)'); ax.set_title('Public notebook lineage: each step is one idea'); ax.set_xlim(-.5, len(LB) + 2.5)
plt.tight_layout(); plt.show()
print(LB.to_string(index=False))
''')

md(r'''
**Reading.** The two big jumps came from **data** (Nishant: public pool 35k -> 84k, +0.04 per member) and from
**stronger fold-averaged LoRA members** (0.639 -> 0.651). Everything after 0.651 is single knobs worth 0.001-0.006,
which is about the noise of one leaderboard reading (B2). The gap to the top private teams (0.70+) is not explained by
any public notebook.

## B2. One leaderboard reading is noisy
''')
code(r'''
NOISE = {'nlm-v27': [0.651, 0.651, 0.643, 0.651], '"Who am I?"': [0.645, 0.654], 'V31': [0.660, 0.660], 'omer 0.663': [0.663, 0.662],
         'LoRA x AC': [0.547, 0.549], 'lora-instyle': [0.563, 0.561], 'Teach + ATLAS': [0.607, 0.606]}
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, (k, v) in enumerate(NOISE.items()):
    ax.scatter([i] * len(v), np.array(v) - np.mean(v), s=30, color='#4c78a8'); ax.plot([i, i], [min(v) - np.mean(v), max(v) - np.mean(v)], color='#4c78a8', lw=1)
ax.axhspan(-.0045, .0045, color='#e45756', alpha=.12, label='+-0.0045 (spread 0.009)')
ax.set_xticks(range(len(NOISE)), NOISE.keys(), rotation=20); ax.set_ylabel('score - mean of re-runs'); ax.legend(frameon=False, fontsize=8)
ax.set_title('Identical code, re-submitted (alexchilton et al., claim)'); plt.tight_layout(); plt.show()
''')
md(r'''
Up to **0.009** apart on identical code. One passage of the ~1,600 public rows is worth about 0.0006, and the class mix of
the public and private parts differs (organisers). Any single-change gain below ~0.01 cannot be read from one
submission. A forum test (discussion 753177) found that out-of-fold scoring on **2 answers per prompt** tracks the board
with correlation **0.91**, against **0.13** for scoring all 18 per prompt.

## B3. Single changes that were tried in public
''')
code(r'''
DELTAS = [  # (change, base LB, new LB, source)
    ('Hungarian inside recovered test clusters (LoRA+AC)', 0.541, 0.524, 'alexchilton'),
    ('Hungarian inside recovered test clusters (saddamaditya)', 0.542, 0.525, 'alexchilton'),
    ('SVC + recovered-group assignment', 0.336, 0.308, 'yangkuangou'),
    ('Ministral-3-8B two-stage member, w3', 0.651, 0.635, 'omer'),
    ('swap q8v5 for Qwen3.5-27B NF4', 0.651, 0.638, 'alexchilton'),
    ('Qwen3.5-9B token attention back on', 0.651, 0.640, 'alexchilton'),
    ('Ministral-3-8B two-stage member, w2', 0.651, 0.641, 'omer'),
    ('drop Llama-3.1-8B (nlm-v27)', 0.651, 0.643, 'omer'),
    ('ModernBERT-large member', 0.651, 0.645, 'omer'),
    ('cluster threshold 0.15 -> 0.08', 0.658, 0.652, 'alekseyosipenko'),
    ('drop Llama-3.1-8B (V31 + ATLAS 20%)', 0.663, 0.658, 'alekseyosipenko'),
    ('svc_w 12 -> 6', 0.664, 0.659, 'zbigniewkirchner'),
    ('own Qwen3-8B two-stage at 10% (no Llama)', 0.643, 0.639, 'alekseyosipenko'),
    ('own five-backbone blend', 0.651, 0.648, 'alexchilton'),
    ('prvsiyan two-stage adapters as member', 0.658, 0.655, 'alekseyosipenko'),
    ('extra seed-11 LoRA24', 0.556, 0.553, 'alexchilton'),
    ('typography margin gate', 0.663, 0.661, 'omer'),
    ('second ATLAS (Qwen3-4B)', 0.651, 0.650, 'omer'),
    ('Sinkhorn 0.45 -> 0.30', 0.663, 0.664, 'judith007'),
    ('SVC pool -> 198k open pool', 0.663, 0.664, 'zbigniewkirchner'),
    ('soft Sinkhorn 0.5', 0.651, 0.653, 'community'),
    ('ATLAS 10% -> 20% (+ faster run)', 0.660, 0.663, 'omer'),
    ('4th LoRA member', 0.552, 0.556, 'alexchilton'),
    ("prvsiyan's blend as one member", 0.592, 0.597, 'alexchilton'),
]
Dd = pd.DataFrame(DELTAS, columns=['change', 'base', 'new', 'source']); Dd['delta'] = Dd.new - Dd.base
Dd = Dd.sort_values('delta')
fig, ax = plt.subplots(figsize=(11, 7.2))
ax.barh(range(len(Dd)), Dd.delta, color=np.where(Dd.delta > 0, '#54a24b', '#e45756'))
ax.set_yticks(range(len(Dd)), [f'{c}  ({s})' for c, s in zip(Dd.change, Dd.source)], fontsize=7.5)
ax.axvspan(-.0045, .0045, color='grey', alpha=.15, label='inside one re-run spread')
ax.axvline(0, color='k', lw=.6); ax.set_xlabel('change in public LB (claim)'); ax.legend(frameon=False, loc='lower right')
ax.set_title('Single changes on public notebooks'); plt.tight_layout(); plt.show()
print(f"{(Dd.delta <= -0.005).sum()} of {len(Dd)} changes lost >= 0.005; {(Dd.delta >= 0.005).sum()} gained >= 0.005")
''')
md(r'''
**Pattern.** Adding a strong new member at a sizeable weight loses, even when it scores 0.60+ alone and helps offline
(Ministral, Qwen3.5-27B, ModernBERT, a second ATLAS). Removing Llama costs 0.005-0.008, so diversity matters, but only
in members whose weights were tuned with the rest. Hard assignment inside recovered prompt groups loses everywhere
(-0.017 to -0.028).

## B4. The prompt-recovery trap
''')
code(r'''
TRAP = pd.DataFrame({'dev OOF (all 18 per prompt)': [0.327, 0.419, 0.501], 'public LB': [0.336, 0.308, 0.351]},
                    index=['SVC, independent', 'SVC + group assignment', 'style LR + assignment'])
ax = TRAP.plot.bar(figsize=(7, 3.4), color=['#9ecae1', '#e45756'], rot=0); ax.set_ylabel('balanced accuracy')
ax.set_title('yangkuangou: CV rewards assignment, the board punishes it (claim)'); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
''')
md(r'''
Part A8 shows why: centring on the other answers to a prompt pays only with many of them (18), and is useless or
harmful with 2-3. The test evidently has few passages per prompt, while the hidden batches can merge near-topic prompts. Only *soft* use of
recovered groups (twin heads at 0.35, the Gaussian head) survives in the 0.66 notebooks.

## B5. Honest re-scoring of the public 0.664 stack (ours)

The 0.664 recipe's `--sim` mode averages all five fold adapters plus the full-data adapter, so its LoRA members see
the held-out rows (leaky). We re-scored each member on rows its model never saw: the CPU members are refit per fold,
and each fold adapter `*_f{k}` is applied only to its own held-out fold. The fold split was confirmed per adapter
(e.g. q8v5_f0: 0.624 on its held-out fold vs 0.882 in-sample).
''')
code(r'''
HONEST = pd.DataFrame([
    ('svc4 (4-view n-gram SVC, 198k open pool)', 0.5042, 'ours, E065'),
    ('tfidf (char + word LR)', 0.4209, 'ours, E065'),
    ('distort (topic-masked TF-IDF)', 0.3219, 'ours, E065'),
    ('CPU part at the 0.664 weights (0.25 / 0.10 / 12)', 0.5071, 'ours, E065'),
    ('svc4, author CV on the 193k pool version', 0.5008, 'zbigniewkirchner, claim'),
    ('q8v5_f0 on its held-out fold (one adapter, one fold)', 0.624, 'ours'),
], columns=['member', 'balanced accuracy (5-fold, prompt-held-out)', 'source'])
print(HONEST.to_string(index=False))
''')
md(r'''
The SVC alone reproduces its author's number, so the open-pool SVC is genuinely worth ~0.50 on unseen prompts, and in
the 0.664 blend it carries almost all of the CPU part. The GPU members (Qwen3-14B, Qwen3-8B, Llama-3.1-8B fold
adapters) are being re-scored the same way; that lets the blend weights, the Sinkhorn strength and any extra member be
fitted on honest out-of-fold predictions instead of single noisy leaderboard readings.

## Takeaways

1. Topic dominates style; style is recoverable when the other answers to a prompt are known, but the test rarely has
   them (A5, A8, B4). Use groups softly or not at all.
2. Typography habits are strong, cheap, family-level signals (A3); length is not (A2). Shared typography explains
   cross-lab confusions such as claude-fable-5 vs GLM.
3. Some pairs are close to indistinguishable on this data (A6, A9): gemini-3.7 vs 3.8-flash, glm-5.3 vs flash,
   gpt-5.6-sol vs terra, and, across labs, claude-fable-5 vs glm-5.3. Effort there has a low ceiling. Claude siblings,
   gpt-5.6-luna vs terra and gemini-3.6 vs 3.8 do separate, much better when both answers to a prompt are seen.
4. The 6 labels with no Benchmarks data have the lowest recall (A7).
5. Public progress came from data and fold-averaged LoRA members; later knobs sit inside a 0.009 noise band (B1-B3).
   Decide with honest, test-like out-of-fold scores, not one public score.
''')


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == 'md' else nbf.v4.new_code_cell(s) for t, s in C]
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    nbf.write(nb, D / 'ntl-eda.ipynb')
    (D / 'kernel-metadata.json').write_text(json.dumps({
        'id': 'mdmuqtadirfuad/ntl-eda-and-public-results', 'title': 'NTL EDA and public results',
        'code_file': 'ntl-eda.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False, 'dataset_sources': [], 'kernel_sources': [],
        'competition_sources': ['name-that-llm'], 'model_sources': [],
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'},
        indent=1), encoding='utf-8')
    print('built', D / 'ntl-eda.ipynb', len(C), 'cells')


if __name__ == '__main__':
    if '--test' in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        _plt.show = lambda *a, **k: _plt.close('all')
        g = {}
        for i, (t, s) in enumerate(C):
            if t == 'code':
                t0 = time.time()
                exec(compile(s, f'cell{i}', 'exec'), g)
                print(f'--- cell {i} ok ({time.time() - t0:.0f}s)', flush=True)
    build()
