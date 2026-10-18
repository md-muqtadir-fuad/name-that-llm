"""Sibling signatures inside each lab family, mined with PROMPT-PAIRED contrasts (all siblings answer every train
prompt, so model-vs-siblings differences on the same prompt cancel the topic).
Feature pool (per-passage relative frequencies, min_df 20): char 1-3-grams (case kept), word 1-2-grams (lowercased,
punctuation tokens kept), function-word sequences 1-3 (content words -> CONTENT), plus the collapsed spectrogram bands.
For model m in family F and feature j: d_p = x[m,p] - mean_{s in F, s != m} x[s,p] over prompts p;
paired t = mean(d) / (sd(d)/sqrt(n)); sign consistency = share of prompts with d of the majority sign (ties excluded).
1) Descriptive signatures on all train: experiments/spectro/out/signatures_<family>.csv (top 40 per model) and
   research/spectro/SIGNATURES.md (top 8 per model).
2) Within-family CV (prompt-grouped 5 folds, family rows only; selection INSIDE each fold): signature LR (top K per
   model by |t| on training prompts) vs the current gate model (4-view TF-IDF LinearSVC on family rows) vs both.
   experiments/spectro/out/signature_cv.csv, paper/experiments.csv E039_<family> (idempotent)."""
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT, FIG = ROOT / 'experiments/spectro/out', ROOT / 'research/spectro'
K_PER_MODEL = 40
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
TX = tr.text.astype(str).tolist()
P = tr.prompt_id.values
FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))

# ---------------------------------------------------------------- feature pool (relative frequencies)
views, names = [], []
for nm, o, tf in [('char', dict(analyzer='char', ngram_range=(1, 3), lowercase=False), str),
                  ('word', dict(analyzer='word', ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]"), str),
                  ('func', dict(analyzer='word', ngram_range=(1, 3), lowercase=False, token_pattern=r"\S+"), func)]:
    v = TfidfVectorizer(use_idf=False, norm='l1', min_df=20, dtype=np.float32, **o)
    views.append(v.fit_transform([tf(t) for t in TX]))
    names += [f'{nm}:{f}' for f in v.get_feature_names_out()]
for f in sorted(OUT.glob('*_passage_shares.npz')):
    z = np.load(f, allow_pickle=False)
    lev = f.name.replace('_passage_shares.npz', '')
    views.append(np.nan_to_num(z['F'], nan=0.0).astype(np.float32))
    names += [f'spec_{lev}:{b}' for b in z['bands']]
X = hstack([v if hasattr(v, 'tocsr') else v for v in views], format='csr')
X = X.toarray().astype(np.float32)
names = np.array(names)
print('features', X.shape, flush=True)

fams = {}
for i, l in enumerate(L):
    fams.setdefault(l.split('-')[0], []).append(i)
fams = {k: v for k, v in fams.items() if len(v) >= 2}
row = {(p, m): r for r, (p, m) in enumerate(zip(P, y))}


def paired(idx, prompts):
    """-> t (models x F), mean diff, sign consistency, n prompts; over prompts where all siblings are present."""
    ps = [p for p in prompts if all((p, k) in row for k in idx)]
    A = np.stack([X[[row[(p, k)] for p in ps]] for k in idx])          # models x prompts x F
    T, MD, SC = [], [], []
    for c in range(len(idx)):
        d = A[c] - (A.sum(0) - A[c]) / (len(idx) - 1)
        md, sd = d.mean(0), d.std(0, ddof=1)
        T.append(md / (sd / np.sqrt(len(ps)) + 1e-9))
        pos, neg = (d > 0).sum(0), (d < 0).sum(0)
        SC.append(np.maximum(pos, neg) / np.maximum(pos + neg, 1))
        MD.append(md)
    return np.array(T), np.array(MD), np.array(SC), len(ps)


# ---------------------------------------------------------------- 1) descriptive signatures
md_lines = ['# Sibling signatures (prompt-paired, all train)\n',
            'rate = mean relative frequency in the model\'s passages; vs sib = mean over its siblings; t = paired t over prompts;',
            'cons = share of prompts where the difference has the majority sign. Full tables: experiments/spectro/out/signatures_<family>.csv\n']
for fam, idx in fams.items():
    T, MD, SC, n = paired(idx, sorted(set(P)))
    rows = []
    for c, k in enumerate(idx):
        top = np.argsort(-np.abs(T[c]))[:K_PER_MODEL]
        m_rate = X[y == k].mean(0)
        s_rate = X[np.isin(y, [j for j in idx if j != k])].mean(0)
        for j in top:
            rows.append(dict(family=fam, model=L[k], feature=names[j], t_paired=round(float(T[c, j]), 2),
                             mean_diff=float(MD[c, j]), rate_model=float(m_rate[j]), rate_siblings=float(s_rate[j]),
                             ratio=round(float((m_rate[j] + 1e-7) / (s_rate[j] + 1e-7)), 3),
                             sign_consistency=round(float(SC[c, j]), 3), n_prompts=n))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f'signatures_{fam}.csv', index=False)
    md_lines.append(f'\n## {fam} ({len(idx)} models, {n} paired prompts)\n')
    md_lines.append('| model | top signatures (feature: ratio vs siblings, t, consistency) |\n|---|---|')
    for k in idx:
        d = D[D.model == L[k]].head(8)
        md_lines.append(f'| {L[k]} | ' + '; '.join(f"`{r.feature}` {r.ratio:.2f}x t={r.t_paired:+.1f} c={r.sign_consistency:.2f}"
                                                  for r in d.itertuples()) + ' |')
    print(fam, 'signatures done', flush=True)
(FIG / 'SIGNATURES.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

# ---------------------------------------------------------------- 2) within-family CV: signature LR vs gate SVC
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def svc_dec(tr_idx, te_idx):
    A, B = [], []
    for an, lo, hi, mf, wt in SPEC:
        tf = shape if an == 'shape' else func if an == 'function' else str
        act = 'char' if an == 'shape' else 'word' if an == 'function' else an
        o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
        if act == 'word':
            o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([tf(TX[i]) for i in tr_idx]) * wt)
        B.append(v.transform([tf(TX[i]) for i in te_idx]) * wt)
    m = LinearSVC(C=0.5, max_iter=4000).fit(hstack(A, format='csr'), y[tr_idx])
    d = m.decision_function(hstack(B, format='csr'))
    return np.column_stack([-d, d]) if d.ndim == 1 else d


def calib(Z, yy):
    t = minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(yy)), yy].mean(), bounds=(0.05, 50), method='bounded').x
    return log_softmax(Z * t, 1)


res = []
for fam, idx in fams.items():
    m = np.where(np.isin(y, idx))[0]
    yy = np.searchsorted(idx, y[m])
    Dsig, Dsvc = np.zeros((len(m), len(idx))), np.zeros((len(m), len(idx)))
    nsel = []
    for a, b in GroupKFold(5).split(m, yy, P[m]):
        T, _, _, _ = paired(idx, sorted(set(P[m[a]])))
        sel = np.unique(np.concatenate([np.argsort(-np.abs(T[c]))[:K_PER_MODEL] for c in range(len(idx))]))
        nsel.append(len(sel))
        sc = StandardScaler().fit(X[m[a]][:, sel])
        lr = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[m[a]][:, sel]), yy[a])
        Dsig[b] = lr.predict_log_proba(sc.transform(X[m[b]][:, sel]))
        Dsvc[b] = svc_dec(m[a], m[b])
    Ls, Lv = calib(Dsig, yy), calib(Dsvc, yy)
    both = max((bacc(yy, (Lv + w * Ls).argmax(1)), w) for w in (0.25, 0.5, 1.0, 2.0))
    r = dict(family=fam, n_models=len(idx), chance=round(1 / len(idx), 3), signature_lr=round(bacc(yy, Dsig.argmax(1)), 4),
             gate_svc=round(bacc(yy, Dsvc.argmax(1)), 4), svc_plus_signature=round(both[0], 4), sig_weight=both[1],
             features_selected=int(np.mean(nsel)))
    res.append(r)
    print(r, flush=True)
R = pd.DataFrame(res)
R.to_csv(OUT / 'signature_cv.csv', index=False)
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E039_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in res:
    log_row(id=f"E039_{r['family']}", category='features',
            description=f"prompt-paired sibling signatures ({r['family']}): signature LR vs gate SVC, within-family CV",
            config='experiments/spectro/signatures.py', metric='within_family_bacc', baseline=f"{r['gate_svc']} (gate SVC)",
            value=str(r['svc_plus_signature']), delta=str(round(r['svc_plus_signature'] - r['gate_svc'], 4)),
            files='experiments/spectro/out/signature_cv.csv',
            notes=f"signature LR alone {r['signature_lr']}; chance {r['chance']}; ~{r['features_selected']} features; weight {r['sig_weight']}")
print('done')
