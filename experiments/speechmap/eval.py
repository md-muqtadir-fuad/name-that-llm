"""Value of SpeechMap paragraphs for this competition (prompt-grouped CV on competition train, 5-fold GroupKFold).
External sets (prompts disjoint from train by construction):
  SM   = SpeechMap non-refusal paragraphs, first eligible paragraph per response, <= K per label (16 labels)
  PV   = prvsiyan passages_clean (18 labels)
  SM+PV
Tests:
  A) 18-way stylegram LinearSVC trained on train-fold (+ external, sample weight w): standalone OOF and gain when added
     to the strong frozen-LLM OOF blend (member weight on a grid)
  B) within-lab gates (claude, gemini, gpt) as in experiments/gates/gate_test.py (lam 1.5), base = strong blend
Writes experiments/speechmap/out/eval_results.csv and appends paper/experiments.csv."""
import json
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

K_PER_LABEL = int(sys.argv[1]) if len(sys.argv) > 1 else 500
T0 = time.time()
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
lid = {l: i for i, l in enumerate(L)}
y = tr.model.map(lid).values
g = tr.prompt_id.values
folds = list(GroupKFold(5).split(tr, y, g))

sm = pd.read_csv(ROOT / 'research/ext_data/speechmap/paragraphs.csv')
sm = sm[(sm.refusal == 0) & (sm.para_idx == 0) & sm.label.isin(L)]
sm = pd.concat([d.sample(min(len(d), K_PER_LABEL), random_state=0) for _, d in sm.groupby('label')])
SM = pd.DataFrame({'text': sm.text.astype(str).values, 'y': sm.label.map(lid).values})
pv = pd.read_csv(ROOT / 'research/ext_data/prvsiyan/passages_clean.csv')
pv = pv[pv.model.isin(L)]
PV = pd.DataFrame({'text': pv.text.astype(str).values, 'y': pv.model.map(lid).values})
SETS = {'none': SM.iloc[:0], 'SM': SM, 'PV': PV, 'SM+PV': pd.concat([SM, PV], ignore_index=True)}
print({k: len(v) for k, v in SETS.items()}, flush=True)

zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
S = cal(S)
base = bacc(y, S.argmax(1))

FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))


def svc_dec(txt, yy, test_txt, sw=None):
    A, B = [], []
    for an, lo, hi, mf, wt in SPEC:
        tf = shape if an == 'shape' else func if an == 'function' else str
        act = 'char' if an == 'shape' else 'word' if an == 'function' else an
        o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
        if act == 'word':
            o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([tf(t) for t in txt]) * wt)
        B.append(v.transform([tf(t) for t in test_txt]) * wt)
    m = LinearSVC(C=0.5, max_iter=4000).fit(hstack(A, format='csr'), yy, sample_weight=sw)
    d = m.decision_function(hstack(B, format='csr'))
    if d.ndim == 1:
        d = np.column_stack([-d, d])
    return m.classes_, d


res = []
# ---- A) 18-way member
for src, wext in (('none', 0.0), ('SM', 0.3), ('SM', 1.0), ('PV', 1.0), ('SM+PV', 1.0)):
    E = SETS[src]
    D = np.zeros((len(y), 18))
    for a, b in folds:
        txt = list(tr.text.values[a]) + list(E.text)
        yy = np.concatenate([y[a], E.y.values])
        sw = np.concatenate([np.ones(len(a)), np.full(len(E), wext)]) if len(E) else None
        cls, dd = svc_dec(txt, yy, list(tr.text.values[b]), sw)
        D[np.ix_(b, cls)] = dd
    Ls = cal(D)
    best = max((bacc(y, (S + m * Ls).argmax(1)), m) for m in (0.25, 0.5, 0.75, 1.0))
    fd = [bacc(y[b], (S + best[1] * Ls)[b].argmax(1)) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
    r = dict(test='A_18way_svc', source=src, w_ext=wext, n_ext=len(E), standalone=round(bacc(y, D.argmax(1)), 4),
             on_strong=round(best[0], 4), delta_vs_strong=round(best[0] - base, 4), member_w=best[1],
             folds_pos=f'{sum(x > 0 for x in fd)}/5', seconds=round(time.time() - T0))
    res.append(r)
    print(r, flush=True)
    pd.DataFrame(res).to_csv(ROOT / 'experiments/speechmap/out/eval_results.csv', index=False)
# ---- B) gates
for src in ('none', 'SM', 'PV', 'SM+PV'):
    E = SETS[src]
    P = S.argmax(1).copy()
    per = {}
    for lab in ('claude', 'gemini', 'gpt'):
        G = np.array([i for i, l in enumerate(L) if l.startswith(lab)])
        dec = np.zeros((len(y), len(G)))
        for a, b in folds:
            ta = a[np.isin(y[a], G)]
            e = E[E.y.isin(G)]
            cls, dd = svc_dec(list(tr.text.values[ta]) + list(e.text), np.concatenate([y[ta], e.y.values]),
                              list(tr.text.values[b]))
            dec[np.ix_(b, np.searchsorted(G, cls))] = dd
        m = np.isin(S.argmax(1), G)
        Pl = S.argmax(1).copy()
        Pl[m] = G[(S[:, G] + 1.5 * dec)[m].argmax(1)]
        per[lab] = round(bacc(y, Pl) - base, 4)
        P[m] = Pl[m]
    fd = [bacc(y[b], P[b]) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
    r = dict(test='B_gates', source=src, n_ext=len(E), on_strong=round(bacc(y, P), 4),
             delta_vs_strong=round(bacc(y, P) - base, 4), folds_pos=f'{sum(x > 0 for x in fd)}/5',
             per_lab=json.dumps(per), seconds=round(time.time() - T0))
    res.append(r)
    print(r, flush=True)
    pd.DataFrame(res).to_csv(ROOT / 'experiments/speechmap/out/eval_results.csv', index=False)
for r in res:
    log_row(id=f"E030_{r['test']}_{r['source']}_{r.get('w_ext', '')}", category='data',
            description=f"SpeechMap eval ({r['test']}, source {r['source']}, K={K_PER_LABEL}/label)",
            config='experiments/speechmap/eval.py', metric='oof_bacc_on_strong', baseline=f'{base:.4f}',
            value=str(r['on_strong']), folds_pos=r['folds_pos'], files='experiments/speechmap/out/eval_results.csv',
            notes=f"standalone {r.get('standalone', '')}; per_lab {r.get('per_lab', '')}")
print('done', round(time.time() - T0), 's')
