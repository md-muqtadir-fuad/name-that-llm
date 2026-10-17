"""Within-lab 'gates' (Claude-gate style) with and without external passages, prompt-grouped CV.
Base = strong public frozen-LLM OOF blend. For rows whose base argmax is in lab G, re-pick the class within G by
  argmax_{c in G} [ log P_base(c) + lam * d_c ],  d = within-lab 4-view TF-IDF LinearSVC decision (as v1 stylegram),
trained on the training-fold rows of lab G (+ optional external rows of lab G; external prompts are disjoint from train)."""
import json
import os
import pathlib
import re
import sys

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
tr = pd.read_csv(ROOT / 'data/train.csv')
LABELS = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
g = tr.prompt_id.values
LABS = {'claude': [i for i, l in enumerate(LABELS) if l.startswith('claude')],
        'gemini': [i for i, l in enumerate(LABELS) if l.startswith('gemini')],
        'gpt': [i for i, l in enumerate(LABELS) if l.startswith('gpt')],
        'glm': [i for i, l in enumerate(LABELS) if l.startswith('glm')],
        # data-driven clusters from research/model_graph.md (confusion + style-centroid views agree)
        'c_newclaude_glm': [LABELS.index(m) for m in ('claude-fable-5', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash')],
        'c_oldclaude': [LABELS.index(m) for m in ('claude-opus-4-6', 'claude-opus-4-8', 'claude-sonnet-5')],
        'c_gpt55_kimi': [LABELS.index(m) for m in ('gpt-5.5', 'kimi-k3')],
        'c_gpt56_astra': [LABELS.index(m) for m in ('gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra')]}

# ---------------- external passages
ext = []
p = ROOT / 'research/ext_data/prvsiyan/passages_clean.csv'
if p.exists():
    d = pd.read_csv(p)
    import os
    if os.environ.get('PRVSIYAN_NEW_ONLY'):   # drop proxy_v1 rows (possible multi-turn contamination)
        d = d[d['split'] == 'new']
    ext.append(pd.DataFrame({'text': d['text'].astype(str), 'model': d['model'], 'src': 'prvsiyan'}))
p = ROOT / 'research/ext_data/extra_api/ntl_extra_passages_claude.csv'
if p.exists():
    d = pd.read_csv(p)
    ext.append(pd.DataFrame({'text': d['text'].astype(str), 'model': d['model'], 'src': 'seyit'}))
ext = pd.concat(ext, ignore_index=True)
ext = ext[ext.model.isin(LABELS)]
ext['y'] = ext.model.map({l: i for i, l in enumerate(LABELS)})
print('external rows', ext.groupby('src').size().to_dict())

# ---------------- base: strong OOF blend (calibrated)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
S = log_softmax(S * minimize_scalar(lambda t: -log_softmax(S * t, 1)[np.arange(len(y)), y].mean(),
                                    bounds=(0.1, 20), method='bounded').x, 1)
base = bacc(y, S.argmax(1))

# ---------------- stylegram views (as v1)
FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))


def svc_decision(train_txt, train_y, test_txt):
    A, B = [], []
    for an, lo, hi, mf, wt in SPEC:
        tf = shape if an == 'shape' else func if an == 'function' else str
        act = 'char' if an == 'shape' else 'word' if an == 'function' else an
        o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
        if act == 'word':
            o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([tf(t) for t in train_txt]) * wt)
        B.append(v.transform([tf(t) for t in test_txt]) * wt)
    m = LinearSVC(C=0.5, max_iter=4000).fit(hstack(A, format='csr'), train_y)
    d = m.decision_function(hstack(B, format='csr'))
    if d.ndim == 1:   # binary: column for classes_[1] is +d
        d = np.column_stack([-d, d])
    return m.classes_, d


folds = list(GroupKFold(5).split(tr, y, g))
labs_run = sys.argv[1].split(',') if len(sys.argv) > 1 else list(LABS)
LAMS = (0.5, 1.0, 1.5, 3.0)
D = {}
for lab in labs_run:
    G = np.array(LABS[lab])
    for use_ext in (False, True):
        dec = np.zeros((len(y), len(G)))
        for a, b in folds:
            ta = a[np.isin(y[a], G)]
            txt, yy = list(tr.text.values[ta]), list(y[ta])
            if use_ext:
                e = ext[ext.y.isin(G)]
                txt += list(e.text)
                yy += list(e.y)
            cls, dd = svc_decision(txt, np.array(yy), list(tr.text.values[b]))
            assert list(cls) == list(G)
            dec[b] = dd
        D[(lab, use_ext)] = dec
        np.save(ROOT / f'experiments/gates/dec_{lab}_{int(use_ext)}{os.environ.get("DEC_SUFFIX", "")}.npy', dec)
        print(f'{lab} ext={use_ext} done', flush=True)

print(f'\nbase strong blend {base:.4f}')
for lab in labs_run:
    G = np.array(LABS[lab])
    inG = np.isin(S.argmax(1), G)
    truthG = np.isin(y, G)
    for use_ext in (False, True):
        for lam in LAMS:
            P = S.argmax(1).copy()
            sc = S[:, G] + lam * D[(lab, use_ext)]
            P[inG] = G[sc[inG].argmax(1)]
            fd = [bacc(y[b], P[b]) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
            rec = np.mean([np.mean(P[y == c] == c) for c in G])
            print(f'{lab:7s} ext={str(use_ext):5s} lam={lam:3.1f}: bacc {bacc(y, P):.4f} ({bacc(y, P) - base:+.4f}) '
                  f'folds+ {sum(x > 0 for x in fd)}/5 | lab recall {rec:.3f} (base {np.mean([np.mean(S.argmax(1)[y == c] == c) for c in G]):.3f})',
                  flush=True)
