"""E046: does MAGNIFYING the fingerprint features help as a feature group inside the stylegram LinearSVC?
Uniform scaling is a no-op for standardised models, but inside the 4-view TF-IDF hstack the relative loudness of a
group matters. Group = collapsed spectrogram band shares (all levels, or token level only), standardised on the fold's
training rows and scaled by m / sqrt(n_features); m in {0, 1, 3, 10}. Default stylegram views/weights unchanged.
Prompt-grouped 5-fold CV: standalone and gain on the strong blend.
Writes experiments/spectro/out/magnify_feature_test.csv; paper/experiments.csv E046_* (idempotent)."""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT = ROOT / 'experiments/spectro/out'
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
TX = tr.text.astype(str).tolist()
folds = list(GroupKFold(5).split(tr, y, tr.prompt_id.values))
LEVS = [p.name.replace('_passage_shares.npz', '') for p in sorted(OUT.glob('*_passage_shares.npz'))]
DENSE = {'all_spectro': np.hstack([np.nan_to_num(np.load(OUT / f'{l}_passage_shares.npz')['F'], nan=0.0) for l in LEVS]),
         'token_only': np.nan_to_num(np.load(OUT / 'token_passage_shares.npz')['F'], nan=0.0)}
VARIANTS = [('all_spectro', 0)] + [(g, m) for g in DENSE for m in (1, 3, 10)]
FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))
D = {v: np.zeros((len(y), 18)) for v in VARIANTS}
for fi, (a, b) in enumerate(folds):
    A, B = [], []
    for an, lo, hi, mf, wt in SPEC:
        tf = shape if an == 'shape' else func if an == 'function' else str
        act = 'char' if an == 'shape' else 'word' if an == 'function' else an
        o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
        if act == 'word':
            o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([tf(TX[i]) for i in a]) * wt)
        B.append(v.transform([tf(TX[i]) for i in b]) * wt)
    for g, m in VARIANTS:
        if m == 0:
            Xa, Xb = hstack(A, format='csr'), hstack(B, format='csr')
        else:
            sc = StandardScaler().fit(DENSE[g][a])
            k = m / np.sqrt(DENSE[g].shape[1])
            Xa = hstack(A + [csr_matrix(sc.transform(DENSE[g][a]) * k)], format='csr')
            Xb = hstack(B + [csr_matrix(sc.transform(DENSE[g][b]) * k)], format='csr')
        D[(g, m)][b] = LinearSVC(C=0.5, max_iter=4000).fit(Xa, y[a]).decision_function(Xb)
    print(f'fold {fi} done', flush=True)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
S = cal(sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()))
ref = D[('all_spectro', 0)]
res = []
for g, m in VARIANTS:
    Dm = D[(g, m)]
    best = max((bacc(y, (S + mw * cal(Dm)).argmax(1)), mw) for mw in (0.25, 0.5, 1.0))
    fd = [bacc(y[b], Dm[b].argmax(1)) - bacc(y[b], ref[b].argmax(1)) for _, b in folds]
    res.append(dict(group=g if m else 'none (default stylegram)', magnify=m, standalone=round(bacc(y, Dm.argmax(1)), 4),
                    delta_standalone_vs_default=round(bacc(y, Dm.argmax(1)) - bacc(y, ref.argmax(1)), 4),
                    folds_pos=f'{sum(d > 0 for d in fd)}/5', on_strong=round(best[0], 4), member_w=best[1]))
    print(res[-1], flush=True)
pd.DataFrame(res).to_csv(OUT / 'magnify_feature_test.csv', index=False)
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E046_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in res:
    log_row(id=f"E046_{r['group'].split(' ')[0]}_x{r['magnify']}", category='features',
            description=f"magnified fingerprint group in stylegram SVC ({r['group']}, x{r['magnify']})",
            config='experiments/spectro/magnify_feature_test.py', metric='oof_bacc_standalone', baseline=str(res[0]['standalone']),
            value=str(r['standalone']), folds_pos=r['folds_pos'], files='experiments/spectro/out/magnify_feature_test.csv',
            notes=f"on strong {r['on_strong']} at w {r['member_w']}")
