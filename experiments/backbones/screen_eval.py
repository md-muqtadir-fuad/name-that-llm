"""E047 model performance study: evaluate the frozen-backbone features from nb/backbone_screen (Kaggle outputs in
nb/backbone_screen/out/feat_<model>.npz) with prompt-grouped GroupKFold(5).
Per model: (1) every depth: PCA-256 (fit on fold train) + logistic regression -> standalone balanced accuracy;
(2) best 3 depths concatenated; (3) LM statistics alone (log-prob, entropy, curvature, rank shares, fertility) incl.
within-family accuracy (the review's claim: curvature separates the GLM pair); (4) best config: gain on the strong
blend, agreement with it, within-family accuracy per lab; (5) cost (seconds per passage on 2xT4, peak GB).
Outputs: experiments/backbones/out/layer_scores.csv, summary.csv, research/backbones/depth_curves.png,
paper/experiments.csv rows E047_<model> (idempotent)."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import log_softmax  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

SRC = ROOT / 'nb/backbone_screen/out'
OUT, FIG = ROOT / 'experiments/backbones/out', ROOT / 'research/backbones'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
folds = list(GroupKFold(5).split(tr, y, tr.prompt_id.values))
pos = {i: k for k, i in enumerate(tr.id.values)}
FAM = {f: np.array([i for i, l in enumerate(L) if l.startswith(f)]) for f in ('claude', 'gemini', 'gpt', 'glm')}
rep = {}
for _r in sorted(SRC.glob('screen_report*.json')):   # GPU runs (v1, v2)
    rep.update({k: v for k, v in json.loads(_r.read_text()).items() if v.get('status') == 'ok'})
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
S = cal(sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()))
base = bacc(y, S.argmax(1))


def oof(X, pca=256, C=0.5):
    D = np.zeros((len(y), 18))
    for a, b in folds:
        sc = StandardScaler().fit(X[a])
        Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
        if pca and X.shape[1] > pca:
            p = PCA(pca, random_state=0).fit(Xa)
            Xa, Xb = p.transform(Xa), p.transform(Xb)
        D[b] = LogisticRegression(C=C, max_iter=3000).fit(Xa, y[a]).predict_log_proba(Xb)
    return D


def within(D, fam):
    g = FAM[fam]
    m = np.isin(y, g)
    return bacc(y[m], g[D[m][:, g].argmax(1)])


def within_fit(X, fam):   # classifier trained on the family's rows only (direct sibling test)
    g = FAM[fam]
    pred = np.zeros(len(y), int)
    for a, b in folds:
        a, b = a[np.isin(y[a], g)], b[np.isin(y[b], g)]
        sc = StandardScaler().fit(X[a])
        pred[b] = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(X[a]), y[a]).predict(sc.transform(X[b]))
    m = np.isin(y, g)
    return bacc(y[m], pred[m])


layer_rows, summ = [], []
for f in sorted(SRC.glob('feat_*.npz')):
    name = f.stem.replace('feat_', '')
    z = np.load(f, allow_pickle=True)   # ids are strings (object array); file = our own Kaggle kernel output
    order = np.array([pos[i] for i in z['ids']])
    F = np.zeros(z['feats'].shape, np.float32)
    F[order] = z['feats'].astype(np.float32)
    LM = np.zeros(z['lm'].shape, np.float32)
    LM[order] = np.nan_to_num(z['lm'])
    layers, nl = list(z['layers']), int(z['n_layers'])
    sc_l, Ds = [], {}
    for j, li in enumerate(layers):
        D = oof(F[:, j])
        Ds[j] = D
        s = bacc(y, D.argmax(1))
        sc_l.append(s)
        layer_rows.append(dict(model=name, layer=li, rel_depth=round(li / nl, 3), standalone=round(s, 4)))
        print(name, li, round(s, 4), flush=True)
    top3 = list(np.argsort(sc_l)[::-1][:3])
    Dc = oof(np.hstack([F[:, j] for j in sorted(top3)]))
    Dlm = oof(LM, pca=0, C=1.0)
    best = max([(sc_l[j], Ds[j], f'L{layers[j]}') for j in range(len(layers))] + [(bacc(y, Dc.argmax(1)), Dc, 'top3')], key=lambda t: t[0])
    Lb = cal(best[1])
    on = max((bacc(y, (S + mw * Lb).argmax(1)), mw) for mw in (0.25, 0.5, 1.0))
    fd = [bacc(y[b], (S + on[1] * Lb)[b].argmax(1)) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
    r = rep.get(name, {})
    summ.append(dict(model=name, n_layers=nl, best_config=best[2], best_standalone=round(best[0], 4),
                     best_rel_depth=round(layers[int(np.argmax(sc_l))] / nl, 2), top3_concat=round(bacc(y, Dc.argmax(1)), 4),
                     on_strong=round(on[0], 4), delta_on_strong=round(on[0] - base, 4), folds_pos=f'{sum(d > 0 for d in fd)}/5',
                     agreement_with_strong=round(float(np.mean(best[1].argmax(1) == S.argmax(1))), 3),
                     **{f'within_{k}': round(within(best[1], k), 3) for k in FAM},
                     lm_stats_standalone=round(bacc(y, Dlm.argmax(1)), 4),
                     **{f'lm_stats_within_{k}': round(within_fit(LM, k), 3) for k in FAM},
                     s_per_passage_2xT4=r.get('s_per_passage'), peak_gb=r.get('peak_gb'), eight_bit=r.get('eight_bit'),
                     tokens_per_word=r.get('tokens_per_word')))
    print(summ[-1], flush=True)
pd.DataFrame(layer_rows).to_csv(OUT / 'layer_scores.csv', index=False)
SM = pd.DataFrame(summ).sort_values('delta_on_strong', ascending=False)
SM.to_csv(OUT / 'summary.csv', index=False)
LR = pd.DataFrame(layer_rows)
fig, ax = plt.subplots(figsize=(8, 5))
for name, d in LR.groupby('model'):
    ax.plot(d.rel_depth, d.standalone, marker='o', label=name)
ax.set_xlabel('relative depth of layer')
ax.set_ylabel('balanced accuracy (frozen, PCA-256 + LR, grouped 5-fold)')
ax.set_title('Backbone screening: which depth carries authorship signal')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.savefig(FIG / 'depth_curves.png', dpi=120, bbox_inches='tight')
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E047_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in summ:
    log_row(id=f"E047_{r['model']}", category='backbone', description=f"backbone screening, frozen {r['model']} ({r['best_config']})",
            config='nb/backbone_screen/screen.py; experiments/backbones/screen_eval.py', metric='oof_bacc_standalone',
            baseline=f'{base:.4f} (strong blend)', value=str(r['best_standalone']), delta=str(r['delta_on_strong']), folds_pos=r['folds_pos'],
            files='experiments/backbones/out/summary.csv;experiments/backbones/out/layer_scores.csv',
            notes=f"on strong {r['on_strong']}; lm-stats {r['lm_stats_standalone']}, glm pair {r['lm_stats_within_glm']}; {r['s_per_passage_2xT4']} s/passage")
print(SM.to_string(index=False))
