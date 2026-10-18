"""Within-lab-family contrast of the spectrogram fingerprints: the family's shared pattern is removed and only the
strongest reliable sibling differences are kept.
For each level (outputs of spectro.py: experiments/spectro/out/<level>_passage_shares.npz) and family with >= 2 models
(claude, gemini, gpt, glm):
  log2 ratio = log2(model share / mean share of the family's models)
  z          = Welch z of the model's passages vs its siblings' passages
  cells with |z| < Z_MIN (or |log2| < LOG_MIN) are blanked -> only the contrasting colours remain.
Also: within-family logistic-regression CV on the collapsed features (prompt-grouped 5 folds, family rows only).
Outputs: research/spectro/family_<level>.png, experiments/spectro/out/family_contrast_<level>.csv (every cell),
experiments/spectro/out/family_cv.csv, paper/experiments.csv rows E038_<level>_<family> (idempotent)."""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT, FIG = ROOT / 'experiments/spectro/out', ROOT / 'research/spectro'
Z_MIN, LOG_MIN = 3.0, 0.25
tr = pd.read_csv(ROOT / 'data/train.csv')
groups = tr.prompt_id.values
cv_rows = []
for f in sorted(OUT.glob('*_passage_shares.npz')):
    lev = f.name.replace('_passage_shares.npz', '')
    z = np.load(f, allow_pickle=False)
    F, y, bands, L = np.nan_to_num(z['F'], nan=0.0), z['y'], list(z['bands']), list(z['labels'])
    fams = {}
    for i, l in enumerate(L):
        fams.setdefault(l.split('-')[0], []).append(i)
    fams = {k: v for k, v in fams.items() if len(v) >= 2}
    rows = []
    fig, axs = plt.subplots(1, len(fams), figsize=(3.2 * len(fams) + 2, 0.32 * len(bands) + 2.5),
                            gridspec_kw={'width_ratios': [len(v) for v in fams.values()]})
    for ax, (fam, idx) in zip(axs, fams.items()):
        Rm = np.full((len(bands), len(idx)), np.nan)
        fam_mean = np.mean([F[y == k].mean(0) for k in idx], 0)
        for c, k in enumerate(idx):
            a, b = F[y == k], F[np.isin(y, [j for j in idx if j != k])]
            lr = np.log2((a.mean(0) + 1e-4) / (fam_mean + 1e-4))
            zz = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-12)
            keep = (np.abs(zz) >= Z_MIN) & (np.abs(lr) >= LOG_MIN)
            Rm[keep, c] = lr[keep]
            for i, bd in enumerate(bands):
                rows.append(dict(level=lev, family=fam, model=L[k], band=bd, model_share=round(float(a.mean(0)[i]), 6),
                                 family_mean_share=round(float(fam_mean[i]), 6), log2_ratio=round(float(lr[i]), 4),
                                 z=round(float(zz[i]), 2), kept=bool(keep[i])))
        im = ax.imshow(Rm, aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        ax.set_facecolor('#f2f2f2')
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([L[k] for k in idx], rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands if ax is axs[0] else [], fontsize=8)
        for c in range(len(idx)):
            for i in range(len(bands)):
                if not np.isnan(Rm[i, c]):
                    ax.text(c, i, f'{2 ** Rm[i, c]:.2f}x', ha='center', va='center', fontsize=6)
        ax.set_title(f'{fam} ({len(idx)})', fontsize=10)
        # within-family CV on the collapsed features
        m = np.isin(y, idx)
        D = np.zeros(m.sum(), dtype=int)
        for a_, b_ in GroupKFold(5).split(F[m], y[m], groups[m]):
            D[b_] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(F[m][a_], y[m][a_]).predict(F[m][b_])
        cv_rows.append(dict(level=lev, family=fam, n_models=len(idx), within_family_bacc=round(bacc(y[m], D), 4),
                            chance=round(1 / len(idx), 4), cells_kept=int(np.sum(~np.isnan(Rm)))))
    fig.suptitle(f'{lev}: within-family contrast, log2(model / family mean); blank = |z| < {Z_MIN:g} or |log2| < {LOG_MIN}',
                 fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.7, label='log2 ratio vs family mean')
    fig.savefig(FIG / f'family_{lev}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / f'family_contrast_{lev}.csv', index=False)
    print(lev, 'done', flush=True)
C = pd.DataFrame(cv_rows)
C.to_csv(OUT / 'family_cv.csv', index=False)
print(C.to_string(index=False))
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E038_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in cv_rows:
    log_row(id=f"E038_{r['level']}_{r['family']}", category='features',
            description=f"within-family CV on collapsed {r['level']} spectrogram ({r['family']}, {r['n_models']} models)",
            config='experiments/spectro/family_contrast.py', metric='within_family_bacc', baseline=f"{r['chance']} (chance)",
            value=str(r['within_family_bacc']), files=f"experiments/spectro/out/family_cv.csv",
            notes=f"cells kept |z|>={Z_MIN:g}: {r['cells_kept']}")
