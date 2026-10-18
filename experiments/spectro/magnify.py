"""Magnified fingerprints (visual only; the classifiers standardise features, so magnifying adds no information).
For each level (experiments/spectro/out/<level>_passage_shares.npz from spectro.py):
  <level>_fingerprint_zoom.png  log2(model / all-model mean) with the colour range fitted to the data (98th pct of |value|)
  <level>_fingerprint_z.png     per-band z-score across the 18 model means (every band uses the full colour range)
  star = reliable cell: Welch |z| >= 3 for the model's passages vs all other passages on that band
CSV: experiments/spectro/out/<level>_fingerprint_zoom.csv (every number behind both figures)."""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT, FIG = ROOT / 'experiments/spectro/out', ROOT / 'research/spectro'
for f in sorted(OUT.glob('*_passage_shares.npz')):
    lev = f.name.replace('_passage_shares.npz', '')
    z = np.load(f, allow_pickle=False)
    F, y, bands, L = np.nan_to_num(z['F'], nan=0.0), z['y'], list(z['bands']), list(z['labels'])
    M = np.stack([F[y == k].mean(0) for k in range(len(L))])
    G = M.mean(0)
    R = np.log2((M + 1e-4) / (G + 1e-4))
    ZB = (M - M.mean(0)) / (M.std(0) + 1e-12)
    W = np.stack([(F[y == k].mean(0) - F[y != k].mean(0)) /
                  np.sqrt(F[y == k].var(0, ddof=1) / (y == k).sum() + F[y != k].var(0, ddof=1) / (y != k).sum() + 1e-12)
                  for k in range(len(L))])
    sig = np.abs(W) >= 3
    pd.DataFrame([dict(level=lev, model=L[k], band=b, model_share=round(float(M[k, i]), 6), all_model_mean=round(float(G[i]), 6),
                       log2_ratio=round(float(R[k, i]), 4), ratio=round(float(2 ** R[k, i]), 4),
                       band_z_across_models=round(float(ZB[k, i]), 3), welch_z_vs_rest=round(float(W[k, i]), 2), significant=bool(sig[k, i]))
                  for k in range(len(L)) for i, b in enumerate(bands)]).to_csv(OUT / f'{lev}_fingerprint_zoom.csv', index=False)
    v = float(np.quantile(np.abs(R), 0.98))
    for kind, Mat, vmax, cl, fmt in [('zoom', R, v, f'log2 ratio (colour range ±{v:.2f})', lambda k, i: f'{2 ** R[k, i]:.2f}x'),
                                     ('z', ZB, 2.5, 'z across models (per band)', lambda k, i: f'{ZB[k, i]:+.1f}')]:
        fig, ax = plt.subplots(figsize=(11, 0.42 * len(bands) + 2.2))
        im = ax.imshow(Mat.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(L)))
        ax.set_xticklabels(L, rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands, fontsize=8)
        for k in range(len(L)):
            for i in range(len(bands)):
                ax.text(k, i, fmt(k, i) + ('★' if sig[k, i] else ''), ha='center', va='center',
                        fontsize=5.5 if sig[k, i] else 4.5, color='black' if sig[k, i] else '#777777',
                        fontweight='bold' if sig[k, i] else 'normal')
        ax.set_title(f'{lev} fingerprint, magnified ({kind}); ★ = |Welch z| >= 3 vs all other models', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, label=cl)
        fig.savefig(FIG / f'{lev}_fingerprint_{kind}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    print(lev, 'zoom range ±', round(v, 3), '| significant cells', int(sig.sum()), 'of', sig.size, flush=True)
