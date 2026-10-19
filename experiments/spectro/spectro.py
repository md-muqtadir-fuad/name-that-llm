"""Text 'spectrograms' per model (log-mel analogue): x = position in the passage (time), y = feature bands on a
log-like scale (mel-like), colour = log2(model mean / all-model mean). Levels:
  char      character classes (letters, digits, space, each punctuation mark incl. curly quotes and dashes)
  word      word-length bands (1..8, 9-10, 11-13, 14+)
  sentence  x = sentence index (1..8+), y = sentence-length bands (words)
  token     Qwen3 tokenizer: token-frequency-rank bands (1-10 ... 10k+, log-spaced)
  surprisal per-token surprisal bands under Qwen3-0.6B-Base (needs surprisal_cache.py output; skipped if missing)
Per level: figure research/spectro/<level>.png (3x6 panels), matrices experiments/spectro/out/<level>.npz,
prompt-grouped 5-fold CV of a logistic regression on the per-passage spectrogram (standalone + gain on the strong
blend), logged to paper/experiments.csv (E037_<level>), and per-model most distinctive cells in research/spectro/SUMMARY.md."""
import json
import pathlib
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import log_softmax  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bacc  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT, FIG = ROOT / 'experiments/spectro/out', ROOT / 'research/spectro'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
folds = list(GroupKFold(5).split(tr, y, tr.prompt_id.values))
TX = tr.text.astype(str).tolist()
PB = 20   # position bins (the "time" axis)

# ---------------------------------------------------------------- per-passage spectrograms: (bands, cols), NaN = empty
CHAR_BANDS = ['a-z', 'A-Z', '0-9', 'space', ',', '.', ';', ':', '!', '?', "'", '’', '"', '“', '”', '—', '–', '-', '(', ')',
              '…', 'other']
CMAP = {c: CHAR_BANDS.index(c) for c in CHAR_BANDS[4:-1]}
CMAP[' '] = 3


def cband(c):
    return 0 if c.islower() else 1 if c.isupper() else 2 if c.isdigit() else CMAP.get(c, len(CHAR_BANDS) - 1)


def binned(band_idx, n_bands, cols=PB):
    """band_idx: band per element in order -> share of each band within each position bin."""
    a = np.zeros((n_bands, cols))
    n = len(band_idx)
    if n == 0:
        return np.full((n_bands, cols), np.nan)
    pos = (np.arange(n) * cols // n)
    np.add.at(a, (np.asarray(band_idx), pos), 1)
    tot = a.sum(0, keepdims=True)
    return np.where(tot > 0, a / np.maximum(tot, 1), np.nan)


WEDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 999]
WBANDS = ['1', '2', '3', '4', '5', '6', '7', '8', '9-10', '11-13', '14+']
SEDGES = [1, 6, 11, 16, 21, 26, 31, 41, 9999]
SBANDS = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41+']
SCOLS = 8


def char_spec(t):
    return binned([cband(c) for c in t], len(CHAR_BANDS))


def word_spec(t):
    ws = [len(re.sub(r'\W', '', w)) for w in t.split()]
    ws = [w for w in ws if w > 0]
    return binned(np.searchsorted(WEDGES, ws, side='right') - 1, len(WBANDS))


def sent_spec(t):
    ss = [s for s in re.split(r'(?<=[.!?…])\s+', t.strip()) if s.strip()]
    a = np.full((len(SBANDS), SCOLS), np.nan)
    for i, s in enumerate(ss):
        c = min(i, SCOLS - 1)
        if i < SCOLS:
            a[:, c] = 0
        a[np.searchsorted(SEDGES, len(s.split()), side='right') - 1, c] += 1
    return a / np.nansum(a, 0, keepdims=True).clip(1)


LEVELS = {'char': (CHAR_BANDS, [char_spec(t) for t in TX], 'position in passage'),
          'word': (WBANDS, [word_spec(t) for t in TX], 'position in passage'),
          'sentence': (SBANDS, [sent_spec(t) for t in TX], 'sentence index (8 = 8th and later)')}

REDGES = [0, 10, 30, 100, 300, 1000, 3000, 10000, 10 ** 9]
RBANDS = ['1-10', '11-30', '31-100', '101-300', '301-1k', '1k-3k', '3k-10k', '10k+']
SUEDGES = [0, 0.5, 1, 2, 3, 4, 6, 8, 1e9]
SUBANDS = ['<0.5', '0.5-1', '1-2', '2-3', '3-4', '4-6', '6-8', '8+']
cache = ROOT / 'research/ext_data/spectro/surprisal_qwen3_0.6b.npz'
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'research/ext_data/hf/Qwen3-0.6B-Base')
    toks = [tok(t, add_special_tokens=False).input_ids for t in TX]
    freq = Counter(i for ids in toks for i in ids)
    rank = {i: r for r, (i, _) in enumerate(freq.most_common())}
    LEVELS['token'] = (RBANDS, [binned(np.searchsorted(REDGES, [rank[i] for i in ids], side='right') - 1, len(RBANDS))
                                for ids in toks], 'position in passage (tokens)')
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        assert list(z['ids']) == list(tr.id)
        LEVELS['surprisal'] = (SUBANDS, [binned(np.searchsorted(SUEDGES, np.nan_to_num(s[1:], nan=0.0), side='right') - 1,
                                                len(SUBANDS)) for s in z['surprisal']], 'position in passage (tokens)')
except Exception as e:   # tokenizer missing -> token levels skipped
    print('token levels skipped:', repr(e)[:200])

# ---------------------------------------------------------------- strong blend (for the gain test)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
S = cal(sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()))
base = bacc(y, S.argmax(1))


def cv(X):
    D = np.zeros((len(y), len(L)))
    for a, b in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[a], y[a])
        D[b] = m.predict_log_proba(X[b])
    Ls = cal(D)
    best = max((bacc(y, (S + mw * Ls).argmax(1)), mw) for mw in (0.1, 0.25, 0.5, 1.0))
    fd = [bacc(y[b], (S + best[1] * Ls)[b].argmax(1)) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
    return bacc(y, D.argmax(1)), best[0] - base, best[1], sum(x > 0 for x in fd)


summary, res, allX = [], [], []
for lev, (bands, specs, xlabel) in LEVELS.items():
    A = np.stack(specs)                                   # passages x bands x cols
    M = np.stack([np.nanmean(A[y == k], 0) for k in range(len(L))])   # models x bands x cols
    G = np.nanmean(M, 0)
    R = np.log2((M + 1e-4) / (G + 1e-4))
    np.savez_compressed(OUT / f'{lev}.npz', model_mean=M, global_mean=G, log2_ratio=R, bands=np.array(bands), labels=np.array(L))
    nk = np.stack([(~np.isnan(A[y == k])).sum(0) for k in range(len(L))])   # passages contributing to each cell
    pd.DataFrame([dict(level=lev, model=L[k], band=bands[i], col=j, n_passages=int(nk[k, i, j]),
                       model_mean=round(float(M[k, i, j]), 6), all_model_mean=round(float(G[i, j]), 6),
                       log2_ratio=round(float(R[k, i, j]), 4))
                  for k in range(len(L)) for i in range(len(bands)) for j in range(R.shape[2])]
                 ).to_csv(OUT / f'{lev}_cells.csv', index=False)   # every number behind the figure
    fig, axs = plt.subplots(3, 6, figsize=(19, 8.5), sharex=True, sharey=True)
    for k, ax in enumerate(axs.flat):
        im = ax.imshow(R[k], aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1.5, vmax=1.5, interpolation='nearest')
        ax.set_title(L[k], fontsize=9)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands, fontsize=6)
        ax.set_xticks([0, R.shape[2] // 2, R.shape[2] - 1])
        ax.set_xticklabels(['start', 'mid', 'end'] if lev != 'sentence' else ['1', str(SCOLS // 2 + 1), f'{SCOLS}+'], fontsize=7)
    fig.suptitle(f'{lev} spectrogram: log2(model / all-model mean), x = {xlabel}', fontsize=11)
    fig.colorbar(im, ax=axs, shrink=0.6, label='log2 ratio (±1 = 2x more / less)')
    fig.savefig(FIG / f'{lev}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    X = np.nan_to_num(A.reshape(len(A), -1), nan=0.0)
    allX.append(X)
    sa, d, mw, fp = cv(X)
    res.append(dict(level=lev, standalone=round(sa, 4), delta_on_strong=round(d, 4), member_w=mw, folds_pos=f'{fp}/5'))
    print(res[-1], flush=True)
    # position-free fingerprint: whole-passage band shares (weighted by elements per position bin is not needed:
    # bins are equal-width, so the mean over bins is the passage share)
    F = np.nanmean(A, 2)                                  # passages x bands
    FM = np.stack([np.nanmean(F[y == k], 0) for k in range(len(L))])
    FR = np.log2((FM + 1e-4) / (np.nanmean(FM, 0) + 1e-4))
    np.savez_compressed(OUT / f'{lev}_passage_shares.npz', F=F, y=y, bands=np.array(bands), labels=np.array(L))
    pd.DataFrame(FR, index=L, columns=bands).round(4).to_csv(OUT / f'{lev}_fingerprint_log2.csv')
    pd.DataFrame(FM, index=L, columns=bands).round(6).to_csv(OUT / f'{lev}_fingerprint_share.csv')
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(bands) + 2))
    im = ax.imshow(FR.T, aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(L)))
    ax.set_xticklabels(L, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels(bands, fontsize=8)
    for k in range(len(L)):
        for i in range(len(bands)):
            if abs(FR[k, i]) >= 0.75:
                ax.text(k, i, f'{2 ** FR[k, i]:.1f}x', ha='center', va='center', fontsize=5.5, color='w')
    ax.set_title(f'{lev} fingerprint: whole-passage share, log2(model / all-model mean)', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label='log2 ratio')
    fig.savefig(FIG / f'{lev}_fingerprint.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    sa, d, mw, fp = cv(np.nan_to_num(F, nan=0.0))
    res.append(dict(level=f'{lev}_collapsed', standalone=round(sa, 4), delta_on_strong=round(d, 4), member_w=mw, folds_pos=f'{fp}/5'))
    print(res[-1], flush=True)
    for k in range(len(L)):   # most distinctive cells: large |log ratio| on bands that are not near-empty
        ok = G > (1e-3 if lev == 'char' else 1e-2)
        Rk = np.where(ok, R[k], 0)
        top = np.argsort(-np.abs(Rk), axis=None)[:3]
        cells = [f"{bands[i // Rk.shape[1]]} @ {'sent ' + str(i % Rk.shape[1] + 1) if lev == 'sentence' else str(round(100 * (i % Rk.shape[1]) / Rk.shape[1])) + '%'}: "
                 f"x{2 ** Rk.flat[i]:.2f}" for i in top]
        summary.append((L[k], lev, '; '.join(cells)))
sa, d, mw, fp = cv(np.hstack(allX))
res.append(dict(level='all', standalone=round(sa, 4), delta_on_strong=round(d, 4), member_w=mw, folds_pos=f'{fp}/5'))
print(res[-1], flush=True)
pd.DataFrame(res).to_csv(OUT / 'cv_results.csv', index=False)
pd.DataFrame(summary, columns=['model', 'level', 'top_cells']).to_csv(OUT / 'top_cells.csv', index=False)
_ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)   # idempotent: replace earlier E037 rows
_ex[~_ex['id'].str.startswith('E037_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in res:
    log_row(id=f"E037_{r['level']}", category='features', description=f"text spectrogram ({r['level']} level) + logistic regression",
            config='experiments/spectro/spectro.py', metric='oof_bacc_standalone', baseline=f'{base:.4f} (strong blend)',
            value=str(r['standalone']), delta=str(r['delta_on_strong']), folds_pos=r['folds_pos'],
            files='experiments/spectro/out/cv_results.csv', notes=f"delta is gain on strong blend at member weight {r['member_w']}")
with open(FIG / 'SUMMARY.md', 'w', encoding='utf-8') as f:
    f.write('# Text spectrograms: CV and most distinctive cells per model\n\n')
    f.write('Figures: ' + ', '.join(f'[{l}]({l}.png) / [{l} fingerprint]({l}_fingerprint.png)' for l in LEVELS)
            + '. Colour = log2(model / all-model mean). Numbers: experiments/spectro/out/*_cells.csv, *_fingerprint_*.csv.\n\n')
    f.write('| level | standalone | delta on strong | member w | folds+ |\n|---|---|---|---|---|\n')
    for r in res:
        f.write(f"| {r['level']} | {r['standalone']} | {r['delta_on_strong']:+.4f} | {r['member_w']} | {r['folds_pos']} |\n")
    f.write('\n')
    f.write('| model | level | top cells (band @ position: ratio vs all-model mean) |\n|---|---|---|\n')
    for m, lev, c in summary:
        f.write(f'| {m} | {lev} | {c} |\n')
print('wrote', FIG)
