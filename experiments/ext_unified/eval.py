"""Value of unified external sources under prompt-grouped CV (5-fold GroupKFold).
External rows are capped at 500 per (source, label), sampled with random_state=0.
Tests: A) 18-way stylegram SVC blended into the strong frozen-LLM OOF base;
B) within-lab gates (lam 1.5). PV-only Test A is checked against the reference first."""
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
OUT = ROOT / 'experiments/ext_unified/out'
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
lid = {l: i for i, l in enumerate(L)}
y = tr.model.map(lid).values
g = tr.prompt_id.values
folds = list(GroupKFold(5).split(tr, y, g))

external = pd.read_parquet(ROOT / 'research/ext_data/unified/ext_all.parquet')
external = external[external.label.isin(L)]
external = pd.concat([d.sample(min(len(d), K_PER_LABEL), random_state=0)
                      for _, d in external.groupby(['source', 'label'], sort=False)], ignore_index=True)
external['y'] = external.label.map(lid).astype(int)
SETS = {'none': external.iloc[:0], 'PV': external[external.source == 'prvsiyan'], 'ALL': external,
        'ALL_balanced': external[external.balanced == 1]}
for source, n in external.source.value_counts().items():
    if n >= 200:
        SETS[f'ALL_minus_{source}'] = external[external.source != source]
SETS['ALL_no_SM'] = external[external.source != 'speechmap']
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
def save_result(r):
    res.append(r)
    pd.DataFrame(res).to_csv(OUT / 'eval_results.csv', index=False)
    log_row(id=f"E033_{r['test']}_{r['set']}", category='data',
            description=f"Unified external eval ({r['test']}, set {r['set']})",
            config=f"experiments/ext_unified/eval.py; w_ext={r.get('w_ext', '')}; lam={1.5 if r['test'] == 'B_gates' else ''}",
            metric='oof_bacc_on_strong', baseline=f'{base:.4f}', value=str(r['on_strong']),
            folds_pos=r['folds_pos'], files='experiments/ext_unified/out/eval_results.csv',
            notes=f"n_ext={r['n_ext']}; standalone={r.get('standalone', '')}; member_w={r.get('member_w', '')}; per_lab={r.get('per_lab', '')}")


def append_summary():
    summary_path = OUT / 'SUMMARY.md'
    old = summary_path.read_text(encoding='utf-8') if summary_path.exists() else ''
    marker = '## External source value (E033)'
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    by_key = {(r['test'], r['set']): r for r in res}
    lines = [marker, '', '| Set | A delta (folds) | B delta (folds) |', '|---|---:|---:|']
    for name in SETS:
        a = by_key.get(('A_18way_svc', name))
        b = by_key.get(('B_gates', name))
        af = f"{a['delta_vs_strong']:+.4f} ({a['folds_pos']})" if a else '—'
        bf = f"{b['delta_vs_strong']:+.4f} ({b['folds_pos']})" if b else '—'
        lines.append(f'| {name} | {af} | {bf} |')
    all_a = by_key.get(('A_18way_svc', 'ALL'))
    all_b = by_key.get(('B_gates', 'ALL'))
    hurt_a = [name.removeprefix('ALL_minus_') for name in SETS
              if name.startswith('ALL_minus_') and all_a and by_key.get(('A_18way_svc', name), {}).get('delta_vs_strong', 0) < all_a['delta_vs_strong']]
    hurt_b = [name.removeprefix('ALL_minus_') for name in SETS
              if name.startswith('ALL_minus_') and all_b and by_key.get(('B_gates', name), {}).get('delta_vs_strong', 0) < all_b['delta_vs_strong']]
    lines += ['', f"Removals that hurt vs ALL (source helps): A: {', '.join(hurt_a) or 'none'}; B: {', '.join(hurt_b) or 'none'}."]
    pv = by_key.get(('A_18way_svc', 'PV'))
    lines.append(f"PV-only A reference: {'passed' if pv and abs(pv['delta_vs_strong'] - 0.0133) <= 0.002 else 'not reached'}; target +0.0133 ±0.002.")
    section = '\n'.join(lines)
    if len(section.splitlines()) > 20:
        raise RuntimeError(f'E033 summary section exceeds 20 lines: {len(section.splitlines())}')
    summary_path.write_text((old + '\n' if old else '') + section + '\n', encoding='utf-8')

# ---- A) 18-way member
for src in ('PV',):  # reference gate: stop before other sets if the PV-only A result misses tolerance
    E = SETS[src]
    wext = 1.0
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
    r = dict(test='A_18way_svc', set=src, w_ext=wext, n_ext=len(E), standalone=round(bacc(y, D.argmax(1)), 4),
             on_strong=round(best[0], 4), delta_vs_strong=round(best[0] - base, 4), member_w=best[1],
             folds_pos=f'{sum(x > 0 for x in fd)}/5', seconds=round(time.time() - T0))
    print(r, flush=True)
    save_result(r)
    if abs((best[0] - base) - 0.0133) > 0.002:
        print('PV-only A reference missed tolerance; stopping as requested.', flush=True)
        sys.exit(2)
for src, E in SETS.items():
    if src == 'PV':
        continue
    wext = 0.0 if src == 'none' else 1.0
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
    r = dict(test='A_18way_svc', set=src, w_ext=wext, n_ext=len(E), standalone=round(bacc(y, D.argmax(1)), 4),
             on_strong=round(best[0], 4), delta_vs_strong=round(best[0] - base, 4), member_w=best[1],
             folds_pos=f'{sum(x > 0 for x in fd)}/5', seconds=round(time.time() - T0))
    print(r, flush=True)
    save_result(r)
# ---- B) gates
for src, E in SETS.items():
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
    r = dict(test='B_gates', set=src, n_ext=len(E), on_strong=round(bacc(y, P), 4),
             delta_vs_strong=round(bacc(y, P) - base, 4), folds_pos=f'{sum(x > 0 for x in fd)}/5',
             per_lab=json.dumps(per), seconds=round(time.time() - T0))
    print(r, flush=True)
    save_result(r)
append_summary()
print('done', round(time.time() - T0), 's')
