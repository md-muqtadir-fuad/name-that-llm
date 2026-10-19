"""Does SpeechMap help the 0.563 public notebook's in-style stylegram scheme? (prompt-grouped 5-fold OOF on train)
Scheme (vnton37nguyn v14-v16): extra passages from the seyit prompt bank (seyit Claude API, Lucas / abdullah / dophuchung
proxy generations) + prvsiyan, one passage per (prompt, label), prompts with >= 12 labels only, TF-IDF vocabulary from
train + extra, per class the extra rows share EXTRA_W = 0.5 of the class weight. Plain stylegram SVC only (no twin).
Variants: none | pool | pool+SM<q> (q SpeechMap questions with >= 12 labels, first paragraph) | SM<q> only.
Reports standalone OOF bacc and gain on the strong frozen-LLM OOF blend (member weight grid, as experiments/speechmap/eval.py).
Writes experiments/instyle/out/results.csv and appends paper/experiments.csv (E034_*)."""
import glob
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

OUT = ROOT / 'experiments/instyle/out'
OUT.mkdir(parents=True, exist_ok=True)
RUN_ID = __import__('os').environ.get('RUN_ID', 'E034')
T0 = time.time()
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
lid = {l: i for i, l in enumerate(L)}
y = tr.model.map(lid).values
K = 18
folds = list(GroupKFold(5).split(tr, y, tr.prompt_id.values))
EXTRA_MIN_LABELS, EXTRA_W = 12, 0.5

zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
S = cal(S)
base = bacc(y, S.argmax(1))


def clean(x, k=EXTRA_MIN_LABELS):
    x = x[x.model.isin(L)].copy()
    x['text'] = x.text.astype(str).map(lambda t: re.sub(r'\s+', ' ', re.sub(r'^\*\*(.*)\*\*$', r'\1', t.strip())).strip())
    x = x[x.text.str.split().str.len().between(30, 300)].drop_duplicates(subset=['model', 'text'])
    x = x.drop_duplicates(subset=['prompt_id', 'model'], keep='first')
    return x[x.groupby('prompt_id').model.transform('nunique') >= k].reset_index(drop=True)


E = ROOT / 'research/ext_data'
fr = [pd.read_csv(E / 'extra_api/ntl_extra_passages_claude.csv')[['prompt_id', 'model', 'text']],
      pd.read_csv(E / 'lucas/gen_passages.csv')[['prompt_id', 'model', 'text']]]
d = pd.read_csv(E / 'abdullah/ntl_generations.csv')
fr.append(d[d.finish_reason.eq('stop')].rename(columns={'label': 'model'})[['prompt_id', 'model', 'text']])
fr.append(pd.read_csv(E / 'dophuchung/ntl_proxy_generations.csv').rename(columns={'label': 'model'})[['prompt_id', 'model', 'text']])
d = pd.read_csv(E / 'prvsiyan/passages_clean.csv')[['prompt_id', 'model', 'text']]
fr.append(d.assign(prompt_id='pv_' + d.prompt_id.astype(str)))
RAW = pd.concat(fr, ignore_index=True)
POOL = clean(RAW)
sm = pd.read_csv(E / 'speechmap/kaggle_ds/speechmap_p0.csv')
sm = clean(sm.rename(columns={'label': 'model'}).assign(prompt_id=lambda d: 'sm_' + d.question_id.astype(str))
           [['prompt_id', 'model', 'text']])
qs = sm.prompt_id.drop_duplicates().sample(frac=1, random_state=0).tolist()
SM = {q: sm[sm.prompt_id.isin(qs[:q])] for q in (300, 900)}
SETS = {'none': POOL.iloc[:0], 'pool': POOL, 'pool+SM300': pd.concat([POOL, SM[300]]),
        'pool+SM900': pd.concat([POOL, SM[900]]), 'SM900': SM[900]}
# our Kaggle Benchmarks generations (seyit prompt bank, 'about N words' protocol); b001 excluded (contaminated)
kb = []
for f in sorted(glob.glob(str(E / 'kbench/b0*/**/ntl_gen_b*.csv'), recursive=True)):
    if '/b001' in f.replace(chr(92), '/'):
        continue
    d = pd.read_csv(f)
    d = d[d.error.fillna('').eq('') & d.response.fillna('').astype(str).str.len().gt(0)]
    kb.append(pd.DataFrame({'prompt_id': d.prompt_id, 'text': d.response.astype(str),
                            'model': d.llm.str.split('/').str[-1].str.replace('@default', '').str.replace('-2026-04-23', '')}))
KB = pd.concat(kb, ignore_index=True) if kb else POOL.iloc[:0]
for _k in (10, 11, 12):   # min labels per prompt (pool rows first, so duplicates keep the public passage)
    SETS[f'pool@{_k}'] = clean(RAW, _k)
    SETS[f'pool+kb@{_k}'] = clean(pd.concat([RAW, KB], ignore_index=True), _k)
if len(sys.argv) > 1:
    SETS = {k: SETS[k] for k in sys.argv[1].split(',')}
print({k: len(v) for k, v in SETS.items()}, '| pool per label', POOL.model.value_counts().to_dict(), flush=True)

FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))

res = []
for name, X in SETS.items():
    ex_t, ex_y = X.text.tolist(), X.model.map(lid).values.astype(int)
    D = np.zeros((len(y), K))
    for a, b in folds:
        ya = y[a]
        A, B = [], []
        for an, lo, hi, mf, wt in SPEC:
            tf = shape if an == 'shape' else func if an == 'function' else str
            act = 'char' if an == 'shape' else 'word' if an == 'function' else an
            o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
            if act == 'word':
                o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
            v = TfidfVectorizer(**o)
            A.append(v.fit_transform([tf(t) for t in list(tr.text.values[a]) + ex_t]) * wt)
            B.append(v.transform([tf(t) for t in tr.text.values[b]]) * wt)
        ntr = np.bincount(ya, minlength=K).astype(float)
        nex = np.bincount(ex_y, minlength=K).astype(float)
        sw = np.concatenate([np.where(nex[ya] > 0, 1 - EXTRA_W, 1.0), EXTRA_W * ntr[ex_y] / np.maximum(nex[ex_y], 1)])
        m = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A, format='csr'), np.concatenate([ya, ex_y]),
                                                                 sample_weight=sw)
        D[np.ix_(b, m.classes_)] = m.decision_function(hstack(B, format='csr'))
    Ls = cal(D)
    best = max((bacc(y, (S + mw * Ls).argmax(1)), mw) for mw in (0.25, 0.5, 0.75, 1.0, 1.5))
    fd = [bacc(y[b], D[b].argmax(1)) for _, b in folds]
    r = dict(variant=name, n_ext=len(X), standalone=round(bacc(y, D.argmax(1)), 4), fold_standalone=json.dumps([round(f, 4) for f in fd]),
             on_strong=round(best[0], 4), delta_vs_strong=round(best[0] - base, 4), member_w=best[1], seconds=round(time.time() - T0))
    res.append(r)
    np.save(OUT / f'oof_{name}.npy', D)
    print(r, flush=True)
    pd.DataFrame(res).to_csv(OUT / f'results_{RUN_ID}.csv', index=False)
ref = {r['variant']: r for r in res}
ref.setdefault('pool', res[0])
for r in res:
    fs, fp = json.loads(r['fold_standalone']), json.loads(ref['pool']['fold_standalone'])
    log_row(id=f"{RUN_ID}_{r['variant']}", category='data', description=f"in-style stylegram scheme (0.563 notebook), extra set {r['variant']}",
            config='experiments/instyle/stylegram_extra_cv.py', metric='oof_bacc_standalone', baseline=str(ref['pool']['standalone']),
            value=str(r['standalone']), delta=str(round(r['standalone'] - ref['pool']['standalone'], 4)),
            folds_pos=f"{sum(a > b for a, b in zip(fs, fp))}/5 vs pool", files='experiments/instyle/out/results.csv',
            notes=f"n_ext {r['n_ext']}; on_strong {r['on_strong']} (delta {r['delta_vs_strong']}, w {r['member_w']})")
print('done', round(time.time() - T0), 's')
