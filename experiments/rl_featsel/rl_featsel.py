"""E041: reinforcement learning (REINFORCE) for feature-set optimisation of the stylegram LinearSVC.
Action per episode: a weight level for each feature group (0 = group dropped) and the SVC C.
Reward: mean standalone balanced accuracy on outer folds 0-2 (prompt-grouped GroupKFold(5); each fold trained on the
other 4 folds, as in the standard CV). Selection never sees folds 3-4; the final comparison is on folds 3-4 only:
  default config (the notebooks' SPEC) vs best random-search config vs best RL config, same episode budget.
Feature groups (fitted per fold on that fold's training rows):
  char35 | word13 | shape26 | func13 (the 4 default views) | char12 | lchar35 (lowercased char 3-5) |
  punct (punctuation/space skeleton, char 1-4) | spectro (collapsed char/word/sentence/token band shares, standardised)
Logs: experiments/rl_featsel/out/episodes.csv (every episode: method, config, reward), final.csv, run log;
paper/experiments.csv rows E041_* (idempotent)."""
import json
import pathlib
import re
import sys
import time

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

OUT = ROOT / 'experiments/rl_featsel/out'
OUT.mkdir(parents=True, exist_ok=True)
EPISODES, BATCH, LR, SEED = 60, 6, 0.6, 0
LEVELS = [0.0, 0.25, 0.5, 0.7, 1.0, 1.5]
CS = [0.25, 0.5, 1.0]
T0 = time.time()
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
TX = tr.text.astype(str).tolist()
folds = list(GroupKFold(5).split(tr, y, tr.prompt_id.values))
FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))
punct = lambda t: re.sub(r"[A-Za-z0-9]+", "w", t)
WP = r"(?u)\b\w+\b|[^\w\s]"
GROUPS = {'char35': (dict(analyzer='char', ngram_range=(3, 5), lowercase=False, max_features=250000), str),
          'word13': (dict(analyzer='word', ngram_range=(1, 3), lowercase=False, max_features=180000, token_pattern=WP), str),
          'shape26': (dict(analyzer='char', ngram_range=(2, 6), lowercase=False, max_features=60000), shape),
          'func13': (dict(analyzer='word', ngram_range=(1, 3), lowercase=False, max_features=50000, token_pattern=WP), func),
          'char12': (dict(analyzer='char', ngram_range=(1, 2), lowercase=False), str),
          'lchar35': (dict(analyzer='char', ngram_range=(3, 5), lowercase=True, max_features=150000), str),
          'punct': (dict(analyzer='char', ngram_range=(1, 4), lowercase=False, max_features=30000), punct),
          'spectro': None}
NAMES = list(GROUPS)
DEFAULT = dict(char35=1.0, word13=0.7, shape26=0.5, func13=0.5, char12=0.0, lchar35=0.0, punct=0.0, spectro=0.0, C=0.5)
SPEC = np.hstack([np.nan_to_num(np.load(ROOT / f'experiments/spectro/out/{lev}_passage_shares.npz')['F'], nan=0.0)
                  for lev in ('char', 'word', 'sentence', 'token')])

# ---------------------------------------------------------------- per-fold group matrices (fitted on fold-train rows)
FEAT = []
for fi, (a, b) in enumerate(folds):
    d = {}
    for g, spec in GROUPS.items():
        if spec is None:
            sc = StandardScaler().fit(SPEC[a])
            k = 1 / np.sqrt(SPEC.shape[1])
            d[g] = (csr_matrix(sc.transform(SPEC[a]) * k), csr_matrix(sc.transform(SPEC[b]) * k))
            continue
        o, tf = spec
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, dtype=np.float32, **o)
        d[g] = (v.fit_transform([tf(TX[i]) for i in a]), v.transform([tf(TX[i]) for i in b]))
    FEAT.append(d)
    print(f'fold {fi} features ready {time.time() - T0:.0f}s', flush=True)


def run(cfg, fold_ids):
    """standalone bacc per fold for a config."""
    out = []
    for fi in fold_ids:
        a, b = folds[fi]
        A = hstack([FEAT[fi][g][0] * cfg[g] for g in NAMES if cfg[g] > 0], format='csr')
        B = hstack([FEAT[fi][g][1] * cfg[g] for g in NAMES if cfg[g] > 0], format='csr')
        m = LinearSVC(C=cfg['C'], max_iter=4000).fit(A, y[a])
        out.append((bacc(y[b], m.decision_function(B).argmax(1)), m.decision_function(B), b))
    return out


CACHE, LOGROWS = {}, []


def reward(cfg, method, ep):
    key = json.dumps(cfg, sort_keys=True)
    if key not in CACHE:
        if all(cfg[g] == 0 for g in NAMES):
            CACHE[key] = 0.0
        else:
            CACHE[key] = float(np.mean([r[0] for r in run(cfg, [0, 1, 2])]))
    LOGROWS.append(dict(method=method, episode=ep, reward=round(CACHE[key], 4), seconds=round(time.time() - T0), **cfg))
    pd.DataFrame(LOGROWS).to_csv(OUT / 'episodes.csv', index=False)
    return CACHE[key]


rng = np.random.default_rng(SEED)
r_default = reward(DEFAULT, 'default', 0)
print('default reward (folds 0-2):', round(r_default, 4), flush=True)
# ---------------------------------------------------------------- random search (same budget)
best_rs = (r_default, DEFAULT)
for ep in range(1, EPISODES + 1):
    cfg = {g: float(rng.choice(LEVELS)) for g in NAMES}
    cfg['C'] = float(rng.choice(CS))
    r = reward(cfg, 'random', ep)
    best_rs = max(best_rs, (r, cfg), key=lambda t: t[0])
print('random search best:', round(best_rs[0], 4), best_rs[1], flush=True)
# ---------------------------------------------------------------- REINFORCE: independent softmax policy per group + C
theta = {g: np.zeros(len(LEVELS)) for g in NAMES}
theta['C'] = np.zeros(len(CS))
best_rl = (r_default, DEFAULT)
ep = 0
while ep < EPISODES:
    batch = []
    for _ in range(BATCH):
        acts = {}
        for g in theta:
            p = np.exp(theta[g] - theta[g].max())
            p /= p.sum()
            acts[g] = (rng.choice(len(p), p=p), p)
        cfg = {g: LEVELS[acts[g][0]] for g in NAMES}
        cfg['C'] = CS[acts['C'][0]]
        ep += 1
        r = reward(cfg, 'reinforce', ep)
        best_rl = max(best_rl, (r, cfg), key=lambda t: t[0])
        batch.append((acts, r))
    base = np.mean([r for _, r in batch])
    sd = np.std([r for _, r in batch]) + 1e-6
    for acts, r in batch:   # grad of log softmax = onehot - p
        adv = (r - base) / sd
        for g, (i, p) in acts.items():
            gvec = -p
            gvec[i] += 1
            theta[g] += LR / BATCH * adv * gvec
    print(f'RL ep {ep}: batch mean {base:.4f}, best {best_rl[0]:.4f}', flush=True)
pd.DataFrame({g: np.round(np.exp(t) / np.exp(t).sum(), 3) for g, t in theta.items() if g != 'C'}, index=LEVELS).to_csv(OUT / 'policy_final.csv')
print('RL best:', round(best_rl[0], 4), best_rl[1], flush=True)

# ---------------------------------------------------------------- held-out comparison on folds 3-4 (never used for selection)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
hold = np.concatenate([folds[3][1], folds[4][1]])
yh = y[hold]
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(Z)), yh].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
Sh = cal(S[hold])
final = []
for method, (_, cfg) in [('default', (r_default, DEFAULT)), ('random_best', best_rs), ('reinforce_best', best_rl)]:
    rr = run(cfg, [3, 4])
    D = np.zeros((len(y), 18))
    for _, dec, b in rr:
        D[b] = dec
    Dh = cal(D[hold])
    on = max((bacc(yh, (Sh + mw * Dh).argmax(1)), mw) for mw in (0.25, 0.5, 1.0))
    final.append(dict(method=method, config=json.dumps(cfg), search_reward_folds012=round(CACHE[json.dumps(cfg, sort_keys=True)], 4),
                      holdout_standalone_folds34=round(bacc(yh, D[hold].argmax(1)), 4),
                      holdout_fold_scores=json.dumps([round(r[0], 4) for r in rr]),
                      holdout_on_strong=round(on[0], 4), strong_alone=round(bacc(yh, Sh.argmax(1)), 4), member_w=on[1]))
    print(final[-1], flush=True)
F = pd.DataFrame(final)
F.to_csv(OUT / 'final.csv', index=False)
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E041_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
d0 = final[0]['holdout_standalone_folds34']
for r in final:
    log_row(id=f"E041_{r['method']}", category='features', description=f"RL feature-set optimisation: {r['method']} config "
            f"(search on folds 0-2, {EPISODES} episodes; evaluated on held-out folds 3-4)",
            config=r['config'], metric='holdout_bacc_standalone_folds34', baseline=str(d0),
            value=str(r['holdout_standalone_folds34']), files='experiments/rl_featsel/out/final.csv;experiments/rl_featsel/out/episodes.csv',
            notes=f"search reward {r['search_reward_folds012']}; on strong {r['holdout_on_strong']} (strong alone {r['strong_alone']})")
print('done', round(time.time() - T0), 's')
