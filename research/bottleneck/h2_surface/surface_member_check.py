"""Follow-up: is the 18-class surface member's gain on the OOF blend real?
Nested weight choice (w picked on 4 prompt folds, scored on the 5th), prompt-bootstrap CI of the gain at fixed w,
and feature-group variants (all / length only / dash-quote-unicode only / all minus dash-quote-unicode).
Reuses out/features.parquet from surface_features.py."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

OUT = Path('research/bottleneck/h2_surface/out')
train = pd.read_csv('data/train.csv')
pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet')
F = pd.read_parquet(OUT / 'features.parquet')
FEATS = [c for c in F.columns if c != 'id']
X = F[FEATS].to_numpy(float); X = np.sign(X) * np.log1p(np.abs(X))
n_tr = len(train)
assert (F.id.values[:n_tr] == train.id.values).all() and len(F) == n_tr + len(pool)
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)
CL = list(z['classes']); REF = z['reference']
y_tr = train.model.map({c: i for i, c in enumerate(CL)}).to_numpy()
y_pool = pool.model.map({c: i for i, c in enumerate(CL)}).to_numpy()
GROUPS = {'length': ['n_words', 'n_chars', 'n_sents', 'sent_mean', 'sent_std', 'sent_max', 'sent_min', 'commas_per_sent'],
          'dashquote': ['emdash', 'emdash_spaced', 'endash', 'dash_ascii_spaced', 'apos_curly', 'apos_straight', 'dq_curly', 'dq_straight',
                        'ellipsis', 'nonascii_other']}
VARIANTS = {'all': FEATS, 'length_only': GROUPS['length'], 'dashquote_only': GROUPS['dashquote'],
            'all_minus_dashquote': [f for f in FEATS if f not in GROUPS['dashquote']]}
rng = np.random.default_rng(0)
sub_pool = rng.choice(len(pool), 40000, replace=False) + n_tr      # same CPU subsample size as surface_features.py
folds = list(GroupKFold(5).split(np.zeros(n_tr), y_tr, train.prompt_id))
SIB = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.6-luna', 'gpt-5.6-sol',
       'gpt-5.6-terra', 'claude-opus-4-8', 'claude-sonnet-5']
sib = np.isin(y_tr, [CL.index(c) for c in SIB])
W = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
prompts = train.prompt_id.to_numpy(); up = np.unique(prompts)
rows = []
for vname, cols in VARIANTS.items():
    ci = [FEATS.index(c) for c in cols]
    oof = np.zeros((n_tr, 18))
    for a_i, v_i in folds:
        fit_i = np.r_[sub_pool, a_i]
        y_fit = np.r_[y_pool[sub_pool - n_tr], y_tr[a_i]]
        sw = np.r_[np.full(len(sub_pool), 0.25), np.ones(len(a_i))]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(X[fit_i][:, ci], y_fit, logisticregression__sample_weight=sw)
        oof[v_i] = m.predict_log_proba(X[v_i][:, ci])
    for s, sname in enumerate(['s1', 's3', 's6']):
        base = REF[s].argmax(1)
        # nested: choose w on the other 4 folds, apply on the held-out fold
        pred = np.zeros(n_tr, int); chosen = []
        for a_i, v_i in folds:
            sc = [balanced_accuracy_score(y_tr[a_i], (REF[s][a_i] + w * oof[a_i]).argmax(1)) for w in W]
            w = W[int(np.argmax(sc))]; chosen.append(w)
            pred[v_i] = (REF[s][v_i] + w * oof[v_i]).argmax(1)
        fixed = (REF[s] + 0.2 * oof).argmax(1)
        # prompt bootstrap of the gain at fixed w=0.2
        gains = []
        idx_by_p = {p: np.flatnonzero(prompts == p) for p in up}
        for _ in range(300):
            bi = np.concatenate([idx_by_p[p] for p in rng.choice(up, len(up))])
            gains.append(balanced_accuracy_score(y_tr[bi], fixed[bi]) - balanced_accuracy_score(y_tr[bi], base[bi]))
        rows.append(dict(variant=vname, scenario=sname, base=balanced_accuracy_score(y_tr, base),
                         nested=balanced_accuracy_score(y_tr, pred), nested_w=str(chosen),
                         fixed_w02=balanced_accuracy_score(y_tr, fixed), gain_ci_lo=np.percentile(gains, 2.5), gain_ci_hi=np.percentile(gains, 97.5),
                         sib_base=(base[sib] == y_tr[sib]).mean(), sib_w02=(fixed[sib] == y_tr[sib]).mean(),
                         nonsib_base=(base[~sib] == y_tr[~sib]).mean(), nonsib_w02=(fixed[~sib] == y_tr[~sib]).mean()))
        print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / 'surface_member_nested.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(res.round(4).to_string())
