"""H2 surface-feature analysis.

Step 1: ~34 surface features per passage (train + pool_v4) -> features.parquet
Step 2: per sibling pair, prompts answered by both models only: LogisticRegression on the surface
        features, GroupKFold(5) by prompt_id, balanced accuracy (pool+train, train only, pool->train transfer),
        feature-group ablations, top-5 univariate features with per-model means.
Step 3: complementarity on train: does the surface logit add to the blend's pairwise logit difference?
Step 4: 18-class surface member added to the OOF blend (weight grid).
Outputs -> research/bottleneck/h2_surface/out/
"""
import re, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

T0 = time.time()
say = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
OUT = Path('research/bottleneck/h2_surface/out'); OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv('data/train.csv')
pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet')
norm = lambda s: s.astype(str).str.lower().str.split().str.join(' ')
pool = pool[~norm(pool.text).isin(set(norm(train.text)))]          # same leakage guard as the trainers
df = pd.concat([train.assign(src='train', source='host')[['id', 'prompt_id', 'text', 'model', 'src', 'source']],
                pool.assign(src='pool', prompt_id='pool_' + pool.prompt_id.astype(str))[['id', 'prompt_id', 'text', 'model', 'src', 'source']]],
               ignore_index=True)

# ------------------------------------------------------------------ features
SENT = re.compile(r'(?<=[.!?])["”’)]?\s+(?=["“‘(]?[A-Z0-9])')
WORD = re.compile(r"[A-Za-z0-9’'\-]+")
CAPW = re.compile(r'(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+')
QUOTE_DASH = set('—–’‘“”…')


def feats(t):
    words = WORD.findall(t)
    nw = max(len(words), 1); nc = max(len(t), 1)
    sents = [s for s in SENT.split(t) if s.strip()]
    sl = np.array([len(WORD.findall(s)) for s in sents] or [0], float)
    wl = np.array([len(w) for w in words] or [0], float)
    per = lambda c: 100.0 * c / nw
    f = dict(
        n_words=len(words), n_chars=len(t), n_sents=len(sents),
        sent_mean=sl.mean(), sent_std=sl.std(), sent_max=sl.max(), sent_min=sl.min(),
        word_len=wl.mean(), long_words=(wl >= 9).mean(), ttr=len(set(w.lower() for w in words)) / nw,
        emdash=per(t.count('—')), emdash_spaced=per(t.count(' — ')), endash=per(t.count('–')),
        hyphen_word=per(len(re.findall(r'\w-\w', t))), dash_ascii_spaced=per(t.count(' - ') + t.count('--')),
        apos_curly=per(t.count('’') + t.count('‘')), apos_straight=per(t.count("'")),
        dq_curly=per(t.count('“') + t.count('”')), dq_straight=per(t.count('"')),
        semicolon=per(t.count(';')), colon=per(t.count(':')), comma=per(t.count(',')),
        exclam=per(t.count('!')), question=per(t.count('?')), paren=per(t.count('(')),
        markdown=per(t.count('*') + t.count('_') + t.count('#') + t.count('`')),
        ellipsis=per(t.count('…') + t.count('...')), slash=per(t.count('/')),
        digits=100.0 * sum(c.isdigit() for c in t) / nc,
        cap_words=per(len(CAPW.findall(t))), allcaps=per(len(re.findall(r'\b[A-Z]{2,}\b', t))),
        nonascii_other=100.0 * sum(ord(c) > 127 and c not in QUOTE_DASH for c in t) / nc,
        inline_enum=per(len(re.findall(r'\(\w\)|\b\d\)|\b(?:first|second|third|finally),', t, flags=re.I))),
        commas_per_sent=t.count(',') / max(len(sents), 1),
        ends_period=float(t.rstrip().endswith('.')),
    )
    return f


fp = OUT / 'features.parquet'
if fp.exists():
    F = pd.read_parquet(fp)
    assert (F.id.values == df.id.values).all()
else:
    say(f'features for {len(df)} passages')
    F = pd.DataFrame([feats(t) for t in df.text])
    F.insert(0, 'id', df.id.values)
    F.to_parquet(fp, index=False)
FEATS = [c for c in F.columns if c != 'id']
say(f'{len(FEATS)} features: {FEATS}')
X_all = F[FEATS].to_numpy(float)
X_all = np.sign(X_all) * np.log1p(np.abs(X_all))   # heavy-tailed rates -> log1p

GROUPS = {
    'length': ['n_words', 'n_chars', 'n_sents', 'sent_mean', 'sent_std', 'sent_max', 'sent_min', 'commas_per_sent'],
    'dash_quote_unicode': ['emdash', 'emdash_spaced', 'endash', 'dash_ascii_spaced', 'apos_curly', 'apos_straight', 'dq_curly',
                           'dq_straight', 'ellipsis', 'nonascii_other'],
    'other_punct': ['semicolon', 'colon', 'comma', 'exclam', 'question', 'paren', 'markdown', 'slash', 'hyphen_word', 'ends_period'],
    'lexical_shape': ['word_len', 'long_words', 'ttr', 'digits', 'cap_words', 'allcaps', 'inline_enum'],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATS), set(FEATS) ^ set(sum(GROUPS.values(), []))

PAIRS = [('gemini-3.6-flash', 'gemini-3.7-flash'), ('gemini-3.6-flash', 'gemini-3.8-flash'), ('gemini-3.7-flash', 'gemini-3.8-flash'),
         ('glm-5.3', 'glm-5.3-flash'),
         ('gpt-5.6-luna', 'gpt-5.6-sol'), ('gpt-5.6-luna', 'gpt-5.6-terra'), ('gpt-5.6-sol', 'gpt-5.6-terra'),
         ('claude-opus-4-8', 'claude-sonnet-5')]

lr = lambda C=1.0: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight='balanced'))


def cv_bacc(X, y, g, model_fn, k=5):
    pred = np.zeros(len(y), int); score = np.zeros(len(y))
    for tr_i, va_i in GroupKFold(min(k, len(np.unique(g)))).split(X, y, g):
        m = model_fn().fit(X[tr_i], y[tr_i])
        pred[va_i] = m.predict(X[va_i]); score[va_i] = m.predict_proba(X[va_i])[:, 1]
    return balanced_accuracy_score(y, pred), score


# blend OOF (for complementarity)
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)
CL = list(z['classes']); REF = z['reference']
assert (z['ids'] == train.id.values).all()
ytr18 = train.model.map({c: i for i, c in enumerate(CL)}).to_numpy()

rows, top_rows = [], []
is_train = (df.src == 'train').to_numpy()
for a, b in PAIRS:
    m = df.model.isin([a, b]).to_numpy()
    sub = df[m]
    both = sub.groupby('prompt_id').model.nunique()
    shared = set(both[both == 2].index)
    keep = m & df.prompt_id.isin(shared).to_numpy()
    idx = np.flatnonzero(keep)
    X, y, g = X_all[idx], (df.model.to_numpy()[idx] == b).astype(int), df.prompt_id.to_numpy()[idx]
    tmask = is_train[idx]
    r = dict(pair=f'{a} | {b}', n_rows=len(idx), n_prompts=len(shared), n_pool=int((~tmask).sum()), n_train=int(tmask.sum()))
    r['lr_all'], _ = cv_bacc(X, y, g, lr)
    r['hgb_all'], _ = cv_bacc(X, y, g, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                                               class_weight='balanced', random_state=0))
    r['lr_train_only'], _ = cv_bacc(X[tmask], y[tmask], g[tmask], lambda: lr(0.3))
    # pool -> train transfer (fit on pool shared prompts only, score host rows)
    mt = lr().fit(X[~tmask], y[~tmask])
    r['lr_pool_to_train'] = balanced_accuracy_score(y[tmask], mt.predict(X[tmask]))
    # feature-group ablations (pool+train CV)
    for gname, cols in GROUPS.items():
        ci = [FEATS.index(c) for c in cols]
        keep_c = [j for j in range(len(FEATS)) if j not in ci]
        r[f'only_{gname}'], _ = cv_bacc(X[:, ci], y, g, lr)
        r[f'drop_{gname}'], _ = cv_bacc(X[:, keep_c], y, g, lr)
    # blend pairwise accuracy on the same host rows (scenario '1' = index 0 and '6' = index 2)
    tri = idx[tmask]   # positions in df == positions in train (train rows come first)
    ia, ib = CL.index(a), CL.index(b)
    for s, sname in [(0, 's1'), (2, 's6')]:
        d = REF[s][tri, ib] - REF[s][tri, ia]
        r[f'blend_pair_{sname}'] = balanced_accuracy_score(y[tmask], (d > 0).astype(int))
    # complementarity on host rows: [blend diff] vs [blend diff + surface logit from the pool-fitted model]
    d = (REF[0][tri, ib] - REF[0][tri, ia])[:, None]
    sl = mt.decision_function(X[tmask])[:, None]
    gt = g[tmask]
    r['comp_blend_only'], _ = cv_bacc(d, y[tmask], gt, lambda: lr(1.0))
    r['comp_blend_plus_surface'], _ = cv_bacc(np.hstack([d, sl]), y[tmask], gt, lambda: lr(1.0))
    r['comp_blend_plus_allfeats'], _ = cv_bacc(np.hstack([d, X[tmask]]), y[tmask], gt, lambda: lr(0.1))
    rows.append(r)
    # univariate discriminativeness (pool+train shared prompts) + per-model means on raw features
    raw = F[FEATS].to_numpy(float)[idx]
    for j, c in enumerate(FEATS):
        if raw[:, j].std() == 0:
            auc = 0.5
        else:
            auc = roc_auc_score(y, raw[:, j])
        top_rows.append(dict(pair=r['pair'], feature=c, auc=auc, abs_eff=abs(auc - 0.5),
                             mean_a=raw[y == 0, j].mean(), mean_b=raw[y == 1, j].mean(),
                             train_mean_a=raw[tmask & (y == 0), j].mean(), train_mean_b=raw[tmask & (y == 1), j].mean()))
    say(f"{r['pair']}: rows {len(idx)} prompts {len(shared)}  LR {r['lr_all']:.3f}  HGB {r['hgb_all']:.3f}  train-only {r['lr_train_only']:.3f}  "
        f"pool->train {r['lr_pool_to_train']:.3f}  blend s1 {r['blend_pair_s1']:.3f}  blend+surf {r['comp_blend_only']:.3f}->{r['comp_blend_plus_surface']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / 'pair_separability.csv', index=False)
top = pd.DataFrame(top_rows).sort_values(['pair', 'abs_eff'], ascending=[True, False])
top.to_csv(OUT / 'pair_feature_auc_all.csv', index=False)
top.groupby('pair').head(5).to_csv(OUT / 'pair_top5_features.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(res.round(3).to_string())
print(top.groupby('pair').head(5).round(3).to_string())

# ------------------------------------------------------------------ 18-class surface member on the blend
say('18-class surface member')
ytr = ytr18
pool_idx = np.flatnonzero(~is_train); tr_idx = np.flatnonzero(is_train)
y_all = df.model.map({c: i for i, c in enumerate(CL)}).to_numpy()
oof = np.zeros((len(tr_idx), 18))
rng = np.random.default_rng(0)
sub_pool = pool_idx if len(pool_idx) <= 40000 else rng.choice(pool_idx, 40000, replace=False)   # CPU budget
for k, (a_i, v_i) in enumerate(GroupKFold(5).split(tr_idx, ytr, train.prompt_id)):
    fit_i = np.r_[sub_pool, tr_idx[a_i]]
    sw = np.r_[np.full(len(sub_pool), 0.25), np.ones(len(a_i))]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X_all[fit_i], y_all[fit_i], logisticregression__sample_weight=sw)
    oof[v_i] = m.predict_log_proba(X_all[tr_idx[v_i]])
    say(f'  fold {k} done')
bacc = lambda p: balanced_accuracy_score(ytr, p.argmax(1))
SIB = {'gemini': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'], 'glm': ['glm-5.3', 'glm-5.3-flash'],
       'gpt56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'], 'claude48s5': ['claude-opus-4-8', 'claude-sonnet-5']}
sib_cls = [CL.index(c) for v in SIB.values() for c in v]
mem = []
for s, sname in enumerate(['s1', 's3', 's6']):
    for w in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        P = REF[s] + w * oof
        pr = P.argmax(1)
        sm = np.isin(ytr, sib_cls)
        mem.append(dict(scenario=sname, w=w, bacc=bacc(P), sib_acc=(pr[sm] == ytr[sm]).mean()))
mem.append(dict(scenario='surface_alone', w=1, bacc=bacc(oof), sib_acc=(oof.argmax(1)[np.isin(ytr, sib_cls)] == ytr[np.isin(ytr, sib_cls)]).mean()))
mem = pd.DataFrame(mem)
mem.to_csv(OUT / 'surface_member_blend.csv', index=False)
np.save(OUT / 'surface_member_oof_logp.npy', oof)
print(mem.round(4).to_string())
say('done')
