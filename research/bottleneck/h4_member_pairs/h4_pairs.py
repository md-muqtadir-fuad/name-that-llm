"""H4: are sibling errors a combination problem (members separate pairs better than the blend)?

Step 1/2: pairwise accuracy of every member vs the blend on rows whose true label is in the pair.
Step 2b: honest best-member selection (pick on 4 folds, score on the 5th) + McNemar for the in-sample best.
Step 3:  pairwise expert, gated on blend top-2 == pair, leave-one-fold-out logistic regression on
         per-member log-prob differences. Reported per scenario, pooled + per fold, two fold schemes.
Read-only on inputs; writes only into this directory.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
T0 = time.time()

d = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
assert (tr.id.values == d['ids']).all()
CL = [str(c) for c in d['classes']]
CI = {c: i for i, c in enumerate(CL)}
Y = tr.model.map(CI).values
N = len(Y)

comps = d['components']                        # (37, 3, N, 18)
names = [str(n) for n in d['component_names']]
extra = []
for k in ('lstm', 'cnn'):
    e = np.load(ROOT / f'experiments/cnn_lstm/out/oof_{k}.npz', allow_pickle=True)
    assert (e['ids'] == tr.id.values).all() and [str(x) for x in e['labels']] == CL
    extra.append(np.broadcast_to(e['oof'][None], (3, N, 18)))
    names.append(k)
COMP = np.concatenate([comps, np.stack(extra)], 0)  # (39, 3, N, 18)
COMP = np.maximum(COMP, -50.0)
BLENDS = {'reference': d['reference'], 'nested': d['nested']}

PAIRS = [('gemini-3.6-flash', 'gemini-3.7-flash'), ('gemini-3.6-flash', 'gemini-3.8-flash'),
         ('gemini-3.7-flash', 'gemini-3.8-flash'), ('glm-5.3', 'glm-5.3-flash'),
         ('gpt-5.6-luna', 'gpt-5.6-sol'), ('gpt-5.6-luna', 'gpt-5.6-terra'),
         ('gpt-5.6-sol', 'gpt-5.6-terra'), ('claude-opus-4-8', 'claude-sonnet-5'),
         ('claude-fable-5', 'glm-5.3')]
PI = [(CI[a], CI[b]) for a, b in PAIRS]
PN = [f'{a}|{b}' for a, b in PAIRS]

fold_default = np.zeros(N, int)
for k, (_, v) in enumerate(GroupKFold(5).split(tr, groups=tr.prompt_id)):
    fold_default[v] = k
plan = json.loads((ROOT / 'research/ext_oof/q35heads/fold_plan.json').read_text())
fold_plan = np.zeros(N, int)
for k in range(5):
    fold_plan[np.array(plan['jobs'][f'outer{k}']['valid'])] = k
FOLDS = {'groupkfold_default': fold_default, 'fold_plan_outer': fold_plan}

log_lines = []
def log(msg):
    line = f'[{time.time() - T0:7.1f}s] {msg}'
    print(line); log_lines.append(line)


def top2_set(P):
    t = np.argsort(-P, 1)[:, :2]
    return np.sort(t, 1)


# ---------------- Step 1/2: pairwise accuracies ----------------
rows = []
summary = []
for s in range(3):
    ref = BLENDS['reference'][s]
    t2 = top2_set(ref)
    for (a, b), pn in zip(PI, PN):
        m = (Y == a) | (Y == b)
        ya = (Y[m] == a)
        gate = (t2[:, 0] == min(a, b)) & (t2[:, 1] == max(a, b))
        mg = m & gate
        blend_ok = ((ref[m, a] > ref[m, b]) == ya)
        blend_acc = blend_ok.mean()
        blend_gate_acc = ((ref[mg, a] > ref[mg, b]) == (Y[mg] == a)).mean()
        accs = []
        for j, nm in enumerate(names):
            P = COMP[j, s]
            ok = ((P[m, a] > P[m, b]) == ya)
            okg = ((P[mg, a] > P[mg, b]) == (Y[mg] == a))
            accs.append(ok.mean())
            rows.append(dict(scenario=s, pair=pn, member=nm, n=int(m.sum()), pair_acc=ok.mean(),
                             blend_pair_acc=blend_acc, delta=ok.mean() - blend_acc,
                             n_gated=int(mg.sum()), gated_acc=okg.mean(), blend_gated_acc=blend_gate_acc))
        accs = np.array(accs)
        jb = int(accs.argmax())
        okb = ((COMP[jb, s][m, a] > COMP[jb, s][m, b]) == ya)
        n10 = int((okb & ~blend_ok).sum()); n01 = int((~okb & blend_ok).sum())
        p_mc = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        # honest selection: pick best member on 4 folds, score on held-out fold (both fold schemes)
        hon = {}
        for fk, F in FOLDS.items():
            sel_ok = np.zeros(m.sum(), bool)
            Fm = F[m]
            for k in range(5):
                trm, tem = Fm != k, Fm == k
                acc_tr = [(((COMP[j, s][m, a] > COMP[j, s][m, b]) == ya)[trm]).mean() for j in range(len(names))]
                acc_tr.append(blend_ok[trm].mean())  # allow choosing the blend itself
                jj = int(np.argmax(acc_tr))
                if jj == len(names):
                    sel_ok[tem] = blend_ok[tem]
                else:
                    sel_ok[tem] = (((COMP[jj, s][m, a] > COMP[jj, s][m, b]) == ya)[tem])
            hon[fk] = sel_ok.mean() - blend_acc
        summary.append(dict(scenario=s, pair=pn, n=int(m.sum()), blend_pair_acc=blend_acc,
                            best_member=names[jb], best_acc=accs[jb], best_delta=accs[jb] - blend_acc,
                            n_members_beating_blend=int((accs > blend_acc).sum()),
                            n_members_beating_by_2pts=int((accs > blend_acc + 0.02).sum()),
                            mcnemar_best_vs_blend_p=p_mc, best_wins=n10, best_losses=n01,
                            honest_select_delta_gkf=hon['groupkfold_default'],
                            honest_select_delta_plan=hon['fold_plan_outer'],
                            svm_acc=accs[names.index('svm')], style_acc=accs[names.index('style')],
                            lstm_acc=accs[names.index('lstm')], cnn_acc=accs[names.index('cnn')],
                            n_gated_true_in_pair=int(mg.sum()), blend_gated_acc=blend_gate_acc,
                            pair_errors_total=int((~blend_ok).sum()),
                            ))
pd.DataFrame(rows).to_csv(OUT / 'pair_member_accuracy.csv', index=False)
S = pd.DataFrame(summary)
S.to_csv(OUT / 'pair_summary.csv', index=False)
log('step1/2 done')
with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.precision', 3):
    print(S[S.scenario == 0][['pair', 'blend_pair_acc', 'best_member', 'best_acc', 'best_delta',
                              'n_members_beating_blend', 'mcnemar_best_vs_blend_p',
                              'honest_select_delta_gkf', 'honest_select_delta_plan',
                              'svm_acc', 'style_acc', 'lstm_acc', 'n_gated_true_in_pair', 'blend_gated_acc']])
    print(S.groupby('scenario')[['best_delta', 'honest_select_delta_gkf', 'honest_select_delta_plan']].mean())


# ---------------- Step 3: pairwise expert ----------------
CGRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def fit_lr(X, y, groups):
    """Pick C by inner 3-fold GroupKFold (log-loss), refit on all training rows."""
    if len(np.unique(y)) < 2:
        return None
    best, bestC = np.inf, CGRID[0]
    ng = len(np.unique(groups))
    if ng >= 3:
        for C in CGRID:
            ll = 0.0
            for a_, b_ in GroupKFold(3).split(X, y, groups):
                if len(np.unique(y[a_])) < 2:
                    continue
                sc = StandardScaler().fit(X[a_])
                lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[a_]), y[a_])
                p = np.clip(lr.predict_proba(sc.transform(X[b_]))[:, 1], 1e-6, 1 - 1e-6)
                ll -= np.sum(y[b_] * np.log(p) + (1 - y[b_]) * np.log(1 - p))
            if ll < best:
                best, bestC = ll, C
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=bestC, max_iter=2000).fit(sc.transform(X), y)
    return sc, lr, bestC


def features(s, blend, a, b, idx, kind):
    bd = (blend[idx, a] - blend[idx, b])[:, None]
    if kind == 'blend_only':
        return bd
    cd = (COMP[:, s][:, idx, a] - COMP[:, s][:, idx, b]).T  # (n, 39)
    return np.hstack([bd, cd])


exp_rows, fold_rows, pair_rows = [], [], []
VARIANTS = [('blend_only', 'all_true_in_pair'), ('all_members', 'all_true_in_pair'),
            ('all_members', 'gated_true_in_pair')]
groups = tr.prompt_id.values
for blend_name in ('reference', 'nested'):
    for s in range(3):
        blend = BLENDS[blend_name][s]
        base = blend.argmax(1)
        t2 = top2_set(blend)
        for fk, F in FOLDS.items():
            if blend_name == 'nested' and fk != 'groupkfold_default':
                continue
            for kind, train_on in VARIANTS:
                pred = base.copy()
                Cs = []
                for (a, b), pn in zip(PI, PN):
                    gate = (t2[:, 0] == min(a, b)) & (t2[:, 1] == max(a, b))
                    inpair = (Y == a) | (Y == b)
                    for k in range(5):
                        trn = (F != k) & inpair
                        if train_on == 'gated_true_in_pair':
                            trn &= gate
                        tst = (F == k) & gate
                        if tst.sum() == 0:
                            continue
                        it, ie = np.where(trn)[0], np.where(tst)[0]
                        res = fit_lr(features(s, blend, a, b, it, kind), (Y[it] == a).astype(int), groups[it])
                        if res is None:
                            continue
                        sc, lr, C = res
                        Cs.append(C)
                        pa = lr.predict_proba(sc.transform(features(s, blend, a, b, ie, kind)))[:, 1]
                        pred[ie] = np.where(pa > 0.5, a, b)
                changed = pred != base
                fixed = int((changed & (pred == Y)).sum()); broke = int((changed & (base == Y)).sum())
                per_fold = [bas(Y[F == k], pred[F == k]) - bas(Y[F == k], base[F == k]) for k in range(5)]
                r = dict(blend=blend_name, scenario=s, folds=fk, features=kind, train_on=train_on,
                         base_bacc=bas(Y, base), expert_bacc=bas(Y, pred), delta=bas(Y, pred) - bas(Y, base),
                         n_changed=int(changed.sum()), fixed=fixed, broke=broke,
                         folds_improved=int(sum(x > 1e-12 for x in per_fold)),
                         folds_worse=int(sum(x < -1e-12 for x in per_fold)),
                         median_C=float(np.median(Cs)) if Cs else np.nan)
                exp_rows.append(r)
                for k, x in enumerate(per_fold):
                    fold_rows.append(dict(blend=blend_name, scenario=s, folds=fk, features=kind,
                                          train_on=train_on, fold=k, delta=x))
                for (a, b), pn in zip(PI, PN):
                    gate = (t2[:, 0] == min(a, b)) & (t2[:, 1] == max(a, b))
                    pair_rows.append(dict(blend=blend_name, scenario=s, folds=fk, features=kind, train_on=train_on,
                                          pair=pn, n_gated=int(gate.sum()),
                                          gated_true_in_pair=int((gate & ((Y == a) | (Y == b))).sum()),
                                          changed=int((changed & gate).sum()),
                                          fixed=int((changed & gate & (pred == Y)).sum()),
                                          broke=int((changed & gate & (base == Y)).sum())))
                log(f'{blend_name} s{s} {fk} {kind}/{train_on}: delta={r["delta"]:+.4f} '
                    f'changed={r["n_changed"]} fixed={fixed} broke={broke} folds+={r["folds_improved"]} '
                    f'folds-={r["folds_worse"]} C~{r["median_C"]}')

E = pd.DataFrame(exp_rows)
E.to_csv(OUT / 'expert_results.csv', index=False)
pd.DataFrame(fold_rows).to_csv(OUT / 'expert_per_fold.csv', index=False)
pd.DataFrame(pair_rows).to_csv(OUT / 'expert_per_pair.csv', index=False)
(OUT / 'run.log').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
log('done')
