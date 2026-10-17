"""Does a small within-set specialist (hand stylometry + char/word TF-IDF) improve the strong OOF blend?
Specialist redistributes probability mass only INSIDE its class set S (total mass of S kept from the base):
    L'[S] = logsumexp(L[S]) + log_softmax((1 - w) * L[S] + w * log p_spec)
Evaluated with prompt-disjoint folds (same GroupKFold(5) for specialist; base = public prompt-disjoint OOF)."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from scipy.special import logsumexp, log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from stylo_feats import feats

ROOT = pathlib.Path(__file__).resolve().parents[2]
tr = pd.read_csv(ROOT / 'data/train.csv')
LABELS = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
g = tr.prompt_id.values

zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
assert (zz['ids'] == tr.id.values).all()
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
L = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
t = minimize_scalar(lambda t: -log_softmax(L * t, 1)[np.arange(len(y)), y].mean(), bounds=(0.1, 20), method='bounded').x
L = log_softmax(L * t, 1)
base = bacc(y, L.argmax(1))
print(f'base strong OOF bacc {base:.4f}')

SETS = {'glm_vs_opus5': ['glm-5.3', 'glm-5.3-flash', 'claude-opus-5'],
        'glm_pair': ['glm-5.3', 'glm-5.3-flash'],
        'gemini_flash3': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
        'google4': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
        'gpt56_3': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
        'openai5': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],
        'anthropic5': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],
        'fable_vs_opus5': ['claude-fable-5', 'claude-opus-5'],
        'kimi_vs_gpt55': ['kimi-k3', 'gpt-5.5']}
only = sys.argv[1].split(',') if len(sys.argv) > 1 else list(SETS)

H = np.array([list(feats(x).values()) for x in tr.text], dtype=np.float64)
folds = list(GroupKFold(5).split(tr, y, g))


def spec_oof(cls_idx, C_lr, use_tfidf):
    """OOF log-probs over the set (n x |S|) for ALL rows; specialist trained only on rows of S in the train folds."""
    out = np.zeros((len(tr), len(cls_idx)))
    for a, b in folds:
        a = a[np.isin(y[a], cls_idx)]
        sc = StandardScaler().fit(H[a])
        parts_a, parts_b = [csr_matrix(sc.transform(H[a]) * 0.3)], [csr_matrix(sc.transform(H[b]) * 0.3)]
        if use_tfidf:
            for an, rng, mf in (('char', (2, 5), 120000), ('word', (1, 2), 60000)):
                v = TfidfVectorizer(analyzer=an, ngram_range=rng, sublinear_tf=True, min_df=2, max_features=mf,
                                    lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
                parts_a.append(v.fit_transform(tr.text.values[a]))
                parts_b.append(v.transform(tr.text.values[b]))
        clf = LogisticRegression(C=C_lr, max_iter=5000).fit(hstack(parts_a, format='csr'), y[a])
        out[b] = np.log(np.clip(clf.predict_proba(hstack(parts_b, format='csr')), 1e-9, 1))
        assert list(clf.classes_) == list(cls_idx)
    return out


def combine(L, S, ls, wt):
    L2 = L.copy()
    sub = L[:, S]
    L2[:, S] = logsumexp(sub, 1, keepdims=True) + log_softmax((1 - wt) * sub + wt * ls, 1)
    return L2


rows = []
for name in only:
    S = np.array(sorted(LABELS.index(m) for m in SETS[name]))
    in_s = np.isin(y, S)
    for use_tfidf, C_lr in ((False, 0.3), (True, 3.0), (True, 10.0)):
        ls = spec_oof(S, C_lr, use_tfidf)
        within_base = bacc(y[in_s], S[L[in_s][:, S].argmax(1)])
        within_spec = bacc(y[in_s], S[ls[in_s].argmax(1)])
        res = {wt: bacc(y, combine(L, S, ls, wt).argmax(1)) - base for wt in (0.1, 0.2, 0.3, 0.5)}
        # fold stability of the best weight: per-fold delta (prompt-disjoint folds)
        bw = max(res, key=res.get)
        L2 = combine(L, S, ls, bw)
        fd = [bacc(y[b], L2[b].argmax(1)) - bacc(y[b], L[b].argmax(1)) for _, b in folds]
        rows.append(dict(set=name, feats='hand+tfidf' if use_tfidf else 'hand', C=C_lr, within_base=round(within_base, 3),
                         within_spec=round(within_spec, 3), **{f'd@w{k}': round(v, 4) for k, v in res.items()},
                         best_w=bw, folds_pos=f'{sum(d > 0 for d in fd)}/5', fold_min=round(min(fd), 4)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
(ROOT / 'experiments/lexical/out').mkdir(exist_ok=True)
df.to_csv(ROOT / 'experiments/lexical/out/specialist.csv', index=False)
print(df.to_string())
