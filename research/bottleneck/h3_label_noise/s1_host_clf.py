"""H3 step 1: host-train-only TF-IDF + LinearSVC; 5-fold GroupKFold(prompt_id) BA; predict pool_v4.
Outputs (this folder): s1_cv_oof.npz, s1_pool_scores.npz, s1_cv_report.txt, s1_pool_meta.csv
"""
import time, hashlib, re, sys
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

OUT = 'research/bottleneck/h3_label_noise/'
t0 = time.time()
tr = pd.read_csv('data/train.csv')
pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet')
classes = sorted(tr.model.unique())
cix = {c: i for i, c in enumerate(classes)}
y = tr.model.map(cix).values


def feats():
    ch = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), lowercase=False, sublinear_tf=True,
                         min_df=2, max_features=400000, dtype=np.float32)
    wd = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True, sublinear_tf=True,
                         min_df=2, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)
    return ch, wd


def fit_tf(texts):
    ch, wd = feats()
    X = hstack([ch.fit_transform(texts), wd.fit_transform(texts)]).tocsr()
    return (ch, wd), X


def tf(vecs, texts):
    return hstack([vecs[0].transform(texts), vecs[1].transform(texts)]).tocsr()


C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
oof = np.zeros((len(tr), len(classes)), np.float32)
gkf = GroupKFold(n_splits=5)
for f, (a, b) in enumerate(gkf.split(tr, y, tr.prompt_id)):
    vecs, Xa = fit_tf(tr.text.values[a])
    Xb = tf(vecs, tr.text.values[b])
    m = LinearSVC(C=C, class_weight='balanced', max_iter=5000).fit(Xa, y[a])
    oof[b] = m.decision_function(Xb)
    print(f'fold {f} done {time.time()-t0:.0f}s', flush=True)
pred = oof.argmax(1)
ba = balanced_accuracy_score(y, pred)
rec = confusion_matrix(y, pred, normalize='true')
lines = [f'C={C} 5-fold GroupKFold(prompt_id) BA={ba:.4f}', 'per-class recall:']
for i, c in enumerate(classes):
    top = np.argsort(-rec[i])[:4]
    lines.append(f'  {c:24s} {rec[i,i]:.3f}  top: ' + ', '.join(f'{classes[j]}={rec[i,j]:.2f}' for j in top))
print('\n'.join(lines))
open(OUT + 's1_cv_report.txt', 'w').write('\n'.join(lines) + '\n')
np.savez(OUT + 's1_cv_oof.npz', ids=tr.id.values, classes=np.array(classes), scores=oof, y=y)

# full fit, predict pool
vecs, X = fit_tf(tr.text.values)
m = LinearSVC(C=C, class_weight='balanced', max_iter=5000).fit(X, y)
print(f'full fit {time.time()-t0:.0f}s', flush=True)
S = np.zeros((len(pool), len(classes)), np.float32)
bs = 10000
for s in range(0, len(pool), bs):
    S[s:s+bs] = m.decision_function(tf(vecs, pool.text.values[s:s+bs]))
    print(f'pool {s+bs} {time.time()-t0:.0f}s', flush=True)
np.savez(OUT + 's1_pool_scores.npz', ids=pool.id.values, classes=np.array(classes), scores=S)

# overlap diagnostics: exact text dup with train, prompt_id overlap with train
norm = lambda t: hashlib.md5(re.sub(r'\s+', ' ', t.strip().lower()).encode()).hexdigest()
trh = set(tr.text.map(norm))
meta = pool[['id', 'source', 'model', 'prompt_id', 'n_words']].copy()
meta['dup_train_text'] = pool.text.map(norm).isin(trh)
meta['train_prompt'] = pool.prompt_id.astype(str).isin(set(tr.prompt_id.astype(str)))
meta.to_csv(OUT + 's1_pool_meta.csv', index=False)
print('dup with train text:', meta.dup_train_text.sum(), ' prompt_id overlap:', meta.train_prompt.sum())
print(meta.groupby('source')[['dup_train_text', 'train_prompt']].sum().query('dup_train_text>0 or train_prompt>0'))
print(f'done {time.time()-t0:.0f}s')
