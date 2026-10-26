"""H3 step 4: effect proxy of dropping flagged (source,label) groups / sources.
Train a pool-only TF-IDF+LinearSVC on a capped subsample of pool_v4 (<= CAP rows per source x label),
evaluate on the full host train (pool and train are disjoint generations; exact dups removed).
Variants given as "name:source/label,source/label|..." on argv; '*' as label = whole source.
Outputs: s4_drop_effect.csv
"""
import sys, time, os
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, recall_score

OUT = 'research/bottleneck/h3_label_noise/'
CAP = int(sys.argv[1])
variants = [('all', [])]
for v in sys.argv[2:]:
    name, spec = v.split(':', 1)
    variants.append((name, [tuple(x.split('/')) for x in spec.split(',')]))
t0 = time.time()
tr = pd.read_csv('data/train.csv')
pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet', columns=['id', 'text', 'model', 'source'])
meta = pd.read_csv(OUT + 's1_pool_meta.csv')
pool = pool[~meta.dup_train_text.values]
pool = pool.sample(frac=1, random_state=0); pool = pool[pool.groupby(['source', 'model']).cumcount() < CAP]
pool = pool.reset_index(drop=True)
print('subsample', len(pool), flush=True)
classes = sorted(tr.model.unique()); cix = {c: i for i, c in enumerate(classes)}
ch = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), lowercase=False, sublinear_tf=True, min_df=3,
                     max_features=300000, dtype=np.float32)
wd = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=3,
                     token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)
Xp = hstack([ch.fit_transform(pool.text.values), wd.fit_transform(pool.text.values)]).tocsr()
Xt = hstack([ch.transform(tr.text.values), wd.transform(tr.text.values)]).tocsr()
print(f'features {time.time()-t0:.0f}s', flush=True)
yp = pool.model.map(cix).values; yt = tr.model.map(cix).values
rows = []
for name, drops in variants:
    keep = np.ones(len(pool), bool)
    for s, l in drops:
        keep &= ~((pool.source.values == s) & ((pool.model.values == l) | (l == '*')))
    m = LinearSVC(C=0.3, class_weight='balanced', max_iter=5000).fit(Xp[keep], yp[keep])
    p = m.decision_function(Xt).argmax(1)
    rec = recall_score(yt, p, labels=range(18), average=None)
    r = dict(variant=name, n_train_rows=int(keep.sum()), host_BA=balanced_accuracy_score(yt, p))
    r.update({f'rec_{c}': rec[cix[c]] for c in (classes if os.environ.get('S4_ALLREC') else ['claude-fable-5', 'glm-5.3', 'glm-5.3-flash', 'claude-opus-5'])})
    r['fable_to_glm53'] = ((yt == cix['claude-fable-5']) & (p == cix['glm-5.3'])).sum() / (yt == cix['claude-fable-5']).sum()
    rows.append(r)
    print(r, f'{time.time()-t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(OUT + os.environ.get('S4_OUT', 's4_drop_effect.csv'), index=False, float_format='%.4f')
