# E072 layer-by-layer screen (CPU part, Kaggle CPU). Input: kernel output of mdmuqtadirfuad/ntl-layer-screen
# (feat_<name>.npy = (n_layers+1, n_rows, hidden) fp16 mean-pooled hidden states, rows in train.csv order).
# Per model and layer: StandardScaler + PCA-256 + LogisticRegression(C=0.5) (the E047 probe), fit on 4 folds of the
# seeded split used by the honest-OOF harness (RandomState(0) shuffle of sorted prompt ids, fold = i % 5), scored on the 5th.
# Output: oof_<name>.npy (n_layers+1, n_rows, 18) float16 log-probs, layer_scores.csv (balanced accuracy per layer/fold).
import glob, json, os, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.preprocessing import StandardScaler

T0 = time.time()
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
train = pd.read_csv(os.path.join(DATA, 'train.csv'))
labels = sorted(train.model.unique()); y = train.model.map({l: i for i, l in enumerate(labels)}).to_numpy()
rng = np.random.RandomState(0); pids = np.array(sorted(train.prompt_id.unique())); rng.shuffle(pids)
fold = train.prompt_id.map({q: i % 5 for i, q in enumerate(pids)}).to_numpy()
ids = np.load(glob.glob('/kaggle/input/**/ids.npy', recursive=True)[0], allow_pickle=True).astype(str)
assert (ids == train.id.astype(str).to_numpy()).all()
rows = []
for f in sorted(glob.glob('/kaggle/input/**/feat_*.npy', recursive=True)):
    name = os.path.basename(f)[5:-4]
    F = np.load(f, mmap_mode='r')
    O = np.zeros((F.shape[0], len(y), 18), np.float16)
    for l in range(F.shape[0]):
        X = np.asarray(F[l], dtype=np.float32)
        for k in range(5):
            a, b = fold != k, fold == k
            sc = StandardScaler().fit(X[a]); p = PCA(256, random_state=0).fit(sc.transform(X[a]))
            Xa, Xb = p.transform(sc.transform(X[a])), p.transform(sc.transform(X[b]))
            O[l, b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xa, y[a]).predict_log_proba(Xb)
            rows.append(dict(model=name, layer=l, rel_depth=round(l / (F.shape[0] - 1), 3), fold=k, bacc=bas(y[b], O[l, b].argmax(1))))
        print(f'[{time.time() - T0:6.0f}s] {name} layer {l}/{F.shape[0] - 1}: {bas(y, O[l].argmax(1)):.4f}', flush=True)
    np.save(f'/kaggle/working/oof_{name}.npy', O)
    pd.DataFrame(rows).to_csv('/kaggle/working/layer_scores.csv', index=False)
print('done', flush=True)
