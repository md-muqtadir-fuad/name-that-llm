"""H3 step 3d: dedicated binary classifiers (host train only, char1-5+word1-2 TF-IDF + LinearSVC,
5-fold GroupKFold by prompt) for selected pairs. Output: s3d_binary_pairs.csv"""
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
OUT = 'research/bottleneck/h3_label_noise/'
tr = pd.read_csv('data/train.csv')
pairs = [('claude-fable-5', 'glm-5.3'), ('claude-fable-5', 'glm-5.3-flash'), ('glm-5.3', 'glm-5.3-flash'),
         ('claude-fable-5', 'claude-opus-4-8'), ('claude-fable-5', 'claude-opus-5'), ('claude-fable-5', 'claude-sonnet-5'),
         ('claude-opus-5', 'glm-5.3'), ('claude-sonnet-5', 'glm-5.3'), ('claude-opus-4-8', 'glm-5.3'),
         ('claude-fable-5', 'kimi-k3'), ('gpt-5.5', 'qwen3.8-max'), ('gpt-5.5', 'kimi-k3'),
         ('gemini-3.7-flash', 'gemini-3.8-flash'), ('gpt-5.6-sol', 'gpt-5.6-terra')]
rows = []
for a, b in pairs:
    d = tr[tr.model.isin([a, b])].reset_index(drop=True); y = (d.model == a).values.astype(int)
    sc = np.zeros(len(d))
    for tri, tei in GroupKFold(5).split(d, y, d.prompt_id):
        ch = TfidfVectorizer(analyzer='char', ngram_range=(1, 5), lowercase=False, sublinear_tf=True, min_df=2)
        wd = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
        Xa = hstack([ch.fit_transform(d.text[tri]), wd.fit_transform(d.text[tri])]).tocsr()
        Xb = hstack([ch.transform(d.text[tei]), wd.transform(d.text[tei])]).tocsr()
        sc[tei] = LinearSVC(C=0.3).fit(Xa, y[tri]).decision_function(Xb)
    rows.append(dict(a=a, b=b, acc=((sc > 0) == y).mean(), auc=roc_auc_score(y, sc)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT + 's3d_binary_pairs.csv', index=False, float_format='%.4f')
