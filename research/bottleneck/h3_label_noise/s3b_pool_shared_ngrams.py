"""H3 step 3b: shared distinctive 1-4grams of claude-fable-5 & glm-5.3 vs the other 16 labels, on pool sources
that contain both labels (arxitekton, drjcfmd, lxd, prvsiyan); rates = doc frequency per label. Output: s3b_pool_shared_ngrams.csv"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
OUT = 'research/bottleneck/h3_label_noise/'
p = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet', columns=['source', 'model', 'text'])
p = p[p.source.isin(['arxitekton', 'drjcfmd', 'lxd', 'prvsiyan'])]
cv = CountVectorizer(ngram_range=(1, 4), binary=True, lowercase=True, min_df=20,
                     token_pattern=r"(?u)\b\w[\w']*\b|[—–;:!?()\"*]")
X = cv.fit_transform(p.text.values).tocsc()
voc = cv.get_feature_names_out()
models = sorted(p.model.unique())
R = np.vstack([np.asarray(X[(p.model == m).values].mean(0)).ravel() for m in models])
A, B = models.index('claude-fable-5'), models.index('glm-5.3')
rest = [i for i in range(len(models)) if i not in (A, B)]
cl = [i for i in rest if models[i].startswith('claude')]
nonclg = [i for i in rest if not models[i].startswith(('claude', 'glm'))]
sm = 1e-3
o = pd.DataFrame(dict(ngram=voc, rate_fable=R[A], rate_glm53=R[B], rest_mean=R[rest].mean(0), rest_max=R[rest].max(0),
                      other_claude_mean=R[cl].mean(0), non_claude_glm_mean=R[nonclg].mean(0)))
o['min_lo_vs_rest'] = np.minimum(np.log((o.rate_fable + sm) / (o.rest_mean + sm)), np.log((o.rate_glm53 + sm) / (o.rest_mean + sm)))
o['excl'] = o.rest_max < o[['rate_fable', 'rate_glm53']].min(axis=1)
o = o.sort_values('min_lo_vs_rest', ascending=False)
o.head(300).to_csv(OUT + 's3b_pool_shared_ngrams.csv', index=False, float_format='%.4f')
x = o[o.excl & (o[['rate_fable', 'rate_glm53']].min(axis=1) >= 0.005)].head(25).copy()
x['ngram'] = x.ngram.str.replace('—', '<EMDASH>')
print('docs', len(p), 'vocab', len(voc), 'n exclusive shared (both > every other label, min rate>=0.5%):',
      int((o.excl & (o[['rate_fable', 'rate_glm53']].min(axis=1) >= 0.005)).sum()))
print(x[['ngram', 'rate_fable', 'rate_glm53', 'rest_mean', 'rest_max', 'other_claude_mean', 'non_claude_glm_mean']].round(4).to_string(index=False))
# compare: how many n-grams are exclusive-shared for other cross-family pairs (same criterion)
cnt = []
mr = R.max(0)
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        if models[i].split('-')[0] == models[j].split('-')[0]:
            continue
        rr = np.delete(R, [i, j], 0).max(0)
        cnt.append((models[i], models[j], int(((np.minimum(R[i], R[j]) > rr) & (np.minimum(R[i], R[j]) >= 0.005)).sum())))
c = pd.DataFrame(cnt, columns=['a', 'b', 'n_excl_shared']).sort_values('n_excl_shared', ascending=False)
c.to_csv(OUT + 's3b_pool_excl_shared_counts.csv', index=False)
print('cross-family pairs: median', c.n_excl_shared.median(), '\n', c.head(10).to_string(index=False))
