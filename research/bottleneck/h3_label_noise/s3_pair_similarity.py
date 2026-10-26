"""H3 step 3: same-prompt similarity of every model pair in host train (and replication on pool sources
with full 18-model prompt coverage). Shared distinctive n-grams for claude-fable-5 vs glm-5.3.
Outputs: s3_pair_similarity_train.csv, s3_pair_similarity_pool.csv, s3_shared_ngrams_fable_glm53.csv,
         s3_shared_ngrams_pairs.csv, s3_report.txt
"""
import re, itertools, sys
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

OUT = 'research/bottleneck/h3_label_noise/'
FAM = lambda m: m.split('-')[0].replace('qwen3.8', 'qwen')
tok = lambda t: re.findall(r"[a-z0-9']+|[^\sa-z0-9']", t.lower().replace('’', "'"))


def ngrams(toks, n):
    return {' '.join(toks[i:i+n]) for i in range(len(toks) - n + 1)}


def pair_table(df, label):
    """df: prompt_id, model, text (one text per prompt x model). Returns per-pair means."""
    df = df.drop_duplicates(['prompt_id', 'model'])
    models = sorted(df.model.unique())
    chv = TfidfVectorizer(analyzer='char', ngram_range=(5, 5), sublinear_tf=True, min_df=2, lowercase=True)
    X = chv.fit_transform(df.text.values)
    df = df.reset_index(drop=True)
    df['row'] = np.arange(len(df))
    rows = defaultdict(list)
    for pid, g in df.groupby('prompt_id'):
        if g.model.nunique() < 10:
            continue
        T = {m: tok(t) for m, t in zip(g.model, g.text)}
        R = dict(zip(g.model, g.row))
        G1 = {m: set(t) for m, t in T.items()}
        G3 = {m: ngrams(t, 3) for m, t in T.items()}
        G4 = {m: ngrams(t, 4) for m, t in T.items()}
        # n-gram -> number of models in this prompt using it
        c3 = Counter(x for s in G3.values() for x in s)
        c4 = Counter(x for s in G4.values() for x in s)
        ms = sorted(T)
        Xg = X[[R[m] for m in ms]]
        cos = (Xg @ Xg.T).toarray()
        for i, j in itertools.combinations(range(len(ms)), 2):
            a, b = ms[i], ms[j]
            j1 = len(G1[a] & G1[b]) / max(1, len(G1[a] | G1[b]))
            j3 = len(G3[a] & G3[b]) / max(1, len(G3[a] | G3[b]))
            ex3 = sum(1 for x in G3[a] & G3[b] if c3[x] == 2)   # shared by exactly these two answers
            ex4 = sum(1 for x in G4[a] & G4[b] if c4[x] == 2)
            lr = abs(np.log(len(T[a]) / len(T[b])))
            rows[(a, b)].append((j1, j3, cos[i, j], ex3, ex4, lr))
    out = []
    for (a, b), v in rows.items():
        v = np.array(v)
        out.append(dict(a=a, b=b, fam_a=FAM(a), fam_b=FAM(b), cross=FAM(a) != FAM(b), n=len(v),
                        jacc_w1=v[:, 0].mean(), jacc_w3=v[:, 1].mean(), cos_c5=v[:, 2].mean(),
                        excl_3g=v[:, 3].mean(), excl_4g=v[:, 4].mean(), abs_loglen=v[:, 5].mean()))
    t = pd.DataFrame(out)
    for c in ['jacc_w1', 'jacc_w3', 'cos_c5', 'excl_3g', 'excl_4g']:
        t[c + '_rank_cross'] = np.nan
        cr = t.cross
        t.loc[cr, c + '_rank_cross'] = t.loc[cr, c].rank(ascending=False).astype(int)
        t[c + '_z_cross'] = (t[c] - t.loc[cr, c].mean()) / t.loc[cr, c].std()
    t['source'] = label
    return t.sort_values('cos_c5', ascending=False)


def shared_distinctive(tr, A, B, nmax=3, min_df=4):
    """n-grams over-represented in both A and B vs the other 16 models (document frequency)."""
    models = sorted(tr.model.unique())
    cv = CountVectorizer(analyzer='word', ngram_range=(1, nmax), lowercase=True, binary=True,
                         token_pattern=r"(?u)\b\w[\w']*\b|[—–;:!?()\"*]", min_df=min_df)
    X = cv.fit_transform(tr.text.values)
    voc = np.array(cv.get_feature_names_out())
    ndoc = tr.model.value_counts()
    df = {m: np.asarray(X[(tr.model == m).values].sum(0)).ravel() for m in models}
    rest = [m for m in models if m not in (A, B)]
    dfr = np.mean([df[m] / ndoc[m] for m in rest], 0)
    pa, pb = df[A] / ndoc[A], df[B] / ndoc[B]
    sm = 1.0 / 150
    lo_a = np.log((pa + sm) / (dfr + sm)); lo_b = np.log((pb + sm) / (dfr + sm))
    # max rate among the rest (to exclude n-grams owned by a third model)
    mx_rest = np.max([df[m] / ndoc[m] for m in rest], 0)
    s = np.minimum(lo_a, lo_b)
    o = pd.DataFrame(dict(ngram=voc, df_A=df[A], df_B=df[B], rate_A=pa, rate_B=pb, rate_rest_mean=dfr,
                          rate_rest_max=mx_rest, min_logodds=s))
    return o, (lo_a, lo_b, pa, pb, dfr)


if __name__ == '__main__':
    tr = pd.read_csv('data/train.csv')
    rep = []
    t = pair_table(tr[['prompt_id', 'model', 'text']], 'train')
    t.to_csv(OUT + 's3_pair_similarity_train.csv', index=False)
    cross = t[t.cross]
    rep.append(f'TRAIN: {len(t)} pairs, {len(cross)} cross-family. Cross-family mean: ' +
               ', '.join(f'{c}={cross[c].mean():.4f}' for c in ['jacc_w1', 'jacc_w3', 'cos_c5', 'excl_3g', 'excl_4g', 'abs_loglen']))
    within = t[~t.cross]
    rep.append('Within-family mean: ' + ', '.join(f'{c}={within[c].mean():.4f}' for c in ['jacc_w1', 'jacc_w3', 'cos_c5', 'excl_3g', 'excl_4g', 'abs_loglen']))
    focus = [('claude-fable-5', 'glm-5.3'), ('claude-fable-5', 'glm-5.3-flash'), ('claude-opus-5', 'glm-5.3'),
             ('claude-opus-5', 'glm-5.3-flash'), ('claude-sonnet-5', 'glm-5.3'), ('claude-opus-4-8', 'glm-5.3'),
             ('claude-opus-4-6', 'glm-5.3'), ('claude-fable-5', 'claude-opus-5'), ('glm-5.3', 'glm-5.3-flash'),
             ('claude-fable-5', 'kimi-k3'), ('claude-fable-5', 'gpt-5.5')]
    cols = ['a', 'b', 'n', 'jacc_w1', 'jacc_w3', 'cos_c5', 'excl_3g', 'excl_4g', 'abs_loglen',
            'cos_c5_rank_cross', 'cos_c5_z_cross', 'excl_4g_rank_cross', 'excl_4g_z_cross', 'jacc_w3_rank_cross']
    tt = t.set_index(['a', 'b'])
    rep.append('Focus pairs (rank among 126 cross-family pairs, 1 = most similar):')
    for a, b in focus:
        k = (a, b) if (a, b) in tt.index else (b, a)
        r = tt.loc[k]
        rep.append(f'  {a} ~ {b}: jw1={r.jacc_w1:.4f} jw3={r.jacc_w3:.4f} cos5={r.cos_c5:.4f} ex3={r.excl_3g:.3f} '
                   f'ex4={r.excl_4g:.3f} |dloglen|={r.abs_loglen:.3f} | cross-rank cos5={r.cos_c5_rank_cross} '
                   f'ex4={r.excl_4g_rank_cross} jw3={r.jacc_w3_rank_cross} z(cos5)={r.cos_c5_z_cross:+.2f}')
    rep.append('Top 12 cross-family pairs by cos_c5:')
    for _, r in cross.sort_values('cos_c5', ascending=False).head(12).iterrows():
        rep.append(f'  {r.a} ~ {r.b}: cos5={r.cos_c5:.4f} ex4={r.excl_4g:.3f} jw3={r.jacc_w3:.4f}')
    rep.append('Top 12 cross-family pairs by excl_4g (4-grams shared by exactly this pair on the same prompt):')
    for _, r in cross.sort_values('excl_4g', ascending=False).head(12).iterrows():
        rep.append(f'  {r.a} ~ {r.b}: ex4={r.excl_4g:.3f} ex3={r.excl_3g:.3f} cos5={r.cos_c5:.4f}')

    # per-model length
    tr['nw'] = tr.text.str.split().str.len()
    rep.append('Mean words: ' + ', '.join(f'{m}={v:.1f}' for m, v in tr.groupby('model').nw.mean().items()
                                          if m in ('claude-fable-5', 'glm-5.3', 'glm-5.3-flash', 'claude-opus-5')) +
               f' (all={tr.nw.mean():.1f})')

    # shared distinctive n-grams
    o, _ = shared_distinctive(tr, 'claude-fable-5', 'glm-5.3')
    o = o[(o.df_A >= 4) & (o.df_B >= 4)].sort_values('min_logodds', ascending=False)
    o.head(200).to_csv(OUT + 's3_shared_ngrams_fable_glm53.csv', index=False)
    rep.append('Top shared distinctive n-grams fable & glm-5.3 (rate_A, rate_B, rest mean, rest max):')
    for _, r in o.head(25).iterrows():
        rep.append(f'  "{r.ngram}": {r.rate_A:.3f} {r.rate_B:.3f} vs {r.rate_rest_mean:.3f} (max {r.rate_rest_max:.3f})')

    # count of jointly over-represented n-grams per pair (both >= 2x rest, df>=6 each)
    models = sorted(tr.model.unique())
    cv = CountVectorizer(analyzer='word', ngram_range=(1, 3), lowercase=True, binary=True,
                         token_pattern=r"(?u)\b\w[\w']*\b|[—–;:!?()\"*]", min_df=6)
    X = cv.fit_transform(tr.text.values)
    nd = tr.model.value_counts()
    R = np.vstack([np.asarray(X[(tr.model == m).values].sum(0)).ravel() / nd[m] for m in models])
    D = np.vstack([np.asarray(X[(tr.model == m).values].sum(0)).ravel() for m in models])
    pr = []
    for i, j in itertools.combinations(range(len(models)), 2):
        rest = [k for k in range(len(models)) if k not in (i, j)]
        rr = R[rest].mean(0) + 1 / 150
        ok = (R[i] >= 2 * rr) & (R[j] >= 2 * rr) & (D[i] >= 6) & (D[j] >= 6)
        pr.append(dict(a=models[i], b=models[j], cross=FAM(models[i]) != FAM(models[j]), n_joint_overrep=int(ok.sum())))
    pr = pd.DataFrame(pr).sort_values('n_joint_overrep', ascending=False)
    pc = pr[pr.cross].reset_index(drop=True)
    pc['rank_cross'] = np.arange(1, len(pc) + 1)
    pr = pr.merge(pc[['a', 'b', 'rank_cross']], on=['a', 'b'], how='left')
    pr.to_csv(OUT + 's3_shared_ngrams_pairs.csv', index=False)
    rep.append(f'Jointly over-represented 1-3grams (both >=2x rest mean): cross-family median={pc.n_joint_overrep.median():.0f}, '
               f'max={pc.n_joint_overrep.max()}; top cross pairs:')
    for _, r in pc.head(10).iterrows():
        rep.append(f'  {r.a} ~ {r.b}: {r.n_joint_overrep}')
    for a, b in focus[:4]:
        r = pr[((pr.a == a) & (pr.b == b)) | ((pr.a == b) & (pr.b == a))].iloc[0]
        rep.append(f'  focus {a} ~ {b}: {r.n_joint_overrep} (cross rank {r.rank_cross})')

    # replication on pool sources with many models per prompt (independent generations)
    if len(sys.argv) > 1 and sys.argv[1] == 'pool':
        pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet')
        res = []
        for src in ['drjcfmd', 'lxd', 'prvsiyan', 'arxitekton']:
            p = pool[pool.source == src][['prompt_id', 'model', 'text']].copy()
            p['prompt_id'] = src + '_' + p.prompt_id.astype(str)
            cnt = p.groupby('prompt_id').model.nunique()
            keep = cnt[cnt >= (10 if src == 'lxd' else 14)].index
            p = p[p.prompt_id.isin(keep)]
            if p.prompt_id.nunique() > 400:
                rs = np.random.RandomState(0)
                p = p[p.prompt_id.isin(rs.choice(p.prompt_id.unique(), 400, replace=False))]
            if len(p) == 0:
                continue
            tp = pair_table(p, src)
            res.append(tp)
            tps = tp.set_index(['a', 'b'])
            rep.append(f'POOL {src}: prompts={p.prompt_id.nunique()} rows={len(p)}')
            for a, b in focus[:5] + [('claude-fable-5', 'claude-opus-5')]:
                k = (a, b) if (a, b) in tps.index else (b, a)
                if k in tps.index:
                    r = tps.loc[k]
                    rep.append(f'  {a} ~ {b}: cos5={r.cos_c5:.4f} rank_cross={r.cos_c5_rank_cross} '
                               f'z={r.cos_c5_z_cross:+.2f} ex4={r.excl_4g:.3f} rank_ex4={r.excl_4g_rank_cross}')
        pd.concat(res).to_csv(OUT + 's3_pair_similarity_pool.csv', index=False)
    open(OUT + 's3_report.txt', 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
