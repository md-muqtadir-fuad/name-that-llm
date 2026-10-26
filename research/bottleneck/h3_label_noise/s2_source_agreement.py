"""H3 step 2: per pool source x label agreement with the host-train-only classifier (s1).
Outputs: s2_source_label_agreement.csv, s2_focus_setting_agreement.csv, s2_report.txt
"""
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

OUT = 'research/bottleneck/h3_label_noise/'
FAM = lambda m: m.split('-')[0]
FOCUS = ['claude-fable-5', 'glm-5.3', 'glm-5.3-flash']

pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet', columns=['id', 'source', 'model', 'setting', 'prompt_id'])
meta = pd.read_csv(OUT + 's1_pool_meta.csv')
z = np.load(OUT + 's1_pool_scores.npz', allow_pickle=True)
classes = list(z['classes']); S = z['scores']
assert (z['ids'] == pool.id.values).all() and (meta.id.values == pool.id.values).all()
cix = {c: i for i, c in enumerate(classes)}
yl = pool.model.map(cix).values
rank = (S > S[np.arange(len(S)), yl][:, None]).sum(1)          # 0 = top-1
pred = S.argmax(1)
pool['pred'] = np.array(classes)[pred]
pool['top1'] = rank == 0
pool['top3'] = rank < 3
pool['fam_ok'] = [FAM(a) == FAM(b) for a, b in zip(pool.pred, pool.model)]
pool['dup'] = meta.dup_train_text.values
pool['trp'] = meta.train_prompt.values

# host CV reference (clean labels)
cv = np.load(OUT + 's1_cv_oof.npz', allow_pickle=True)
oy, osc = cv['y'], cv['scores']
orank = (osc > osc[np.arange(len(oy)), oy][:, None]).sum(1)
host = pd.DataFrame({'model': np.array(classes)[oy], 'top1': orank == 0, 'top3': orank < 3}).groupby('model').mean()
Rhost = confusion_matrix(oy, osc.argmax(1), labels=range(18), normalize='true')  # rows = true label signature

g = pool.groupby(['source', 'model'])
t = g.agg(n=('top1', 'size'), top1=('top1', 'mean'), top3=('top3', 'mean'), fam=('fam_ok', 'mean'),
          n_dup_train=('dup', 'sum'), n_train_prompt=('trp', 'sum')).reset_index()
t['top_pred'] = g.pred.agg(lambda s: s.value_counts().index[0]).values
t['top_pred_share'] = g.pred.agg(lambda s: s.value_counts(normalize=True).iloc[0]).values
# leave-one-source-out label norm (n-weighted over other sources)
tot = pool.groupby('model').agg(N=('top1', 'size'), T1=('top1', 'sum'), T3=('top3', 'sum'))
t = t.merge(tot, left_on='model', right_index=True)
t['norm1'] = (t.T1 - t.top1 * t.n) / (t.N - t.n)
t['norm3'] = (t.T3 - t.top3 * t.n) / (t.N - t.n)
t['ratio1'] = t.top1 / t.norm1
t['ratio3'] = t.top3 / t.norm3
t['z1'] = (t.top1 - t.norm1) / np.sqrt(t.norm1 * (1 - t.norm1) / t.n)
t['host_cv_top1'] = t.model.map(host.top1); t['host_cv_top3'] = t.model.map(host.top3)
# source-level factor (median ratio over that source's labels) -> label-specific residual
sf = t.groupby('source').ratio3.median().rename('src_med_ratio3')
t = t.merge(sf, left_on='source', right_index=True)
t['resid3'] = t.ratio3 / t.src_med_ratio3
# confusion signature: which host label's OOF confusion row is closest (JS divergence) to this group's prediction distribution
def js(p, q):
    p = p + 1e-6; q = q + 1e-6; p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return 0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum()
look, jsl, jsb = [], [], []
for _, r in t.iterrows():
    p = np.bincount(pred[(pool.source == r.source).values & (pool.model == r.model).values], minlength=18).astype(float)
    p /= p.sum()
    d = np.array([js(p, Rhost[k]) for k in range(18)])
    look.append(classes[d.argmin()]); jsb.append(d.min()); jsl.append(d[cix[r.model]])
t['looks_like_host'] = look; t['js_own'] = jsl; t['js_best'] = jsb
t['flag'] = ((t.n >= 30) & (((t.ratio3 < 0.6) & (t.z1 < -3)) | ((t.looks_like_host != t.model) & (t.js_own - t.js_best > 0.05))))
t['flag_label_specific'] = t.flag & (t.resid3 < 0.7)
cols = ['source', 'model', 'n', 'top1', 'top3', 'fam', 'norm1', 'norm3', 'ratio1', 'ratio3', 'z1', 'src_med_ratio3',
        'resid3', 'top_pred', 'top_pred_share', 'looks_like_host', 'js_own', 'js_best', 'host_cv_top1', 'host_cv_top3',
        'n_dup_train', 'n_train_prompt', 'flag', 'flag_label_specific']
t = t[cols].sort_values(['model', 'ratio3'])
t.to_csv(OUT + 's2_source_label_agreement.csv', index=False, float_format='%.4f')

rep = []
rep.append(f'Pool rows={len(pool)}; exact dup with train text={int(pool.dup.sum())}; rows on a train prompt_id={int(pool.trp.sum())}')
rep.append('Overall pool agreement with host-trained clf: top1=%.3f top3=%.3f family=%.3f (host CV top1 mean=%.3f)' %
           (pool.top1.mean(), pool.top3.mean(), pool.fam_ok.mean(), host.top1.mean()))
lab = pool.groupby('model')[['top1', 'top3', 'fam_ok']].mean().join(host, rsuffix='_hostcv')
rep.append('Per label pool agreement vs host CV:\n' + lab.round(3).to_string())
rep.append('Per source agreement (all labels):\n' + pool.groupby('source')[['top1', 'top3', 'fam_ok']].mean().join(sf).round(3).sort_values('top3').to_string())
pd.set_option('display.width', 250)
show = ['source', 'model', 'n', 'top1', 'top3', 'fam', 'norm1', 'norm3', 'ratio3', 'z1', 'resid3', 'top_pred', 'looks_like_host']
rep.append('FOCUS labels by source:\n' + t[t.model.isin(FOCUS)][show].round(3).to_string(index=False))
rep.append('FLAGGED (n>=30; ratio3<0.6 & z1<-3, or signature closer to another host label):\n' +
           t[t.flag][show + ['flag_label_specific']].round(3).to_string(index=False))

# focus label x source x setting
f = pool[pool.model.isin(FOCUS)].groupby(['model', 'source', 'setting']).agg(
    n=('top1', 'size'), top1=('top1', 'mean'), top3=('top3', 'mean'), fam=('fam_ok', 'mean'),
    top_pred=('pred', lambda s: s.value_counts().index[0])).reset_index()
f = f[f.n >= 20]
f.to_csv(OUT + 's2_focus_setting_agreement.csv', index=False, float_format='%.4f')
rep.append('Focus label x source x setting (n>=20):\n' + f.round(3).to_string(index=False))

# where do pool fable / glm passages go, by source
for m in FOCUS:
    q = pool[pool.model == m]
    ct = pd.crosstab(q.source, q.pred, normalize='index')
    keep = ct.mean().sort_values(ascending=False).index[:7]
    rep.append(f'Prediction distribution of pool "{m}" by source (top columns):\n' + ct[keep].round(2).to_string())
rep.append('Host CV confusion rows (this clf):')
for m in FOCUS + ['claude-opus-5']:
    i = cix[m]; o = np.argsort(-Rhost[i])[:5]
    rep.append(f'  {m}: ' + ', '.join(f'{classes[j]}={Rhost[i, j]:.2f}' for j in o))

# blend OOF (our model) confusion for the focus labels
zb = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)
tr = pd.read_csv('data/train.csv').set_index('id').loc[zb['ids']]
yb = tr.model.map(cix).values
pb = zb['reference'].mean(0).argmax(1)
Rb = confusion_matrix(yb, pb, labels=range(18), normalize='true')
rep.append('Blend OOF (reference, mean of 3) confusion rows:')
for m in FOCUS + ['claude-opus-5', 'claude-sonnet-5', 'claude-opus-4-8']:
    i = cix[m]; o = np.argsort(-Rb[i])[:5]
    rep.append(f'  {m}: ' + ', '.join(f'{classes[j]}={Rb[i, j]:.2f}' for j in o))
rep.append('Blend: P(pred glm-5.3 | fable)=%.3f  P(pred fable | glm-5.3)=%.3f  P(pred glm* | fable)=%.3f  P(pred claude* | glm-5.3)=%.3f' %
           (Rb[cix['claude-fable-5'], cix['glm-5.3']], Rb[cix['glm-5.3'], cix['claude-fable-5']],
            Rb[cix['claude-fable-5'], [cix['glm-5.3'], cix['glm-5.3-flash']]].sum(),
            Rb[cix['glm-5.3'], [i for i, c in enumerate(classes) if c.startswith('claude')]].sum()))
open(OUT + 's2_report.txt', 'w', encoding='utf-8').write('\n\n'.join(rep) + '\n')
print('\n\n'.join(rep))
