"""H3 step 3c: pairwise separability (host-only, OOF). For each pair (a,b): accuracy of sign(score_a - score_b) on
host rows of a and b, from s1 CV OOF (TF-IDF+SVC) and from our blend OOF (reference mean). Output: s3c_pairwise_separability.csv"""
import itertools, numpy as np, pandas as pd
OUT = 'research/bottleneck/h3_label_noise/'
z = np.load(OUT + 's1_cv_oof.npz', allow_pickle=True); cl = list(z['classes']); y = z['y']; S = z['scores']
zb = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)
tr = pd.read_csv('data/train.csv').set_index('id').loc[zb['ids']]
yb = tr.model.map({c: i for i, c in enumerate(cl)}).values; B = zb['reference'].mean(0)
fam = lambda m: m.split('-')[0]
rows = []
for i, j in itertools.combinations(range(18), 2):
    r = dict(a=cl[i], b=cl[j], cross=fam(cl[i]) != fam(cl[j]))
    for nm, sc, yy in [('svc', S, y), ('blend', B, yb)]:
        m = (yy == i) | (yy == j)
        d = sc[m, i] - sc[m, j]
        r[f'acc_{nm}'] = ((d > 0) == (yy[m] == i)).mean()
    rows.append(r)
t = pd.DataFrame(rows)
for nm in ['svc', 'blend']:
    t[f'rank_cross_{nm}'] = t[t.cross][f'acc_{nm}'].rank(method='min')  # 1 = least separable
t.sort_values('acc_blend').to_csv(OUT + 's3c_pairwise_separability.csv', index=False, float_format='%.4f')
c = t[t.cross]
print('cross-family pairwise acc: svc median %.3f, blend median %.3f' % (c.acc_svc.median(), c.acc_blend.median()))
print('within-family pairwise acc: svc median %.3f, blend median %.3f' % (t[~t.cross].acc_svc.median(), t[~t.cross].acc_blend.median()))
print('least separable cross-family pairs (blend):')
print(c.sort_values('acc_blend').head(10)[['a', 'b', 'acc_svc', 'acc_blend', 'rank_cross_svc', 'rank_cross_blend']].round(3).to_string(index=False))
print(t[(t.a == 'claude-fable-5') & t.b.str.startswith(('claude', 'glm'))][['a', 'b', 'acc_svc', 'acc_blend']].round(3).to_string(index=False))
