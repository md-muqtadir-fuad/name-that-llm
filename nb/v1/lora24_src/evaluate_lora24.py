"""Fixed-mixture evaluation; no readout/ensemble fitting or outer-label tuning."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
assert Path('/kaggle/input').is_dir()
root=Path('/kaggle/working');inputs=Path('/kaggle/input')
spec=json.loads((root/'training_spec.json').read_text());plan=spec['plan']
train_path=[p for p in inputs.rglob('train.csv') if (p.parent/'test.csv').exists()]
assert len(train_path)==1 and hashlib.sha256(train_path[0].read_bytes()).hexdigest()==plan['train_sha256']
train=pd.read_csv(train_path[0],dtype=str);labels=np.asarray(plan['labels'],dtype=str)
y=np.searchsorted(labels,train.model.to_numpy());groups=train.prompt_id.to_numpy(dtype=str)
texts=train.text.to_numpy(dtype=object)
reference={}
for name,digest in spec['reference_hashes'].items():
    matches=list(inputs.rglob(name));assert len(matches)==1
    path=matches[0];assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    reference['report' if name.endswith('.json') else 'oof']=path
baseline_report=json.loads(reference['report'].read_text())
assert baseline_report['train_sha256']==plan['train_sha256']
with np.load(reference['oof'],allow_pickle=False) as z:
    assert np.array_equal(z['ids'],train.id.to_numpy(dtype=str)) and np.array_equal(z['labels'],labels)
    assert np.array_equal(z['y'],y) and np.array_equal(z['groups'],groups)
    baseline=z['nested'].copy()
assert baseline.shape==(3,len(train),18) and np.isfinite(baseline).all()
def scores(values):return [float(balanced_accuracy_score(y,part.argmax(1))) for part in values]
np.testing.assert_allclose(scores(baseline),[baseline_report['scenario_scores']['nested_candidate'][s] for s in ('1','3','6')],rtol=0,atol=1e-12)
def clusters(rows):
    if len(rows)<2:return np.zeros(len(rows),int)
    x=TfidfVectorizer(stop_words='english',sublinear_tf=True).fit_transform(texts[rows])
    distance=np.clip(1-(x@x.T).toarray(),0,2);np.fill_diagonal(distance,0)
    return fcluster(linkage(squareform(distance,checks=False),method='average'),.9,criterion='distance')
def context(rows,scenario,name):
    rng=np.random.default_rng(20261015+int(hashlib.sha256(name.encode()).hexdigest()[:6],16))
    batches=np.empty(len(rows),int)
    for group in np.unique(groups[rows]):
        members=rng.permutation(np.flatnonzero(groups[rows]==group))
        batches[members]=np.arange(len(members))//scenario
    lab=np.empty(len(rows),int);offset=0
    for batch in np.unique(batches):
        members=np.flatnonzero(batches==batch);recovered=clusters(rows[members])
        lab[members]=recovered+offset;offset+=int(recovered.max())+1
    return lab
def affine(x,coef,bias):
    z=np.asarray(x,dtype=np.float64)@np.asarray(coef,dtype=np.float64).T+np.asarray(bias,dtype=np.float64)
    return z-logsumexp(z,axis=-1,keepdims=True)
plain=np.empty((3,len(train),18));center=np.empty_like(plain);coverage=np.zeros(len(train),int)
for name in spec['jobs']:
    checkpoint=root/'lora24'/name
    completion=json.loads((checkpoint/'COMPLETE.json').read_text())
    assert completion['job']==name and completion['train_sha256']==plan['train_sha256']
    for file,digest in completion['files'].items():
        assert hashlib.sha256((checkpoint/file).read_bytes()).hexdigest()==digest,(name,file)
    if name=='final':continue
    rows=np.asarray(plan['jobs'][name]['valid'],int);coverage[rows]+=1
    with np.load(checkpoint/'validation_features.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],train.id.iloc[rows].to_numpy(dtype=str))
        assert np.array_equal(z['classes'],labels) and z['train_sha256'].item()==plan['train_sha256']
        features=z['features'].copy()
    with np.load(checkpoint/'readouts.npz',allow_pickle=False) as z:
        assert np.array_equal(z['classes'],labels) and z['train_sha256'].item()==plan['train_sha256']
        head={key:z[key].copy() for key in ('plain_coef','plain_intercept','center_coef','center_intercept')}
    p=affine(features,head['plain_coef'],head['plain_intercept'])
    for index,scenario in enumerate((1,3,6)):
        recovered=context(rows,scenario,name);sizes=np.bincount(recovered)[recovered]
        residual=features.copy()
        for group in np.unique(recovered):
            members=np.flatnonzero(recovered==group);residual[members]-=features[members].mean(0)
        c=affine(residual,head['center_coef'],head['center_intercept']);c[sizes<2]=p[sizes<2]
        plain[index,rows]=p;center[index,rows]=c
assert np.all(coverage==1) and np.isfinite(plain).all() and np.isfinite(center).all()
half=.5*plain+.5*center
unique=np.unique(groups)
den=np.stack([np.bincount(y[groups==g],minlength=18) for g in unique])
draws=np.random.default_rng(20261015).integers(len(unique),size=(3000,len(unique)))
def compare(logits):
    diff=((logits.argmax(-1)==y[None]).astype(float)-(baseline.argmax(-1)==y[None]).astype(float)).mean(0)
    num=np.stack([np.bincount(y[groups==g],weights=diff[groups==g],minlength=18) for g in unique])
    bootstrap=[float((num[d].sum(0)/den[d].sum(0).clip(1)).mean()) for d in draws]
    return dict(scores=dict(zip(('1','3','6'),scores(logits))),mean_gain=float(np.mean(scores(logits))-np.mean(scores(baseline))),
                paired_prompt_gain_interval=np.quantile(bootstrap,[.025,.975]).tolist())
assert spec['fixed_mixture_weights']==[.05,.1,.2]
mixtures={str(weight):(1-weight)*baseline+weight*half for weight in spec['fixed_mixture_weights']}
report=dict(train_sha256=plan['train_sha256'],classes=labels.tolist(),
            training='Five independent outer fits and one final fit, official pretrained base only, fixed3epochs.',
            baseline_tag='gemma_tail_half_current_calibrated',baseline_scores=scores(baseline),
            standalone=dict(plain=scores(plain),center=scores(center),half=scores(half)),
            fixed_mixtures={weight:compare(values) for weight,values in mixtures.items()},
            inner_weight_fitting=False,outer_label_model_selection=False,submitted=False,
            limitations=['Development data reused; three fixed mixtures are descriptive comparisons.',
                         'Own two-head sampler/loss does not exactly reproduce unavailable public training source.',
                         'Lexical context recovery .9; singleton plain fallback; no class uniqueness assumption.',
                         'UltraBERT excluded from this validation. No competition submission created.'])
(root/'lora24_comparison.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(root/'lora24_oof.npz',ids=train.id.to_numpy(dtype=str),labels=labels,y=y,groups=groups,
                    plain=plain,center=center,half=half,**{f'mixture_{key}':value for key,value in mixtures.items()})
print(json.dumps(report,indent=2),flush=True)
