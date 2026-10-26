"""v10 = v8 (private copy of public 0.639) + our Gemma-4-12B two-stage LoRA (E056, plain readout) at a fixed 15%
probability mixture before Sinkhorn. Single change against v8: one new cell (nb/v10/gemma_cell.py, runs in every mode)
and six lines in the final cell. Run from the repo root: python nb/v10/make_v10.py [weight]"""
import json
import sys
from pathlib import Path

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
nb = json.loads(Path('nb/v8/ntl-v8.ipynb').read_text(encoding='utf-8'))
cell = Path('nb/v10/gemma_cell.py').read_text(encoding='utf-8')
anchor = "    print('NIGHT24_FIXED_BLEND top1 .90 + original readout bank .10',flush=True)\n"
mix = ("    _gz=np.load('/kaggle/working/gemma_probabilities.npz',allow_pickle=False)\n"
       "    _gi=pd.Series(np.arange(len(_gz['ids'])),index=_gz['ids']).loc[test.id.astype(str)].to_numpy()\n"
       "    _gp=_gz['probabilities'][_gi][:,[_gz['labels'].tolist().index(l) for l in LABELS]]\n"
       "    assert _gp.shape==P.shape and np.isfinite(_gp).all() and np.allclose(_gp.sum(1),1)\n"
       f"    P={1 - W:.2f}*P+{W:.2f}*_gp\n"
       f"    print('GEMMA_MEMBER mixed at {W:.2f} before balancing',flush=True)\n")
last = ''.join(nb['cells'][-1]['source'])
assert last.count(anchor) == 1
nb['cells'][-1]['source'] = last.replace(anchor, anchor + mix).splitlines(True)
nb['cells'].insert(2, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                       'source': cell.splitlines(True)})
compile(cell, 'gemma_cell.py', 'exec')
Path('nb/v10/ntl-v10.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads(Path('nb/v8/kernel-metadata.json').read_text(encoding='utf-8'))
meta.update(id='mdmuqtadirfuad/ntl-v10', title='ntl-v10', code_file='ntl-v10.ipynb')
meta['kernel_sources'] = meta['kernel_sources'] + ['mdmuqtadirfuad/ntl-gemma-ts-s2']
if 'google/gemma-4/Transformers/gemma-4-12b/2' not in meta['model_sources']:
    meta['model_sources'] = meta['model_sources'] + ['google/gemma-4/Transformers/gemma-4-12b/2']
Path('nb/v10/kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('written nb/v10/ntl-v10.ipynb', len(nb['cells']), 'cells; weight', W)
