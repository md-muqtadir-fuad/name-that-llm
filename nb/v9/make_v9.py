"""v9 = v8 (private copy of public 0.639) + the E050 word-BiLSTM at a fixed 10% probability mixture before Sinkhorn.
Single change against v8: one new cell (nb/v9/lstm_cell.py, runs in every mode) and five lines in the final cell.
Run from the repo root: python nb/v9/make_v9.py"""
import json
from pathlib import Path

nb = json.loads(Path('nb/v8/ntl-v8.ipynb').read_text(encoding='utf-8'))
cell = Path('nb/v9/lstm_cell.py').read_text(encoding='utf-8')
anchor = "    print('NIGHT24_FIXED_BLEND top1 .90 + original readout bank .10',flush=True)\n"
mix = ("    _lz=np.load('/kaggle/working/lstm_probabilities.npz',allow_pickle=False)\n"
       "    _li=pd.Series(np.arange(len(_lz['ids'])),index=_lz['ids']).loc[test.id.astype(str)].to_numpy()\n"
       "    _lp=_lz['probabilities'][_li][:,[_lz['labels'].tolist().index(l) for l in LABELS]]\n"
       "    assert _lp.shape==P.shape and np.isfinite(_lp).all() and np.allclose(_lp.sum(1),1)\n"
       "    P=.90*P+.10*_lp\n"
       "    print('LSTM_MEMBER mixed at .10 before balancing',flush=True)\n")
last = ''.join(nb['cells'][-1]['source'])
assert last.count(anchor) == 1
nb['cells'][-1]['source'] = last.replace(anchor, anchor + mix).splitlines(True)
nb['cells'].insert(2, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                       'source': cell.splitlines(True)})
Path('nb/v9/ntl-v9.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads(Path('nb/v8/kernel-metadata.json').read_text(encoding='utf-8'))
meta.update(id='mdmuqtadirfuad/ntl-v9', title='ntl-v9', code_file='ntl-v9.ipynb')
meta['dataset_sources'] = meta['dataset_sources'] + ['mdmuqtadirfuad/ntl-lstm-final']
Path('nb/v9/kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('written nb/v9/ntl-v9.ipynb', len(nb['cells']), 'cells')
