"""v11 = avazaku's verified reproduction of dhruvtibarewal/nlm-v27 (public 0.651; Llama-3.1-8B + Qwen3-14B + Qwen3-8B
LoRA24 v5 fold ensembles, comp stack, ATLAS, Sinkhorn, dense-cluster Hungarian) + our Gemma-4-12B two-stage member
(E056, plain readout) mixed into P after ATLAS and before Sinkhorn. Single change against the 0.651 base.
Uses avazaku's public Llama asset cache (Kaggle's own Llama is gated for this account).
Run from the repo root: python nb/v11/make_v11.py [weight]"""
import glob
import json
import sys
from pathlib import Path

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
src = glob.glob('research/kernels_2026-10-26/avazaku__name-that-llm-0-651-verified-reproduction/*.ipynb')[0]
nb = json.loads(Path(src).read_text(encoding='utf-8'))
cell = Path('nb/v10/gemma_cell.py').read_text(encoding='utf-8')
SCRIPT = 3   # the inference script cell
script = ''.join(nb['cells'][SCRIPT]['source'])
anchor = '    Q = P.copy()\n'
assert script.count(anchor) == 1 and 'P = mixed / mixed.sum(1, keepdims=True)' in script
mix = ("    _gz = np.load('/kaggle/working/gemma_probabilities.npz', allow_pickle=False)   # our Gemma member (v11)\n"
       "    _gi = pd.Series(np.arange(len(_gz['ids'])), index=_gz['ids']).loc[test.id.astype(str)].to_numpy()\n"
       "    _gp = _gz['probabilities'][_gi][:, [_gz['labels'].tolist().index(l) for l in LABELS]]\n"
       "    assert _gp.shape == P.shape and np.isfinite(_gp).all() and np.allclose(_gp.sum(1), 1)\n"
       f"    P = {1 - W:.2f} * P + {W:.2f} * _gp; P /= P.sum(1, keepdims=True); log('GEMMA_MEMBER mixed at {W:.2f}')\n")
nb['cells'][SCRIPT]['source'] = script.replace(anchor, mix + anchor).splitlines(True)
nb['cells'].insert(SCRIPT, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                            'source': cell.splitlines(True)})
compile(cell, 'gemma_cell.py', 'exec')
out = Path('nb/v11')
(out / 'ntl-v11.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads((Path(src).parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
for k in ('id_no', 'keywords'):
    meta.pop(k, None)
meta.update(id='mdmuqtadirfuad/ntl-v11', title='ntl-v11', code_file='ntl-v11.ipynb', is_private=True)
meta['kernel_sources'] = meta.get('kernel_sources', []) + ['mdmuqtadirfuad/ntl-gemma-ts-s2']
if 'google/gemma-4/Transformers/gemma-4-12b/2' not in meta.get('model_sources', []):
    meta['model_sources'] = meta.get('model_sources', []) + ['google/gemma-4/Transformers/gemma-4-12b/2']
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('written nb/v11/ntl-v11.ipynb', len(nb['cells']), 'cells; weight', W)
