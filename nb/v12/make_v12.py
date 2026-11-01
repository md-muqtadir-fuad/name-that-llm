"""v12 = public 0.664 notebook (zbigniewkirchner, V31 code family: hard ATLAS deadline, dual-GPU 8B, open-pool SVC)
+ our Gemma-4-12B two-stage LoRA (kernel output mdmuqtadirfuad/ntl-gemma-ts-s2, final adapter) as an infer.py MODELS row:
plain weight 2.0 and centred twin 2.8 (= adapt_w 1.0 x w 2.0 x twin 1.4) on infer.py's own estimated clusters.
Evidence: honest seeded-split OOF of the 0.664 members we can score (E065-E067b): +0.011, 31/40 test-like draws.
Parity: kernel ntl-gemma-parity (infer.py loader vs training-time OOF: argmax agreement 1.0 on 216 rows; ~15 min/4,000 rows).
Patches: infer.py LoRA-name fallback for Gemma-4 ('language_model.' prefix); setup cell finds the Gemma base + adapter;
MODELS gets the Gemma row last. Metadata: the 0.664's sources + Gemma, on the GPU byod image of the GPU-scored 0.66x
notebooks (the pulled 0.664 metadata lists a CPU image, gcr.io/kaggle-images/81b1e7...).
Run from the repo root: python nb/v12/make_v12.py [plain_w] [twin]"""
import glob
import json
import sys
from pathlib import Path

W = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
TWIN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.4
BASE = Path(glob.glob('research/kernels_2026-11-01/zbigniewkirchner__0-664-one-swap-198k-clean-pool-fully-public')[0])
GPU_IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
nb = json.loads(next(BASE.glob('*.ipynb')).read_text(encoding='utf-8'))
src = lambda i: ''.join(nb['cells'][i]['source'])


def setsrc(i, s):
    nb['cells'][i]['source'] = s.splitlines(True)


# 1) infer.py: Gemma-4 nests its decoder under language_model.
I = next(i for i, c in enumerate(nb['cells']) if src(i).startswith('%%writefile infer.py'))
s = src(I)
key = "if name not in mods and ('model.' + name) in mods: name = 'model.' + name\n"
assert s.count(key) == 1
line = next(l for l in s.split('\n') if l.strip() == key.strip())
ind = line[:len(line) - len(line.lstrip())]
setsrc(I, s.replace(line + '\n', line + '\n' + ind + "if name not in mods and ('language_model.' + name) in mods: name = 'language_model.' + name   # Gemma-4 (v12)\n"))

# 2) setup cell: find the Gemma base and our final two-stage adapter
S = next(i for i, c in enumerate(nb['cells']) if "POOL = '/kaggle/working/open_pool.parquet'" in src(i))
setsrc(S, src(S).rstrip('\n') + '\n\n'
       "# v12: our Gemma-4-12B two-stage LoRA (final adapter) and its base\n"
       "GEMMA_DIR = next(os.path.dirname(c) for c in cfgs if 'gemma-4-12b' in c.lower().replace('_', '-')\n"
       "                 and os.path.basename(os.path.dirname(c)) != 'base_config' and glob.glob(os.path.join(os.path.dirname(c), '*.safetensors')))\n"
       "GEMMA_FINAL = os.path.dirname(sorted(glob.glob('/kaggle/input/**/lora_twostage/final/COMPLETE.json', recursive=True))[0])\n"
       "assert os.path.exists(os.path.join(GEMMA_FINAL, 'adapter', 'adapter_model.safetensors')) and os.path.exists(os.path.join(GEMMA_FINAL, 'heads.safetensors'))\n"
       "print('gemma', GEMMA_DIR, GEMMA_FINAL)\n")

# 3) MODELS: Gemma row last (plain w, centred twin = adapt_w * w * twin)
M = next(i for i, c in enumerate(nb['cells']) if 'MODELS = [' in src(i))
s = src(M)
anchor = 'import pandas as pd\n'
assert s.count(anchor) == 1
row = (f"MODELS.append({{'name': 'g12ts', 'type': 'l24', 'path': GEMMA_DIR, 'adapters': [GEMMA_FINAL], 'keep': 24,\n"
       f"               'layers': [2, 4, 8, 12, 14, 16, 20, 24], 'w': {W}, 'bs': 8, 'max_len': 448, 'twin': {TWIN}, 'eos_id': 1}})   # v12\n")
setsrc(M, s.replace(anchor, row + anchor))

# 4) a short note on top (private notebook)
nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
    f'# ntl-v12\n', '\n', f'The public 0.664 notebook plus our Gemma-4-12B two-stage LoRA as one more LoRA row (plain {W}, centred twin '
    f'{W * TWIN:.1f}). Everything else is unchanged.\n']})

out = Path('nb/v12')
(out / 'ntl-v12.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads((BASE / 'kernel-metadata.json').read_text(encoding='utf-8'))
for k in ('id_no', 'keywords'):
    meta.pop(k, None)
meta.update(id='mdmuqtadirfuad/ntl-v12', title='ntl-v12', code_file='ntl-v12.ipynb', is_private=True, docker_image=GPU_IMAGE)
meta['kernel_sources'] = meta.get('kernel_sources', []) + ['mdmuqtadirfuad/ntl-gemma-ts-s2']
meta['model_sources'] = meta.get('model_sources', []) + ['google/gemma-4/Transformers/gemma-4-12b/2']
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(f'written nb/v12/ntl-v12.ipynb (gemma plain {W}, twin {TWIN} -> centred {W * TWIN:.1f})')
