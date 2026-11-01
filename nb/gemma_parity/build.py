"""Builds nb/gemma_parity/kernel/: does our Gemma-4-12B two-stage member reproduce through the public 0.664 infer.py
l24 loader? Runs infer.py --sim 0 (seeded fold 0 held out, all 18 passages per prompt) with ONLY the Gemma row on its
fold-0 adapter (our GroupKFold fold 0), CPU members off, and compares the saved plain log-probs with our training-time
val_outputs on the rows both hold out. Also logs the member's wall time (for the 4,000-row test budget).
infer.py patches: 'language_model.' LoRA-name fallback (Gemma-4 nests its decoder), and --sim saves member log-probs.
Run from the repo root: python nb/gemma_parity/build.py"""
import json
from pathlib import Path

D = Path('nb/gemma_parity')
K = D / 'kernel'
K.mkdir(parents=True, exist_ok=True)
inf = Path('nb/honest_cpu/infer_0664.py').read_text(encoding='utf-8')
old = "    np.save(os.path.join(a.cache, f'sim{a.sim}_lp.npy'), tot)\n"
assert inf.count(old) == 1
inf = inf.replace(old, old + "    np.savez_compressed(os.path.join(a.cache, f'sim{a.sim}_members.npz'), ids=np.array(test.id.astype(str).tolist()), "
                  "**{k_: v_.astype(np.float32) for k_, v_ in LP.items()})   # parity patch\n")
key = "if name not in mods and ('model.' + name) in mods: name = 'model.' + name\n"
assert inf.count(key) == 1
line = next(l for l in inf.split('\n') if l.strip() == key.strip())
ind = line[:len(line) - len(line.lstrip())]
inf = inf.replace(line + '\n', line + '\n' + ind + "if name not in mods and ('language_model.' + name) in mods: name = 'language_model.' + name   # Gemma-4\n")
run = f'''import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), 'no GPU in this session'
open('/kaggle/working/infer.py', 'w').write({inf!r})
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
GEMMA_DIR = next(os.path.dirname(c) for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
                 if 'gemma-4-12b' in c.lower().replace('_', '-') and os.path.basename(os.path.dirname(c)) != 'base_config'
                 and glob.glob(os.path.join(os.path.dirname(c), '*.safetensors')))
F0 = os.path.dirname(sorted(glob.glob('/kaggle/input/**/lora_twostage/fold0/COMPLETE.json', recursive=True))[0])
print('gemma', GEMMA_DIR, '| adapter', F0, flush=True)
row = {{'name': 'g12ts', 'type': 'l24', 'path': GEMMA_DIR, 'adapters': [F0], 'keep': 24, 'layers': [2, 4, 8, 12, 14, 16, 20, 24],
       'w': 1.0, 'bs': 8, 'max_len': 448, 'twin': 1.0, 'eos_id': 1}}
t0 = time.time()
subprocess.run([sys.executable, '/kaggle/working/infer.py', '--data', DATA, '--models', json.dumps([row]), '--sim', '0',
                '--perpassage', '1', '--tfidf_w', '0', '--distort_w', '0', '--svc_w', '0', '--adapt_w', '1.0', '--adapt_thr', '0.15',
                '--sinkhorn', '0', '--post', '0', '--out', '/kaggle/working/sim0.parquet', '--cache', '/kaggle/working/cache'],
               check=True, cwd='/kaggle/working')
wall = time.time() - t0
z = np.load('/kaggle/working/cache/sim0_members.npz')
v = np.load(os.path.join(F0, 'val_outputs.npz'), allow_pickle=True)
lsm = lambda x: x - x.max(1, keepdims=True) - np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True))
pos = {{i: k for k, i in enumerate(z['ids'])}}
both = [k for k, i in enumerate(v['ids']) if i in pos]
a = z['g12ts'][[pos[v['ids'][k]] for k in both]]
b = lsm(v['plain_logits'][both].astype(np.float64))
train = pd.read_csv(os.path.join(DATA, 'train.csv')); lab = sorted(train.model.unique())
yt = np.array([lab.index(m) for m in train.set_index('id').loc[list(z['ids']), 'model']])
print(f'PARITY rows held out by both: {{len(both)}} | max abs diff of log-probs {{np.abs(a - b).max():.4f}} | '
      f'argmax agreement {{(a.argmax(1) == b.argmax(1)).mean():.4f}} | acc infer.py {{(a.argmax(1) == v["y"][both]).mean():.4f}} '
      f'vs training-time {{(b.argmax(1) == v["y"][both]).mean():.4f}}', flush=True)
print(f'all {{len(yt)}} sim rows (fold-0 adapter, part in-sample): acc {{(z["g12ts"].argmax(1) == yt).mean():.4f}} | '
      f'infer.py wall {{wall:.0f}}s for {{len(yt)}} rows -> ~{{wall / len(yt) * 4000 / 60:.0f}} min per adapter on 4,000 rows (upper bound, incl. load)', flush=True)
'''
(K / 'run.py').write_text(run, encoding='utf-8', newline='\n')
(K / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'mdmuqtadirfuad/ntl-gemma-parity', 'title': 'ntl-gemma-parity', 'code_file': 'run.py', 'language': 'python',
    'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
    'dataset_sources': [], 'competition_sources': ['name-that-llm'], 'kernel_sources': ['mdmuqtadirfuad/ntl-gemma-ts-s2'],
    'model_sources': ['google/gemma-4/Transformers/gemma-4-12b/2'],
    'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
    'machine_shape': 'NvidiaTeslaT4'}, indent=1), encoding='utf-8')
print('built', K)
