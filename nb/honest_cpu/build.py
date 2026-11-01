"""Builds nb/honest_cpu/kernel/: a Kaggle CPU script that runs the 0.664 infer.py (one patch: --sim also saves every
member's log-probs) in --sim k mode for k = 0..4 with no GPU members, all 18 passages of each held-out prompt kept
(--sim_k off) and member weights 1, so tfidf / distort / svc4 come out as raw, honest per-row OOF log-probs on the
seeded split (RandomState(0) shuffle of sorted prompt ids, fold = i % 5). Pool = the open pool minus train-prompt
regenerations, as in the 0.664 notebook. Run from the repo root.
Variant `own` (E068): our Kaggle Benchmarks passages (dataset mdmuqtadirfuad/ntl-own-passages, 12 labels) appended to
the extra file, so tfidf / distort / svc4 train on them at the extra weight 0.5: python nb/honest_cpu/build.py own"""
import json
import sys
from pathlib import Path

OWN = len(sys.argv) > 1 and sys.argv[1] == 'own'

D = Path('nb/honest_cpu')
K = D / ('kernel_own' if OWN else 'kernel')
K.mkdir(exist_ok=True)
inf = (D / 'infer_0664.py').read_text(encoding='utf-8')
old = "    np.save(os.path.join(a.cache, f'sim{a.sim}_lp.npy'), tot)\n"
assert inf.count(old) == 1
inf = inf.replace(old, old + "    np.savez_compressed(os.path.join(a.cache, f'sim{a.sim}_members.npz'), ids=np.array(test.id.astype(str).tolist()), "
                  "**{k_: v_.astype(np.float32) for k_, v_ in LP.items()})   # honest-OOF patch\n")
run = f'''import glob, os, subprocess, sys
import pandas as pd
open('/kaggle/working/infer.py', 'w').write({inf!r})
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
EXTRA = next(iter(glob.glob('/kaggle/input/**/nlm_matched_clean.parquet', recursive=True)))
OWN = {OWN!r}
if OWN:   # E068: + our Benchmarks passages
    ex = pd.read_parquet(EXTRA)
    own = pd.read_parquet(glob.glob('/kaggle/input/**/own_passages.parquet', recursive=True)[0])[['text', 'model']]
    print('extra', len(ex), '+ own', len(own), own.model.value_counts().to_dict(), flush=True)
    pd.concat([ex[['text', 'model']], own], ignore_index=True).to_parquet('/kaggle/working/extra.parquet')
    EXTRA = '/kaggle/working/extra.parquet'
op = pd.read_parquet(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)[0], columns=['text', 'model', 'flag_train_prompt_regen'])
op[~op.flag_train_prompt_regen.astype(bool)][['text', 'model']].to_parquet('/kaggle/working/open_pool.parquet')
print('pool', len(op), 'extra', EXTRA, flush=True)
for k in range(5):
    subprocess.run([sys.executable, '/kaggle/working/infer.py', '--data', DATA, '--models', '', '--sim', str(k),
                    '--perpassage', '1', '--extra', EXTRA, '--extra_w', '0.5', '--tfidf_w', '1', '--distort_w', '1',
                    '--svc_w', '1', '--pool', '/kaggle/working/open_pool.parquet', '--adapt_w', '1.0', '--adapt_thr', '0.15',
                    '--sinkhorn', '0', '--post', '0', '--out', f'/kaggle/working/sim{{k}}.parquet',
                    '--cache', '/kaggle/working/cache'], check=True, cwd='/kaggle/working')
os.remove('/kaggle/working/open_pool.parquet')
'''
(K / 'run.py').write_text(run, encoding='utf-8', newline='\n')
(K / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'mdmuqtadirfuad/ntl-honest-cpu' + ('-own' if OWN else ''), 'title': 'ntl-honest-cpu' + ('-own' if OWN else ''), 'code_file': 'run.py', 'language': 'python',
    'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
    'dataset_sources': ['dhruvtibarewal/nlm-matched-clean', 'crtproduction/name-that-llm-open-pool'] + (['mdmuqtadirfuad/ntl-own-passages'] if OWN else []),
    'competition_sources': ['name-that-llm'], 'kernel_sources': [], 'model_sources': [],
    'docker_image': 'gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a'},
    indent=1), encoding='utf-8')
print('built', K)
