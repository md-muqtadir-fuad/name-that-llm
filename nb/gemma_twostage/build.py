"""Kaggle kernel for the two-stage Gemma-4-12B LoRA (nb/colab_twostage/twostage_train.py, NTL_BACKBONE=gemma4-12b):
stage 1 on nishantdahal/ntl-handwriting-blend pool_v4.parquet (75,783 passages), stage 2 on host folds 0-2 + final.
MODE 'smoke' = tiny stage1a + fold0 with a time estimate; 'jobs:<a,b>' = run those jobs (later sessions resume from the
previous kernel's output, attached as a kernel source). Run from the repo root: python nb/gemma_twostage/build.py smoke"""
import json
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
KID = sys.argv[2] if len(sys.argv) > 2 else 'ntl-gemma-twostage'        # kernel id (one per session)
PREV = sys.argv[3] if len(sys.argv) > 3 else ''                          # earlier session's kernel (resume source)
BACKBONE = sys.argv[4] if len(sys.argv) > 4 else 'gemma4-12b'
MODEL_SRC = {'gemma4-12b': 'google/gemma-4/Transformers/gemma-4-12b/2', 'llama31-8b': None}[BACKBONE]
# Llama-3.1-8B: Kaggle's copy is gated for this account, so the weights come from our HF-download kernel's output
WEIGHTS_KERNEL = {'llama31-8b': 'mdmuqtadirfuad/ntl-llama31-8b-weights'}.get(BACKBONE)
MATCH = {'gemma4-12b': ('/gemma-4-12b/',), 'llama31-8b': ('llama-3.1', '/8b/')}[BACKBONE]   # all must be in the path
import base64
SRC = Path('nb/colab_twostage/twostage_train.py').read_text(encoding='utf-8')
B64 = base64.b64encode(SRC.encode('utf-8')).decode('ascii')   # the trainer embeds its worker in ''' strings
RUN = r'''import base64, glob, json, os, shutil, subprocess, sys
SRC = base64.b64decode(B64).decode('utf-8')
from pathlib import Path
MODE, BACKBONE, MATCH = '%s', '%s', %r
os.chdir('/kaggle/working')
Path('twostage_train.py').write_text(SRC)
cfg = [p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if all(m in p.lower() for m in MATCH)]
pool = glob.glob('/kaggle/input/**/pool_v4.parquet', recursive=True)
train = [p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p]
assert cfg and pool and train, (cfg, pool, train)
work = Path('/kaggle/working/smoke' if MODE == 'smoke' else '/kaggle/working/ts')
for prev in glob.glob('/kaggle/input/**/lora_twostage', recursive=True):   # resume: earlier sessions' completed jobs
    if MODE != 'smoke' and Path(prev).is_dir() and not (work / 'lora_twostage').exists():
        shutil.copytree(prev, work / 'lora_twostage')
# stage-1 pool = pool_v4 + darragh1walsh/ntl-kaggle-ai-gen-run-1 (CC0, 1,481 in-style passages, none in pool_v4)
import pandas as pd
pv = pd.read_parquet(pool[0])
dw = [pd.read_csv(f) for f in glob.glob('/kaggle/input/**/ntl-kaggle-ai-gen-*.csv', recursive=True)]
if dw:
    dw = pd.concat(dw, ignore_index=True)
    dw = dw[dw.passed_filter.astype(str).str.lower().isin(['true', '1'])].reset_index(drop=True)
    dw = pd.DataFrame({'id': 'dw_' + dw.index.astype(str), 'prompt_id': 'dw_' + dw.prompt_id.astype(str),
                       'text': dw.text.astype(str), 'model': dw.model.astype(str)})
    pv = pd.concat([pv[['id', 'prompt_id', 'text', 'model']].astype(str), dw], ignore_index=True)
    print('pool_v4 +', len(dw), 'darragh rows ->', len(pv), flush=True)
pv.to_parquet('/kaggle/working/pool_merged.parquet', index=False)
env = dict(os.environ, NTL_BACKBONE=BACKBONE, NTL_WORK=str(work), NTL_DATA=str(Path(train[0]).parent),
           NTL_POOL='/kaggle/working/pool_merged.parquet', NTL_MODEL_DIR=str(Path(cfg[0]).parent))
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
args = [sys.executable, '-u', 'twostage_train.py'] + (['--smoke'] if MODE == 'smoke' else ['--jobs', MODE.split(':', 1)[1]])
r = subprocess.run(args, env=env)
for f in glob.glob(str(work / 'lora_twostage' / '**' / 'lora_state.pt'), recursive=True):
    if MODE == 'smoke':
        os.remove(f)
os.remove('twostage_train.py')
for f in ('pool_merged.parquet', str(work / 'train_plus_pool.csv')):   # large inputs; rebuilt each session
    if os.path.exists(f):
        os.remove(f)
assert r.returncode == 0, 'trainer failed'
'''
d = Path('nb/gemma_twostage') / (f'smoke_{BACKBONE}' if MODE == 'smoke' else KID)
d.mkdir(parents=True, exist_ok=True)
(d / 'run.py').write_text(f"B64 = '{B64}'\n" + RUN % (MODE, BACKBONE, MATCH), encoding='utf-8', newline='\n')
compile((d / 'run.py').read_text(encoding='utf-8'), 'run.py', 'exec')
meta = {'id': f'mdmuqtadirfuad/{KID}', 'title': KID, 'code_file': 'run.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False,
        'dataset_sources': ['nishantdahal/ntl-handwriting-blend', 'darragh1walsh/ntl-kaggle-ai-gen-run-1'],
        'kernel_sources': ([f'mdmuqtadirfuad/{PREV}'] if PREV else []) + ([WEIGHTS_KERNEL] if WEIGHTS_KERNEL else []),
        'competition_sources': ['name-that-llm'], 'model_sources': [MODEL_SRC] if MODEL_SRC else [],
        'machine_shape': 'NvidiaTeslaT4',
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'}
(d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('written', d / 'run.py', 'mode', MODE)
