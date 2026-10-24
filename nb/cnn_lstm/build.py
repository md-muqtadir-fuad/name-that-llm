"""Builds two Kaggle CPU kernels (ntl-cnn-cpu, ntl-lstm-cpu) that run experiments/cnn_lstm/cnn_lstm.py: a tiny smoke pass
first (fails fast), then the full two-stage run. Inputs: competition train.csv + nishantdahal/ntl-handwriting-blend
(pool_v4.parquet). Output: /kaggle/working/experiments/cnn_lstm/out/oof_<arch>.npz. Run from the repo root."""
import json
from pathlib import Path

SRC = Path('experiments/cnn_lstm/cnn_lstm.py').read_text(encoding='utf-8')
assert "'''" not in SRC
RUN = r'''import glob, os, subprocess, sys
from pathlib import Path
ARCH = '%s'
os.chdir('/kaggle/working')
Path('cnn_lstm.py').write_text(SRC)
for dst, pat in (('data/train.csv', '/kaggle/input/**/name-that-llm/train.csv'),
                 ('research/ext_data/twostage/pool_v4.parquet', '/kaggle/input/**/pool_v4.parquet')):
    hits = glob.glob(pat, recursive=True)
    assert hits, pat
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if not Path(dst).exists():
        os.symlink(hits[0], dst)
print('cpus', os.cpu_count(), flush=True)
for smoke in ('1', '0'):
    r = subprocess.run([sys.executable, '-u', 'cnn_lstm.py', ARCH], env=dict(os.environ, SMOKE=smoke))
    assert r.returncode == 0, f'{ARCH} smoke={smoke} failed'
os.remove('cnn_lstm.py')
'''
for arch in ('cnn', 'lstm'):
    d = Path(f'nb/cnn_lstm/{arch}')
    d.mkdir(parents=True, exist_ok=True)
    (d / 'run.py').write_text(f"SRC = r'''{SRC}'''\n" + RUN % arch, encoding='utf-8', newline='\n')
    meta = {'id': f'mdmuqtadirfuad/ntl-{arch}-cpu', 'title': f'ntl-{arch}-cpu', 'code_file': 'run.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False,
            'enable_tpu': False, 'enable_internet': False, 'dataset_sources': ['nishantdahal/ntl-handwriting-blend'],
            'competition_sources': ['name-that-llm'], 'kernel_sources': [], 'model_sources': []}
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    compile((d / 'run.py').read_text(encoding='utf-8'), str(d / 'run.py'), 'exec')
    print('written', d)
