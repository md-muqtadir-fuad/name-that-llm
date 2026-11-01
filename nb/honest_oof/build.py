"""Builds nb/honest_oof/kernel/ (run.py with gauss.py embedded + kernel-metadata.json). Run from the repo root."""
import json
from pathlib import Path

D = Path('nb/honest_oof')
K = D / 'kernel'
K.mkdir(exist_ok=True)
src = (D / 'run.py').read_text(encoding='utf-8')
src = src.replace('GAUSS_SRC = None', 'GAUSS_SRC = ' + repr((D / 'gauss.py').read_text(encoding='utf-8')), 1)
import sys
if len(sys.argv) > 1:   # e.g. build.py q14v5 -> run only that member
    src = src.replace('RUN = None', 'RUN = ' + repr(sys.argv[1].split(',')), 1)
(K / 'run.py').write_text(src, encoding='utf-8', newline='\n')
(K / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'mdmuqtadirfuad/ntl-honest-oof', 'title': 'ntl-honest-oof', 'code_file': 'run.py', 'language': 'python',
    'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
    'dataset_sources': ['dhruvtibarewal/nlm-l24-adapters-v5'], 'competition_sources': ['name-that-llm'], 'kernel_sources': [],
    'model_sources': ['qwen-lm/qwen-3/Transformers/8b-base/1', 'qwen-lm/qwen-3/Transformers/14b-base/1',
                      'alexchilton/llama-3-1-8b-base/Transformers/default/1'],
    'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
    'machine_shape': 'NvidiaTeslaT4'}, indent=1), encoding='utf-8')
print('built', K)
