"""Builds nb/honest_oof/kernel/ (run.py with gauss.py embedded + kernel-metadata.json). Run from the repo root."""
import json
from pathlib import Path

D = Path('nb/honest_oof')
K = D / 'kernel'
K.mkdir(exist_ok=True)
src = (D / 'run.py').read_text(encoding='utf-8')
src = src.replace('GAUSS_SRC = None', 'GAUSS_SRC = ' + repr((D / 'gauss.py').read_text(encoding='utf-8')), 1)
(K / 'run.py').write_text(src, encoding='utf-8', newline='\n')
(K / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'mdmuqtadirfuad/ntl-honest-oof', 'title': 'ntl-honest-oof', 'code_file': 'run.py', 'language': 'python',
    'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
    'dataset_sources': ['dhruvtibarewal/nlm-l24-adapters-v5'], 'competition_sources': ['name-that-llm'], 'kernel_sources': [],
    'model_sources': ['qwen-lm/qwen-3/Transformers/8b-base/1', 'qwen-lm/qwen-3/Transformers/14b-base/1',
                      'alexchilton/llama-3-1-8b-base/Transformers/default/1'],
    'docker_image': 'gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a',
    'machine_shape': 'NvidiaTeslaT4'}, indent=1), encoding='utf-8')
print('built', K)
