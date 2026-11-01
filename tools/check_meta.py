"""Pre-push check: diff our kernel-metadata.json against the base notebook's and flag anything that can change how or
how fast the notebook runs (image, machine, accelerator, internet) or what it can read (data / kernel / model sources).
Identity fields (id, title, code_file, is_private, keywords, id_no) are expected to differ and are only listed.
Usage: python tools/check_meta.py <our kernel dir> <base kernel dir>   (exit code 1 if a runtime field differs)"""
import json
import sys
from pathlib import Path

RUNTIME = ('docker_image', 'machine_shape', 'enable_gpu', 'enable_tpu', 'enable_internet', 'kernel_type', 'language')
SOURCES = ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources')
IDENTITY = ('id', 'id_no', 'title', 'code_file', 'is_private', 'keywords')

ours = json.loads((Path(sys.argv[1]) / 'kernel-metadata.json').read_text(encoding='utf-8'))
base = json.loads((Path(sys.argv[2]) / 'kernel-metadata.json').read_text(encoding='utf-8'))
bad = False
for k in RUNTIME:
    if ours.get(k) != base.get(k):
        bad = True
        print(f'RUNTIME CHANGE  {k}: ours={ours.get(k)!r}  base={base.get(k)!r}')
for k in SOURCES:
    a, b = set(ours.get(k) or []), set(base.get(k) or [])
    if a - b:
        print(f'added   {k}: {sorted(a - b)}')
    if b - a:
        print(f'REMOVED {k}: {sorted(b - a)}')
for k in IDENTITY:
    if ours.get(k) != base.get(k):
        print(f'identity        {k}: ours={str(ours.get(k))[:60]!r}  base={str(base.get(k))[:60]!r}')
img = ours.get('docker_image') or ''
if ours.get('enable_gpu') and img.startswith('gcr.io/kaggle-images/'):
    # 2026-11-01: gcr.io/kaggle-images/* are CPU images (torch sees no GPU, the T4 session is still charged). Pulled
    # metadata of a public notebook can carry one (its latest version saved on CPU): v11a copied 17b2... and timed out.
    bad = True
    print(f'RUNTIME CHANGE  docker_image is a CPU image with enable_gpu=true: {img[:60]}... '
          f'use the GPU byod image gcr.io/kaggle-private-byod/python@sha256:2757e0c7...')
other = sorted((set(ours) | set(base)) - set(RUNTIME) - set(SOURCES) - set(IDENTITY))
for k in other:
    if ours.get(k) != base.get(k):
        print(f'other           {k}: ours={str(ours.get(k))[:60]!r}  base={str(base.get(k))[:60]!r}')
print('RESULT:', 'runtime fields DIFFER - check before pushing' if bad else 'runtime fields identical to the base')
sys.exit(1 if bad else 0)
