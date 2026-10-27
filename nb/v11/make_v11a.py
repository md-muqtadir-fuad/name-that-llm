"""v11a = exact copy of avazaku's verified reproduction of dhruvtibarewal/nlm-v27 (public 0.651) + a preview shortcut:
when test.csv is exactly the 3-row public sample (same ids, byte lengths and blake2s hashes as in v8's check), write a
3-row placeholder submission and skip the two heavy GPU cells (comp_stack, infer). Any other test file (the hidden
rerun) runs the original notebook unchanged. Saves ~20-30 GPU-minutes per commit.
Run from the repo root: python nb/v11/make_v11a.py"""
import glob
import json
from pathlib import Path

src = glob.glob('research/kernels_2026-10-26/avazaku__name-that-llm-0-651-verified-reproduction/*.ipynb')[0]
nb = json.loads(Path(src).read_text(encoding='utf-8'))
v8 = json.loads(Path('nb/v8/ntl-v8.ipynb').read_text(encoding='utf-8'))
v8c1 = ''.join(v8['cells'][1]['source'])
fixture = v8c1[v8c1.index('def exact_public_fixture'):v8c1.index('DATA=next(')]   # v8's exact 3-row check, verbatim
preview = ('# preview shortcut (v11a): skip the heavy GPU cells on the exact 3-row public sample only\n'
           'import csv, hashlib, json\nfrom pathlib import Path\n' + fixture +
           "PUBLIC_PREVIEW = exact_public_fixture(Path(DATA) / 'test.csv')\n"
           'if PUBLIC_PREVIEW:\n'
           '    import pandas as pd\n'
           "    _f = pd.read_csv(Path(DATA) / 'test.csv', dtype=str)\n"
           "    pd.DataFrame({'id': _f.id, 'model': ['claude-fable-5'] * len(_f)}).to_parquet('/kaggle/working/submission.parquet', index=False)\n"
           "    Path('/kaggle/working/submission.mode.json').write_text(json.dumps(dict(mode='exact_public_format_preview')))\n"
           "    print('EXACT_PUBLIC_FORMAT_PREVIEW_ONLY: the hidden rerun executes the full pipeline.')\n")
assert 'def exact_public_fixture' in preview and 'expected=' in preview.replace(' ', '')
heavy = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and
         any(l.lstrip().startswith('!python') for l in ''.join(c['source']).split('\n'))]
assert len(heavy) == 2, heavy
for i in heavy:
    body = ''.join(nb['cells'][i]['source']).rstrip('\n').split('\n')
    nb['cells'][i]['source'] = ('if not PUBLIC_PREVIEW:\n' + '\n'.join('    ' + l if l.strip() else l for l in body) + '\n').splitlines(True)
setup = next(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and 'DATA = next(' in ''.join(c['source']))
nb['cells'].insert(setup + 1, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                               'source': preview.splitlines(True)})
compile(preview, 'preview', 'exec')
out = Path('nb/v11a')
out.mkdir(parents=True, exist_ok=True)
(out / 'ntl-v11a.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads((Path(src).parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
for k in ('id_no', 'keywords'):
    meta.pop(k, None)
meta.update(id='mdmuqtadirfuad/ntl-v11a', title='ntl-v11a', code_file='ntl-v11a.ipynb', is_private=True)
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('written nb/v11a/ntl-v11a.ipynb', len(nb['cells']), 'cells; heavy cells wrapped:', [h + (1 if h > setup else 0) for h in heavy])
