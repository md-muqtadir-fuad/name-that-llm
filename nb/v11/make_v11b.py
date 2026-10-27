"""v11b = avazaku's verified reproduction of nlm-v27 (public 0.651) + our Gemma-4-12B two-stage LoRA (E056) as a proper
MODELS row of infer.py: same loader (LoRA merged into the base, 8-layer mean pool through the checkpoint's norms),
plain + centred-twin readouts, adaptive centring and the log-space blend, weight 1.0 (as the Llama row).
Patch: Gemma-4 nests its decoder under `language_model.`, so infer.py's LoRA module-name lookup gets one fallback line.
No preview shortcut: the commit run must exercise infer.py with Gemma (3 rows through every model).
GEMMA STOP RULE (paper/DECISIONS.md, 2026-10-27): if v11b does not beat the 0.651 base, Gemma is dropped for submissions.
Run from the repo root: python nb/v11/make_v11b.py [weight]"""
import glob
import json
import sys
from pathlib import Path

W = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
src = glob.glob('research/kernels_2026-10-26/avazaku__name-that-llm-0-651-verified-reproduction/*.ipynb')[0]
nb = json.loads(Path(src).read_text(encoding='utf-8'))

# 1) infer.py: LoRA module-name fallback for Gemma-4 (decoder under language_model.)
SCRIPT = next(i for i, c in enumerate(nb['cells']) if ''.join(c['source']).startswith('%%writefile infer.py'))
script = ''.join(nb['cells'][SCRIPT]['source'])
key = "if name not in mods and ('model.' + name) in mods: name = 'model.' + name\n"
assert script.count(key) == 1
line = next(l for l in script.split('\n') if l.strip() == key.strip())
indent = line[:len(line) - len(line.lstrip())]
fallback = indent + "if name not in mods and ('language_model.' + name) in mods: name = 'language_model.' + name   # Gemma-4 (v11b)\n"
script = script.replace(line + '\n', line + '\n' + fallback)
nb['cells'][SCRIPT]['source'] = script.splitlines(True)

# 2) MODELS: append the Gemma row (paths discovered at run time)
MC = next(i for i, c in enumerate(nb['cells']) if 'MODELS = [' in ''.join(c['source']))
mc = ''.join(nb['cells'][MC]['source'])
anchor = 'import pandas as pd\n'
assert mc.count(anchor) == 1
gemma = (
    "# v11b: our Gemma-4-12B two-stage LoRA (kernel output mdmuqtadirfuad/ntl-gemma-ts-s2), same treatment as the other LoRA rows\n"
    "GEMMA_DIR = next(os.path.dirname(c) for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))\n"
    "                 if 'gemma-4-12b' in c.lower().replace('_', '-') and os.path.basename(os.path.dirname(c)) != 'base_config'\n"
    "                 and glob.glob(os.path.join(os.path.dirname(c), '*.safetensors')))\n"
    "GEMMA_FINAL = os.path.dirname(sorted(glob.glob('/kaggle/input/**/lora_twostage/final/COMPLETE.json', recursive=True))[0])\n"
    f"MODELS.append({{'name': 'g12ts', 'type': 'l24', 'path': GEMMA_DIR, 'adapters': [GEMMA_FINAL], 'keep': 24,\n"
    f"               'layers': [2, 4, 8, 12, 14, 16, 20, 24], 'w': {W}, 'bs': 8, 'max_len': 448, 'twin': 0.35, 'eos_id': 1}})\n"
    "print('v11b gemma row:', GEMMA_DIR, GEMMA_FINAL)\n")
nb['cells'][MC]['source'] = mc.replace(anchor, gemma + anchor).splitlines(True)

out = Path('nb/v11b')
out.mkdir(parents=True, exist_ok=True)
(out / 'ntl-v11b.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads((Path(src).parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
for k in ('id_no', 'keywords'):
    meta.pop(k, None)
meta.update(id='mdmuqtadirfuad/ntl-v11b', title='ntl-v11b', code_file='ntl-v11b.ipynb', is_private=True)
meta['kernel_sources'] = meta.get('kernel_sources', []) + ['mdmuqtadirfuad/ntl-gemma-ts-s2']
if 'google/gemma-4/Transformers/gemma-4-12b/2' not in meta.get('model_sources', []):
    meta['model_sources'] = meta.get('model_sources', []) + ['google/gemma-4/Transformers/gemma-4-12b/2']
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('written nb/v11b/ntl-v11b.ipynb; gemma weight', W)
