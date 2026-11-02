"""Public version of v12 (user, 2026-11-02): same code as nb/v12/ntl-v12.ipynb; the base author's first-person intro (cell 1)
and our working note (cell 0) are replaced by a factual description that links the base, and one first-person line in
section 6 is made factual. Saved as public under the existing kernel (id_no) with a new title, without running.
Run from the repo root: python nb/v12/make_v12_public.py <id_no>"""
import json
import sys
from pathlib import Path

ID_NO = int(sys.argv[1])
TITLE = 'Name That LLM v12: 0.664 + Gemma two-stage (0.658)'
SLUG = 'name-that-llm-v12-0-664-gemma-two-stage-0-658'
nb = json.loads(Path('nb/v12/ntl-v12.ipynb').read_text(encoding='utf-8'))
src = lambda i: ''.join(nb['cells'][i]['source'])
assert src(0).startswith('# ntl-v12') and src(1).startswith('# 0.664 | Open Pool')
intro = ("# Name That LLM v12\n\n"
         "The public 0.664 notebook ([0.664 | One Swap: 198k Clean Pool, Fully Public]"
         "(https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public)) with one change: a "
         "Gemma-4-12B two-stage LoRA added to `infer.py`'s `MODELS` as one more LoRA row (plain weight 2, centred twin 2.8), "
         "plus a one-line LoRA-name fallback for Gemma-4 (its decoder sits under `language_model.`). Everything else is "
         "unchanged. Public LB 0.658.\n\n"
         "The Gemma adapter is read from a private kernel output, so the notebook does not run as is on other accounts.\n")
nb['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': intro.splitlines(True)}] + nb['cells'][2:]
i6 = next(i for i, c in enumerate(nb['cells']) if src(i).startswith('## 6. Members and run'))
s6 = src(i6)
old = "My one change to the numbers here: `--svc_w 12` (recipe: 1.5)."
assert s6.count(old) == 1
nb['cells'][i6]['source'] = s6.replace(old, "The 0.664 notebook sets `--svc_w 12` (recipe: 1.5). v12 appends the Gemma row to `MODELS`.").splitlines(True)
out = Path('nb/v12_public'); out.mkdir(exist_ok=True)
(out / 'ntl-v12.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
meta = json.loads(Path('nb/v12/kernel-metadata.json').read_text(encoding='utf-8'))
meta.update(id=f'mdmuqtadirfuad/{SLUG}', id_no=ID_NO, title=TITLE, is_private=False)
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
md = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown']
import re
print('written nb/v12_public;', len(nb['cells']), 'cells; first-person hits in markdown:',
      sum(len(re.findall(r"\b(I|I'm|I've|my|My)\b", m)) for m in md))
