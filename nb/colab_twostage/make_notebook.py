"""Build nb/colab_twostage/twostage_colab.ipynb (embeds twostage_train.py). Run from the repo root after build_trainer.py.
The notebook runs on a Colab GPU runtime (VS Code Colab extension, colab.research.google.com, or the Colab CLI)."""
import json
from pathlib import Path

trainer = Path('nb/colab_twostage/twostage_train.py').read_text(encoding='utf-8')
md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(True)}
code = lambda s: {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.strip('\n').splitlines(True)}
cells = [
    md('''
# Two-stage LoRA (OLMo-2-7B, 24 layers) on Colab - Name That LLM
Stage 1 trains on the public pool (`nishantdahal/ntl-two-stage-lora` `pool_v1.parquet`, 34,940 passages) in 3 resumable
parts; stage 2 fine-tunes on host train for fold0-2 + final. Outputs go to Google Drive when it can be mounted
(otherwise `/content`, lost on disconnect), then to a private Kaggle dataset.

**Run:** Runtime = GPU (T4 or better). Run all cells. After a disconnect, run all cells again: finished jobs are skipped.
**Secrets:** type your Kaggle username and API key when asked (input boxes at the top of VS Code).
Never print them.
'''),
    code('''
import os, subprocess, sys
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout)
!pip install -q -U "transformers>=4.48" peft accelerate safetensors huggingface_hub kaggle
'''),
    code('''
import getpass, json, os
from pathlib import Path
# plain prompts (Colab secrets block in VS Code); the input boxes appear at the TOP of the VS Code window
_u = input('Kaggle username: ').strip()
_k = getpass.getpass('Kaggle API key: ').strip()
Path('~/.kaggle').expanduser().mkdir(exist_ok=True)
_p = Path('~/.kaggle/kaggle.json').expanduser()
_p.write_text(json.dumps({'username': _u, 'key': _k}))
_p.chmod(0o600)
del _u, _k
print('kaggle credentials written')
'''),
    code('''
WORK = '/content/ntl_twostage'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/ntl_twostage'
except Exception as e:
    print('Drive not mounted (outputs are lost on disconnect):', repr(e)[:200])
os.makedirs(WORK, exist_ok=True)
os.environ.update(NTL_WORK=WORK, NTL_DATA='/content/data', NTL_MODEL_DIR='/content/models/olmo2-7b')
print('work dir', WORK)
'''),
    code('''
!mkdir -p /content/data
!kaggle competitions download -c name-that-llm -f train.csv -p /content/data -q
!kaggle datasets download -d nishantdahal/ntl-two-stage-lora -f pool_v1.parquet -p /content/data -q
!cd /content/data && for z in *.zip; do [ -f "$z" ] && unzip -o -q "$z" && rm "$z"; done; ls -la /content/data
'''),
    code('''
from huggingface_hub import snapshot_download
snapshot_download('allenai/OLMo-2-1124-7B', local_dir='/content/models/olmo2-7b',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*'])
!ls /content/models/olmo2-7b | head
'''),
    code('%%writefile /content/twostage_train.py\n' + trainer),
    md('## Smoke test (about 5-10 min): tiny stage1a + fold0, separate work dir; prints memory, sec/row and a full-run estimate'),
    code('''
!cd /content && NTL_WORK=/content/ntl_smoke python twostage_train.py --smoke
'''),
    md('## Full run (resumable): stage1a, stage1b, stage1c, fold0, fold1, fold2, final'),
    code('''
!cd /content && python twostage_train.py
'''),
    md('## Upload the stage-2 checkpoints + fold outputs to a private Kaggle dataset'),
    code('''
import shutil
src = Path(WORK) / 'lora_twostage'
dst = Path('/content/upload/ntl-olmo2-twostage')
if dst.exists():
    shutil.rmtree(dst)
for job in ('fold0', 'fold1', 'fold2', 'final'):
    if (src / job / 'COMPLETE.json').exists():
        shutil.copytree(src / job, dst / 'lora_twostage' / job, ignore=shutil.ignore_patterns('lora_state.pt'))
for f in ('run_config.json', 'folds.csv', 'summary_run.json'):
    if (src / f).exists():
        shutil.copy(src / f, dst / 'lora_twostage' / f)
user = json.loads(Path('~/.kaggle/kaggle.json').expanduser().read_text())['username']
meta = {'title': 'ntl-olmo2-twostage', 'id': f'{user}/ntl-olmo2-twostage', 'licenses': [{'name': 'CC-BY-SA-4.0'}]}
(dst / 'dataset-metadata.json').write_text(json.dumps(meta))
r = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(dst), '-m', 'colab two-stage run', '--dir-mode', 'zip'], capture_output=True, text=True)
if 'not found' in (r.stdout + r.stderr).lower() or r.returncode:
    r = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(dst), '--dir-mode', 'zip'], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
'''),
]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'T4'},
                                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 5}
Path('nb/colab_twostage/twostage_colab.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('written nb/colab_twostage/twostage_colab.ipynb', len(cells), 'cells')
