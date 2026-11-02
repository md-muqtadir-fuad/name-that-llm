# E072 layer-by-layer screen (GPU part): frozen Phi-4-14B, OLMo-2-13B and Qwen3-8B-Base (reference: the backbone of the
# public LoRA members). For every train passage and EVERY layer (embeddings + each block): masked mean of hidden states
# (BOS position excluded when the tokenizer prepends one), max 384 tokens. 13-14B models in 8-bit (bitsandbytes), the
# 8B in fp16 split over the two T4s. Output: /kaggle/working/feat_<name>.npy (n_layers+1, n_rows, hidden) fp16 + meta.
# v2: each model runs in its own process (the GPU is fully freed on exit; v1 ran out of memory loading Phi-4 after
# OLMo-2) and a failed model does not fail the run (Kaggle discards all outputs of an ERROR run).
# Phi-4 is not an official Kaggle model: downloaded from Hugging Face (microsoft/phi-4, MIT) with internet on.
import json, subprocess, sys, time
import numpy as np, pandas as pd, glob, os, torch
assert torch.cuda.is_available(), 'no GPU in this session'
print('gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes>=0.46.1'], check=True)

ONE = r'''
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
T0 = time.time()
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
TX = pd.read_csv(os.path.join(DATA, 'train.csv')).text.fillna('').astype(str).tolist()
def kaggle_dir(pattern):
    h = sorted(glob.glob('/kaggle/input/' + pattern, recursive=True)); assert h, pattern; return os.path.dirname(h[0])
def phi4():
    from huggingface_hub import snapshot_download
    return snapshot_download('microsoft/phi-4', local_dir='/tmp/phi4', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'])
CANDS = {'qwen3-8b': (lambda: kaggle_dir('**/qwen-3/transformers/8b-base/*/config.json'), False, 16),
         'olmo2-13b': (lambda: kaggle_dir('**/olmo-2/transformers/1124-13b/*/config.json'), True, 8),
         'phi4-14b': (phi4, True, 8)}
name = sys.argv[1]; where, eight, bs = CANDS[name]
path = where(); log(name, path)
tok = AutoTokenizer.from_pretrained(path)
if tok.pad_token is None: tok.pad_token = tok.eos_token
kw = dict(device_map='auto', max_memory={0: '13GiB', 1: '13GiB'})
if eight: kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
else: kw['dtype'] = torch.float16
m = AutoModel.from_pretrained(path, **kw).eval()
log(name, 'loaded', sorted({str(p.device) for p in m.parameters()}))
enc = tok(TX, truncation=True, max_length=384)['input_ids']
bos = tok.bos_token_id is not None and all(len(e) and e[0] == tok.bos_token_id for e in enc[:20])
order = np.argsort([len(e) for e in enc]); first = next(m.parameters()).device
F = None
with torch.no_grad():
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; Lm = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
        for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states
        mask = att.unsqueeze(-1).float()
        if bos: mask[:, 0] = 0
        if F is None:
            F = np.zeros((len(hs), len(TX), hs[0].shape[-1]), np.float16)
            log(name, 'layers', len(hs) - 1, 'hidden', hs[0].shape[-1], 'bos masked', bos)
        for l, h in enumerate(hs):
            mk = mask.to(h.device)
            F[l, b] = ((h.float() * mk).sum(1) / mk.sum(1)).cpu().numpy().astype(np.float16)
        if i % (bs * 40) == 0: log(f'  {name} {i}/{len(TX)}')
assert np.isfinite(F).all(), name
np.save(f'/kaggle/working/feat_{name}.npy', F)
json.dump(dict(path=str(path), layers=int(F.shape[0] - 1), hidden=int(F.shape[2]), eight_bit=eight, bos_masked=bool(bos),
               seconds=round(time.time() - T0)), open(f'/kaggle/working/meta_{name}.json', 'w'), indent=1)
log(name, 'saved')
'''
open('/kaggle/working/one_model.py', 'w').write(ONE)
T0 = time.time()
status = {}
for name in ('qwen3-8b', 'olmo2-13b', 'phi4-14b'):
    r = subprocess.run([sys.executable, '/kaggle/working/one_model.py', name])
    status[name] = r.returncode
    print(f'[{time.time() - T0:6.0f}s] {name} exit code {r.returncode}', flush=True)
    subprocess.run(['rm', '-rf', '/tmp/phi4'])
json.dump(status, open('/kaggle/working/status.json', 'w'))
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
np.save('/kaggle/working/ids.npy', pd.read_csv(os.path.join(DATA, 'train.csv')).id.astype(str).to_numpy())
os.remove('/kaggle/working/one_model.py')
print('done', status, flush=True)
