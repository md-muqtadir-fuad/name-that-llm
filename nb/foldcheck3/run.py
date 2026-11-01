# Fold-split check v2 (per-row correctness saved) for the public LoRA24 v5 fold adapters (dhruvtibarewal/nlm-l24-adapters-v5, q8v5_f0..f4 on
# Qwen3-8B-Base, keep 24 layers, pool [2,4,6,8,12,16,20,24], max_len 384), reproducing the public infer.py l24 loader:
# merge W += B A * alpha/r, masked mean-pool (position 0 excluded) through the checkpoint's LayerNorms, plain head.
# Rows: 6 prompts x 18 answers from each of OUR folds (sklearn GroupKFold(5) on train.csv grouped by prompt_id, default
# order). If adapter f_j is clearly worst on our fold j for every j, their folds are ours and honest OOF is possible.
import glob, json, os, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
assert torch.cuda.is_available(), 'no GPU in this session'   # v2 stalled >2 h after loading (CPU fallback suspected)
print('gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
train = pd.read_csv(os.path.join(DATA, 'train.csv'), dtype=str)
LABELS = sorted(train.model.unique())
y = np.searchsorted(LABELS, train.model.to_numpy())
fold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k
rng = np.random.default_rng(0)
ps = rng.choice(sorted(train.prompt_id.unique()), 60, replace=False)   # v2: 60 random prompts, any split testable offline
rows = list(np.flatnonzero(train.prompt_id.isin(ps).to_numpy()))
rows = np.array(rows)
base = next(os.path.dirname(c) for c in glob.glob('/kaggle/input/**/config.json', recursive=True)
            if '8b-base' in c.lower() and 'qwen' in c.lower() and glob.glob(os.path.join(os.path.dirname(c), '*.safetensors')))
ads = sorted(d for d in glob.glob('/kaggle/input/**/q8v5_f[0-4]', recursive=True) if os.path.isdir(d))
print('base', base, '| adapters', ads, '| rows', len(rows), flush=True)
assert len(ads) == 5
L, KEEP, MAXLEN, BS = [2, 4, 6, 8, 12, 16, 20, 24], 24, 384, 16
tok = AutoTokenizer.from_pretrained(base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(base)
tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
tc.num_hidden_layers, tc.use_cache = KEEP, False
if getattr(tc, 'layer_types', None) is not None:
    tc.layer_types = tc.layer_types[:KEEP]
m = AutoModel.from_pretrained(base, config=cfg, dtype=torch.float16, device_map='auto').eval()
print('param devices', sorted({str(p.device) for p in m.parameters()}), f'({time.time() - T0:.0f}s)', flush=True)
assert all(p.device.type == 'cuda' for p in m.parameters()), 'weights not all on GPU'
mods = dict(m.named_modules())
first = next(m.parameters()).device
enc = [x for x in tok(list(train.text.iloc[rows]), truncation=True, max_length=MAXLEN)['input_ids']]
order = np.argsort([len(x) for x in enc])
acc = np.zeros((5, 5))
OK = np.zeros((5, len(rows)), dtype=bool)
with torch.no_grad():
    for ai, ad in enumerate(ads):
        adp = os.path.join(ad, 'adapter') if os.path.exists(os.path.join(ad, 'adapter', 'adapter_config.json')) else ad
        acfg = json.load(open(os.path.join(adp, 'adapter_config.json')))
        scale = acfg['lora_alpha'] / acfg['r']
        sd = load_file(os.path.join(adp, 'adapter_model.safetensors'))
        deltas = []
        for kk, v in sd.items():
            if '.lora_A.' not in kk:
                continue
            name = kk.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods:
                name = 'model.' + name
            W = mods[name].weight
            B = sd[kk.replace('.lora_A.', '.lora_B.')]
            W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
            deltas.append((W, B, v))
        H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu', weights_only=True)
        hid = H['norms.0.weight'].shape[0]
        norms = [torch.nn.LayerNorm(hid) for _ in L]
        for i, nm in enumerate(norms):
            nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        Z = np.zeros((len(enc), hid * len(L)), np.float32)
        for i in range(0, len(order), BS):
            if i % (BS * 20) == 0: print(f'  batch {i // BS} ({time.time() - T0:.0f}s)', flush=True)
            b = order[i:i + BS]
            Lm = max(len(enc[j]) for j in b)
            ids = torch.full((len(b), Lm), tok.pad_token_id)
            att = torch.zeros((len(b), Lm), dtype=torch.long)
            for r, j in enumerate(b):
                ids[r, :len(enc[j])] = torch.tensor(enc[j])
                att[r, :len(enc[j])] = 1
            hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states
            mask = att.unsqueeze(-1).float()
            mask[:, 0] = 0
            Z[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(L, norms)], -1).numpy()
        for W, B, A_ in deltas:
            W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)
        pred = (Z @ H['plain.weight'].float().numpy().T + H['plain.bias'].float().numpy()).argmax(1)
        ok = pred == y[rows]
        OK[ai] = ok
        for k in range(5):
            acc[ai, k] = ok[fold[rows] == k].mean()
        print(f'adapter {os.path.basename(ad)}: acc on our folds 0-4 = {np.round(acc[ai], 3).tolist()} ({time.time() - T0:.0f}s)', flush=True)
np.save('/kaggle/working/foldcheck_acc.npy', acc)
np.savez('/kaggle/working/foldcheck_rows.npz', ids=train.id.to_numpy(dtype=str)[rows], prompt_id=train.prompt_id.to_numpy(dtype=str)[rows], ok=OK, adapters=np.array([os.path.basename(a) for a in ads]))
held = acc.argmin(1)
print('held-out fold per adapter (argmin):', held.tolist(), '| matches our GroupKFold:', held.tolist() == list(range(5)), flush=True)
