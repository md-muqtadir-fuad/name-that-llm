# Feature cache for offline (CPU) experiments: frozen Qwen3-8B-Base, mean-pooled hidden states (first token = BOS
# excluded from nothing; same pooling as the 0.541 notebook's 'mean' heads) of several layers for all train passages.
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

LAYERS = [4, 7, 12, 16, 22]
T0 = time.time()
train = pd.read_csv(next(p for p in ['/kaggle/input/competitions/name-that-llm/train.csv',
                                     '/kaggle/input/name-that-llm/train.csv'] if os.path.exists(p)))
TR = train['text'].tolist()


def find_model_dir(model_type, hidden):
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get('model_type') == model_type and c.get('hidden_size') == hidden:
            return os.path.dirname(cfg)


path = find_model_dir('qwen3', 4096)
print('model', path, flush=True)
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = 'right'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(path)
cfg.num_hidden_layers = max(LAYERS) + 1
if getattr(cfg, 'layer_types', None):
    cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
model = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()


@torch.no_grad()
def embed(texts, bs=8, max_len=512):
    H = 4096
    out = np.zeros((len(texts), len(LAYERS), H), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    dev = next(model.parameters()).device
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        batch = [(tok.bos_token + texts[i]) if tok.bos_token else texts[i] for i in idx]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[L]).all() for L in LAYERS):
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].float().unsqueeze(-1)
        for li, L in enumerate(LAYERS):
            h = hs[L].float()
            out[idx, li] = ((h * m.to(h.device)).sum(1) / m.sum(1).clamp(min=1).to(h.device)).cpu().numpy()
        if b0 % 400 == 0:
            print(b0, f'{time.time() - T0:.0f}s', flush=True)
    return out


E = embed(TR)
assert np.isfinite(E.astype(np.float32)).all()
np.savez_compressed('qwen3_8b_train_feats.npz', E=E, layers=np.array(LAYERS), ids=train['id'].values.astype(str))
print('saved', E.shape, f'{time.time() - T0:.0f}s')
