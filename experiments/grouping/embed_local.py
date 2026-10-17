"""CPU sentence-embedding cache for the grouping study (all universe passages; L2-normalised, fp16).
usage: python embed_local.py name[,name...]     (env NT = torch threads, default 3)
'q06b' = Qwen3-0.6B-Base mean-pooled hidden states of layers 10/17/24 (a CPU-feasible LLM-layer proxy for the v1
notebook's GLM-4-9B layer-24 signal, which needs a GPU), stored un-normalised as keys L10/L17/L24."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import FEATS, ROOT

MODELS = {   # name: (HF id, prompt, max_seq_length, trust_remote_code)
    'minilm': ('sentence-transformers/all-MiniLM-L6-v2', '', 256, False),
    'mpnet': ('sentence-transformers/all-mpnet-base-v2', '', 384, False),
    'bge_s': ('BAAI/bge-small-en-v1.5', '', 512, False),
    'bge_b': ('BAAI/bge-base-en-v1.5', '', 512, False),
    'e5_b': ('intfloat/e5-base-v2', 'query: ', 512, False),
    'gte_b': ('thenlper/gte-base', '', 512, False),
    'nomic': ('nomic-ai/nomic-embed-text-v1.5', 'clustering: ', 512, True),
    # Qwen3-Embedding: instruction-prefixed query side (last-token pooling, handled by sentence-transformers)
    'q3e06': ('Qwen/Qwen3-Embedding-0.6B',
              'Instruct: Identify the question or writing prompt that this passage was written in response to\nQuery: ',
              512, False),
}
torch.set_num_threads(int(os.environ.get('NT', 3)))
U = pd.read_parquet(FEATS / 'universe.parquet')
texts = U['text'].tolist()
order = np.argsort([len(t) for t in texts])   # length-sorted batches: far less padding on CPU


@torch.no_grad()
def qwen_base_layers(layers=(10, 17, 24), bs=16, max_len=512):
    from transformers import AutoModel, AutoTokenizer
    path = ROOT / 'research/ext_data/hf/Qwen3-0.6B-Base'
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    m.layers = m.layers[:max(layers)]      # hidden_states[L] needs the first L blocks only
    m.config.num_hidden_layers = max(layers)
    if getattr(m.config, 'layer_types', None):
        m.config.layer_types = m.config.layer_types[:max(layers)]
    m.eval()
    H = m.config.hidden_size
    out = {L: np.zeros((len(texts), H), dtype=np.float32) for L in layers}
    t0 = time.time()
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        hs = m(**enc, output_hidden_states=True).hidden_states
        msk = enc['attention_mask'].float().unsqueeze(-1)
        for L in layers:
            out[L][idx] = ((hs[L] * msk).sum(1) / msk.sum(1).clamp(min=1)).numpy()
        if b0 % (bs * 50) == 0:
            print(f'  q06b {b0}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
    return {f'L{L}': v.astype(np.float16) for L, v in out.items()}


for name in sys.argv[1].split(','):
    out = FEATS / f'st_{name}.npz'
    if out.exists():
        continue
    t0 = time.time()
    if name == 'q06b':
        np.savez_compressed(out, uid=U['uid'].values.astype(str), **qwen_base_layers())
        print(name, f'{time.time() - t0:.0f}s', flush=True)
        continue
    from sentence_transformers import SentenceTransformer
    hf, pr, ml, trc = MODELS[name]
    m = SentenceTransformer(hf, device='cpu', trust_remote_code=trc)
    m.max_seq_length = ml
    E = m.encode([texts[i] for i in order], batch_size=16, normalize_embeddings=True, prompt=pr or None,
                 show_progress_bar=False, convert_to_numpy=True)
    X = np.empty_like(E)
    X[order] = E
    np.savez_compressed(out, uid=U['uid'].values.astype(str), emb=X.astype(np.float16))
    print(name, X.shape, f'{time.time() - t0:.0f}s', flush=True)
    del m
