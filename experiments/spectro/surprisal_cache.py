"""Per-token surprisal (nats) of every train passage under a small local base LM (Qwen3-0.6B-Base, CPU, fp32).
Output: research/ext_data/spectro/surprisal_qwen3_0.6b.npz  (git-ignored) with ids, token ids and surprisal per passage
(first token has no context -> NaN). Used by experiments/spectro/spectro.py for the token-surprisal spectrogram."""
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
LM = ROOT / 'research/ext_data/hf/Qwen3-0.6B-Base'
OUT = ROOT / 'research/ext_data/spectro'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(LM)
model = AutoModelForCausalLM.from_pretrained(LM, dtype=torch.float32).eval()
tr = pd.read_csv(ROOT / 'data/train.csv')
enc = [tok(t, add_special_tokens=False).input_ids for t in tr.text]
order = np.argsort([len(e) for e in enc])
S = [None] * len(enc)
T0 = time.time()
BS = 2
with torch.inference_mode():
    for k in range(0, len(order), BS):
        idx = order[k:k + BS]
        n = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), n), tok.pad_token_id or 0)
        att = torch.zeros((len(idx), n), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(enc[i])] = torch.tensor(enc[i])
            att[r, :len(enc[i])] = 1
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
        s = -lp.gather(-1, ids[:, 1:, None])[..., 0]
        for r, i in enumerate(idx):
            S[i] = np.concatenate([[np.nan], s[r, :len(enc[i]) - 1].numpy()]).astype(np.float32)
        if k % 200 == 0:
            print(f'{k}/{len(order)} {time.time() - T0:.0f}s', flush=True)
np.savez_compressed(OUT / 'surprisal_qwen3_0.6b.npz', ids=tr.id.values,
                    tokens=np.array([np.array(e, dtype=np.int32) for e in enc], dtype=object),
                    surprisal=np.array(S, dtype=object))
print('done', round(time.time() - T0), 's')
