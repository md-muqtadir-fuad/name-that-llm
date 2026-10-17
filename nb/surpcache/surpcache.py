# ntl-surprisal-cache: per-token surprisal profile of every TRAIN passage under frozen causal LMs (observers).
# Output (/kaggle/working/surp_<tag>.npz): lens (n,), lp (sum lens,) log p(actual token | prefix), ent (entropy of the
# predictive distribution), rank (rank of the actual token), plus timing, for the offline style-member experiments.
# No token ids / text are saved. Internet off; models attached from Kaggle Models.
import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
BUDGET_S = 45 * 60          # stop starting new observers after 45 min
SMOKE = os.environ.get('SURP_SMOKE')   # local logic test: tiny model dir, 20 rows, CPU
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm', 'data']
            if os.path.exists(p + '/train.csv'))
tr = pd.read_csv(DATA + '/train.csv')
if SMOKE:
    tr = tr.iloc[:20]
TEXTS = tr.text.fillna('').astype(str).tolist()
print('train', tr.shape, 'gpus', torch.cuda.device_count(), flush=True)

OUTDIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
OBS = [('q3_8b', 'qwen3', 4096), ('q25_7b', 'qwen2', 3584), ('q3_4b', 'qwen3', 2560)]


def find_model_dir(model_type, hidden):
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get('model_type') == model_type and c.get('hidden_size') == hidden:
            return os.path.dirname(cfg)
    return None


@torch.no_grad()
def run(tag, path, bs=8, max_len=448):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    if SMOKE:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map='auto').eval()
    print(tag, 'loaded', f'{time.time() - t0:.0f}s', flush=True)
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids('<|endoftext|>')
    pad = tok.pad_token_id if tok.pad_token_id is not None else start
    enc_all = [tok(t, add_special_tokens=False)['input_ids'][:max_len - 1] for t in TEXTS]
    order = np.argsort([len(e) for e in enc_all])
    res = [None] * len(TEXTS)
    dev = next(model.parameters()).device
    bad = 0
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        L = max(len(enc_all[i]) for i in idx) + 1
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            e = enc_all[i]
            ids[j, 0] = start
            ids[j, 1:len(e) + 1] = torch.tensor(e)
            att[j, :len(e) + 1] = 1
        ids, att = ids.to(dev), att.to(dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        if not torch.isfinite(logits[att.bool().to(logits.device)]).all():
            bad += 1
            model.to(torch.bfloat16)
            logits = model(input_ids=ids, attention_mask=att).logits
            model.to(torch.float16)
        for j, i in enumerate(idx):
            n = len(enc_all[i])
            lsm = torch.log_softmax(logits[j, :n].float(), -1)
            tgt = ids[j, 1:n + 1].to(lsm.device)
            lp = lsm.gather(-1, tgt[:, None])[:, 0]
            ent = -(lsm.exp() * lsm).sum(-1)
            rank = (lsm > lp[:, None]).sum(-1)
            res[i] = (lp.cpu().numpy().astype(np.float32), ent.cpu().numpy().astype(np.float32),
                      rank.cpu().numpy().astype(np.int32))
        del logits
        if b0 % 800 == 0:
            print(tag, b0, f'{time.time() - t0:.0f}s', flush=True)
    secs = time.time() - t0
    np.savez_compressed(f'{OUTDIR}/surp_{tag}.npz', lens=np.array([len(r[0]) for r in res]),
                        lp=np.concatenate([r[0] for r in res]), ent=np.concatenate([r[1] for r in res]),
                        rank=np.concatenate([r[2] for r in res]), seconds=np.array(secs), bad=np.array(bad))
    print(tag, 'done', f'{secs:.0f}s', 'bf16 redo batches', bad, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


for tag, mt, hid in (OBS[:1] if SMOKE else OBS):
    if time.time() - T0 > BUDGET_S:
        print('budget reached, skipping', tag, flush=True)
        continue
    p = SMOKE if SMOKE else find_model_dir(mt, hid)
    print(tag, p, flush=True)
    if p is None:
        continue
    run(tag, p)
print('total', f'{time.time() - T0:.0f}s', flush=True)
