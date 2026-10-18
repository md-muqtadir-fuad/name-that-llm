# Name That LLM - backbone screening (E047, model performance study). Frozen base LMs, no training.
# For every candidate attached as a Kaggle model source and for all train passages:
#   * mean-pooled hidden states at 7 relative depths (0.1 ... 1.0), BOS excluded, fp16
#   * LM statistics from the model's own next-token distribution (full models only): mean / std token log-prob,
#     mean entropy, Fast-DetectGPT-style curvature (l - E_p[log p]) / sd_p[log p], rank-1 / <=5 / <=20 / <=100 shares,
#     tokens per word (tokenizer fertility)
#   * timing (load, seconds per passage) and peak memory
# Output: /kaggle/working/feat_<name>.npz + screen_report.json. Evaluated locally by experiments/backbones/screen_eval.py.
import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL = not os.path.exists('/kaggle/input')
SMOKE = LOCAL                      # local code-path test on a tiny model and 24 passages
OUT = 'smoke_out' if LOCAL else '/kaggle/working'
os.makedirs(OUT, exist_ok=True)
DEPTHS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
MAXLEN, BS = 512, 4
# (name, config.json glob, load in 8 bit, has a usable LM head)
CANDS = [('qwen3-8b', '**/qwen-3/transformers/8b-base/*/config.json', False, True),
         ('gemma4-12b', '**/gemma-4/transformers/gemma-4-12b/*/config.json', False, True),
         ('llama31-8b', '**/llama-3.1/transformers/8b/*/config.json', False, True),
         ('qwen3-14b', '**/qwen-3/transformers/14b-base/*/config.json', True, True),
         ('olmo2-7b', '**/olmo-2/transformers/1124-7b/*/config.json', False, True),
         ('olmo2-13b', '**/olmo-2/transformers/1124-13b/*/config.json', True, True),
         ('mistral-7b', '**/mistral/transformers/7b-v0.1-hf/*/config.json', False, True)]
if LOCAL:
    CANDS = [('qwen3-0.6b-smoke', '../../research/ext_data/hf/Qwen3-0.6B-Base/config.json', False, True)]
hits = glob.glob('/kaggle/input/**/train.csv', recursive=True) if not LOCAL else ['../../data/train.csv']
tr = pd.read_csv([h for h in hits if 'name-that-llm' in h or LOCAL][0])
if SMOKE:
    tr = tr.iloc[:24]
TX = tr.text.astype(str).tolist()
NW = np.array([len(t.split()) for t in TX])
NGPU = torch.cuda.device_count()
print('passages', len(TX), 'gpus', NGPU, flush=True)
report = {}


def find(pattern):
    if LOCAL:
        return os.path.dirname(pattern) if os.path.exists(pattern) else None
    h = sorted(glob.glob('/kaggle/input/' + pattern, recursive=True))
    return os.path.dirname(h[0]) if h else None


def load(path, eight_bit):
    kw = dict(dtype=torch.float16 if NGPU else torch.float32)
    if NGPU:
        kw['device_map'] = 'auto'
        kw['max_memory'] = {i: '14GiB' for i in range(NGPU)}
        if eight_bit:
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(path, **kw).eval()


for name, pat, eight, has_lm in CANDS:
    path = find(pat)
    if path is None:
        print(f'{name}: not attached, skipped', flush=True)
        report[name] = dict(status='not attached')
        continue
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        model = load(path, eight)
        t_load = time.time() - t0
        enc = [tok(t, truncation=True, max_length=MAXLEN).input_ids for t in TX]
        bos = tok.bos_token_id
        n_layers = model.config.get_text_config().num_hidden_layers if hasattr(model.config, 'get_text_config') else model.config.num_hidden_layers
        layers = sorted({max(1, int(round(d * n_layers))) for d in DEPTHS})
        feats, lm = None, np.full((len(TX), 9), np.nan, np.float32)
        order = np.argsort([len(e) for e in enc])
        t1 = time.time()
        dev = next(model.parameters()).device
        with torch.inference_mode():
            for k in range(0, len(order), BS):
                idx = order[k:k + BS]
                n = max(len(enc[i]) for i in idx)
                ids = torch.full((len(idx), n), tok.pad_token_id if tok.pad_token_id is not None else 0)
                att = torch.zeros((len(idx), n), dtype=torch.long)
                for r, i in enumerate(idx):
                    ids[r, :len(enc[i])] = torch.tensor(enc[i])
                    att[r, :len(enc[i])] = 1
                out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
                hs = out.hidden_states
                if feats is None:
                    feats = np.zeros((len(TX), len(layers), hs[0].shape[-1]), np.float16)
                m = att.clone()
                if bos is not None:
                    m[(ids == bos)] = 0
                mf = m.unsqueeze(-1).float()
                for j, li in enumerate(layers):
                    h = hs[li].float().cpu()
                    h = torch.nan_to_num(h, nan=0.0, posinf=6e4, neginf=-6e4)
                    feats[idx, j] = ((h * mf).sum(1) / mf.sum(1).clamp(min=1)).numpy().astype(np.float16)
                if has_lm:
                    lg = out.logits[:, :-1].float()
                    lp = torch.log_softmax(lg, -1)
                    tgt = ids[:, 1:].to(lp.device)
                    lt = lp.gather(-1, tgt[..., None])[..., 0]
                    p = lp.exp()
                    ent = -(p * lp).sum(-1)
                    var = (p * lp * lp).sum(-1) - ent * ent
                    curv = (lt + ent) / var.clamp(min=1e-6).sqrt()
                    rank = (lp > lt[..., None]).sum(-1)
                    vm = att[:, 1:].to(lp.device).bool()
                    for r, i in enumerate(idx):
                        v = vm[r]
                        a, e_, c_, rk = lt[r][v], ent[r][v], curv[r][v], rank[r][v].float()
                        lm[i] = [a.mean().item(), a.std().item(), e_.mean().item(), c_.mean().item(),
                                 (rk == 0).float().mean().item(), (rk < 5).float().mean().item(),
                                 (rk < 20).float().mean().item(), (rk < 100).float().mean().item(), len(enc[i]) / max(NW[i], 1)]
                    del lg, lp, p
                del out, hs
        t_fwd = time.time() - t1
        mem = max((torch.cuda.max_memory_allocated(i) for i in range(NGPU)), default=0) / 2 ** 30
        np.savez_compressed(f'{OUT}/feat_{name}.npz', feats=feats, layers=np.array(layers), n_layers=n_layers, ids=tr.id.values,
                            lm=lm, lm_names=np.array(['lp_mean', 'lp_std', 'entropy_mean', 'curvature_mean', 'rank1', 'rank_le5',
                                                      'rank_le20', 'rank_le100', 'tokens_per_word']))
        report[name] = dict(status='ok', path=path, eight_bit=eight, n_layers=int(n_layers), layers=layers, hidden=int(feats.shape[-1]),
                            load_s=round(t_load), forward_s=round(t_fwd), s_per_passage=round(t_fwd / len(TX), 4),
                            peak_gb=round(mem, 2), nan_rows=int(np.isnan(lm[:, 0]).sum()) if has_lm else None,
                            tokens_per_word=round(float(np.nanmean(lm[:, 8])), 3))
        print(name, report[name], flush=True)
    except Exception as e:   # one failing backbone must not stop the study
        report[name] = dict(status='error', error=repr(e)[:400])
        print(name, 'ERROR', repr(e)[:400], flush=True)
    finally:
        for v in ('model', 'out', 'hs'):
            globals().pop(v, None)
        gc.collect()
        if NGPU:
            torch.cuda.empty_cache()
            for i in range(NGPU):
                torch.cuda.reset_peak_memory_stats(i)
json.dump(report, open(f'{OUT}/screen_report.json', 'w'), indent=1)
print('done', flush=True)
