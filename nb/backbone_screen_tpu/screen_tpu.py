# Name That LLM - backbone screening on TPU (Kaggle TPU v3-8, PyTorch/XLA SPMD). Port of nb/backbone_screen/screen.py:
# same features and output format (feat_<name>.npz + screen_report.json), read by experiments/backbones/screen_eval.py.
# Differences: bf16 instead of fp16 / 8-bit; every 2-D weight sharded over the 8 TPU cores along dim 0 (so 13-14B
# models fit unquantised); fixed shapes (batch 8 x 512 tokens) -> one XLA compilation per model; LM statistics
# computed vectorised on device. Off-TPU (no torch_xla) it runs the same code on CPU for a local smoke test.
# Run plan (validation of the port): qwen3-0.6b smoke (64 passages) -> qwen3-8b (parity vs the GPU run) -> qwen3-14b bf16.
import gc
import glob
import json
import os
import time

import importlib.metadata as _im

if os.path.exists('/kaggle/input'):   # TPU image may ship an older transformers (Qwen3 needs >= 4.51); internet is on
    try:
        _v = tuple(int(x) for x in _im.version('transformers').split('.')[:2])
    except Exception:
        _v = (0, 0)
    if _v < (4, 51):
        os.system('pip install -q "transformers>=4.51,<5" accelerate')
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    xr.use_spmd()
    XLA = True
except Exception as e:   # local smoke test
    print('torch_xla not available -> CPU path:', repr(e)[:120])
    XLA = False
LOCAL = not os.path.exists('/kaggle/input')
OUT = 'smoke_out' if LOCAL else '/kaggle/working'
os.makedirs(OUT, exist_ok=True)
DEPTHS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
MAXLEN, BS = 512, 8
CANDS = [('qwen3-0.6b-smoke', '**/qwen-3/transformers/0.6b-base/*/config.json', 64),
         ('qwen3-8b-tpu', '**/qwen-3/transformers/8b-base/*/config.json', None),       # parity vs the GPU run
         ('qwen3-14b', '**/qwen-3/transformers/14b-base/*/config.json', None),         # GPU run needed bitsandbytes
         ('olmo2-13b', '**/olmo-2/transformers/1124-13b/*/config.json', None),         # GPU run needed bitsandbytes
         ('gemma4-12b', '**/gemma-4/transformers/gemma-4-12b/*/config.json', None),    # GPU run: OOM (262k vocab)
         ('llama31-8b', '**/llama-3.1/transformers/8b/*/config.json', None),           # needs Meta licence accepted
         ('mistral-7b', '**/mistral/*/7b-v0.1-hf/*/config.json', None)]
if LOCAL:
    CANDS = [('qwen3-0.6b-smoke', '../../research/ext_data/hf/Qwen3-0.6B-Base/config.json', 4)]
    BS = 2   # local CPU check only; the TPU run uses 8
hits = ['../../data/train.csv'] if LOCAL else glob.glob('/kaggle/input/**/train.csv', recursive=True)
TR = pd.read_csv([h for h in hits if 'name-that-llm' in h or LOCAL][0])
if XLA:
    NDEV = xr.global_runtime_device_count()
    MESH = xs.Mesh(np.arange(NDEV), (NDEV, 1), ('model', 'data'))
    DEV = xm.xla_device()
    print('XLA devices', NDEV, xm.xla_device_hw(DEV), flush=True)
else:
    NDEV, DEV = 1, torch.device('cpu')
report = {}


def find(pattern):
    if LOCAL:
        return os.path.dirname(pattern) if os.path.exists(pattern) else None
    h = sorted(glob.glob('/kaggle/input/' + pattern, recursive=True))
    return os.path.dirname(h[0]) if h else None


def load_sharded(path):
    """bf16 on host, then move parameter by parameter to the XLA device and shard dim 0 over all cores (peak device
    memory ~ model/NDEV + one replicated parameter)."""
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if XLA else torch.float32, low_cpu_mem_usage=True).eval()
    if not XLA:
        return model
    k = 0
    for mod in model.modules():
        for pname, p in list(mod.named_parameters(recurse=False)):
            t = p.data.to(DEV)
            if t.dim() == 2 and t.shape[0] % NDEV == 0:
                xs.mark_sharding(t, MESH, ('model', None))
            mod._parameters[pname] = torch.nn.Parameter(t, requires_grad=False)
            k += 1
            if k % 16 == 0:
                xm.mark_step()
        for bname, b in list(mod.named_buffers(recurse=False)):
            mod._buffers[bname] = b.to(DEV)
    xm.mark_step()
    return model


for name, pat, nmax in CANDS:
    path = find(pat)
    if path is None:
        report[name] = dict(status='not attached')
        print(name, 'not attached', flush=True)
        continue
    tr = TR.iloc[:nmax] if nmax else TR
    TX = tr.text.astype(str).tolist()
    NW = np.array([len(t.split()) for t in TX])
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        model = load_sharded(path)
        t_load = time.time() - t0
        cfg = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
        n_layers = cfg.num_hidden_layers
        layers = sorted({max(1, int(round(d * n_layers))) for d in DEPTHS})
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        bos = tok.bos_token_id
        enc = [tok(t, truncation=True, max_length=MAXLEN).input_ids for t in TX]
        feats, lm = None, np.zeros((len(TX), 9), np.float32)
        t1, n_comp = time.time(), 0
        with torch.no_grad():
            for k in range(0, len(TX), BS):
                idx = list(range(k, min(k + BS, len(TX))))
                ids = torch.full((BS, MAXLEN), pad, dtype=torch.long)       # fixed shape: one compilation
                att = torch.zeros((BS, MAXLEN), dtype=torch.long)
                for r, i in enumerate(idx):
                    ids[r, :len(enc[i])] = torch.tensor(enc[i])
                    att[r, :len(enc[i])] = 1
                m = att.clone()
                if bos is not None:
                    m[ids == bos] = 0
                ids_d, att_d, m_d = ids.to(DEV), att.to(DEV), m.to(DEV).unsqueeze(-1).float()
                out = model(input_ids=ids_d, attention_mask=att_d, output_hidden_states=True)
                pooled = torch.stack([(out.hidden_states[li].float() * m_d).sum(1) / m_d.sum(1).clamp(min=1) for li in layers], 1)
                lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
                tgt = ids_d[:, 1:]
                lt = lp.gather(-1, tgt[..., None])[..., 0]
                p = lp.exp()
                ent = -(p * lp).sum(-1)
                var = (p * lp * lp).sum(-1) - ent * ent
                curv = (lt + ent) / var.clamp(min=1e-6).sqrt()
                rank = (lp > lt[..., None]).sum(-1).float()
                v = att_d[:, 1:].float()
                nv = v.sum(1).clamp(min=1)
                mean = lambda x: (x * v).sum(1) / nv
                mu = mean(lt)
                stats = torch.stack([mu, ((lt - mu[:, None]) ** 2 * v).sum(1).div((nv - 1).clamp(min=1)).sqrt(), mean(ent), mean(curv),
                                     mean((rank == 0).float()), mean((rank < 5).float()), mean((rank < 20).float()),
                                     mean((rank < 100).float())], 1)
                pooled_c, stats_c = pooled.cpu().numpy(), stats.cpu().numpy()      # .cpu() executes the graph
                n_comp += 1
                if feats is None:
                    feats = np.zeros((len(TX), len(layers), pooled_c.shape[-1]), np.float16)
                feats[idx] = pooled_c[:len(idx)].astype(np.float16)
                lm[idx, :8] = stats_c[:len(idx)]
                lm[idx, 8] = [len(enc[i]) / max(NW[i], 1) for i in idx]
                if k == 0:
                    print(f'{name}: first batch (compile) {time.time() - t1:.0f}s', flush=True)
        t_fwd = time.time() - t1
        np.savez_compressed(f'{OUT}/feat_{name}.npz', feats=feats, layers=np.array(layers), n_layers=n_layers, ids=tr.id.values,
                            lm=lm, lm_names=np.array(['lp_mean', 'lp_std', 'entropy_mean', 'curvature_mean', 'rank1', 'rank_le5',
                                                      'rank_le20', 'rank_le100', 'tokens_per_word']))
        report[name] = dict(status='ok', path=path, device='tpu' if XLA else 'cpu', n_dev=NDEV, n_layers=int(n_layers), layers=layers,
                            hidden=int(feats.shape[-1]), passages=len(TX), load_s=round(t_load), forward_s=round(t_fwd),
                            s_per_passage=round(t_fwd / len(TX), 4), nan_feats=int(np.isnan(feats.astype(np.float32)).sum()),
                            tokens_per_word=round(float(lm[:, 8].mean()), 3))
        print(name, report[name], flush=True)
        if name.endswith('smoke') and report[name]['nan_feats']:
            raise SystemExit('smoke check failed: NaN features')
    except SystemExit:
        raise
    except Exception as e:
        report[name] = dict(status='error', error=repr(e)[:500])
        print(name, 'ERROR', repr(e)[:500], flush=True)
        if name.endswith('smoke'):
            json.dump(report, open(f'{OUT}/screen_report_tpu.json', 'w'), indent=1)
            raise SystemExit('smoke check failed - stopping before the big models')
    finally:
        globals().pop('model', None)
        gc.collect()
json.dump(report, open(f'{OUT}/screen_report_tpu.json', 'w'), indent=1)
print('done', flush=True)
