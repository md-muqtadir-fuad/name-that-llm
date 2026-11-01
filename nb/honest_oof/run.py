# Honest per-row OOF of the public fold-averaged LoRA24 members of the 0.664 stack (dhruvtibarewal/nlm-l24-adapters-v5:
# q14v5, q8v5, ll8v5). Each train row is read only by the adapter f_k whose training excluded it, k = the row's fold in
# the split the adapters were trained on (infer.py --sim split: RandomState(0) shuffle of sorted prompt ids, fold = i % 5;
# checked by ntl-foldcheck2). Loader = public infer.py l24_predict (merge W += B A * alpha/r, masked mean-pool without
# position 0, per-layer LayerNorm, linear heads). Every row is featurized by every adapter, so the log also shows
# in-sample vs held-out accuracy per adapter (a second split check, now for all three members).
# Saved per member (rows in train.csv order):
#   plain  = Z @ Wp.T + bp of the honest adapter (18 logits)
#   rc, bc = Z @ Wc.T and bc of the honest adapter: centred head on any grouping g is rc - mean_g(rc) + bc (exact, linear)
#   U, m, lam = the shared-prompt Gaussian head (gauss.py, the 0.664 cfg d=128) fit on the adapter's training rows, U for
#               the held-out rows; offline gauss.scores(None, groups, prm, U=U) with T1 = Tg = 3 as in infer.py
import gc, glob, json, os, sys, time
import numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)
assert torch.cuda.is_available(), 'no GPU in this session'   # out of weekly quota, Kaggle runs "GPU" kernels on CPU
log('gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
torch.set_grad_enabled(False)   # in-place LoRA merge into leaf weights (worker threads use their own no_grad)
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
GAUSS_SRC = None   # gauss.py from the 0.664 notebook (Apache-2.0), embedded by build.py
open('/kaggle/working/gauss.py', 'w').write(GAUSS_SRC); sys.path.insert(0, '/kaggle/working')
import gauss as ga
train = pd.read_csv(os.path.join(DATA, 'train.csv'))
LABELS = sorted(train.model.unique()); y = train.model.map({l: i for i, l in enumerate(LABELS)}).values
rng = np.random.RandomState(0); pids = np.array(sorted(train.prompt_id.unique())); rng.shuffle(pids)
fold = train.prompt_id.map({q: i % 5 for i, q in enumerate(pids)}).values
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
def mdir(*parts):
    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts) and glob.glob(os.path.join(os.path.dirname(x), '*.safetensors'))]
    assert c, parts; return c[0]
def adir(name):
    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True); assert h, name; return os.path.dirname(h[0])
L = [2, 4, 6, 8, 12, 16, 20, 24]
MEMBERS = [dict(name='q8v5', path=mdir('/qwen-3/', '/8b-base/'), keep=24, bs=16, dp=True, gauss=True),
           dict(name='ll8v5', path=mdir('/llama-3-1-8b-base/'), keep=24, bs=16, dp=True, gauss=False),
           dict(name='q14v5', path=mdir('/qwen-3/', '/14b-base/'), keep=27, bs=8, dp=False, gauss=True)]
RUN = None   # member names to run (set by build.py); v2 output already has q8v5 and ll8v5
MEMBERS = [m for m in MEMBERS if RUN is None or m['name'] in RUN]
def gpu_mem(): return ' '.join(f'gpu{i}:{torch.cuda.memory_allocated(i) / 1e9:.1f}GB' for i in range(torch.cuda.device_count()))
GCFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
OUT = {'ids': np.array(train.id.astype(str).tolist()), 'fold': fold, 'y': y, 'labels': np.array(LABELS)}

def lsm(z): return z - z.max(1, keepdims=True) - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
for spec in MEMBERS:
    log('member', spec['name'], spec['path'])
    tok = AutoTokenizer.from_pretrained(spec['path'])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(spec['path']); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = spec['keep']; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None: tc.layer_types = tc.layer_types[:spec['keep']]
    ms = []   # as infer.py: a full copy per GPU if the largest batch fits with 1 GiB to spare, else one split copy
    if spec['dp'] and torch.cuda.device_count() >= 2:
        try:
            for d in range(2):
                ms.append(AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map={'': d}).eval())
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(d)
                with torch.no_grad():
                    ms[-1](input_ids=torch.full((spec['bs'], 384), tok.pad_token_id, device=f'cuda:{d}'), output_hidden_states=True, use_cache=False)
                assert torch.cuda.max_memory_allocated(d) < torch.cuda.get_device_properties(d).total_memory - (1 << 30), 'under 1 GiB to spare'
        except Exception as e:
            log(f'  one split copy instead of one per GPU ({e!r:.80})'); ms = []
        gc.collect(); torch.cuda.empty_cache()
    if not ms:
        ms = [AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()]
    mods = [dict(m.named_modules()) for m in ms]
    enc = tok(list(train.text.values), truncation=True, max_length=384)['input_ids']
    order = np.argsort([len(x) for x in enc]); bs = spec['bs']
    log('  param devices', sorted({str(p.device) for m in ms for p in m.parameters()}))
    plain, rc, bc = np.zeros((len(y), 18)), np.zeros((len(y), 18)), np.zeros((len(y), 18))
    Pall, Rall = np.zeros((5, len(y), 18), np.float32), np.zeros((5, len(y), 18), np.float32)   # every row x every adapter
    split_ok = True
    U, gm, glam = (np.zeros((len(y), 128)), np.zeros((5, 18, 128)), np.zeros((5, 128))) if spec['gauss'] else (None, None, None)
    for k in range(5):
        ad = adir(f"{spec['name']}_f{k}")
        adp = os.path.join(ad, 'adapter') if os.path.exists(os.path.join(ad, 'adapter', 'adapter_config.json')) else ad
        acfg = json.load(open(os.path.join(adp, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
        sd = load_file(os.path.join(adp, 'adapter_model.safetensors')); deltas = []
        for kk, v in sd.items():
            if '.lora_A.' not in kk: continue
            name = kk.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods[0] and ('model.' + name) in mods[0]: name = 'model.' + name
            B = sd[kk.replace('.lora_A.', '.lora_B.')]
            for md in mods:
                W = md[name].weight
                W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
                deltas.append((W, B, v))
        H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu', weights_only=True)
        if os.path.exists(os.path.join(ad, 'meta.json')):
            lab = json.load(open(os.path.join(ad, 'meta.json')))['labels']; perm = [lab.index(l) for l in LABELS]
            for p in ('plain', 'centered'): H[f'{p}.weight'] = H[f'{p}.weight'][perm]; H[f'{p}.bias'] = H[f'{p}.bias'][perm]
        hid = H['norms.0.weight'].shape[0]
        norms = [torch.nn.LayerNorm(hid) for _ in L]
        for i, nm in enumerate(norms): nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        Z = np.zeros((len(enc), hid * len(L)), np.float32); ovf = [0, 0]
        def run(c, n):
            mm = ms[c]; dev = next(mm.parameters()).device
            with torch.no_grad():
                for i in range(c * bs, len(order), bs * n):
                    b = order[i:i + bs]; Lm = max(len(enc[j]) for j in b)
                    ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
                    for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
                    hs = mm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
                    if not all(torch.isfinite(hs[l]).all() for l in L):
                        mm.to(torch.bfloat16); hs = mm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states; mm.to(torch.float16); ovf[c] += 1
                    mask = att.unsqueeze(-1).float(); mask[:, 0] = 0; mg = {d_: mask.to(d_) for d_ in {hs[l].device for l in L}}
                    Z[b] = torch.cat([nm(((hs[l].float() * mg[hs[l].device]).sum(1) / mg[hs[l].device].sum(1)).cpu()) for l, nm in zip(L, norms)], -1).numpy()
        with ThreadPoolExecutor(len(ms)) as ex:
            for f in [ex.submit(run, c, len(ms)) for c in range(len(ms))]: f.result()
        for W, B, A_ in deltas:
            W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)
        del deltas, sd; torch.cuda.empty_cache()
        ho = fold == k
        P = Z @ H['plain.weight'].float().numpy().T + H['plain.bias'].float().numpy()
        plain[ho] = P[ho]; rc[ho] = Z[ho] @ H['centered.weight'].float().numpy().T; bc[ho] = H['centered.bias'].float().numpy()
        Pall[k] = P; Rall[k] = Z @ H['centered.weight'].float().numpy().T
        acc = (P.argmax(1) == y); pf = [acc[fold == j].mean() for j in range(5)]
        ok_k = int(np.argmin(pf)) == k and acc[~ho].mean() - acc[ho].mean() > 0.15   # held-out fold clearly worst
        split_ok &= ok_k
        log(f"  {spec['name']}_f{k}: acc held-out fold {acc[ho].mean():.3f} | in-sample {acc[~ho].mean():.3f} | "
            f"per fold {np.round(pf, 3).tolist()} | split ok {ok_k} | bf16 redo {sum(ovf)}")
        if spec['gauss'] and split_ok:   # a wrong split would fit the head on held-out rows; logits still saved for offline
            prm = ga.fit(Z[~ho], y[~ho], train.prompt_id.values[~ho], GCFG)
            U[ho] = ga.transform(Z[ho], prm); gm[k] = prm['m']; glam[k] = prm['lam']
    md = W = B = v = H = None   # loop variables kept the last model copy (md) alive: q14v5 OOM at load in v2
    del ms, mods; gc.collect(); torch.cuda.empty_cache(); log('  gpu memory after free', gpu_mem())
    OUT[spec['name'] + '_plain'] = plain.astype(np.float32); OUT[spec['name'] + '_rc'] = rc.astype(np.float32); OUT[spec['name'] + '_bc'] = bc.astype(np.float32)
    OUT[spec['name'] + '_Pall'] = Pall; OUT[spec['name'] + '_Rall'] = Rall; OUT[spec['name'] + '_split_ok'] = np.array(split_ok)
    if spec['gauss']:
        OUT[spec['name'] + '_U'] = U.astype(np.float32); OUT[spec['name'] + '_m'] = gm; OUT[spec['name'] + '_lam'] = glam
    log(f"{spec['name']} honest plain acc {(plain.argmax(1) == y).mean():.4f}")
    np.savez_compressed('/kaggle/working/honest_oof.npz', **OUT)
json.dump(GCFG, open('/kaggle/working/gauss_cfg.json', 'w'))
log('done')
