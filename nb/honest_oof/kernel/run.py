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
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)
assert torch.cuda.is_available(), 'no GPU in this session'   # out of weekly quota, Kaggle runs "GPU" kernels on CPU
log('gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
DATA = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
GAUSS_SRC = '"""Shared-prompt Gaussian group head (ported, Apache-2.0, from public Kaggle notebooks by prvsiyan / kirilltym / sangam21).\nTwo-way model x = class mean + prompt effect + noise; exact (n<=3) or cavity-EM (n>3) injective group marginals."""\nimport itertools\nimport numpy as np\nK = 18\nDEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "n_exact": 3, "inj": True,\n           "iters": 5, "repel": True, "sp_scale": 1.0}\n\n# =================================================================== primitives ====\ndef _lse(A, axis=None, keepdims=False):\n    m = np.max(A, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)\n    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m\n    if keepdims: return out\n    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())\n\ndef log_softmax(L): return L - _lse(L, axis=1, keepdims=True)\n\ndef _lw_shrinkage(X):\n    """Ledoit-Wolf shrinkage coefficient for rows X that are already centred (sklearn formula, assume_centered)."""\n    from sklearn.covariance import ledoit_wolf_shrinkage\n    return float(ledoit_wolf_shrinkage(X, assume_centered=True))\n\ndef _shrunk(S, a):\n    d = S.shape[0]; return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)\n\ndef _two_way(Z, y, pinv, G, iters=4):\n    """additive fit Z = mu[y] + P[prompt] + E by alternating means (balanced design -> converges in 1-2 rounds)."""\n    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)\n    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))\n    for _ in range(iters):\n        R = Z - P[pinv]\n        for c in cls: mu[c] = R[y == c].mean(0)\n        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]\n    return mu, P, cnt_p\n\ndef fit(X, y, prompts, cfg=None):\n    """fit the shared-prompt Gaussian on training rows. X (N, D) raw features, y (N,) labels in 0..17, prompts (N,) group ids.\n    Returns a params dict (numpy arrays + floats); temperatures default to 1 (set by the caller\'s cross-fit)."""\n    c = dict(DEFAULT); c.update(cfg or {})\n    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape\n    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)\n    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx\n    if c["red"] == "none":\n        V = np.eye(D)\n    else:\n        if c["red"] == "wpca":\n            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]; R = Xs - Pm[pinv]\n        elif c["red"] == "pca": R = Xs\n        else: raise ValueError(c["red"])\n        d = int(min(c["d"], N - G - 1, D))\n        # economy SVD through the (N x N) Gram matrix (N << D for the real views)\n        if N < D:\n            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]; w, U = np.clip(w[o], 1e-12, None), U[:, o]\n            V = (R.T @ U) / np.sqrt(w)\n        else:\n            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T\n    Z = Xs @ V; d = Z.shape[1]\n    mu, P, ng = _two_way(Z, y, pinv, G)\n    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)\n    Se_raw = E.T @ E / dof\n    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"]); Se = _shrunk(Se_raw, ae)\n    Pc = P - P.mean(0); Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw\n    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])\n    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se)); Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]\n    # joint diagonalisation: A Se A\' = I, A Sp A\' = diag(lam)\n    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T; M = 0.5 * (M + M.T)\n    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li\n    W = V @ A.T\n    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,\n            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4), "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()}, "cfg": c}\n\ndef transform(X, prm): return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]\n\n_MASK = {}\ndef _inj_mask(n, C):\n    if (n, C) not in _MASK:\n        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij"); bad = np.zeros([C] * n, bool)\n        for i, j in itertools.combinations(range(n), 2): bad |= grids[i] == grids[j]\n        _MASK[(n, C)] = bad\n    return _MASK[(n, C)]\n\ndef singleton_ll(U, m, lam):\n    """(n, C) log N(u; m_c, I + diag lam) up to a constant (p integrated out)."""\n    R = U[:, None, :] - m[None]; return -0.5 * (R ** 2 / (1 + lam)).sum(-1)\n\ndef joint_terms(U, m, lam):\n    """a (n, C) and B (n, n, C, C) of the exact joint log-likelihood of one group (constant in c dropped)."""\n    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]\n    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1); G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)\n    return a, B\n\ndef joint_table(a, B):\n    """full tuple table Q[c_1, ..., c_n] = sum_i a_i + sum_{i<j} B_ij."""\n    n, C = a.shape; Q = np.zeros([C] * n)\n    for i in range(n):\n        sh = [1] * n; sh[i] = C; Q = Q + a[i].reshape(sh)\n    for i, j in itertools.combinations(range(n), 2):\n        sh = [1] * n; sh[i] = C; sh[j] = C; Q = Q + B[i, j].reshape(sh)\n    return Q\n\ndef exact_marginals(Q, T=1.0, inj=True):\n    """per-passage log marginal posteriors (n, C) from a tuple table Q (unnormalised log joint), tempered by 1/T."""\n    n = Q.ndim; C = Q.shape[0]; Qt = Q / T\n    if inj and n >= 2: Qt = np.where(_inj_mask(n, C), -np.inf, Qt)\n    lz = _lse(Qt); out = np.empty((n, C))\n    for i in range(n):\n        ax = tuple(k for k in range(n) if k != i); out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz\n    return out\n\ndef em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):\n    """cavity EM for one group (any n >= 2): returns (n, C) log q."""\n    n = len(U); v = lam / (1 + (n - 1) * lam)\n    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))\n    q = np.exp(log_softmax(L))\n    for _ in range(iters):\n        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v                  # cavity prompt-effect means (n, d)\n        R = U[:, None, :] - Pm[:, None, :] - m[None]; L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T\n        if repel:   # mean-field at-most-once: discourage classes the other members already hold\n            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)\n        q = np.exp(log_softmax(L))\n    return log_softmax(L)\n\nLAM_DIFF = 1e6\ndef lam_g(prm, cfg=None):\n    """lam used for groups: the fitted one, or ~infinity in the paired-difference ablation (cfg \'diff\'): then the joint likelihood\n    only sees within-group differences (x_i - x_j around mu_a - mu_b with covariance 2 S_e; triples: centred observations vs\n    centred prototypes) and the prompt term cancels exactly."""\n    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {})\n    return np.full_like(prm["lam"], LAM_DIFF) if c.get("diff") else prm["lam"]\n\ndef group_logpost(U, prm, T1=None, Tg=None, cfg=None):\n    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U); lg = lam_g(prm, cfg)\n    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg\n    if n == 1: return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)\n    if n <= c["n_exact"]:\n        a, B = joint_terms(U, prm["m"], lg); return exact_marginals(joint_table(a, B), Tg, c["inj"])\n    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])\n\ndef scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):\n    """(n, 18) log-probabilities for every pool row; lab = detected group labels of the pool rows (singletons allowed)."""\n    U = transform(X_pool, prm) if U is None else U; lab = np.asarray(lab); n = len(U); out = np.empty((n, prm["m"].shape[0]))\n    sizes = np.bincount(lab); s1 = sizes[lab] == 1\n    T1 = prm["T1"] if T1 is None else T1\n    if s1.any(): out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)\n    for g in np.where(sizes >= 2)[0]:\n        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)\n    return out\n'   # gauss.py from the 0.664 notebook (Apache-2.0), embedded by build.py
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
        torch.cuda.empty_cache()
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
    del ms, mods; torch.cuda.empty_cache()
    OUT[spec['name'] + '_plain'] = plain.astype(np.float32); OUT[spec['name'] + '_rc'] = rc.astype(np.float32); OUT[spec['name'] + '_bc'] = bc.astype(np.float32)
    OUT[spec['name'] + '_Pall'] = Pall; OUT[spec['name'] + '_Rall'] = Rall; OUT[spec['name'] + '_split_ok'] = np.array(split_ok)
    if spec['gauss']:
        OUT[spec['name'] + '_U'] = U.astype(np.float32); OUT[spec['name'] + '_m'] = gm; OUT[spec['name'] + '_lam'] = glam
    log(f"{spec['name']} honest plain acc {(plain.argmax(1) == y).mean():.4f}")
    np.savez_compressed('/kaggle/working/honest_oof.npz', **OUT)
json.dump(GCFG, open('/kaggle/working/gauss_cfg.json', 'w'))
log('done')
