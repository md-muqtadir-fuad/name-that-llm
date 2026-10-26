# Gemma member (E056): Gemma-4-12B two-stage LoRA (24 layers; stage 1 pool_v4 + darragh run-1, stage 2 all host rows),
# plain readout. Runs in every mode (also the 3-row preview, so the commit run checks loading + the anchor parity),
# while the GPUs are still free; frees them afterwards. The final cell mixes it in before balancing.
import gc as _gc, glob as _glob, json as _json, os as _os, pathlib as _pl
import numpy as _np, pandas as _pd, torch as _torch


@_torch.no_grad()
def _gemma_member():
    import importlib.util as _iu
    _fs = _iu.find_spec
    _iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _fs(name, *a, **k)
    from peft import PeftModel
    import peft.import_utils as _pi
    _pi.is_torchao_available = lambda *a, **k: False
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    hits = sorted(_glob.glob('/kaggle/input/**/lora_twostage/final/COMPLETE.json', recursive=True))
    assert hits, 'attach kernel output mdmuqtadirfuad/ntl-gemma-ts-s2'
    ck = _pl.Path(_os.path.dirname(hits[0]))
    meta = _json.load(open(ck / 'model_meta.json'))
    ro = _np.load(ck / 'readouts.npz')
    labels = [str(c) for c in ro['classes']]
    gdir = next((_os.path.dirname(c) for c in sorted(_glob.glob('/kaggle/input/**/config.json', recursive=True))
                 if 'gemma-4-12b' in c.lower().replace('_', '-') and _os.path.basename(_os.path.dirname(c)) != 'base_config'
                 and _glob.glob(_os.path.join(_os.path.dirname(c), '*.safetensors'))), None)
    assert gdir, 'attach model google/gemma-4/Transformers/gemma-4-12b/2'
    print('gemma member', ck, 'base', gdir, flush=True)
    a = _np.load(ck / 'anchor_features.npz', allow_pickle=True)
    _data = _pl.Path(DATA)   # DATA is a Path in v8, a str in the nlm-v27 line
    tr = _pd.read_csv(_data / 'train.csv', dtype=str).set_index('id')
    te = _pd.read_csv(_data / 'test.csv', dtype=str)
    gtok = AutoTokenizer.from_pretrained(ck / 'tokenizer')
    BOS, EOS, PAD = list(meta['bos']), int(meta['eos']), int(meta['pad'])
    DEPTH, POOL = int(meta['depth']), [int(v) for v in meta['pool_layers']]

    def toks(tl):
        raw = gtok(list(tl), add_special_tokens=False, truncation=True, max_length=meta['max_len'] - 1 - len(BOS))['input_ids']
        return [BOS + list(x) + [EOS] for x in raw]
    seq_te, seq_pa = toks(te.text), toks(tr.loc[[str(i) for i in a['ids']]].text)
    cfg = AutoConfig.from_pretrained(gdir)
    tcfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
    orig = int(tcfg.num_hidden_layers)
    tcfg.num_hidden_layers = DEPTH
    try:
        if getattr(tcfg, 'layer_types', None) is not None:
            tcfg.layer_types = list(tcfg.layer_types)[:DEPTH]
    except Exception:
        pass
    try:
        sh = getattr(tcfg, 'num_kv_shared_layers', 0)
        if isinstance(sh, int) and sh:
            tcfg.num_kv_shared_layers = max(0, DEPTH - (orig - sh))
    except Exception:
        pass
    tcfg.use_cache = False
    full = AutoModel.from_pretrained(gdir, config=cfg, dtype=_torch.float16, device_map={'': 0})
    bb = full.language_model if hasattr(full, 'language_model') else full
    for c in (tcfg, bb.config):
        try:
            c.allow_global_per_layer_attribute_access = True
        except Exception:
            pass
    del full
    _gc.collect()
    _torch.cuda.empty_cache()
    vocab = bb.get_input_embeddings().num_embeddings
    used = _np.unique(_np.concatenate([_np.concatenate(seq_te + seq_pa), [PAD, EOS] + BOS])).astype(_np.int64)
    remap = _np.full(max(vocab, int(used.max()) + 1), -1, dtype=_np.int64)
    remap[used] = _np.arange(len(used))
    dev = bb.get_input_embeddings().weight.device
    keep = _torch.as_tensor(used, device=dev)
    for _, mod in bb.named_modules():
        if isinstance(mod, _torch.nn.Embedding) and mod.num_embeddings == vocab:
            mod.weight = _torch.nn.Parameter(mod.weight.data[keep].clone(), requires_grad=False)
            mod.num_embeddings, mod.padding_idx = len(used), None
    del keep
    _torch.cuda.empty_cache()
    lm = PeftModel.from_pretrained(bb, ck / 'adapter').eval()
    nl = sum(1 for n, _ in lm.named_parameters() if 'lora_' in n)
    assert nl > 0, 'LoRA weights not attached'
    hw = load_file(str(ck / 'heads.safetensors'))
    H = int(meta['hidden_size'])
    norms = _torch.nn.ModuleList([_torch.nn.LayerNorm(H) for _ in POOL])
    norms.load_state_dict({k[len('norms.'):]: v for k, v in hw.items() if k.startswith('norms.')})
    norms = norms.to(dev).float()
    PAD_T = int(remap[PAD])

    def embed(seqs, bs=8):
        seqs = [remap[_np.array(s)].tolist() for s in seqs]
        Z = _np.zeros((len(seqs), H * len(POOL)), dtype=_np.float32)
        order = _np.argsort([len(s) for s in seqs], kind='stable')
        for b0 in range(0, len(order), bs):
            idx = order[b0:b0 + bs]
            L = max(len(seqs[i]) for i in idx)
            x = _torch.full((len(idx), L), PAD_T, dtype=_torch.long)
            m = _torch.zeros((len(idx), L), dtype=_torch.long)
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])] = _torch.tensor(seqs[i])
                m[r, :len(seqs[i])] = 1
            x, m = x.to(dev), m.to(dev)
            for dt in (_torch.float16, _torch.bfloat16):   # retry non-finite batches in bf16 (as in training)
                with _torch.autocast('cuda', dtype=dt):
                    hs = lm(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False).hidden_states
                mm = m.unsqueeze(-1).float()
                mm[:, 0] = 0
                den = mm.sum(1).clamp(min=1)
                z = _torch.cat([nrm((hs[Lr].float() * mm).sum(1) / den) for nrm, Lr in zip(norms, POOL)], -1)
                if _torch.isfinite(z).all():
                    break
            Z[idx] = _np.nan_to_num(z.cpu().numpy())
            del hs
        return Z

    Za = embed(seq_pa, 3)
    rel = float(_np.abs(Za - a['features']).mean() / (_np.abs(a['features']).mean() + 1e-9))
    print(f'GEMMA_MEMBER parity vs anchors: mean rel {rel:.5f} (lora tensors {nl})', flush=True)
    assert rel < 0.02, 'Gemma LoRA features do not match the checkpoint anchors'
    Z = embed(seq_te)
    del lm, bb
    _gc.collect()
    _torch.cuda.empty_cache()
    logits = Z @ ro['plain_coef'].T + ro['plain_intercept']
    logits -= logits.max(1, keepdims=True)
    p = _np.exp(logits)
    p /= p.sum(1, keepdims=True)
    assert _np.isfinite(p).all() and _np.allclose(p.sum(1), 1)
    _np.savez_compressed('/kaggle/working/gemma_probabilities.npz', ids=te.id.to_numpy(dtype=str),
                         labels=_np.asarray(labels), probabilities=p)
    print(f'GEMMA_MEMBER {len(p)} rows scored', flush=True)


_gemma_member()
