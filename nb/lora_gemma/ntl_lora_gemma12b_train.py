# ntl-lora-gemma12b-train: private training kernel for Kaggle "Name That LLM".
#
# Gemma-4-12B, first 24 decoder blocks, frozen fp16 base; the input embedding is trimmed to the token ids
# that occur (memory trick from qasminumber/name-that-llm-v5). LoRA r16 / alpha 32 / dropout 0.05 on every
# attention and MLP projection. Features z = per-layer LayerNorm of position-0-free mean pools of 8 layers.
# Heads: plain (18 models), centred (18; z minus the mean of its same-prompt micro-batch), auxiliary lab (6).
# Loss = CE_plain + CE_centred + 0.5 * CE_lab. Same-prompt micro-batches of random size 2 or 3, every fitting
# passage once per epoch, about 18 passages per optimizer update (micro-batches are packed up to PACK rows per
# forward pass; centring stays inside each micro-batch). 3 epochs, AdamW (adapter 1e-4, heads 1e-3, wd 0.01),
# 10% warmup + cosine, fp16 autocast with fp32 trainable parameters, gradient checkpointing.
# Jobs: GroupKFold(5) on prompt_id (fold0..fold4) + final (all rows); two worker processes, one per T4.
#
# SMOKE = True : two tiny jobs per GPU (fold0, fold2 on GPU0; final, fold1 on GPU1), 60 fitting rows, ~10 updates,
#                20 validation rows; writes the same files; prints peak memory, sec/update, full-run estimate.
#                Two jobs per worker exercise the sequential path of the full run (adapter restore, training after
#                an extraction, probe skipped); GPU0's first job raises one synthetic OOM to exercise the fallback.
# SMOKE = False: full run (fold0, fold2, fold4 on GPU0; final, fold1, fold3 on GPU1).
SMOKE = False
BACKBONE = 'gemma4-12b'  # fallback: 'qwen3-8b-base' (then attach qwen-lm/qwen-3/Transformers/8b-base/1)

BACKBONES = {
    'gemma4-12b': dict(match='/gemma-4-12b/', slug='google/gemma-4/Transformers/gemma-4-12b/2',
                       depth=24, pool_layers=[2, 4, 8, 12, 14, 16, 20, 24]),
    'qwen3-8b-base': dict(match='/8b-base/', slug='qwen-lm/qwen-3/Transformers/8b-base/1',
                          depth=24, pool_layers=[2, 4, 6, 8, 12, 16, 20, 24]),
}
HP = dict(r=16, alpha=32, lora_dropout=0.05, lr_adapter=1e-4, lr_head=1e-3, weight_decay=0.01, epochs=3,
          warmup=0.1, eff_batch=18, pack=8, length_window=64, max_len=448, head_dropout=0.1, lab_weight=0.5, clip=1.0, seed=42,
          init_scale=256.0, growth_interval=2000, n_folds=5)
LABS = ['claude', 'gemini', 'glm', 'gpt', 'kimi', 'qwen']
WORKER_JOBS_FULL = {0: ['fold0', 'fold2', 'fold4'], 1: ['final', 'fold1', 'fold3']}
WORKER_JOBS_SMOKE = {0: ['fold0', 'fold2'], 1: ['final', 'fold1']}
SMOKE_FIT = (4, 15)    # prompts x passages per prompt = 60 fitting rows
SMOKE_VALID = (2, 10)  # prompts x passages per prompt = 20 validation rows
MONITOR_HOURS = 11.5

WORKER_SRC = r'''
import os, sys
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Kaggle ships an old torchao that PEFT rejects; it is unused here, so hide it before any ML import.
import importlib.util as _iu
_find_spec = _iu.find_spec
_iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _find_spec(name, *a, **k)
import fcntl, gc, hashlib, json, math, random, time, traceback
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

W = int(sys.argv[1])
WORK = Path('/kaggle/working')
OUT = WORK / 'lora_gemma'
RUN = json.loads((OUT / 'run_config.json').read_text())
PLAN = json.loads((OUT / 'plan.json').read_text())
HP, SPEC = RUN['hp'], RUN['backbone_spec']
T0 = time.time()


def log(msg):
    print(f'W{W}| [{time.time() - T0:7.1f}s] {msg}', flush=True)


def jlog(**kw):
    print(f'W{W}| ' + json.dumps(kw), flush=True)


torch.set_num_threads(2)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one visible GPU per worker'
device = torch.device('cuda:0')
torch.cuda.set_device(device)
GB = 2 ** 30
status_path = OUT / f'worker{W}_status.json'
status = dict(worker=W, jobs=RUN['worker_jobs'][str(W)], completed=[], failed={}, status='INIT',
              gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              versions={n: version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')})


def write_status():
    status_path.write_text(json.dumps(status, indent=2))


write_status()
log(f"gpu {status['gpu']} versions {status['versions']}")

# ------------------------------------------------------------------ data and tokens
train = pd.read_csv(RUN['train_csv'], dtype=str)
LABELS = np.array(RUN['labels'])
LABS = np.array(RUN['labs'])
CLASS_LAB = np.array(RUN['class_lab'])
K, KL = len(LABELS), len(LABS)
y = np.searchsorted(LABELS, train.model.to_numpy(dtype=str))
assert (LABELS[y] == train.model.to_numpy(dtype=str)).all()
ylab = CLASS_LAB[y]
groups = train.prompt_id.to_numpy(dtype=str)
ids = train.id.to_numpy(dtype=str)
model_dir = Path(RUN['model_dir'])
tok = AutoTokenizer.from_pretrained(model_dir)
BOS = [tok.bos_token_id] if (tok.bos_token_id is not None and tok('a')['input_ids'][:1] == [tok.bos_token_id]) else []
EOS = tok.eos_token_id
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
raw = tok(train.text.tolist(), add_special_tokens=False, truncation=True,
          max_length=HP['max_len'] - 1 - len(BOS))['input_ids']
seqs_orig = [BOS + list(x) + [EOS] for x in raw]
lengths = np.array([len(s) for s in seqs_orig])
log(f'tokens: bos {BOS} eos {EOS} pad {PAD}; length min {lengths.min()} median {int(np.median(lengths))} '
    f'max {lengths.max()} (cap {HP["max_len"]})')

# ------------------------------------------------------------------ backbone (one load per worker)
def cget(c, key, default=None):
    """getattr that survives transformers' per-layer (heterogeneous) config attributes."""
    try:
        value = getattr(c, key, default)
    except Exception as error:
        return f'<{type(error).__name__}>'
    return default if value is None else value


load_started = time.time()
with open(WORK / '.model_load.lock', 'w') as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)  # one load at a time (host RAM)
    log('loading backbone (lock held)')
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = cfg.get_text_config() if hasattr(cfg, 'get_text_config') else cfg
    orig_layers = int(tcfg.num_hidden_layers)
    DEPTH = int(SPEC['depth'])
    tcfg.num_hidden_layers = DEPTH
    if cget(tcfg, 'layer_types') is not None:
        tcfg.layer_types = list(tcfg.layer_types)[:DEPTH]
    shared = cget(tcfg, 'num_kv_shared_layers', 0)
    if isinstance(shared, int) and shared:  # keep the original KV-sharing boundary after truncation
        tcfg.num_kv_shared_layers = max(0, DEPTH - (orig_layers - shared))
        log(f'num_kv_shared_layers {shared} -> {tcfg.num_kv_shared_layers}')
    tcfg.use_cache = False
    cfg_info = {k: cget(tcfg, k) for k in (
        'model_type', 'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim',
        'sliding_window', 'vocab_size', 'final_logit_softcapping', 'attn_logit_softcapping', 'num_kv_shared_layers',
        'hidden_activation', 'tie_word_embeddings')}
    cfg_info['config_class'] = type(tcfg).__name__
    cfg_info['orig_layers'] = orig_layers
    cfg_info['layer_types'] = list(cget(tcfg, 'layer_types') or [])
    try:  # heterogeneous (per-layer) Gemma-4 configs: record the kept layers' head counts
        plc = cget(tcfg, 'per_layer_config')
        if isinstance(plc, (list, tuple)):
            cfg_info['per_layer_kv_heads'] = [cget(c, 'num_key_value_heads') for c in plc[:DEPTH]]
            cfg_info['per_layer_config_len'] = len(plc)
    except Exception as error:
        cfg_info['per_layer_config_error'] = repr(error)[:200]
    log(f'text config {json.dumps(cfg_info, default=str)}')
    full = AutoModel.from_pretrained(model_dir, config=cfg, dtype=torch.float16, device_map={'': 0})
    log(f'loaded {type(full).__name__}; mem {torch.cuda.memory_allocated(device) / GB:.2f} GiB')
    backbone = full.language_model if hasattr(full, 'language_model') else full
    # The model is built; let library code (PEFT hasattr / save) read global values of per-layer attributes
    # instead of raising AmbiguousGlobalPerLayerAttributeError.
    for c in (tcfg, backbone.config):
        try:
            c.allow_global_per_layer_attribute_access = True
        except Exception as error:
            log(f'could not set allow_global_per_layer_attribute_access: {error!r}')
    del full
    gc.collect()
    torch.cuda.empty_cache()
    backbone.requires_grad_(False)
    # Trim every vocab-sized embedding to the token ids that occur. The embedding is frozen, so inference may
    # re-trim to any superset (train + test tokens) and gets identical features.
    vocab = backbone.get_input_embeddings().num_embeddings
    used = np.unique(np.concatenate([np.concatenate(seqs_orig), [PAD, EOS] + BOS])).astype(np.int64)
    remap = np.full(max(vocab, int(used.max()) + 1), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    keep = torch.as_tensor(used, device=device)
    trimmed = []
    for mname, mod in backbone.named_modules():
        if isinstance(mod, nn.Embedding) and mod.num_embeddings == vocab:
            mod.weight = nn.Parameter(mod.weight.data[keep].clone(), requires_grad=False)
            mod.num_embeddings, mod.padding_idx = len(used), None
            trimmed.append(mname)
    other_emb = {mname: mod.num_embeddings for mname, mod in backbone.named_modules()
                 if isinstance(mod, nn.Embedding) and mname not in trimmed}
    if other_emb:  # an untrimmed token-indexed table would silently read remapped ids
        log(f'WARNING untrimmed embeddings {other_emb}')
    del keep
    gc.collect()
    torch.cuda.empty_cache()
    fcntl.flock(lock_file, fcntl.LOCK_UN)
seqs = [remap[np.array(s)].tolist() for s in seqs_orig]
PAD_T = int(remap[PAD])
H = int(tcfg.hidden_size)
POOL = [int(v) for v in SPEC['pool_layers']]
D = H * len(POOL)
base_mem = torch.cuda.memory_allocated(device) / GB
log(f'{type(backbone).__name__}: vocab {vocab} -> {len(used)} (trimmed {trimmed}); hidden {H}; depth {DEPTH}; '
    f'pool {POOL}; mem {base_mem:.2f} GiB; load {time.time() - load_started:.0f}s')

backbone.config.use_cache = False
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
targets = [n for n, m in backbone.named_modules() if isinstance(m, nn.Linear) and n.startswith('layers.')]
leaf_names = sorted({n.rsplit('.', 1)[-1] for n in targets})
assert targets, 'no Linear layers found in decoder blocks'
log(f'LoRA targets: {len(targets)} Linear layers, names {leaf_names}')
torch.manual_seed(HP['seed'])
model = get_peft_model(backbone, LoraConfig(r=HP['r'], lora_alpha=HP['alpha'], lora_dropout=HP['lora_dropout'],
                                            bias='none', target_modules=targets))
try:
    model.enable_input_require_grads()
except Exception as error:  # not needed with non-reentrant checkpointing
    log(f'enable_input_require_grads skipped: {error!r}')
lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
for p in lora_params:
    p.data = p.data.float()
snapshot = [p.detach().cpu().clone() for p in lora_params]  # identical fresh adapter for every job
snapshot_l1 = float(sum(float(s.double().abs().sum()) for s in snapshot))
n_lora = sum(p.numel() for p in lora_params)
attn_impl = cget(backbone.config, '_attn_implementation')
log(f'LoRA params {n_lora / 1e6:.1f}M; attention {attn_impl}; mem {torch.cuda.memory_allocated(device) / GB:.2f} GiB')
load_seconds = time.time() - load_started

norms = None
PACK = int(HP['pack'])


class SyntheticOOM(RuntimeError):
    """Raised once in SMOKE runs (GPU0, first job, second update) to exercise the out-of-memory fallback."""


OOM_ERRORS = (torch.cuda.OutOfMemoryError, SyntheticOOM)


def collate(rows):
    L = max(len(seqs[i]) for i in rows)
    x = torch.full((len(rows), L), PAD_T, dtype=torch.long)
    m = torch.zeros((len(rows), L), dtype=torch.long)
    for r, i in enumerate(rows):
        s = seqs[i]
        x[r, :len(s)] = torch.tensor(s)
        m[r, :len(s)] = 1
    return x.to(device, non_blocking=True), m.to(device, non_blocking=True)


def represent(rows, dtype=torch.float16):
    x, m = collate(rows)
    with torch.autocast('cuda', dtype=dtype):
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False).hidden_states
    mm = m.unsqueeze(-1).float()
    mm[:, 0] = 0  # position 0 (BOS / attention sink) is excluded from the mean
    den = mm.sum(1).clamp(min=1)
    with torch.autocast('cuda', enabled=False):
        z = torch.cat([nrm((hs[L].float() * mm).sum(1) / den) for nrm, L in zip(norms, POOL)], -1)
    return z, int(m.sum()), int(m.numel())


def size_rule(remaining, rng):
    if remaining <= 3:
        return remaining
    if remaining == 4:
        return 2
    return int(rng.integers(2, 4))


def chunk_labels(rows, seed):
    """Random same-prompt chunks of 2 or 3 (as in training) for held-out centring."""
    rng = np.random.default_rng(seed)
    lab = np.empty(len(rows), dtype=object)
    for g in sorted(set(groups[rows])):
        pos = rng.permutation(np.flatnonzero(groups[rows] == g))
        i = c = 0
        while i < len(pos):
            s = size_rule(len(pos) - i, rng)
            lab[pos[i:i + s]] = f'{g}#{c}'
            i, c = i + s, c + 1
    return lab.astype(str)


def center_np(Z, labels):
    Z = Z.copy()
    for g in np.unique(labels):
        i = labels == g
        Z[i] -= Z[i].mean(0)
    return Z


def make_packs(update):
    # similar-length micro-batches share a forward pass (less padding); centring stays per micro-batch
    update = sorted(update, key=lambda mb: max(len(seqs[i]) for i in mb))
    out, cur, n = [], [], 0
    for mb in update:
        if cur and n + len(mb) > PACK:
            out.append(cur)
            cur, n = [], 0
        cur.append(mb)
        n += len(mb)
    if cur:
        out.append(cur)
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


probe_done = False


def run_job(name):
    global norms, PACK, probe_done
    jp = PLAN[name]
    fit = np.asarray(jp['fit'], dtype=int)
    valid = np.asarray(jp['valid'], dtype=int)
    epochs = jp['epochs']
    seed = int(jp['seed'])
    ck = OUT / name
    ck.mkdir(parents=True, exist_ok=True)
    job_started = time.time()
    with torch.no_grad():
        for p, s in zip(lora_params, snapshot):
            p.copy_(s.to(device))
            p.grad = None  # nothing left over from a failed job
        restored_l1 = float(sum(float(p.detach().double().abs().sum()) for p in lora_params))
    assert abs(restored_l1 - snapshot_l1) <= 1e-6 * max(1.0, snapshot_l1), \
        f'LoRA restore mismatch: {restored_l1} vs snapshot {snapshot_l1}'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in POOL]).to(device)
    plain = nn.Linear(D, K).to(device)
    centered = nn.Linear(D, K).to(device)
    lab = nn.Linear(D, KL).to(device)
    drop = nn.Dropout(HP['head_dropout'])
    head_params = [*norms.parameters(), *plain.parameters(), *centered.parameters(), *lab.parameters()]
    params = lora_params + head_params
    assert all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW([{'params': lora_params, 'lr': HP['lr_adapter']},
                             {'params': head_params, 'lr': HP['lr_head']}], weight_decay=HP['weight_decay'])
    total = sum(len(ep) for ep in epochs)
    warm = max(1, int(total * HP['warmup']))

    def schedule(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scaler = torch.amp.GradScaler('cuda', init_scale=HP['init_scale'], growth_interval=HP['growth_interval'])

    def pack_loss(pack, n_update):
        rows = np.concatenate([np.asarray(mb, dtype=int) for mb in pack])
        gid = np.repeat(np.arange(len(pack)), [len(mb) for mb in pack])
        z, ntok, npad = represent(rows)
        if not bool(torch.isfinite(z).all()):
            return None, None, (ntok, npad)
        g = torch.as_tensor(gid, device=device)
        sums = torch.zeros(len(pack), D, device=device, dtype=z.dtype).index_add(0, g, z)
        cnt = torch.bincount(g, minlength=len(pack)).to(z.dtype)
        zc = z - (sums / cnt[:, None])[g]
        ok = (cnt[g] > 1).to(z.dtype)  # a lone passage has nothing to be centred against
        yt = torch.as_tensor(y[rows], device=device)
        yl = torch.as_tensor(ylab[rows], device=device)
        with torch.autocast('cuda', enabled=False):
            lp = F.cross_entropy(plain(drop(z)), yt, reduction='sum')
            lc = (F.cross_entropy(centered(drop(zc)), yt, reduction='none') * ok).sum()
            ll = F.cross_entropy(lab(drop(z)), yl, reduction='sum')
            loss = (lp + lc + HP['lab_weight'] * ll) / n_update
        return loss, (float(lp.detach()), float(lc.detach()), float(ll.detach())), (ntok, npad)

    # memory probe: the PACK longest passages of the whole training set, forward + backward, no step
    if not probe_done:
        for pack_size in sorted({PACK, 6, 4}, reverse=True):
            if pack_size > PACK:
                continue
            longest = np.argsort(-lengths, kind='stable')[:pack_size]
            probe = [longest[i:i + 3].tolist() for i in range(0, pack_size, 3)]
            PACK = pack_size
            torch.cuda.reset_peak_memory_stats(device)
            oom = None
            try:
                model.train(); norms.train(); plain.train(); centered.train(); lab.train()
                t = time.time()
                loss, _, _ = pack_loss(probe, pack_size)
                finite = loss is not None
                if finite:
                    loss.backward()
                del loss
                torch.cuda.synchronize(device)
            except torch.cuda.OutOfMemoryError as error:
                oom = str(error)[:300]
            opt.zero_grad(set_to_none=True)
            for p in lora_params:
                p.grad = None
            if oom is not None:  # cleanup outside the except block: its traceback pins the activations
                gc.collect()
                torch.cuda.empty_cache()
                log(f'memory probe OOM at pack {pack_size}: {oom}')
                if pack_size == 4:
                    raise RuntimeError(f'out of memory even at pack 4: {oom}')
                continue
            status['probe'] = dict(pack=pack_size, max_len=int(lengths[longest].max()), seconds=time.time() - t,
                                   finite=finite, peak_allocated_gb=torch.cuda.max_memory_allocated(device) / GB,
                                   peak_reserved_gb=torch.cuda.max_memory_reserved(device) / GB)
            log(f"memory probe {status['probe']}")
            break
        probe_done = True
        write_status()

    torch.cuda.reset_peak_memory_stats(device)
    records, nonfinite_packs, skipped_updates, oom_events = [], 0, 0, []
    synthetic_oom = bool(RUN['smoke'] and W == 0 and name == RUN['worker_jobs']['0'][0])
    log(f'job {name}: fit {len(fit)} valid {len(valid)} updates {total} warmup {warm} pack {PACK}; '
        f'adapter restored (L1 {restored_l1:.4f})')
    train_started = time.time()
    for e, ep in enumerate(epochs):
        model.train(); norms.train(); plain.train(); centered.train(); lab.train()
        for update in ep:
            t = time.time()
            n_update = sum(len(mb) for mb in update)
            retries = 0
            while True:  # an out-of-memory update is redone from scratch with smaller packs (same objective)
                parts = np.zeros(3)
                ntoks, npads, bad, packs = 0, 0, 0, make_packs(update)
                oom = None
                try:
                    for pk in packs:
                        loss, met, (ntok, npad) = pack_loss(pk, n_update)
                        ntoks, npads = ntoks + ntok, npads + npad
                        if loss is None:
                            bad += 1
                            continue
                        scaler.scale(loss).backward()
                        parts += np.array(met)
                        del loss
                        if synthetic_oom and len(records) == 1:
                            synthetic_oom = False
                            raise SyntheticOOM('smoke: synthetic OOM after a backward, to exercise the fallback')
                except OOM_ERRORS as error:
                    oom = f'{type(error).__name__}: {str(error)[:300]}'
                if oom is None:
                    break
                # cleanup outside the except block (its traceback pins the activations); drop partial gradients
                loss = None
                opt.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                old = PACK
                if old <= 3:
                    raise RuntimeError(f'out of memory at pack {old}: {oom}')
                PACK = 6 if old > 6 else 4 if old > 4 else 3
                retries += 1
                oom_events.append(dict(job=name, update=len(records) + 1, pack_from=old, pack_to=PACK, error=oom))
                log(f'OOM in update {len(records) + 1} at pack {old}; redoing the update at pack {PACK}: {oom}')
            nonfinite_packs += bad
            scale_before = float(scaler.get_scale())
            if bad == len(packs):
                gn = float('nan')
                opt.zero_grad(set_to_none=True)
                stepped = False
            else:
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, HP['clip']))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                stepped = float(scaler.get_scale()) >= scale_before
            lr_used = opt.param_groups[0]['lr']
            sched.step()
            torch.cuda.synchronize(device)
            skipped_updates += int(not stepped)
            rec = dict(update=len(records) + 1, epoch=e + 1, rows=n_update, packs=len(packs), tokens=ntoks, padded_tokens=npads,
                       sec=time.time() - t, plain=parts[0] / n_update, centred=parts[1] / n_update,
                       lab=parts[2] / n_update, grad_norm=gn, scale=float(scaler.get_scale()), stepped=stepped,
                       nonfinite_packs=bad, oom_retries=retries, pack=PACK, lr=lr_used)
            records.append(rec)
            if RUN['smoke'] or len(records) % 10 == 0 or len(records) == total:
                jlog(job=name, upd=rec['update'], total=total, ep=e + 1, rows=n_update, sec=round(rec['sec'], 2),
                     plain=round(rec['plain'], 4), centred=round(rec['centred'], 4), lab=round(rec['lab'], 4),
                     pad_eff=round(ntoks / max(1, npads), 3), gn=round(gn, 3), scale=rec['scale'], skipped=skipped_updates, nonfinite=nonfinite_packs,
                     oom=len(oom_events), pack=PACK, peak_gb=round(torch.cuda.max_memory_allocated(device) / GB, 2))
    train_seconds = time.time() - train_started
    peak_train = dict(allocated_gb=torch.cuda.max_memory_allocated(device) / GB,
                      reserved_gb=torch.cuda.max_memory_reserved(device) / GB)

    # ------------------------------------------------ features with the fixed final-epoch model
    model.eval(); norms.eval(); plain.eval(); centered.eval(); lab.eval()

    def extract(rows, bs=8):
        rows = np.asarray(rows, dtype=int)
        Z = np.zeros((len(rows), D), np.float32)
        order = np.argsort([len(seqs[i]) for i in rows], kind='stable')
        bad_pos = []
        # no_grad, not inference_mode: the same process trains its next job afterwards, and an inference tensor
        # cached anywhere in the model would break that job's backward
        with torch.no_grad():
            for b in range(0, len(order), bs):
                pos = order[b:b + bs]
                z, _, _ = represent(rows[pos])
                z = z.float().cpu().numpy()
                fin = np.isfinite(z).all(1)
                Z[pos[fin]] = z[fin]
                bad_pos += pos[~fin].tolist()
            for p in bad_pos:  # rare fp16 overflow: retry the row alone under bf16 autocast
                try:
                    z, _, _ = represent(rows[[p]], dtype=torch.bfloat16)
                    Z[p] = np.nan_to_num(z.float().cpu().numpy()[0], posinf=0.0, neginf=0.0)
                except Exception as error:
                    log(f'bf16 retry failed: {error!r}')
                    Z[p] = 0.0
        return Z, len(bad_pos)

    Wp, bp = plain.weight.detach().cpu().numpy(), plain.bias.detach().cpu().numpy()
    Wc, bc = centered.weight.detach().cpu().numpy(), centered.bias.detach().cpu().numpy()
    Wl, bl = lab.weight.detach().cpu().numpy(), lab.bias.detach().cpu().numpy()
    extract_seconds, n_valid_bad = 0.0, 0
    if len(valid):
        t = time.time()
        Zv, n_valid_bad = extract(valid)
        extract_seconds = time.time() - t
        chunks = chunk_labels(valid, seed + 1000)
        lp = Zv @ Wp.T + bp
        lc = center_np(Zv, chunks) @ Wc.T + bc
        lcp = center_np(Zv, groups[valid]) @ Wc.T + bc
        ll = Zv @ Wl.T + bl
        np.savez_compressed(ck / 'val_outputs.npz', ids=ids[valid], rows=valid, prompt_id=groups[valid],
                            y=y[valid], ylab=ylab[valid], classes=LABELS, labs=LABS, class_lab=CLASS_LAB,
                            plain_logits=lp.astype(np.float32), center_logits_chunk=lc.astype(np.float32),
                            chunk_ids=chunks, center_logits_prompt=lcp.astype(np.float32),
                            lab_logits=ll.astype(np.float32), z=Zv.astype(np.float16),
                            pool_layers=np.array(POOL), nonfinite_rows=n_valid_bad)
        acc = dict(plain=float((lp.argmax(1) == y[valid]).mean()), centred_chunk=float((lc.argmax(1) == y[valid]).mean()),
                   half=float(((0.5 * lp + 0.5 * lc).argmax(1) == y[valid]).mean()),
                   lab=float((ll.argmax(1) == ylab[valid]).mean()))
        log(f'job {name}: held-out accuracy {acc}; extract {extract_seconds:.1f}s for {len(valid)} rows')
    else:
        acc = {}

    # ------------------------------------------------ checkpoint (lightningv08 lora24 layout + lab head)
    save_started = time.time()
    np.savez_compressed(ck / 'readouts.npz', classes=LABELS, plain_coef=Wp, plain_intercept=bp,
                        center_coef=Wc, center_intercept=bc, lab_coef=Wl, lab_intercept=bl, lab_classes=LABS,
                        class_lab=CLASS_LAB, pool_layers=np.array(POOL), train_sha256=RUN['train_sha256'],
                        hidden_size=H, depth=DEPTH)
    model.save_pretrained(str(ck / 'adapter'), safe_serialization=True, save_embedding_layers=False)
    tensors = {}
    for prefix, module in (('norms', norms), ('plain', plain), ('centered', centered), ('lab', lab)):
        tensors.update({f'{prefix}.{k}': v.detach().cpu().contiguous() for k, v in module.state_dict().items()})
    save_file(tensors, str(ck / 'heads.safetensors'))
    np.save(ck / 'kept_token_ids.npy', used)
    meta = dict(job=name, backbone=RUN['backbone'], model_slug=SPEC['slug'], model_dir=str(model_dir),
                model_class=type(backbone).__name__, access='AutoModel.from_pretrained(...).language_model if present',
                depth=DEPTH, orig_layers=orig_layers, pool_layers=POOL, hidden_size=H, feature_dim=D,
                max_len=HP['max_len'], bos=BOS, eos=EOS, pad=PAD, bos_prepended=bool(BOS),
                tokenization='[bos] + tok(text, add_special_tokens=False, truncation=True, '
                             'max_length=max_len-1-len(bos)).input_ids + [eos]; mean pool masks position 0',
                embedding_trim=dict(original_vocab=int(vocab), kept=int(len(used)), file='kept_token_ids.npy',
                                    modules=trimmed, note='frozen embedding: re-trim at inference to the ids that '
                                    'occur in train+test (any superset gives identical features)'),
                config=cfg_info, attn_implementation=attn_impl,
                lora=dict(r=HP['r'], alpha=HP['alpha'], dropout=HP['lora_dropout'], targets=leaf_names,
                          n_target_modules=len(targets)),
                heads=dict(plain=K, centered=K, lab=KL, loss=f"CE_plain + CE_centred + {HP['lab_weight']} * CE_lab",
                           centred_input='z - mean(z over same-prompt micro-batch of 2-3)'),
                classes=LABELS.tolist(), labs=LABS.tolist(), class_lab=CLASS_LAB.tolist(),
                dtype='frozen fp16 base, fp32 LoRA/heads, fp16 autocast', train_sha256=RUN['train_sha256'])
    (ck / 'model_meta.json').write_text(json.dumps(meta, indent=2, default=str))
    if name == 'final':
        anchors, n_anchor_bad = extract(np.arange(3), bs=3)
        np.savez_compressed(ck / 'anchor_features.npz', ids=ids[:3], classes=LABELS, features=anchors,
                            train_sha256=RUN['train_sha256'])
        tok.save_pretrained(str(ck / 'tokenizer'))
        try:  # truncated config, for reference only (inference rebuilds it from the base config + model_meta)
            cfg.save_pretrained(str(ck / 'base_config'))
        except Exception as error:
            log(f'base_config not saved: {error!r}')
    save_seconds = time.time() - save_started
    secs = np.array([r['sec'] for r in records])
    rows_per = np.array([r['rows'] for r in records])
    steady = slice(2, None) if len(records) > 4 else slice(1, None) if len(records) > 1 else slice(None)
    report = dict(job=name, status='COMPLETE', fit_rows=int(len(fit)), valid_rows=int(len(valid)),
                  fit_ids=ids[fit].tolist(), valid_ids=ids[valid].tolist(), epochs=len(epochs), total_updates=total,
                  warmup_updates=warm, pack=PACK, load_seconds=load_seconds, train_seconds=train_seconds,
                  extract_seconds=extract_seconds, save_seconds=save_seconds, job_seconds=time.time() - job_started,
                  median_sec_per_update=float(np.median(secs[steady])) if len(secs) else None,
                  median_sec_per_row=float(np.median(secs[steady] / rows_per[steady])) if len(secs) else None,
                  mean_rows_per_update=float(rows_per.mean()) if len(rows_per) else None,
                  extract_sec_per_row=extract_seconds / max(1, len(valid)) if len(valid) else None,
                  peak_train=peak_train, peak_allocated_gb_job=torch.cuda.max_memory_allocated(device) / GB,
                  base_memory_gb=base_mem, nonfinite_packs=nonfinite_packs, skipped_updates=skipped_updates,
                  nonfinite_valid_rows=n_valid_bad, heldout_accuracy=acc, oom_events=oom_events,
                  lora_restore_l1=restored_l1, updates=records, smoke=RUN['smoke'])
    (ck / 'training_report.json').write_text(json.dumps(report, indent=2))
    files = {str(p.relative_to(ck)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(ck.rglob('*')) if p.is_file() and p.name != 'COMPLETE.json'}
    (ck / 'COMPLETE.json').write_text(json.dumps(dict(job=name, train_sha256=RUN['train_sha256'], files=files), indent=2))
    log(f'JOB COMPLETE {name}: train {train_seconds:.0f}s, median {report["median_sec_per_update"]}s/update, '
        f'peak {peak_train["allocated_gb"]:.2f} GiB, skipped {skipped_updates}, nonfinite packs {nonfinite_packs}, '
        f'oom retries {len(oom_events)}, pack {PACK}')
    del opt, sched, scaler, plain, centered, lab, head_params, params
    norms = None
    gc.collect()
    torch.cuda.empty_cache()


status['status'] = 'TRAINING'
status['load_seconds'] = load_seconds
status['base_memory_gb'] = base_mem
write_status()
for job in RUN['worker_jobs'][str(W)]:
    try:
        status['current'] = job
        write_status()
        run_job(job)
        status['completed'].append(job)
    except Exception as error:
        status['failed'][job] = traceback.format_exc()
        log(f'JOB FAILED {job}: {error!r}\n{traceback.format_exc()}')
        gc.collect()
        torch.cuda.empty_cache()
    write_status()
status['status'] = 'COMPLETE' if not status['failed'] else 'FAILED'
status['current'] = None
write_status()
log(f"worker done: completed {status['completed']} failed {list(status['failed'])}")
sys.exit(1 if status['failed'] else 0)
'''

# ====================================================================== launcher
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

T0 = time.time()
DRY = not Path('/kaggle/input').is_dir()  # local dry run: planning only, no GPU work
WORK = Path(os.environ.get('NTL_DRY_OUT', 'dry_out')) if DRY else Path('/kaggle/working')
OUT = WORK / 'lora_gemma'
OUT.mkdir(parents=True, exist_ok=True)


def say(msg):
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


ast.parse(WORKER_SRC, filename='worker.py')
if DRY:
    train_csv = Path(os.environ.get('NTL_TRAIN_CSV', 'data/train.csv'))
else:
    hits = [p for p in Path('/kaggle/input').rglob('train.csv') if (p.parent / 'test.csv').exists()]
    assert len(hits) == 1, hits
    train_csv = hits[0]
train = pd.read_csv(train_csv, dtype=str)
assert train.id.is_unique and train.notna().all().all()
LABELS = np.array(sorted(train.model.unique()))
assert len(LABELS) == 18
CLASS_LAB = np.array([next(i for i, lab in enumerate(LABS) if m.startswith(lab)) for m in LABELS])
y = np.searchsorted(LABELS, train.model.to_numpy(dtype=str))
groups = train.prompt_id.to_numpy(dtype=str)
char_len = train.text.str.len().to_numpy()
n = len(train)
say(f'train {train.shape} from {train_csv}; labs {dict(zip(LABELS.tolist(), np.array(LABS)[CLASS_LAB].tolist()))}')

fold = np.full(n, -1)
for k, (_, va) in enumerate(GroupKFold(n_splits=HP['n_folds']).split(np.zeros(n), y, groups)):
    fold[va] = k
assert (fold >= 0).all()
pd.DataFrame({'id': train.id, 'prompt_id': train.prompt_id, 'model': train.model, 'fold': fold}).to_csv(
    OUT / 'folds.csv', index=False)
say(f'GroupKFold(5) rows per fold {np.bincount(fold).tolist()}, prompts per fold '
    f'{[len(set(groups[fold == k])) for k in range(5)]}')


def size_rule(remaining, rng):
    if remaining <= 3:
        return remaining
    if remaining == 4:
        return 2
    return int(rng.integers(2, 4))


def plan_job(fit, seed):
    """Per epoch: same-prompt micro-batches of random size 2 or 3 covering every fitting row once, shuffled,
    grouped into length-sorted updates of about eff_batch rows (a small remainder joins the previous update),
    update order shuffled."""
    rng = np.random.default_rng(seed)
    epochs = []
    for _ in range(HP['epochs']):
        mbs = []
        for g in sorted(set(groups[fit])):
            members = rng.permutation(fit[groups[fit] == g])
            i = 0
            while i < len(members):
                s = size_rule(len(members) - i, rng)
                mbs.append(members[i:i + s].tolist())
                i += s
        mbs = [mbs[j] for j in rng.permutation(len(mbs))]
        # length grouping (as v5): sort windows of shuffled micro-batches by length before cutting updates,
        # then shuffle the updates, so each forward pass pads little
        win = HP['length_window']
        mbs = [mb for s in range(0, len(mbs), win) for mb in sorted(mbs[s:s + win], key=lambda mb: char_len[mb].max())]
        updates, cur, cnt = [], [], 0
        for mb in mbs:
            cur.append(mb)
            cnt += len(mb)
            if cnt >= HP['eff_batch']:
                updates.append(cur)
                cur, cnt = [], 0
        if cur:
            if updates and cnt < HP['eff_batch'] // 2:
                updates[-1].extend(cur)
            else:
                updates.append(cur)
        updates = [updates[j] for j in rng.permutation(len(updates))]
        flat = [r for u in updates for mb in u for r in mb]
        assert sorted(flat) == sorted(fit.tolist()), 'every fitting row exactly once per epoch'
        assert all(len(set(groups[mb])) == 1 for u in updates for mb in u)
        epochs.append(updates)
    return epochs


def take(rows, n_prompts, per_prompt):
    keep = sorted(set(groups[rows]))[:n_prompts]
    return np.concatenate([rows[groups[rows] == g][:per_prompt] for g in keep])


full_jobs = {f'fold{k}': (np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(HP['n_folds'])}
full_jobs['final'] = (np.arange(n), np.array([], dtype=int))
job_seed = {name: HP['seed'] + i for i, name in enumerate(full_jobs)}
worker_jobs = WORKER_JOBS_SMOKE if SMOKE else WORKER_JOBS_FULL
plan, full_counts = {}, {}
for name, (fit, valid) in full_jobs.items():
    full_plan = plan_job(fit, job_seed[name])
    full_counts[name] = dict(fit=int(len(fit)), valid=int(len(valid)), updates=sum(len(e) for e in full_plan),
                             train_rows=int(HP['epochs'] * len(fit)))
    if not any(name in jobs for jobs in worker_jobs.values()):
        continue
    if SMOKE:
        fit = take(fit, *SMOKE_FIT)
        valid = take(valid, *SMOKE_VALID) if len(valid) else valid
        job_plan = plan_job(fit, job_seed[name])
    else:
        job_plan = full_plan
    plan[name] = dict(fit=fit.tolist(), valid=valid.tolist(), seed=job_seed[name],
                      epochs=[[[[int(r) for r in mb] for mb in u] for u in ep] for ep in job_plan])
    sizes = np.bincount([len(mb) for e in job_plan for u in e for mb in u])
    say(f"plan {name}: fit {len(fit)} valid {len(valid)} updates {sum(len(e) for e in job_plan)} "
        f"micro-batch sizes {dict(enumerate(sizes.tolist()))} rows/update "
        f"{np.mean([sum(len(mb) for mb in u) for e in job_plan for u in e]):.1f}")
say(f'full-run update counts {json.dumps({k: v["updates"] for k, v in full_counts.items()})}')

model_dir = None
if not DRY:
    match = BACKBONES[BACKBONE]['match']
    cands = sorted({p.parent for p in Path('/kaggle/input').rglob('config.json')
                    if match in str(p) and list(p.parent.glob('*.safetensors'))})
    assert len(cands) >= 1, f'attach {BACKBONES[BACKBONE]["slug"]}'
    model_dir = cands[0]
    say(f'backbone {BACKBONE}: {model_dir}')
train_sha = hashlib.sha256(Path(train_csv).read_bytes()).hexdigest()
run = dict(smoke=SMOKE, backbone=BACKBONE, backbone_spec=BACKBONES[BACKBONE], hp=HP, labels=LABELS.tolist(),
           labs=LABS, class_lab=CLASS_LAB.tolist(), worker_jobs={str(k): v for k, v in worker_jobs.items()},
           worker_jobs_full={str(k): v for k, v in WORKER_JOBS_FULL.items()}, full_counts=full_counts,
           job_seed=job_seed, train_csv=str(train_csv), train_sha256=train_sha,
           model_dir=str(model_dir) if model_dir else None,
           fold_method='sklearn GroupKFold(n_splits=5).split(X, y, groups=prompt_id), default ordering; folds.csv')
(OUT / 'run_config.json').write_text(json.dumps(run, indent=2))
(OUT / 'plan.json').write_text(json.dumps(plan))
(WORK / 'worker.py').write_text(WORKER_SRC)
if DRY:
    say('dry run: plan written, workers not launched')
    sys.exit(0)

import torch  # noqa: E402  (only for the device count; workers own the GPUs)
assert torch.cuda.device_count() == 2, f'need T4 x2, got {torch.cuda.device_count()}'
del torch
procs = []
for w in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(w))
    handle = open(WORK / f'worker{w}.log', 'w')
    procs.append((subprocess.Popen([sys.executable, '-u', str(WORK / 'worker.py'), str(w)], env=env, cwd=str(WORK),
                                   stdout=handle, stderr=subprocess.STDOUT), handle))
    say(f'worker {w} started: jobs {worker_jobs[w]}')
seen = [0, 0]
deadline = time.time() + MONITOR_HOURS * 3600


def pump():
    for w in (0, 1):
        lines = (WORK / f'worker{w}.log').read_text(errors='replace').splitlines()
        for line in lines[seen[w]:]:
            if line.startswith(f'W{w}|') or 'Error' in line or 'Traceback' in line:
                print(line[:2000], flush=True)
        seen[w] = len(lines)


while any(p.poll() is None for p, _ in procs) and time.time() < deadline:
    pump()
    time.sleep(15)
pump()
for w, (p, handle) in enumerate(procs):
    if p.poll() is None:
        p.terminate()
        say(f'worker {w} terminated at the monitor deadline')
    handle.close()
codes = [p.wait() for p, _ in procs]
say(f'worker exit codes {codes}')
for w, code in enumerate(codes):
    if code:
        print(f'----- worker{w}.log tail -----\n' + (WORK / f'worker{w}.log').read_text(errors='replace')[-12000:],
              flush=True)

# ---------------------------------------------------------------- summary
summary = dict(smoke=SMOKE, backbone=BACKBONE, worker_exit_codes=codes, jobs={})
statuses = {}
for w in (0, 1):
    path = OUT / f'worker{w}_status.json'
    statuses[w] = json.loads(path.read_text()) if path.exists() else {}
summary['workers'] = {w: {k: s.get(k) for k in ('status', 'completed', 'load_seconds', 'base_memory_gb', 'probe',
                                                'versions', 'gpu')} | {'failed': list(s.get('failed', {}))}
                      for w, s in statuses.items()}
reports = {}
for name in plan:
    path = OUT / name / 'training_report.json'
    if path.exists():
        r = json.loads(path.read_text())
        reports[name] = r
        summary['jobs'][name] = {k: r.get(k) for k in (
            'fit_rows', 'valid_rows', 'total_updates', 'pack', 'train_seconds', 'extract_seconds', 'save_seconds',
            'median_sec_per_update', 'median_sec_per_row', 'mean_rows_per_update', 'extract_sec_per_row',
            'peak_train', 'nonfinite_packs', 'skipped_updates', 'nonfinite_valid_rows', 'heldout_accuracy',
            'oom_events')}
        summary['jobs'][name]['complete_json'] = (OUT / name / 'COMPLETE.json').exists()
fold_outputs = sorted(OUT.glob('fold*/val_outputs.npz'))
if fold_outputs:
    from sklearn.metrics import balanced_accuracy_score
    parts = [np.load(p) for p in fold_outputs]
    yy = np.concatenate([p['y'] for p in parts])
    lp = np.concatenate([p['plain_logits'] for p in parts])
    lc = np.concatenate([p['center_logits_chunk'] for p in parts])
    ll = np.concatenate([p['lab_logits'] for p in parts])
    yl = np.concatenate([p['ylab'] for p in parts])
    summary['oof'] = dict(rows=int(len(yy)), folds=[p.parent.name for p in fold_outputs],
                          plain=float(balanced_accuracy_score(yy, lp.argmax(1))),
                          centred_chunk=float(balanced_accuracy_score(yy, lc.argmax(1))),
                          half=float(balanced_accuracy_score(yy, (0.5 * lp + 0.5 * lc).argmax(1))),
                          lab_accuracy=float((ll.argmax(1) == yl).mean()))
if reports:
    per_row = [r['median_sec_per_row'] for r in reports.values() if r.get('median_sec_per_row')]
    per_upd = [r['median_sec_per_update'] for r in reports.values() if r.get('median_sec_per_update')]
    ext = [r['extract_sec_per_row'] for r in reports.values() if r.get('extract_sec_per_row')]
    save = [r['save_seconds'] for r in reports.values()]
    loads = [s.get('load_seconds') for s in statuses.values() if s.get('load_seconds')]
    sec_row, sec_ext = float(np.mean(per_row)), float(np.mean(ext)) if ext else 0.1
    est = {}
    for w, jobs in WORKER_JOBS_FULL.items():
        t = float(np.mean(loads)) if loads else 120.0
        for name in jobs:
            c = full_counts[name]
            t += c['train_rows'] * sec_row + (c['valid'] + (3 if name == 'final' else 0)) * sec_ext + float(np.max(save))
        est[f'worker{w}_hours'] = t / 3600
    summary['extrapolation'] = dict(sec_per_row=sec_row, median_sec_per_update=float(np.mean(per_upd)),
                                    rows_per_full_update=HP['eff_batch'], extract_sec_per_row=sec_ext,
                                    full_updates=full_counts, **est,
                                    wall_hours=max(est.values()), gpu_hours=sum(est.values()))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print('=' * 30 + (' SMOKE SUMMARY ' if SMOKE else ' RUN SUMMARY ') + '=' * 30)
print(json.dumps(summary, indent=2, default=str), flush=True)
ok = all(c == 0 for c in codes) and all(summary['jobs'].get(nm, {}).get('complete_json') for nm in plan)
say('ALL JOBS COMPLETE' if ok else 'SOME JOBS FAILED')
if not ok:
    raise SystemExit(1)
