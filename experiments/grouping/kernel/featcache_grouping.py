# Feature cache for the offline test-grouping study (experiments/grouping): embeddings of the competition train
# passages and of external DISTRACTOR passages (other prompts) so simulated test pools can be built on CPU.
# Inputs: competition name-that-llm; datasets seyitkaangunes/ntl-extra-api-passages,
# olegzholobov/name-that-llm-response-corpus, wuwenmin/bge-large-en-v1-5, gmhost/e5-large-v2, gmhost/gte-large,
# donkeys/bge-base-en-v1-5, ryotak12/intfloat-e5-base-v2, ddtarski/allmpnetbasev2, inversion/sentence-transformers-222;
# kernel output lucasboesen/name-that-llm-glm4-9b-trunc25; models qwen-lm/qwen-3 8b-base, qwen-lm/qwen-3-embedding 0.6b/4b.
# Two GPU workers run in parallel (one model per T4). Every job saves its own .npz so a late failure loses nothing.
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
W = '/kaggle/working'


def find_file(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    assert hits, name
    return hits[0]


# ---------------------------------------------------------------- universe (same filter as experiments/grouping/universe.py)
def oleg_filter(w):
    long = w.melt(id_vars=['id', 'prompt'], var_name='model', value_name='text').dropna(subset=['text'])
    long = long[long['prompt'].notna()]
    long = long[~long['text'].str.contains('[\r\n]', regex=True)]
    long['text'] = long['text'].str.strip()
    nw = long['text'].str.split().str.len()
    long = long[(nw >= 40) & (nw <= 250)]
    long = long.drop_duplicates('text')
    long = long.assign(uid='oleg:' + long['id'] + '|' + long['model'], pkey='oleg:' + long['prompt'], src='oleg')
    return long[['uid', 'src', 'pkey', 'model', 'text']]


def extra_filter(d):
    return pd.DataFrame({'uid': 'extra:' + d['id'].astype(str), 'src': 'extra', 'pkey': 'extra:' + d['prompt_id'],
                         'model': d['model'], 'text': d['text'].astype(str).str.strip()})


print(os.popen('nvidia-smi --query-gpu=name,memory.total --format=csv').read(), flush=True)
tr = pd.read_csv(next(p for p in ['/kaggle/input/competitions/name-that-llm/train.csv',
                                  '/kaggle/input/name-that-llm/train.csv'] if os.path.exists(p)))
t = pd.DataFrame({'uid': 'train:' + tr['id'].astype(str), 'src': 'train', 'pkey': 'train:' + tr['prompt_id'],
                  'model': tr['model'], 'text': tr['text']})
ex = extra_filter(pd.read_csv(find_file('ntl_extra_passages_claude.csv')))
ol = oleg_filter(pd.read_parquet(find_file('external_short_wide.parquet')))
U = pd.concat([t, ex, ol], ignore_index=True)
assert U['uid'].is_unique
print(U.groupby('src').size(), flush=True)


# ------------------------------------------------------------------------------------- worker
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

TXT = U['text'].tolist()
UID = U['uid'].values.astype(str)
DIST = np.where(U['src'].values != 'train')[0]
PARITY = np.arange(64)   # first 64 train rows: compare against the local Qwen3-8B train cache
import threading


def log(*a):
    print(f'[{threading.current_thread().name}]', *a, f'{time.time() - T0:.0f}s', flush=True)


def configs():
    out = []
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            out.append((os.path.dirname(cfg), json.load(open(cfg))))
        except Exception:
            pass
    return out


def find_dir(pred):
    for d, c in configs():
        if pred(d, c):
            return d
    raise FileNotFoundError


@torch.no_grad()
def mean_pool_layers(gpu, path, idx, layers, keep, bs=8, max_len=512):
    """Same pooling as the 0.541 / v1 notebook 'mean' heads: bos + text, mean over all tokens of hidden_states[L]."""
    dev = f'cuda:{gpu}'
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = keep
    if getattr(cfg, 'layer_types', None):
        cfg.layer_types = cfg.layer_types[:keep]
    model = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map={'': gpu}).eval()
    log('loaded', path)
    texts = [TXT[i] for i in idx]
    out = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    bad = 0
    for b0 in range(0, len(texts), bs):
        ii = order[b0:b0 + bs]
        batch = [(tok.bos_token + texts[i]) if tok.bos_token else texts[i] for i in ii]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[L]).all() for L in layers):
            bad += 1
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
            model.to(torch.float16)
        m = enc['attention_mask'].float().unsqueeze(-1)
        for li, L in enumerate(layers):
            h = hs[L].float()
            out[ii, li] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        del hs, enc
        if b0 % 1600 == 0:
            log(b0, len(texts))
    log('non-finite fp16 batches redone in bf16:', bad)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def st_encode(gpu, path, prompts, bs=32):
    """sentence-transformers encode (L2-normalised) of every universe text, once per prompt string."""
    from sentence_transformers import SentenceTransformer
    try:
        m = SentenceTransformer(path, device=f'cuda:{gpu}', model_kwargs={'dtype': torch.float16})
    except Exception as e:
        log('dtype kwarg failed, retry torch_dtype', repr(e))
        try:
            m = SentenceTransformer(path, device=f'cuda:{gpu}', model_kwargs={'torch_dtype': torch.float16})
        except Exception as e2:
            log('torch_dtype kwarg failed, load fp32 on cpu then half', repr(e2))
            m = SentenceTransformer(path, device='cpu')
            m.half()
            m.to(f'cuda:{gpu}')
    m.max_seq_length = 512
    log('loaded', path, str(m).replace('\n', ' ')[:400])
    res = {}
    for name, pr in prompts.items():
        e = m.encode(TXT, batch_size=bs, normalize_embeddings=True, show_progress_bar=False, prompt=pr or None,
                     convert_to_numpy=True).astype(np.float32)
        bad = np.where(~np.isfinite(e).all(1))[0]
        if len(bad):   # fp16 overflow: redo those rows in bf16 (T4 emulates it; few rows)
            log('non-finite rows', len(bad), 'redo in bf16')
            m.to(torch.bfloat16)
            e[bad] = m.encode([TXT[i] for i in bad], batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
                              prompt=pr or None, convert_to_numpy=True).astype(np.float32)
            m.to(torch.float16)
        res[name] = e.astype(np.float16)
        log('encoded', name, e.shape)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return res


Q3E_PROMPTS = {
    'none': '',
    'dtopic': 'Instruct: Identify the specific topic or question this passage answers\nQuery: ',
    'sameq': 'Instruct: Given a passage, retrieve other passages that answer the same question\nQuery: ',
    'clust': 'Instruct: Identify the main topic of the given text\nQuery: ',
}


ST_KEYS = {'bge_l': 'bge-large', 'e5_l': 'e5-large', 'gte_l': 'gte-large', 'bge_b': 'bge-base', 'e5_b': 'e5-base',
           'mpnet': 'all-mpnet-base-v2', 'minilm': 'all-minilm-l6-v2'}


def run(gpu, jobs):
    for job in jobs:
        if time.time() - T0 > 48 * 60:   # GPU budget guard (< 1 h total)
            log('SKIPPED (time budget)', job)
            continue
        try:
            log('job', job)
            if job == 'glm24':
                p = find_dir(lambda d, c: c.get('model_type') == 'glm4' and c.get('hidden_size') == 4096)
                E = mean_pool_layers(gpu, p, np.arange(len(TXT)), [24], 25)
                np.savez_compressed(f'{W}/glm24.npz', E=E[:, 0], uid=UID)
            elif job == 'q8dist':
                p = find_dir(lambda d, c: c.get('model_type') == 'qwen3' and c.get('hidden_size') == 4096
                             and 'embedding' not in d.lower())
                idx = np.concatenate([PARITY, DIST])
                E = mean_pool_layers(gpu, p, idx, [4, 7, 12, 16, 22], 23)
                np.savez_compressed(f'{W}/q8_dist.npz', E=E, uid=UID[idx], layers=np.array([4, 7, 12, 16, 22]))
            elif job == 'q3e06':
                p = find_dir(lambda d, c: 'embedding' in d.lower() and c.get('hidden_size') == 1024
                             and c.get('model_type') == 'qwen3')
                np.savez_compressed(f'{W}/q3e06.npz', uid=UID, **st_encode(gpu, p, Q3E_PROMPTS, bs=32))
            elif job == 'q3e4b':
                p = find_dir(lambda d, c: 'embedding' in d.lower() and c.get('hidden_size') == 2560)
                np.savez_compressed(f'{W}/q3e4b.npz', uid=UID,
                                    **st_encode(gpu, p, {k: Q3E_PROMPTS[k] for k in ('dtopic', 'sameq')}, bs=16))
            elif job in ST_KEYS:
                key = ST_KEYS[job]
                p = find_dir(lambda d, c: key in d.lower() and 'model_type' in c)
                pr = {'q': 'query: '} if job.startswith('e5') else {'plain': ''}
                np.savez_compressed(f'{W}/{job}.npz', uid=UID, **st_encode(gpu, p, pr, bs=64))
            log('saved', job)
        except Exception as e:
            import traceback
            log('FAILED', job, repr(e))
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()


for d, c in configs():
    print('model dir', d, c.get('model_type'), c.get('hidden_size'), flush=True)
J0, J1 = ['glm24', 'q3e06', 'bge_l', 'e5_l', 'gte_l', 'bge_b', 'e5_b', 'mpnet', 'minilm'], ['q8dist', 'q3e4b']
if torch.cuda.device_count() >= 2:
    th = [threading.Thread(target=run, args=(0, J0), name='gpu0'), threading.Thread(target=run, args=(1, J1), name='gpu1')]
else:
    th = [threading.Thread(target=run, args=(0, ['q8dist', 'q3e06', 'glm24'] + J0[2:] + ['q3e4b']), name='gpu0')]
for x in th:
    x.start()
for x in th:
    x.join()
print('all done', f'{time.time() - T0:.0f}s', sorted(os.listdir(W)), flush=True)
