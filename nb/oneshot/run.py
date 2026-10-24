SRC = r'''"""E052: one-shot style-transfer log-probabilities (Findings of EMNLP 2026, "One-shot Style Transfer LLM log-probabilities
for Authorship Attribution and Verification"), training-free. For passage x and model k:
    s[x, k] = mean over exemplars e of model k of  mean_t log P(x_t | "Text by author A: e. Another text by author A:", x_<t)
Exemplars: the answers of all 18 models to 3 fixed bank prompts (same questions for every class, so differences are style,
not topic). Evaluated on all answers to 30 other prompts (540 rows). Backbone Qwen3-4B-Base fp16, one copy per T4.
Usage: python oneshot.py score <worker> <n_workers> [smoke]     (Kaggle GPU; writes out/oneshot_w<worker>.npz)
       python experiments/oneshot/oneshot.py eval                  (local: standalone + added to the strong OOF blend)"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('experiments/oneshot/out')
OUT.mkdir(parents=True, exist_ok=True)
N_BANK, N_EVAL, MAX_TOK = 3, 30, 256


def plan():
    train = pd.read_csv('data/train.csv', dtype=str)
    labels = np.array(sorted(train.model.unique()))
    full = train.groupby('prompt_id').model.nunique()
    full = np.array(sorted(full[full == 18].index))
    perm = np.random.default_rng(0).permutation(full)
    bank, evp = perm[:N_BANK], perm[N_BANK:N_BANK + N_EVAL]
    ex = [[train[(train.prompt_id == b) & (train.model == m)].text.iloc[0] for b in bank] for m in labels]
    ev = np.flatnonzero(train.prompt_id.isin(evp).to_numpy())
    return train, labels, bank, ex, ev


def score(worker, n_workers, smoke):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = next(str(Path(c).parent) for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
                if json.load(open(c)).get('model_type') == 'qwen3' and json.load(open(c)).get('hidden_size') == 2560)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float16, device_map={'': 0},
                                                 attn_implementation='sdpa').eval()
    train, labels, bank, ex, ev = plan()
    rows = ev[worker::n_workers]
    if smoke:
        rows, ex = rows[:4], [e[:1] for e in ex[:2]]
    xs = [tok(t, add_special_tokens=False).input_ids[:MAX_TOK] for t in train.text.iloc[rows]]
    order = np.argsort([len(x) for x in xs])
    S = np.full((len(rows), len(ex), len(ex[0])), np.nan, dtype=np.float32)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0, B = time.time(), 8
    with torch.no_grad():
        for k in range(len(ex)):
            for e in range(len(ex[k])):
                pre = tok('Text by author A:\n' + tok.decode(tok(ex[k][e], add_special_tokens=False).input_ids[:MAX_TOK])
                          + '\n\nAnother text by author A:\n', add_special_tokens=False).input_ids
                P = len(pre)
                for s in range(0, len(order), B):
                    idx = order[s:s + B]
                    L = max(len(xs[i]) for i in idx)
                    ids = torch.full((len(idx), P + L), pad, dtype=torch.long)
                    am = torch.zeros_like(ids)
                    for r, i in enumerate(idx):
                        ids[r, :P + len(xs[i])] = torch.tensor(pre + xs[i])
                        am[r, :P + len(xs[i])] = 1
                    h = model.model(input_ids=ids.cuda(), attention_mask=am.cuda()).last_hidden_state
                    for r, i in enumerate(idx):
                        n = len(xs[i])
                        lp = torch.log_softmax(model.lm_head(h[r, P - 1:P - 1 + n]).float(), -1)
                        S[i, k, e] = lp.gather(1, torch.tensor(xs[i], device=lp.device)[:, None]).mean().item()
            print(f'W{worker} class {k + 1}/{len(ex)} done, {time.time() - t0:.0f}s', flush=True)
    assert np.isfinite(S).all()
    np.savez(OUT / f"oneshot_w{worker}{'_smoke' if smoke else ''}.npz", rows=rows, S=S, bank=bank, seconds=time.time() - t0)


def evaluate():
    import csv
    from sklearn.model_selection import GroupKFold
    train, labels, bank, ex, ev = plan()
    parts = [np.load(f) for f in sorted(OUT.glob('oneshot_w[0-9].npz'))]
    rows = np.concatenate([p['rows'] for p in parts])
    S = np.concatenate([p['S'] for p in parts])
    assert sorted(rows) == sorted(ev)
    y_all = np.searchsorted(labels, train.model.to_numpy())
    fold = np.full(len(train), -1)
    for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(train)), y_all, train.prompt_id)):
        fold[va] = k
    y, f = y_all[rows], fold[rows]
    bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])
    s = S.mean(2)
    variants = {'raw': s, 'class_centred': s - s.mean(0, keepdims=True)}   # class bias removed over the eval rows
    for name, v in variants.items():
        print(f'{name}: standalone bacc {bacc(y, v.argmax(1)):.4f} (chance 0.056); per exemplar '
              + ', '.join(f'{bacc(y, (S[:, :, e] - S[:, :, e].mean(0)).argmax(1)):.3f}' for e in range(S.shape[2])))
    zb = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
    pos = pd.Series(np.arange(len(train)), index=train.id).loc[zb['ids'].astype(str)].to_numpy()
    ref = np.empty_like(zb['reference'])
    ref[:, pos] = zb['reference']
    ref = ref[:, rows]
    v = variants['class_centred']
    v = (v - v.mean(1, keepdims=True)) / v.std()
    GRID = [0, 0.25, 0.5, 1, 2, 4, 8]
    res = []
    for sc in range(ref.shape[0]):
        score = {w: (ref[sc] + w * v).argmax(1) for w in GRID}
        pred = np.empty(len(y), dtype=int)
        for k in np.unique(f):
            o = f != k
            wk = max(GRID, key=lambda w: bacc(y[o], score[w][o]))
            pred[f == k] = score[wk][f == k]
            res.append(dict(scenario=sc, fold=int(k), w=wk, base=bacc(y[f == k], score[0][f == k]),
                            value=bacc(y[f == k], pred[f == k])))
        res.append(dict(scenario=sc, fold='all', w=None, base=bacc(y, score[0]), value=bacc(y, pred)))
    r = pd.DataFrame(res)
    r.to_csv(OUT / 'blend.csv', index=False)
    al = r[r.fold == 'all']
    pf = r[r.fold != 'all'].groupby('fold')[['base', 'value']].mean()
    fpos = f'{int((pf.value - pf.base).gt(0).sum())}/{len(pf)}'
    alone = bacc(y, variants['class_centred'].argmax(1))
    print(f'blend on {len(rows)} rows: {al.base.mean():.4f} -> {al.value.mean():.4f} '
          f'({al.value.mean() - al.base.mean():+.4f}, {fpos} folds, w {sorted(set(r.w.dropna()))})')
    pd.DataFrame(s, columns=labels).assign(id=train.id.iloc[rows].to_numpy(), model=train.model.iloc[rows].to_numpy()) \
        .to_csv(OUT / 'scores.csv', index=False)
    sys.path.insert(0, 'paper')
    from log import LOG, log_row
    keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E052')]
    with open(LOG, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)
    log_row(id='E052', category='member', metric='oof_bacc_strong_blend_lofo_w_540rows',
            description='One-shot style-transfer log-probs (training-free; Qwen3-4B-Base; 3 bank prompts x 18 classes as '
                        'exemplars; class-centred), added to strong OOF blend on 540 rows of 30 prompts',
            config=f'N_BANK={N_BANK}; N_EVAL={N_EVAL}; MAX_TOK={MAX_TOK}', baseline=f'{al.base.mean():.4f}',
            value=f'{al.value.mean():.4f}', folds_pos=fpos, files='experiments/oneshot/out/*',
            notes=f"standalone class-centred {alone:.4f}, raw {bacc(y, variants['raw'].argmax(1)):.4f}")


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        score(int(sys.argv[2]), int(sys.argv[3]), len(sys.argv) > 4)
    else:
        evaluate()
'''
import glob, os, subprocess, sys
from pathlib import Path
os.chdir('/kaggle/working')
Path('data').mkdir(exist_ok=True)
hit = glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True)
assert hit
if not Path('data/train.csv').exists():
    os.symlink(hit[0], 'data/train.csv')
Path('oneshot.py').write_text(SRC)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
subprocess.run([sys.executable, '-u', 'oneshot.py', 'score', '0', '1', 'smoke'], check=True,
               env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
procs = [subprocess.Popen([sys.executable, '-u', 'oneshot.py', 'score', str(w), '2'],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(w))) for w in range(2)]
codes = [p.wait() for p in procs]
assert codes == [0, 0], codes
os.remove('oneshot.py')
