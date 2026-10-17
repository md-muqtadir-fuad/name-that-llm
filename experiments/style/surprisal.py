"""Per-token surprisal profile of each passage under a frozen causal LM (observer).

  python surprisal.py extract <model_dir> <tag> [threads]   -> cache/surp_<tag>.npz (ragged per-token arrays)
  python surprisal.py eval <tag> [<tag> ...]                 -> features + LR OOF, evaluated on top of the bases

Per token t (given all previous tokens, a start token prepended): log p(actual), entropy of the predictive
distribution, rank of the actual token, and the token-piece class (function word / punctuation / word start /
continuation). Passage features = summary statistics of these sequences (overall and per piece class),
log-prob x entropy 2-D histogram, and 'Fast-DetectGPT'-style normalised curvature (lp + H) / sd.
"""
import json
import re
import sys
import time

import numpy as np

from common import CACHE, K, N, TEXTS, evaluate, get_folds, load_bases, load_oof, log_result, save_oof, y

FUNC = set("a an the and or but if while although because as so than that which who whom whose this these those it "
           "its they them their he him his she her we us our you your i me my in on at to from with without for of by "
           "over under between through into about is are was were be been being have has had do does did can could "
           "will would shall should may might must not no nor all any some each every both either neither other "
           "another such only also just very too more most less least here there when where why how what whether "
           "then now still yet even much many few little".split())


def extract(model_dir, tag, threads=4, max_len=448):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids('<|endoftext|>')
    order = np.argsort([len(t) for t in TEXTS])
    out = [None] * N
    t0 = time.time()
    bs = 8
    with torch.no_grad():
        for b0 in range(0, N, bs):
            idx = order[b0:b0 + bs]
            enc = [tok(TEXTS[i], add_special_tokens=False)['input_ids'][:max_len - 1] for i in idx]
            L = max(len(e) for e in enc) + 1
            ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, 0] = start
                ids[j, 1:len(e) + 1] = torch.tensor(e)
                att[j, :len(e) + 1] = 1
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lsm = torch.log_softmax(logits, -1)
            tgt = ids[:, 1:]
            lp = lsm.gather(-1, tgt[..., None])[..., 0]
            ent = -(lsm.exp() * lsm).sum(-1)
            rank = (lsm > lp[..., None]).sum(-1)
            for j, e in enumerate(enc):
                n = len(e)
                pieces = tok.convert_ids_to_tokens(e)
                out[idx[j]] = (lp[j, :n].numpy(), ent[j, :n].numpy(), rank[j, :n].numpy(), pieces)
            if b0 % 400 == 0:
                print(b0, f'{time.time() - t0:.0f}s', flush=True)
    lens = np.array([len(o[0]) for o in out])
    cls = []
    for o in out:
        c = []
        for p in o[3]:
            s = p.replace('Ġ', ' ').replace('Ċ', '\n')
            w = s.strip().lower()
            if not w:
                c.append(4)
            elif re.fullmatch(r'[^\w\s]+', w):
                c.append(1)                      # punctuation
            elif w in FUNC and s.startswith(' '):
                c.append(0)                      # function word
            elif s.startswith(' ') or s[:1].isupper():
                c.append(2)                      # content word start
            else:
                c.append(3)                      # continuation piece
        cls.append(np.array(c, dtype=np.int8))
    np.savez_compressed(CACHE / f'surp_{tag}.npz', lens=lens, lp=np.concatenate([o[0] for o in out]),
                        ent=np.concatenate([o[1] for o in out]), rank=np.concatenate([o[2] for o in out]),
                        cls=np.concatenate(cls))
    print('saved', tag, f'{time.time() - t0:.0f}s', flush=True)


def stats(v):
    if len(v) == 0:
        return [0.0] * 7
    q = np.quantile(v, [.1, .25, .5, .75, .9])
    return [v.mean(), v.std()] + list(q)


def features(tag):
    z = np.load(CACHE / f'surp_{tag}.npz')
    lens = z['lens']
    st = np.concatenate([[0], np.cumsum(lens)])
    F = []
    for i in range(N):
        sl = slice(st[i], st[i + 1])
        lp, en, rk, cl = z['lp'][sl], z['ent'][sl], z['rank'][sl], z['cls'][sl]
        f = stats(lp) + stats(en) + stats(lp + en)
        f += [(rk == 0).mean(), (rk < 3).mean(), (rk < 10).mean(), (rk >= 100).mean(), np.log1p(rk).mean()]
        f += [(lp + en).mean() / (np.sqrt(np.mean(en)) + 1e-6)]
        for c in range(4):
            m = cl == c
            f += [m.mean(), lp[m].mean() if m.any() else 0, en[m].mean() if m.any() else 0,
                  (rk[m] == 0).mean() if m.any() else 0]
        h, _, _ = np.histogram2d(np.clip(-lp, 0, 15), np.clip(en, 0, 8), bins=[6, 4], range=[[0, 15], [0, 8]])
        f += list((h / max(len(lp), 1)).ravel())
        f += [np.log(len(lp))]
        F.append(f)
    return np.nan_to_num(np.array(F, dtype=np.float64))


def lr_oof(X, folds, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    S = np.zeros((N, K))
    for a, b in folds:
        sc = StandardScaler().fit(X[a])
        m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[a]), y[a])
        S[b] = m.predict_log_proba(sc.transform(X[b]))
    return S


if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    else:
        split = int(sys.argv[2])
        bases = load_bases()
        folds = get_folds(split)
        tags = sys.argv[3:]
        X = np.hstack([features(t) for t in tags])
        print('features', X.shape, flush=True)
        for C in (0.03, 0.1, 0.3, 1.0):
            name = f"surp_{'+'.join(tags)}_C{C}"
            S = lr_oof(X, folds, C)
            save_oof(name, split, S)
            r = evaluate(S, f'{name} [s{split}]', bases, folds)
            r['cand'] = name
            log_result(r, split)
