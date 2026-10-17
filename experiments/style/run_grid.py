"""Run candidates from cands.C on a split:  python run_grid.py <split> name1 name2 ...
OOF scores cached to cache/oof_<name>_s<split>.npy; metrics appended to out/results.jsonl."""
import sys
import time

from cands import C
from common import *
from views import run_oof

split = int(sys.argv[1])
names = sys.argv[2:]
bases = load_bases()
folds = get_folds(split)
for name in names:
    cfg = C[name]
    t0 = time.time()
    S = load_oof(name, split)
    if S is None:
        S = run_oof(cfg['views'], cfg['clf'], split, folds, trans=cfg.get('trans', False))
        save_oof(name, split, S)
    r = evaluate(S, f'{name} [s{split}] ({time.time() - t0:.0f}s)', bases, folds)
    r['cand'] = name
    log_result(r, split)
