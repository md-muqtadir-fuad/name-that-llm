"""Step 1: reproduce the reference stylegram (nb/v1/v1.py section 1) OOF on 3 prompt-disjoint splits."""
import sys
import time

from scipy.special import log_softmax

from common import *
from views import REF_VIEWS, run_oof

bases = load_bases()
splits = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
print('bases: FULL', round(bacc(y, bases['FULL'].argmax(1)), 4), 'NOSVM', round(bacc(y, bases['NOSVM'].argmax(1)), 4))
if 0 in splits:
    r = evaluate(bases['_svm'], 'public svm component (already in FULL)', bases, get_folds(0))
    log_result(r, -1)
for split in splits:
    t0 = time.time()
    folds = get_folds(split)
    S = run_oof(REF_VIEWS, dict(kind='svc', C=0.5), split, folds)
    save_oof('ref', split, S)
    r = evaluate(S, f'REF stylegram svc C0.5 [split {split}] ({time.time() - t0:.0f}s)', bases, folds)
    # the notebook's exact member: softmax(decision) (T=1) at blend weight 1.649
    for bn in ('NOSVM', 'FULL'):
        d = bacc(y, (bases[bn] + 1.649 * log_softmax(S, 1)).argmax(1)) - bacc(y, bases[bn].argmax(1))
        r[f'{bn}_d_nbw1.649_T1'] = round(d, 4)
    print('   notebook-style T=1 w=1.649:', r['NOSVM_d_nbw1.649_T1'], r['FULL_d_nbw1.649_T1'], '| T', r['T'],
          '| curve NOSVM', r['NOSVM_curve'], '| curve FULL', r['FULL_curve'], flush=True)
    log_result(r, split)
