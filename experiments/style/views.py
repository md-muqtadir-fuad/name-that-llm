"""Text views + fold-wise sparse feature building + linear classifiers for the stylegram experiments."""
import hashlib
import json
import re

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC

from common import CACHE, K, N, TEXTS, y

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_text(text):
    return " ".join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
                    for t in re.findall(r"\w+|[^\w\s]", text))


# --- extra transforms ---------------------------------------------------------------------------------------
SUFFIX = [('ly', 'LY'), ('ing', 'ING'), ('ed', 'ED'), ('tion', 'TION'), ('sion', 'TION'), ('ment', 'MENT'),
          ('ness', 'NESS'), ('ity', 'ITY'), ('ous', 'OUS'), ('ive', 'IVE'), ('able', 'ABLE'), ('ible', 'ABLE'),
          ('al', 'AL'), ('er', 'ER'), ('est', 'EST'), ('s', 'S')]
APOS = "'’"
TOK_RE = re.compile(r"[A-Za-z]+(?:[" + APOS + r"][A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]")


def posproxy_text(text):
    """Regex POS proxy: function words kept (lower), content words -> suffix class (+ C if capitalised),
    numbers -> NUM, punctuation kept."""
    out = []
    for t in TOK_RE.findall(text):
        lo = t.lower()
        if lo in FUNCTION_WORDS:
            out.append(lo)
        elif t[0].isdigit():
            out.append('NUM')
        elif not t[0].isalnum():
            out.append(t)
        else:
            tag = 'W'
            if any(ch in lo for ch in APOS):
                tag = 'APOS'
            else:
                for suf, nm in SUFFIX:
                    if lo.endswith(suf) and len(lo) > len(suf) + 2:
                        tag = nm
                        break
            out.append(('C' if t[0].isupper() else '') + tag)
    return ' '.join(out)


def template_text(text):
    """Sentence template: every non-function word -> class by case + length bucket (s<=3, m 4-7, l>=8),
    digits -> 0, punctuation and spaces kept; function words kept verbatim."""
    def rep(m):
        w = m[0]
        if w.lower() in FUNCTION_WORDS:
            return w
        c = 'X' if w[0].isupper() else 'x'
        return c + ('s' if len(w) <= 3 else 'm' if len(w) <= 7 else 'l')
    return re.sub(r"\d+", "0", re.sub(r"[^\W\d_]+", rep, text))


TRANSFORMS = {'raw': lambda t: t, 'shape': shape_text, 'function': function_text, 'posproxy': posproxy_text,
              'template': template_text}
_TCACHE = {}


def transformed(name):
    if name not in _TCACHE:
        if name in TRANSFORMS:
            _TCACHE[name] = [TRANSFORMS[name](t) for t in TEXTS]
        else:   # precomputed (tokenizer pieces, spaCy tags): cache/<name>.json, list of strings
            _TCACHE[name] = json.loads((CACHE / f'{name}.json').read_text(encoding='utf-8'))
    return _TCACHE[name]


# --- view spec -----------------------------------------------------------------------------------------------
# dict(t=transform, an='char'|'word'|'char_wb', ng=(lo,hi), mf=max_features, w=weight, lc=lowercase, md=min_df,
#      sub=sublinear_tf, tp=token_pattern|'split', binary=False, norm='l2', idf=True)
REF_VIEWS = [dict(t='raw', an='char', ng=(3, 5), mf=250000, w=1.0),
             dict(t='raw', an='word', ng=(1, 3), mf=180000, w=0.7),
             dict(t='shape', an='char', ng=(2, 6), mf=60000, w=0.5),
             dict(t='function', an='word', ng=(1, 3), mf=50000, w=0.5)]
WORD_TP = r"(?u)\b\w+\b|[^\w\s]"


def vkey(v):
    d = {k: v[k] for k in sorted(v) if k != 'w'}
    return hashlib.md5(json.dumps(d, default=str).encode()).hexdigest()[:12]


def make_vec(v):
    opts = dict(analyzer=v['an'], ngram_range=tuple(v['ng']), lowercase=v.get('lc', False),
                sublinear_tf=v.get('sub', True), min_df=v.get('md', 2), max_features=v.get('mf'),
                dtype=np.float64, binary=v.get('binary', False), norm=v.get('norm', 'l2'), use_idf=v.get('idf', True))
    if v['an'] == 'word':
        tp = v.get('tp', WORD_TP)
        if tp == 'split':
            opts['tokenizer'] = str.split
            opts['token_pattern'] = None
        else:
            opts['token_pattern'] = tp
    return TfidfVectorizer(**opts)


def view_mats(v, split, fi, a, b, trans=False):
    """(Xa, Xb) for view v on fold fi (fit on train rows a [+ unlabeled b if trans]). Disk cached, unweighted."""
    key = f"{vkey(v)}_s{split}_f{fi}{'_T' if trans else ''}"
    pa, pb = CACHE / f'X_{key}_a.npz', CACHE / f'X_{key}_b.npz'
    if pa.exists() and pb.exists():
        return sp.load_npz(pa), sp.load_npz(pb)
    txt = transformed(v['t'])
    vec = make_vec(v)
    fit_rows = np.concatenate([a, b]) if trans else a
    vec.fit([txt[i] for i in fit_rows])
    Xa = vec.transform([txt[i] for i in a]).tocsr()
    Xb = vec.transform([txt[i] for i in b]).tocsr()
    sp.save_npz(pa, Xa, compressed=False)
    sp.save_npz(pb, Xb, compressed=False)
    return Xa, Xb


def stack_views(views, split, fi, a, b, trans=False):
    A, B = [], []
    for v in views:
        Xa, Xb = view_mats(v, split, fi, a, b, trans)
        A.append(Xa * v['w'])
        B.append(Xb * v['w'])
    return sp.hstack(A, format='csr'), sp.hstack(B, format='csr')


# --- classifiers ---------------------------------------------------------------------------------------------
def nb_ratio(X, yy, c, alpha=1.0):
    p = alpha + np.asarray(X[yy == c].sum(0)).ravel()
    q = alpha + np.asarray(X[yy != c].sum(0)).ravel()
    return np.log((p / p.sum()) / (q / q.sum()))


def fit_predict(clf, Xa, ya, Xb):
    kind = clf['kind']
    C = clf.get('C', 0.5)
    cw = clf.get('cw')
    if kind == 'svc':
        m = LinearSVC(C=C, random_state=42, max_iter=clf.get('it', 4000), class_weight=cw).fit(Xa, ya)
        s = m.decision_function(Xb)
    elif kind == 'svc_cs':   # Crammer-Singer multiclass SVM
        m = LinearSVC(C=C, random_state=42, max_iter=clf.get('it', 4000), multi_class='crammer_singer').fit(Xa, ya)
        s = m.decision_function(Xb)
    elif kind == 'lr':
        m = LogisticRegression(C=C, max_iter=clf.get('it', 3000), class_weight=cw, tol=1e-4).fit(Xa, ya)
        s = m.predict_log_proba(Xb)
    elif kind == 'sgd':
        m = SGDClassifier(loss=clf.get('loss', 'modified_huber'), alpha=clf.get('alpha', 1e-4), max_iter=50,
                          tol=None, random_state=0, class_weight=cw, average=clf.get('avg', False)).fit(Xa, ya)
        s = m.decision_function(Xb)
    elif kind == 'ridge':
        m = RidgeClassifier(alpha=clf.get('alpha', 1.0), class_weight=cw, solver='sparse_cg').fit(Xa, ya)
        s = m.decision_function(Xb)
    elif kind in ('nbsvm', 'nblr'):   # one-vs-rest on NB log-count-ratio scaled features
        binar = clf.get('bin', True)
        Xa_b = (Xa > 0).astype(np.float64) if binar else Xa
        Xb_b = (Xb > 0).astype(np.float64) if binar else Xb
        s = np.zeros((Xb.shape[0], K))
        for c in range(K):
            r = nb_ratio(Xa_b, ya, c, clf.get('alpha', 1.0))
            Ra = Xa_b.multiply(r).tocsr()
            Rb = Xb_b.multiply(r).tocsr()
            yc = (ya == c).astype(int)
            if kind == 'nbsvm':
                m = LinearSVC(C=C, random_state=42, max_iter=4000).fit(Ra, yc)
            else:
                m = LogisticRegression(C=C, max_iter=3000).fit(Ra, yc)
            s[:, c] = m.decision_function(Rb)
    else:
        raise ValueError(kind)
    return np.asarray(s, dtype=np.float64)


def _one_fold(views, clf, split, fi, a, b, trans):
    Xa, Xb = stack_views(views, split, fi, a, b, trans)
    return b, fit_predict(clf, Xa, y[a], Xb)


def run_oof(views, clf, split, folds, trans=False, n_jobs=4):
    S = np.zeros((N, K))
    res = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(views, clf, split, fi, a, b, trans)
                                  for fi, (a, b) in enumerate(folds))
    for b, s in res:
        S[b] = s
    return S
