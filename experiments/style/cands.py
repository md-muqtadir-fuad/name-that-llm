"""Candidate registry: name -> dict(views=[...], clf={...}, trans=bool)."""
import copy

from views import REF_VIEWS

R = REF_VIEWS
SVC = dict(kind='svc', C=0.5)


def V(**kw):
    d = dict(md=2, sub=True, lc=False)
    d.update(kw)
    return d


def ref_with(i, **kw):
    v = copy.deepcopy(R)
    v[i].update(kw)
    return v


CHAR, WORD, SHAPE, FUNC = R
TOKQ = V(t='tok_qwen3', an='word', ng=(1, 3), mf=200000, w=0.7, tp='split')
TOKG = V(t='tok_glm4', an='word', ng=(1, 3), mf=200000, w=0.7, tp='split')
TOKGE = V(t='tok_gemma3', an='word', ng=(1, 3), mf=200000, w=0.7, tp='split')
TOKO = V(t='tok_o200k', an='word', ng=(1, 3), mf=200000, w=0.7, tp='split')
POSPX = V(t='posproxy', an='word', ng=(1, 4), mf=80000, w=0.5, tp='split')
TEMPL = V(t='template', an='char', ng=(2, 8), mf=100000, w=0.5)
UPOS = V(t='pos_upos', an='word', ng=(1, 4), mf=60000, w=0.5, tp='split')
PTAG = V(t='pos_tag', an='word', ng=(1, 4), mf=80000, w=0.5, tp='split')
PDEP = V(t='pos_dep', an='word', ng=(1, 3), mf=80000, w=0.5, tp='split')
PMIX = V(t='pos_mixed', an='word', ng=(1, 4), mf=100000, w=0.5, tp='split')

C = {
    # --- step 1 reference and classifier swaps on the same 4 views
    'ref': dict(views=R, clf=SVC),
    'ref_C0.1': dict(views=R, clf=dict(kind='svc', C=0.1)),
    'ref_C0.25': dict(views=R, clf=dict(kind='svc', C=0.25)),
    'ref_C1': dict(views=R, clf=dict(kind='svc', C=1.0)),
    'ref_C2': dict(views=R, clf=dict(kind='svc', C=2.0)),
    'ref_cs': dict(views=R, clf=dict(kind='svc_cs', C=0.5)),
    'ref_ridge1': dict(views=R, clf=dict(kind='ridge', alpha=1.0)),
    'ref_ridge3': dict(views=R, clf=dict(kind='ridge', alpha=3.0)),
    'ref_ridge0.3': dict(views=R, clf=dict(kind='ridge', alpha=0.3)),
    'ref_sgd1e-4': dict(views=R, clf=dict(kind='sgd', alpha=1e-4)),
    'ref_sgd3e-5': dict(views=R, clf=dict(kind='sgd', alpha=3e-5)),
    'ref_sgd3e-4': dict(views=R, clf=dict(kind='sgd', alpha=3e-4)),
    'ref_sgdlog1e-5': dict(views=R, clf=dict(kind='sgd', loss='log_loss', alpha=1e-5)),
    'ref_lr10': dict(views=R, clf=dict(kind='lr', C=10.0, it=500)),
    'ref_lr30': dict(views=R, clf=dict(kind='lr', C=30.0, it=500)),
    'ref_nbsvm': dict(views=R, clf=dict(kind='nbsvm', C=0.1)),
    'ref_nbsvm_C0.03': dict(views=R, clf=dict(kind='nbsvm', C=0.03)),
    'ref_nblr': dict(views=R, clf=dict(kind='nblr', C=1.0)),
    'ref_nbsvm_tfidf': dict(views=R, clf=dict(kind='nbsvm', C=0.3, bin=False)),
    # --- view ablations / re-tuning (SVC C=0.5)
    'only_char': dict(views=[CHAR], clf=SVC),
    'only_word': dict(views=[WORD], clf=SVC),
    'only_shape': dict(views=[SHAPE], clf=SVC),
    'only_func': dict(views=[FUNC], clf=SVC),
    'drop_char': dict(views=[WORD, SHAPE, FUNC], clf=SVC),
    'drop_word': dict(views=[CHAR, SHAPE, FUNC], clf=SVC),
    'drop_shape': dict(views=[CHAR, WORD, FUNC], clf=SVC),
    'drop_func': dict(views=[CHAR, WORD, SHAPE], clf=SVC),
    'w_style_up': dict(views=ref_with(2, w=1.0) [:3] + [dict(FUNC, w=1.0)], clf=SVC),
    'w_char_down': dict(views=[dict(CHAR, w=0.6), WORD, SHAPE, FUNC], clf=SVC),
    'char_1_5': dict(views=[V(t='raw', an='char', ng=(1, 5), mf=300000, w=1.0), WORD, SHAPE, FUNC], clf=SVC),
    'char_2_6': dict(views=[V(t='raw', an='char', ng=(2, 6), mf=400000, w=1.0), WORD, SHAPE, FUNC], clf=SVC),
    'char_wb_2_5': dict(views=[V(t='raw', an='char_wb', ng=(2, 5), mf=250000, w=1.0), WORD, SHAPE, FUNC], clf=SVC),
    'char_lc': dict(views=[dict(CHAR, lc=True), WORD, SHAPE, FUNC], clf=SVC),
    'word_1_2': dict(views=[CHAR, V(t='raw', an='word', ng=(1, 2), mf=180000, w=0.7), SHAPE, FUNC], clf=SVC),
    'nomax': dict(views=[dict(v, mf=None) for v in R], clf=SVC),
    'mindf1': dict(views=[dict(v, md=1) for v in R], clf=SVC),
    'mindf3': dict(views=[dict(v, md=3) for v in R], clf=SVC),
    'nosublin': dict(views=[dict(v, sub=False) for v in R], clf=SVC),
    'binary': dict(views=[dict(v, binary=True) for v in R], clf=SVC),
    'trans': dict(views=R, clf=SVC, trans=True),
    # --- new views added to the reference
    '+tok_qwen3': dict(views=R + [TOKQ], clf=SVC),
    '+tok_glm4': dict(views=R + [TOKG], clf=SVC),
    '+tok_gemma3': dict(views=R + [TOKGE], clf=SVC),
    '+tok_o200k': dict(views=R + [TOKO], clf=SVC),
    'tok_glm4_repl_word': dict(views=[CHAR, TOKG, SHAPE, FUNC], clf=SVC),
    'tok_qwen3_repl_word': dict(views=[CHAR, TOKQ, SHAPE, FUNC], clf=SVC),
    'only_tok_qwen3': dict(views=[TOKQ], clf=SVC),
    'only_tok_glm4': dict(views=[TOKG], clf=SVC),
    'only_tok_gemma3': dict(views=[TOKGE], clf=SVC),
    'only_tok_o200k': dict(views=[TOKO], clf=SVC),
    '+posproxy': dict(views=R + [POSPX], clf=SVC),
    '+template': dict(views=R + [TEMPL], clf=SVC),
    '+upos': dict(views=R + [UPOS], clf=SVC),
    '+ptag': dict(views=R + [PTAG], clf=SVC),
    '+pdep': dict(views=R + [PDEP], clf=SVC),
    '+pmixed': dict(views=R + [PMIX], clf=SVC),
    'only_posproxy': dict(views=[POSPX], clf=SVC),
    'only_template': dict(views=[TEMPL], clf=SVC),
    'only_upos': dict(views=[UPOS], clf=SVC),
    'only_ptag': dict(views=[PTAG], clf=SVC),
    'only_pdep': dict(views=[PDEP], clf=SVC),
    'only_pmixed': dict(views=[PMIX], clf=SVC),
    'style_only4': dict(views=[SHAPE, FUNC, PMIX, PTAG], clf=SVC),   # topic-free syntax/punctuation views only
}
