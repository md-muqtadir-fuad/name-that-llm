"""Hand-built lexical/stylometric features (shared)."""
import re
from collections import Counter

import numpy as np

WORD = re.compile(r"[A-Za-z][A-Za-z’'\-]*")
SENT = re.compile(r'(?<=[.!?…])\s+(?=[A-Z“"‘(])')
HEDGE = {'perhaps', 'maybe', 'might', 'arguably', 'likely', 'possibly', 'seems', 'suggests', 'often', 'typically'}
INTENS = {'profound', 'profoundly', 'vibrant', 'intricate', 'remarkable', 'remarkably', 'truly', 'deeply', 'incredibly',
          'fascinating', 'elegant', 'stunning', 'breathtaking', 'tapestry', 'symphony', 'delicate', 'beautifully'}
FIRSTP = {'i', 'we', 'us', 'our', 'my', 'me'}
SECONDP = {'you', 'your', 'yours'}
CONJ_START = {'and', 'but', 'so', 'yet', 'or'}


def feats(t):
    w = WORD.findall(t)
    lw = [x.lower() for x in w]
    nw = max(len(w), 1)
    nc = max(len(t), 1)
    sents = [s for s in SENT.split(t.strip()) if s]
    sl = np.array([len(WORD.findall(s)) for s in sents]) if sents else np.array([nw])
    wl = np.array([len(x) for x in w]) if w else np.array([0])
    c = Counter(lw)
    first = [s.split()[0].strip('“"‘(').lower() for s in sents if s.split()]
    f = {
        'words': len(w), 'sentences': len(sents), 'words/sentence': sl.mean(), 'sent_len_sd': sl.std(),
        'sent_len_cv': sl.std() / max(sl.mean(), 1), 'max_sent_len': sl.max(), 'min_sent_len': sl.min(),
        'word_len': wl.mean(), 'long_words(>=10)': (wl >= 10).mean(), 'short_words(<=3)': (wl <= 3).mean(),
        'type_token': len(c) / nw, 'hapax_ratio': sum(1 for v in c.values() if v == 1) / nw,
        'commas/sentence': t.count(',') / max(len(sents), 1), 'commas/100w': 100 * t.count(',') / nw,
        'semicolons/100w': 100 * t.count(';') / nw, 'colons/100w': 100 * t.count(':') / nw,
        'em_dash/100w': 100 * t.count('—') / nw, 'en_dash/100w': 100 * t.count('–') / nw,
        'spaced_hyphen/100w': 100 * t.count(' - ') / nw, 'hyphenated_words/100w': 100 * sum('-' in x for x in w) / nw,
        'parens/100w': 100 * t.count('(') / nw, 'curly_apos': float('’' in t), 'straight_apos': float("'" in t),
        'curly_dquote': float('“' in t or '”' in t), 'straight_dquote': float('"' in t),
        'ellipsis': float('…' in t or '...' in t), 'asterisk': float('*' in t), 'exclaim': float('!' in t),
        'question': float('?' in t), 'ends_with_period': float(t.rstrip()[-1:] in '.!?”"’)'),
        'digits/100c': 100 * sum(ch.isdigit() for ch in t) / nc, 'non_ascii/100c': 100 * sum(ord(ch) > 127 for ch in t) / nc,
        'capitalized_mid/100w': 100 * sum(1 for i, x in enumerate(w) if i and x[0].isupper()) / nw,
        'contractions/100w': 100 * sum(1 for x in lw if re.search(r"[’']", x)) / nw,
        'hedges/100w': 100 * sum(c[x] for x in HEDGE) / nw, 'intensifiers/100w': 100 * sum(c[x] for x in INTENS) / nw,
        'first_person/100w': 100 * sum(c[x] for x in FIRSTP) / nw, 'second_person/100w': 100 * sum(c[x] for x in SECONDP) / nw,
        'the/100w': 100 * c['the'] / nw, 'and/100w': 100 * c['and'] / nw, 'of/100w': 100 * c['of'] / nw,
        'this/100w': 100 * c['this'] / nw, 'not/100w': 100 * (c['not'] + c['no']) / nw,
        'ly_adverbs/100w': 100 * sum(x.endswith('ly') for x in lw) / nw, 'ing/100w': 100 * sum(x.endswith('ing') for x in lw) / nw,
        'tion/100w': 100 * sum(x.endswith(('tion', 'sion')) for x in lw) / nw,
        'sent_starts_conj': np.mean([x in CONJ_START for x in first]) if first else 0,
        'sent_starts_this_it': np.mean([x in ('this', 'it', 'these', 'that') for x in first]) if first else 0,
        'sent_starts_the': np.mean([x == 'the' for x in first]) if first else 0,
        'not_x_but_y': float(bool(re.search(r"\bnot (just |merely |only |simply )?\w+[^.]{0,40}, but\b", t, re.I))),
        'its_not_x_its_y': float(bool(re.search(r"\bit[’']?s not\b|\bis not\b[^.]{0,30}\bit is\b", t, re.I))),
        'imagine/picture': float(bool(re.search(r'\b(imagine|picture)\b', t, re.I))),
    }
    return f


