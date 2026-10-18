"""Parse SpeechMap (github.com/xlr8harder/llm-compliance, Apache-2.0) responses into competition-like paragraphs.
Input : external_dataset/speechmap/us_hard_<provider>_<model>.jsonl  (one row per question; full OpenRouter response)
Output: research/ext_data/speechmap/paragraphs.csv          (text; git-ignored)
        experiments/speechmap/out/summary_by_model.csv      (per-label counts; committed)
        experiments/speechmap/out/style_vs_train.csv        (register comparison with competition train; committed)
Paragraph rules: split on blank lines; drop headings/lists/quotes/tables; strip **bold**/__/`; keep 40-250 words,
single line, ends with sentence punctuation. Responses flagged as refusals (short or refusal phrasing) are kept but
marked (refusal=1) so experiments can filter."""
import glob
import json
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRCS = [ROOT / 'external_dataset/speechmap', ROOT / 'external_dataset/speechmap-data/responses']
OUT_TXT = ROOT / 'research/ext_data/speechmap'
OUT = ROOT / 'experiments/speechmap/out'
OUT_TXT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview',
          'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5', 'claude-opus-4-6',
          'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
REFUSAL = re.compile(r"\b(I (can(?:no|')t|cannot|won't|will not|am not able|'m not able|must decline|'m unable|am unable)|"
                     r"I'm sorry, but|I apologize, but|I'd rather not)\b", re.I)
SKIP_LINE = re.compile(r'^\s*(#{1,6}\s|[-*+•]\s|\d+[.)]\s|>|\|)')
END = re.compile(r'[.!?…]["”’)\]]?\s*$')


def label_of(fname):
    """-> (label, variant) or (None, None). variant: base | reasoning | dated. Non-target versions (e.g. fable-5.1,
    opus-5.5, sonnet-5.5) return None."""
    m = re.match(r'us_hard_[^_]+_(.+)\.jsonl$', fname)
    lab = m.group(1).replace('claude-opus-4.6', 'claude-opus-4-6').replace('claude-opus-4.8', 'claude-opus-4-8')
    variant = 'base'
    for suf, var in (('-reasoning', 'reasoning'), ('-thinking', 'reasoning'), ('-0902', 'dated')):
        if lab.endswith(suf):
            lab, variant = lab[:-len(suf)], var
    return (lab, variant) if lab in LABELS else (None, None)


def content_of(d):
    try:
        c = d['response']['choices'][0]['message'].get('content')
    except Exception:
        return None
    if isinstance(c, list):
        c = ''.join(x.get('text', '') for x in c if isinstance(x, dict))
    return c if isinstance(c, str) and c.strip() else None


def paragraphs(text):
    out = []
    for block in re.split(r'\n\s*\n', text.replace('\r\n', '\n')):
        lines = [l for l in block.strip().split('\n') if l.strip()]
        if not lines or any(SKIP_LINE.match(l) for l in lines) or len(lines) > 1:
            continue
        p = re.sub(r'(\*\*|__|`)', '', lines[0]).strip()
        p = re.sub(r'^\*(.+)\*$', r'\1', p).strip()
        n = len(p.split())
        if 40 <= n <= 250 and END.search(p):
            out.append(p)
    return out


rows, summ = [], []
for f in sorted(glob.glob(str(SRCS[0] / '*.jsonl')) + glob.glob(str(SRCS[1] / '*.jsonl'))):
    fn = pathlib.Path(f).name
    lab, variant = label_of(fn)
    if lab is None:
        print('skip (not a target label):', fn)
        continue
    n = n_text = n_ref = n_par = 0
    served, provs, lens = {}, {}, []
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            n += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = content_of(d)
            if c is None:
                continue
            n_text += 1
            sv = d['response'].get('model', '')
            served[sv] = served.get(sv, 0) + 1
            pv = d['response'].get('provider', '')
            provs[pv] = provs.get(pv, 0) + 1
            words = len(c.split())
            lens.append(words)
            ref = int(words < 60 or bool(REFUSAL.search(c[:400])))
            n_ref += ref
            ps = paragraphs(c)
            n_par += len(ps)
            for i, p in enumerate(ps):
                rows.append(dict(label=lab, variant=variant, question_id=d.get('question_id', ''), para_idx=i, n_paras_resp=len(ps),
                                 words=len(p.split()), refusal=ref, served_model=sv, provider=pv,
                                 timestamp=d.get('timestamp', ''), text=p))
    summ.append(dict(label=lab, variant=variant, file=fn, rows=n, with_text=n_text, refusal_flagged=n_ref, paragraphs=n_par,
                     median_response_words=int(np.median(lens)) if lens else 0,
                     served_models=';'.join(f'{k}:{v}' for k, v in sorted(served.items(), key=lambda x: -x[1])[:3]),
                     providers=';'.join(f'{k}:{v}' for k, v in sorted(provs.items(), key=lambda x: -x[1])[:3])))
    print(f'{lab:24s} {variant:9s} rows {n:5d} text {n_text:5d} refusal {n_ref:5d} paragraphs {n_par:6d}', flush=True)
P = pd.DataFrame(rows)
P.to_csv(OUT_TXT / 'paragraphs.csv', index=False)
S = pd.DataFrame(summ)
S['paragraphs_nonrefusal'] = [int(((P.label == r.label) & (P.variant == r.variant) & (P.refusal == 0)).sum()) for r in S.itertuples()]
S.to_csv(OUT / 'summary_by_model.csv', index=False)
missing = sorted(set(LABELS) - set(S.label))
print('missing labels:', missing)

# ---- register comparison with competition train (same statistics as prvsiyan's table + ours)
tr = pd.read_csv(ROOT / 'data/train.csv')


def stats(texts):
    t = pd.Series(list(texts), dtype=str)
    sent = t.apply(lambda s: np.mean([len(x.split()) for x in re.split(r'(?<=[.!?])\s+', s) if x.strip()] or [0]))
    return dict(n=len(t), median_words=float(t.str.split().str.len().median()),
                emdash_share=float(t.str.contains('—').mean()), curly_apos_share=float(t.str.contains('’').mean()),
                straight_apos_share=float(t.str.contains("'").mean()), curly_dquote_share=float(t.str.contains('[“”]').mean()),
                semicolon_share=float(t.str.contains(';').mean()), colon_share=float(t.str.contains(':').mean()),
                mean_sentence_words=float(sent.mean()))


comp = []
Pn = P[(P.refusal == 0) & (P.variant != 'reasoning')]
for lab in LABELS:
    a = stats(tr.text[tr.model == lab])
    b = stats(Pn.text[Pn.label == lab]) if (Pn.label == lab).any() else {k: np.nan for k in a}
    comp.append({'label': lab, **{f'train_{k}': round(v, 3) for k, v in a.items()},
                 **{f'speechmap_{k}': round(v, 3) if v == v else v for k, v in b.items()}})
C = pd.DataFrame(comp)
C.to_csv(OUT / 'style_vs_train.csv', index=False)
print('wrote', OUT_TXT / 'paragraphs.csv', len(P), 'rows |', OUT / 'summary_by_model.csv', '|', OUT / 'style_vs_train.csv')
