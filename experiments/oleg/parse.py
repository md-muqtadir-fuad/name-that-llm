"""Paragraph view of Oleg Zholobov's response corpus (olegzholobov/name-that-llm-response-corpus), excluding SpeechMap
(parsed separately by experiments/speechmap/parse.py).
Cell filters: view 'all'; source != speechmap; license known; text_only_suitable; not text_shared_across_labels;
not possible_reasoning_in_answer; no U+FFFD (lost characters, e.g. curly apostrophes -> style corrupted).
Paragraph rules = experiments/speechmap/parse.py (single line, 40-250 words, no headings/lists/tables, ends with
sentence punctuation, markdown markers stripped); refusals flagged.
Outputs: research/ext_data/oleg/oleg_paragraphs.parquet (git-ignored), experiments/oleg/out/summary_by_source.csv,
experiments/oleg/out/drops.csv."""
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / 'research/ext_data/oleg'
OUT = ROOT / 'experiments/oleg/out'
OUT.mkdir(parents=True, exist_ok=True)
REFUSAL = re.compile(r"\b(I (can(?:no|')t|cannot|won't|will not|am not able|'m not able|must decline|'m unable|am unable)|"
                     r"I'm sorry, but|I apologize, but|I'd rather not)\b", re.I)
SKIP_LINE = re.compile(r'^\s*(#{1,6}\s|[-*+•]\s|\d+[.)]\s|>|\|)')
END = re.compile(r'[.!?…]["”’)\]]?\s*$')


def paragraphs(text):
    out = []
    for block in re.split(r'\n\s*\n', text.replace('\r\n', '\n')):
        lines = [l for l in block.strip().split('\n') if l.strip()]
        if not lines or any(SKIP_LINE.match(l) for l in lines) or len(lines) > 1:
            continue
        p = re.sub(r'(\*\*|__|`)', '', lines[0]).strip()
        p = re.sub(r'^\*(.+)\*$', r'\1', p).strip()
        if 40 <= len(p.split()) <= 250 and END.search(p):
            out.append(p)
    return out


w = pd.read_parquet(D / 'external_all_wide.parquet')
labels = [c for c in w.columns if c not in ('id', 'prompt')]
cells = w.melt(id_vars=['id'], value_vars=labels, var_name='model', value_name='text').dropna(subset=['text'])
m = pd.read_parquet(D / 'external_cell_metadata.parquet')
m = m[m.view == 'all']
p = pd.read_parquet(D / 'external_cell_provenance.parquet', columns=['response_id', 'source', 'license', 'license_status'])
p = p.drop_duplicates('response_id')
x = cells.merge(m, on=['id', 'model'], how='left').merge(p, on='response_id', how='left')
drops = [('cells', len(x))]
for name, keep in [('metadata_found', x.response_id.notna()), ('not_speechmap', x.source.ne('speechmap')),
                   ('license_known', ~x.license_unknown.fillna(True).astype(bool)),
                   ('text_only', x.text_only_suitable.fillna(False).astype(bool)),
                   ('not_shared_across_labels', ~x.text_shared_across_labels.fillna(True).astype(bool)),
                   ('no_reasoning', ~x.possible_reasoning_in_answer.fillna(True).astype(bool)),
                   ('no_replacement_char', ~x.text.str.contains('�', regex=False))]:
    x = x[keep.loc[x.index]] if len(keep) != len(x) else x[keep]
    drops.append((name, len(x)))
rows = []
for r in x.itertuples():
    ps = paragraphs(r.text)
    ref = int(len(r.text.split()) < 60 or bool(REFUSAL.search(r.text[:400])))
    for i, t in enumerate(ps):
        rows.append(dict(label=r.model, source=r.source, prompt_id='og_' + str(r.prompt_group_id), response_id=r.response_id,
                         para_idx=i, words=len(t.split()), refusal=ref, license=r.license, text=t))
P = pd.DataFrame(rows)
drops.append(('paragraphs_all', len(P)))
drops.append(('p0_nonrefusal', int(((P.para_idx == 0) & (P.refusal == 0)).sum())))
P.to_parquet(D / 'oleg_paragraphs.parquet', index=False)
pd.DataFrame(drops, columns=['step', 'rows']).to_csv(OUT / 'drops.csv', index=False)
q = P[(P.para_idx == 0) & (P.refusal == 0)]
S = q.pivot_table(index='source', columns='label', values='text', aggfunc='count', fill_value=0)
S['total'] = S.sum(1)
S.sort_values('total', ascending=False).to_csv(OUT / 'summary_by_source.csv')
g = q.groupby('prompt_id').label.nunique()
print(pd.DataFrame(drops, columns=['step', 'rows']).to_string(index=False))
print('p0 per label:', q.label.value_counts().to_dict())
print('prompt groups:', len(g), '| >=12 labels:', int((g >= 12).sum()), '| rows in those:', int(q.prompt_id.isin(g[g >= 12].index).sum()))
print(S.total.sort_values(ascending=False).head(15).to_string())
