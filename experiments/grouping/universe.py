"""Passage universe for the grouping harness: competition train + external distractor passages.
The same filter is reproduced verbatim in kernel/featcache_grouping.py (GPU feature cache), so rows align by uid.
Distractors (never labelled by competition prompt): seyitkaangunes/ntl-extra-api-passages (1000 new prompts x 5 Claude
models, generated to resemble the competition prompt mix) and olegzholobov/name-that-llm-response-corpus short view
(40-250 words), restricted to single-paragraph answers with a verified user prompt (prompt string = group key)."""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


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


def build_universe():
    tr = pd.read_csv(ROOT / 'data/train.csv')
    t = pd.DataFrame({'uid': 'train:' + tr['id'].astype(str), 'src': 'train', 'pkey': 'train:' + tr['prompt_id'],
                      'model': tr['model'], 'text': tr['text']})
    ex = extra_filter(pd.read_csv(ROOT / 'research/ext_data/extra_api/ntl_extra_passages_claude.csv'))
    ol = oleg_filter(pd.read_parquet(ROOT / 'research/ext_data/oleg/external_short_wide.parquet'))
    u = pd.concat([t, ex, ol], ignore_index=True)
    assert u['uid'].is_unique
    return u


if __name__ == '__main__':
    u = build_universe()
    print(u.groupby('src').agg(n=('uid', 'size'), prompts=('pkey', 'nunique')))
    print(u.groupby('src')['text'].apply(lambda s: s.str.split().str.len().median()))
    u.to_parquet(ROOT / 'experiments/grouping/feats/universe.parquet', index=False)
