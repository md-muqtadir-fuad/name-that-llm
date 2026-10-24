"""Clean user-pasted public questions (Reddit, Facebook, ...) into a prompts.csv for tools/kbench (same columns as
seyitkaangunes/ntl-extra-api-passages prompts.csv, plus source).
Input: research/ext_data/own_prompts/raw/*.txt; one prompt per line or blank-line separated; optional [source] tag in front.
Steps: strip PII (URLs, emails, phones, @mentions, u/ and r/ names), Reddit/Facebook boilerplate, emojis and hashtags;
keep English questions of 5-120 words; drop refusal-prone topics (wasted quota); exact and near-duplicate removal (TF-IDF
cosine >= 0.85) within the set and against the existing 1,000 prompts; heuristic domain / task_type labels.
Output: research/ext_data/own_prompts/prompts.csv and clean_report.csv (every raw item with its keep/drop reason).
Run from the repo root: python tools/prompts/clean_prompts.py"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

D = Path('research/ext_data/own_prompts')
EXISTING = Path('research/ext_data/extra_api/prompts.csv')
PII = [(r'https?://\S+|www\.\S+', ' '), (r'[\w.+-]+@[\w-]+\.[\w.]+', ' '), (r'\+?\d[\d\s().-]{7,}\d', ' '),
       (r'(?<!\w)@\w+', ' '), (r'(?<!\w)/?u/\w+', ' '), (r'(?<!\w)/?r/\w+', ' '), (r'#\w+', ' ')]
BOILER = re.compile(r'^(edit|update|tl;?dr|eta|ps)\b.*$|\b(upvote|downvote|karma|subreddit|this sub|mods?|repost|'
                    r'link in bio|see more|comments?|share|like and)\b', re.I | re.M)
EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿️‍]+')
UNSAFE = re.compile(r'\b(suicide|kill (myself|him|her|them)|self[- ]harm|porn|nsfw|nude|drugs? (dealer|buy)|weapon|'
                    r'bomb|hack(ing)? into|steal|meth|cocaine|terroris\w*|racist joke)\b', re.I)
DOMAINS = {'relationships': 'boyfriend girlfriend husband wife partner dating marriage friend breakup family parents',
           'health': 'health doctor sleep diet pain anxiety depression exercise symptom medicine weight',
           'money': 'money salary budget invest debt loan rent save tax job career interview boss',
           'technology': 'computer phone software app code programming ai internet laptop data',
           'science': 'physics chemistry biology space planet universe energy evolution atom climate moon sun star '
                      'earth ocean animal brain gravity light',
           'history': 'history war ancient empire century king revolution medieval',
           'philosophy': 'meaning life purpose moral ethics free will consciousness truth happiness',
           'education': 'school college university study exam learn teacher degree student',
           'food': 'cook recipe food eat kitchen bake restaurant',
           'travel': 'travel trip country city visit flight tourist',
           'arts': 'music book movie film art paint novel song write writer',
           'sports': 'sport football soccer game team player gym run',
           'society': 'politics government law society culture religion news social media'}
TASKS = [('advice', r'\b(should i|what should|how do i|how can i|any advice|tips?|help me)\b'),
         ('compare', r'\b(vs\.?|versus|better than|difference between|compare)\b'),
         ('argue/opinion', r'\b(do you think|is it (ok|wrong|right|worth)|unpopular opinion|agree|why do people)\b'),
         ('explain concept', r'\b(why|how does|what is|what are|explain|how do)\b'),
         ('story/fiction', r'\b(story|imagine|what if|write about)\b')]


QSTART = set('what why how is are can could should would will do does did which who where when if explain describe '
              'write tell compare imagine suppose'.split())


def is_question(t):
    """A post title that works as a prompt: a question ('?' at the end or mid-text) or a request opener
    ('How I went ...' style statements are excluded)."""
    w = t.lower().split()
    return t.rstrip().endswith('?') or '? ' in t or (len(w) > 1 and w[0].strip('(') in QSTART and w[1] != 'i')


def reddit_titles(text):
    """Post titles from a saved Reddit search page: the line just above each 'r/<subreddit>' line
    (comment snippets, vote counts and page chrome are ignored)."""
    lines = [x.strip() for x in text.splitlines()]
    out = []
    for i, x in enumerate(lines):
        if re.fullmatch(r'r/\w+', x):
            j = i - 1
            while j >= 0 and not lines[j]:
                j -= 1
            if j >= 0 and not re.fullmatch(r'r/\w+|·|.*\bago', lines[j]) and lines[j] not in out:
                out.append(lines[j])
    return out


def scraped_titles(path):
    """Post titles from a table-scraper CSV of a Reddit listing: a cell is a title when its text matches the title slug of
    a /comments/<id>/<slug>/ URL in the same file (post bodies, usernames, ads and page chrome never match a slug)."""
    import csv
    csv.field_size_limit(10 ** 8)
    cells = [c.strip() for row in csv.reader(open(path, encoding='utf-8', errors='replace')) for c in row if c.strip()]
    slugs = {m for c in cells for m in re.findall(r'reddit\.com/r/\w+/comments/\w+/([a-z0-9_]+)', c)}
    by_head = {}
    for s in slugs:
        by_head.setdefault(s[:12], []).append(s)
    out = []
    for c in cells:
        if c.startswith('http') or len(c) > 400 or c in out:
            continue
        n = re.sub(r'[^a-z0-9]+', '_', re.sub(r"['’]", '', c.lower())).strip('_')
        for s in by_head.get(n[:12], []):
            k = min(len(s), len(n), 40)
            if k >= 12 and n[:k] == s[:k]:
                out.append(c)
                break
    return out


def clean(t):
    t = t.replace('❔', '?')
    t = re.sub(r'\b(\w+ )?of reddit\b,?|\breddit,? ', ' ', t, flags=re.I)   # 'Physicist of reddit, what ...'
    for pat, rep in PII:
        t = re.sub(pat, rep, t)
    t = re.sub(r'\b(edit|update|eta|ps|p\.s\.|tl;?dr)\s*\d*\s*[:\-].*$', ' ', t, flags=re.I | re.S)   # trailing notes
    t = BOILER.sub(' ', EMOJI.sub(' ', t))
    t = re.sub(r'\s+', ' ', t).strip(' -–—*>"\'')
    return t


def label_domain(t):
    w = set(re.findall(r'[a-z]+', t.lower()))
    scores = {d: sum(any(x.startswith(k) for x in w) for k in v.split()) for d, v in DOMAINS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else 'general'


def main():
    items = []
    for f in sorted((D / 'raw').glob('*.csv')):   # table-scraper exports of Reddit listings
        items += [dict(file=f.name, source='reddit', raw=t) for t in scraped_titles(f)]
    for f in sorted((D / 'raw').glob('*.txt')):
        text = f.read_text(encoding='utf-8')
        if len(re.findall(r'(?m)^r/\w+\s*$', text)) >= 3 and 'votes·' in text:   # a saved Reddit search page
            items += [dict(file=f.name, source='reddit', raw=t) for t in reddit_titles(text)]
            continue
        for block in re.split(r'\n\s*\n|\n(?=\s*\[)', text):
            for line in [block] if '\n' not in block.strip() else block.strip().split('\n'):
                m = re.match(r'\s*\[(\w+)\]\s*(.*)', line, re.S)
                src, text = (m.group(1).lower(), m.group(2)) if m else ('unspecified', line)
                if text.strip():
                    items.append(dict(file=f.name, source=src, raw=text.strip()))
    rep = pd.DataFrame(items)
    rep['prompt'] = rep.raw.map(clean)
    nw = rep.prompt.str.split().str.len()
    ascii_share = rep.prompt.map(lambda s: sum(c.isascii() for c in s) / max(len(s), 1))
    rep['reason'] = np.select([nw < 4, nw > 120, ascii_share < 0.95, rep.prompt.map(lambda s: bool(UNSAFE.search(s))),
                               ~rep.prompt.map(is_question)],
                              ['too short', 'too long', 'not English', 'refusal-prone topic', 'not a question/request'], '')
    norm = rep.prompt.str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
    rep.loc[(rep.reason == '') & norm.duplicated(), 'reason'] = 'exact duplicate'
    keep = rep.index[rep.reason == '']
    old = pd.read_csv(EXISTING).prompt.tolist() if EXISTING.exists() else []
    if len(keep):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(old + rep.prompt[keep].tolist())
        X = vec.transform(rep.prompt[keep])
        if old:
            near_old = cosine_similarity(X, vec.transform(old)).max(1) >= 0.85
            rep.loc[keep[near_old], 'reason'] = 'near-duplicate of existing prompts'
        S = cosine_similarity(X)
        np.fill_diagonal(S, 0)
        for a in range(len(keep)):
            if rep.at[keep[a], 'reason'] == '' and (S[a, :a][rep.loc[keep[:a], 'reason'].eq('').to_numpy()] >= 0.85).any():
                rep.at[keep[a], 'reason'] = 'near-duplicate within set'
    out = rep[rep.reason == ''].copy()
    out['domain'] = out.prompt.map(label_domain)
    out['task_type'] = out.prompt.map(lambda t: next((k for k, p in TASKS if re.search(p, t, re.I)),
                                                     'open question' if '?' in t else 'describe/narrate'))
    out['topic'] = out.prompt.str.split().str[:8].str.join(' ')
    out['prompt_id'] = [f'own{i:05d}' for i in range(len(out))]
    out[['prompt_id', 'domain', 'task_type', 'topic', 'prompt', 'source']].to_csv(D / 'prompts.csv', index=False)
    rep.assign(reason=rep.reason.replace('', 'kept')).to_csv(D / 'clean_report.csv', index=False)
    print(f'{len(rep)} raw items -> {len(out)} prompts')
    print(rep.reason.replace('', 'kept').value_counts().to_string())
    print(out.domain.value_counts().to_string())
    print(out.task_type.value_counts().to_string())


if __name__ == '__main__':
    main()
