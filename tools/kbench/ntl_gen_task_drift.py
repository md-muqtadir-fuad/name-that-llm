# FROZEN drift panel (month 0 = b002, 2026-10-18; b001 was discarded as chat-contaminated): the b002 task code
# (verified: reproduces all 836 b002/b002r word targets) on the same window, prompts g0063-g0122 of
# seyitkaangunes/ntl-extra-api-passages. Run monthly via tools/kbench/drift.py; do not edit.
import glob
import hashlib
import json
import random
import time

import kaggle_benchmarks as kbench
import pandas as pd

START, N = 63, 60
MAX_OUT = 4000   # output cap: keeps the proxy's per-call cost reservation small
ONLY = {}   # failed-pair retries only (set by drift.py)
IDS = []
BATCH = 'drift_YYYYMM'   # set at run time only


def word_range(prompt_id, model=''):
    """Per-generation target N ('about N words'), drawn from the train word-count distribution clipped to
    [40, 250] (median ~100), seeded per prompt x model. Returns (N, N) for schema compatibility."""
    r = random.Random(int(hashlib.md5((prompt_id + '|' + model).encode()).hexdigest(), 16))
    n = int(min(250, max(40, round(r.lognormvariate(4.6, 0.45)))))
    return n, n


def build_prompt(text, lo, hi):
    return f"{text.strip()} Write it as a single self-contained paragraph with no headings or lists. (about {lo} words)"


@kbench.task(name="ntl-gen", description="Generate short competition-style passages for each prompt")
def ntl_gen(llm) -> None:
    hits = glob.glob('/kaggle/input/**/prompts.csv', recursive=True)
    prompts = pd.read_csv(hits[0])
    prompts = prompts[prompts.prompt_id.isin(IDS)] if IDS else prompts.iloc[START:START + N]
    name = str(getattr(llm, 'name', None) or getattr(llm, 'model', None) or llm)
    rows = []
    if ONLY:
        prompts = prompts[prompts.prompt_id.isin(ONLY.get(name, []))]
    for p in prompts.itertuples():
        lo, hi = word_range(p.prompt_id, name)
        q = build_prompt(p.prompt, lo, hi)
        t0 = time.time()
        text, err = '', ''
        for extra in ({'max_completion_tokens': MAX_OUT}, {'max_tokens': MAX_OUT}, {'max_output_tokens': MAX_OUT}):
            try:
                with kbench.chats.new(name=p.prompt_id, orphan=True):   # fresh, isolated chat per prompt
                    text, err = llm.prompt(q, extra_api_params=extra), ''
                break
            except Exception as e:  # only a wrong parameter NAME moves on to the next name; keep the first real error
                msg = repr(e)
                wrong_name = isinstance(e, TypeError) and 'unexpected keyword' in msg or                     ('400' in msg and ('nsupported parameter' in msg or 'not supported' in msg))
                if not wrong_name:
                    err = msg[:300]
                    break
                err = err or msg[:300]
        rows.append(dict(batch=BATCH, prompt_id=p.prompt_id, llm=name, prompt_text=q, target_lo=lo, target_hi=hi,
                         response=text if isinstance(text, str) else str(text), error=err,
                         seconds=round(time.time() - t0, 2)))
    safe = ''.join(ch if ch.isalnum() or ch in '-._' else '_' for ch in name)
    pd.DataFrame(rows).to_csv(f'/kaggle/working/ntl_gen_{BATCH}_{safe}.csv', index=False)
    print(json.dumps(dict(llm=name, n=len(rows), errors=sum(bool(r['error']) for r in rows))))


ntl_gen.run(kbench.llm)
