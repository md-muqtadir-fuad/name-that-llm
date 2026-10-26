# Kaggle Benchmarks task: generate competition-style passages with the model this run is bound to.
# Push:  kaggle b t push ntl-gen -f tools/kbench/ntl_gen_task.py -d seyitkaangunes/ntl-extra-api-passages
# Run:   kaggle b t run ntl-gen -m gpt-5.6-luna -m claude-opus-5-default ... --wait
# Fetch: kaggle b t download ntl-gen -o research/ext_data/kbench/<batch>
import glob
import hashlib
import json
import random
import time

import kaggle_benchmarks as kbench
import pandas as pd

START, N = 0, 32          # set by daily.py
MAX_OUT = 4000   # output cap: keeps the proxy's per-call cost reservation small
ONLY = {}   # set by daily.py
IDS = ["own_93d0888704", "own_933ffc7ce9", "own_bdf1e37804", "own_b317985d8a", "own_cf477eb51d", "own_f6bbf84bc4", "own_b98f6b0da9", "own_90852f26a4", "own_ee2076513c", "own_4ffbc8b7d3", "own_961494e383", "own_226ce3022d", "own_b0ebef6b7d", "own_3abd08751f", "own_96fb074273", "own_79a10896ac", "own_084db456dd", "own_4250ca8486", "own_b7843d45fa", "own_52a804cdd6", "own_f16dd42a17", "own_a5b14fb95b", "own_5f020c5449", "own_def3258990", "own_0fd742c05d", "own_d96c88c2d3", "own_9f56383a6d", "own_c13c3b86ac", "own_afb5f81092", "own_6ce156b97a", "own_f10ea1993e", "own_ebe28a9939"]   # set by daily.py
BATCH = 'o002'


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
