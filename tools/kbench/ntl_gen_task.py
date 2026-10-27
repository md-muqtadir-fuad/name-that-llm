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

START, N = 0, 40          # set by daily.py
MAX_OUT = 4000   # output cap: keeps the proxy's per-call cost reservation small
ONLY = {}   # set by daily.py
IDS = ["own_589162e8a4", "own_caaa860085", "own_d80c2f3dd5", "own_46b031c151", "own_4582fc651d", "own_06f76827b2", "own_ba554e87c5", "own_3d97b0ad0e", "own_bd06a2de82", "own_988395951b", "own_a5ce7f776b", "own_c45adee7e7", "own_bb5531e105", "own_585839a607", "own_76ab46509f", "own_4c849552bc", "own_b966a9d0ef", "own_1de85e3e93", "own_122dc2c664", "own_bd0c819600", "own_d5b0f3d949", "own_8e90927922", "own_7568a3bf40", "own_6ecd137bf7", "own_9bcb277b1f", "own_632c62359d", "own_38539e5971", "own_29544b19f4", "own_42e6afdd8f", "own_f188c4f976", "own_d8c589d23f", "own_8c5f20a762", "own_df3190caae", "own_2c9fb46f3b", "own_a7178d7e07", "own_98576fa87c", "own_1058931dd1", "own_d67292d54e", "own_f1280c4961", "own_70e474ab37"]   # set by daily.py
BATCH = 'o003'


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
