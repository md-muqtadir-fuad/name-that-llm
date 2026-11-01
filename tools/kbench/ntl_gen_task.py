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
IDS = ["own_bdf069ff30", "own_696604a337", "own_a7504ec527", "own_7cce095319", "own_237bc7c573", "own_bec312f4e1", "own_efe7866691", "own_83d0d668eb", "own_d07448adb1", "own_3e655d7968", "own_fc005a8278", "own_d6371b0182", "own_fc9936a979", "own_766aa3448d", "own_ed922a793e", "own_ea194c6a20", "own_7740056c08", "own_af56a6279a", "own_c50ec6d6b4", "own_192c5c7454", "own_c6d521487d", "own_cce73ca74e", "own_1970a44aee", "own_c6d938fe17", "own_b3887eb307", "own_df3cf82d2a", "own_f7bb2a941b", "own_ff0c768385", "own_f9d65e602a", "own_51785d225e", "own_7b87248f8e", "own_bbf8de0eab", "own_e366d8c3a6", "own_3da4b86062", "own_b02f8e7e1f", "own_7f8aea36b7", "own_8d3327b13d", "own_e4c01e8a30", "own_42a170b335", "own_347129c355"]   # set by daily.py
BATCH = 'o004'


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
