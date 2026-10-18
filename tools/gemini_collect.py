"""Collect competition-style passages from the Gemini API free tier (gemini-3.6/3.7/3.8-flash).

Setup (PowerShell, do NOT paste the key into any chat):
  $env:GEMINI_API_KEY = "your-key"
Usage (from I:\\venv_python\\that_llm):
  .venv\\Scripts\\python.exe tools\\gemini_collect.py --list            # show exact model ids available to your key
  .venv\\Scripts\\python.exe tools\\gemini_collect.py                   # run today's quota for all 3 models
  .venv\\Scripts\\python.exe tools\\gemini_collect.py --models gemini-3.7-flash gemini-3.8-flash --per-day 20
Respects 5 requests/min and the daily cap per model (counted per Pacific day, like Google's reset); stops a
model early on HTTP 429. Same prompts in the same order for every model -> prompt-matched blocks. Resumable.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ROOT / 'research/ext_data/extra_api/prompts.csv'
OUT = ROOT / 'research/ext_data/gemini_api/responses.csv'
STATE = ROOT / 'research/ext_data/gemini_api/quota_state.json'
API = 'https://generativelanguage.googleapis.com/v1beta'
DEFAULT_MODELS = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
FIELDS = ['id', 'prompt_id', 'model', 'api_model', 'model_version', 'interface', 'prompt_text', 'target_lo',
          'target_hi', 'candidate', 'finish_reason', 'response_raw', 'response_clean', 'n_words', 'single_paragraph',
          'prompt_tokens', 'output_tokens', 'thought_tokens', 'timestamp']


def word_range(prompt_id, model=''):
    """Per-generation target N ('about N words'), drawn from the train word-count distribution clipped to
    [40, 250] (median ~100), seeded per prompt x model. Returns (N, N) for schema compatibility."""
    r = random.Random(int(hashlib.md5((prompt_id + '|' + model).encode()).hexdigest(), 16))
    n = int(min(250, max(40, round(r.lognormvariate(4.6, 0.45)))))
    return n, n


def build_prompt(text, lo, hi):
    return f"{text.strip()} Write it as a single self-contained paragraph with no headings or lists. (about {lo} words)"


def clean(t):
    import re
    t = t.replace('\r\n', '\n').strip()
    t = re.sub(r'^\*\*(.*)\*\*$', r'\1', t, flags=re.S)
    return t, '\n' not in t


def call(path, key, body=None, tries=6):
    for t in range(tries):
        try:
            return _call(path, key, body)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            transient = not isinstance(e, urllib.error.HTTPError) or e.code in (500, 502, 503, 504)
            if not transient or t == tries - 1:
                raise
            time.sleep(20 * (t + 1))


def _call(path, key, body=None):
    req = urllib.request.Request(f'{API}/{path}{"&" if "?" in path else "?"}key={key}',
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def pacific_day():
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=8)).date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='competition labels to collect')
    ap.add_argument('--api-ids', nargs='*', default=None, help='override API model ids (same order as --models)')
    ap.add_argument('--per-day', type=int, default=20)
    ap.add_argument('--rpm', type=int, default=5)
    ap.add_argument('--candidates', type=int, default=4, help='answers per request (falls back to 1 if rejected)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    key = os.environ.get('GEMINI_API_KEY')
    envf = ROOT / '.env'
    if not key and envf.exists():   # accepts KEY=VAL, export KEY=VAL and PowerShell $env:KEY = "VAL"
        import re
        for line in envf.read_text(encoding='utf-8-sig').splitlines():
            m = re.match(r'\s*(?:export\s+|\$env:)?GEMINI_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                key = m.group(1)
    if not key:
        sys.exit('set GEMINI_API_KEY first (PowerShell: $env:GEMINI_API_KEY = "...")')
    if a.list:
        ms = call('models?pageSize=1000', key).get('models', [])
        for m in ms:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(m['name'].split('/')[-1], '|', m.get('displayName'))
        return
    api_ids = a.api_ids or a.models
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    today = pacific_day()
    rows = list(csv.DictReader(open(OUT, encoding='utf-8'))) if OUT.exists() else []
    done = {(r['prompt_id'], r['model']) for r in rows}
    prompts = list(csv.DictReader(open(PROMPTS, encoding='utf-8')))
    f = open(OUT, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if not rows:
        w.writeheader()
    gap = 60.0 / a.rpm + 1.0
    for label, mid in zip(a.models, api_ids):
        used = state.get(label, {}).get(today, 0)
        todo = [p for p in prompts if (p['prompt_id'], label) not in done][:max(0, a.per_day - used)]
        print(f'{label} ({mid}): {used} used today, {len(todo)} to request')
        ncand = a.candidates
        for p in todo:
            lo, hi = word_range(p['prompt_id'], label)
            prompt = build_prompt(p['prompt'], lo, hi)
            body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                    'generationConfig': {'candidateCount': ncand}}
            t0 = time.time()
            try:
                res = call(f'models/{mid}:generateContent', key, body)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors='replace')[:300]
                if e.code == 400 and ncand > 1 and 'andidate' in msg:
                    print('  candidateCount>1 not supported; retrying with 1')
                    ncand = 1
                    body['generationConfig']['candidateCount'] = 1
                    res = call(f'models/{mid}:generateContent', key, body)
                elif e.code == 429:
                    print(f'  429 quota/rate limit for {label}: {msg[:120]} -> next model')
                    break
                else:
                    print(f'  HTTP {e.code}: {msg}')
                    break
            state.setdefault(label, {})[today] = state.get(label, {}).get(today, 0) + 1
            STATE.write_text(json.dumps(state, indent=1))
            ver = res.get('modelVersion', '')
            if not ver.startswith(label):
                print(f'  VERSION MISMATCH: requested {label}, served {ver!r} -> not saved; stopping this model')
                break
            um = res.get('usageMetadata', {})
            for ci, c in enumerate(res.get('candidates', [])):
                raw = ''.join(part.get('text', '') for part in c.get('content', {}).get('parts', [])
                              if not part.get('thought'))
                if not raw.strip():
                    continue
                txt, single = clean(raw)
                w.writerow(dict(id=hashlib.md5((p['prompt_id'] + label + str(ci) + txt).encode()).hexdigest()[:12],
                                prompt_id=p['prompt_id'], model=label, api_model=mid,
                                model_version=res.get('modelVersion', ''), interface='gemini-api-free',
                                prompt_text=prompt, target_lo=lo, target_hi=hi, candidate=ci,
                                finish_reason=c.get('finishReason', ''), response_raw=raw, response_clean=txt,
                                n_words=len(txt.split()), single_paragraph=int(single),
                                prompt_tokens=um.get('promptTokenCount', ''),
                                output_tokens=um.get('candidatesTokenCount', ''),
                                thought_tokens=um.get('thoughtsTokenCount', ''),
                                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')))
            f.flush()
            print(f"  {p['prompt_id']}: {len(res.get('candidates', []))} answer(s), version {res.get('modelVersion', '?')}")
            time.sleep(max(0.0, gap - (time.time() - t0)))
    f.close()
    print('done ->', OUT)


if __name__ == '__main__':
    main()
