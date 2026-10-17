"""Manual collection of competition-style passages from chat apps (no API).

Loop: the prompt is copied to your clipboard -> paste it into a FRESH chat -> copy the model's answer ->
press Enter here -> the answer is read from the clipboard, checked and appended to the CSV.

Usage (from I:\\venv_python\\that_llm):
  .venv\\Scripts\\python.exe tools\\chat_collect.py --model claude-opus-5 --ui "Claude.ai: Claude Opus 5"
Options: --n 20 (prompts this session), --out research/ext_data/chat_manual/responses.csv
Keys after each answer: Enter = save | r = re-read clipboard | s = skip prompt | q = quit
Rules: fresh/temporary chat per prompt, memory + custom instructions OFF, copy the answer exactly (no edits).
"""
import argparse
import csv
import datetime as dt
import hashlib
import pathlib
import random
import re
import sys
import tkinter

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview',
          'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5', 'claude-opus-4-6',
          'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ROOT / 'research/ext_data/extra_api/prompts.csv'   # 1,000 public competition-style prompts (seyitkaangunes)
FIELDS = ['id', 'prompt_id', 'model', 'ui_model', 'interface', 'prompt_text', 'target_lo', 'target_hi',
          'response_raw', 'response_clean', 'n_words', 'single_paragraph', 'timestamp']

_tk = tkinter.Tk()
_tk.withdraw()


def clip_set(text):
    _tk.clipboard_clear()
    _tk.clipboard_append(text)
    _tk.update()


def clip_get():
    try:
        return _tk.clipboard_get()
    except tkinter.TclError:
        return ''


def word_range(prompt_id, model=''):
    """Per-generation target N ('about N words'), drawn from the train word-count distribution clipped to
    [40, 250] (median ~100), seeded per prompt x model. Returns (N, N) for schema compatibility."""
    r = random.Random(int(hashlib.md5((prompt_id + '|' + model).encode()).hexdigest(), 16))
    n = int(min(250, max(40, round(r.lognormvariate(4.6, 0.45)))))
    return n, n


def build_prompt(text, lo, hi):
    return f"{text.strip()} Write it as a single self-contained paragraph with no headings or lists. (about {lo} words)"


def clean(t):
    t = t.replace('\r\n', '\n').strip()
    t = re.sub(r'^\*\*(.*)\*\*$', r'\1', t, flags=re.S)     # whole-passage bold wrapper (host removed these too)
    paras = [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
    return t, len(paras) == 1 and '\n' not in t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=LABELS)
    ap.add_argument('--ui', required=True, help='exact model name the app shows, e.g. "ChatGPT: GPT-5.6 Sol"')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default=str(ROOT / 'research/ext_data/chat_manual/responses.csv'))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(out, encoding='utf-8'))) if out.exists() else []
    done = {(r['prompt_id'], r['model']) for r in rows}
    per_prompt = {}
    for r in rows:
        per_prompt[r['prompt_id']] = per_prompt.get(r['prompt_id'], 0) + 1
    prompts = list(csv.DictReader(open(PROMPTS, encoding='utf-8')))
    # prefer prompts already answered by other models (builds prompt-matched blocks), then file order
    todo = sorted([p for p in prompts if (p['prompt_id'], a.model) not in done],
                  key=lambda p: -per_prompt.get(p['prompt_id'], 0))[:a.n]
    new = not out.exists()
    f = open(out, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    saved = 0
    for i, p in enumerate(todo, 1):
        lo, hi = word_range(p['prompt_id'], a.model)
        prompt = build_prompt(p['prompt'], lo, hi)
        clip_set(prompt)
        print(f"\n[{i}/{len(todo)}] {p['prompt_id']} ({p['domain']}) -> copied to clipboard. Open a FRESH chat, paste, send.")
        print('   ', prompt[:160] + ('...' if len(prompt) > 160 else ''))
        while True:
            k = input('Copy the full answer, then press Enter (s=skip, q=quit): ').strip().lower()
            if k == 'q':
                print(f'saved {saved}')
                return
            if k == 's':
                break
            raw = clip_get()
            if raw.strip() == prompt.strip() or not raw.strip():
                print('  clipboard still holds the prompt / is empty; copy the answer first.')
                continue
            txt, single = clean(raw)
            n = len(txt.split())
            warn = [] if single else ['NOT a single paragraph']
            if not 30 <= n <= 300:
                warn.append(f'{n} words (outside 30-300)')
            print(f'  {n} words | single paragraph: {single} | {" ; ".join(warn) or "ok"}')
            print('  >', txt[:200].replace('\n', ' / ') + ('...' if len(txt) > 200 else ''))
            k2 = input('  Enter = save, r = re-read clipboard, s = skip: ').strip().lower()
            if k2 == 'r':
                continue
            if k2 == 's':
                break
            w.writerow(dict(id=hashlib.md5((p['prompt_id'] + a.model + txt).encode()).hexdigest()[:12],
                            prompt_id=p['prompt_id'], model=a.model, ui_model=a.ui, interface='web-chat',
                            prompt_text=prompt, target_lo=lo, target_hi=hi, response_raw=raw, response_clean=txt,
                            n_words=n, single_paragraph=int(single),
                            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')))
            f.flush()
            saved += 1
            break
    print(f'\nsession done: saved {saved} -> {out}')


if __name__ == '__main__':
    sys.exit(main())
