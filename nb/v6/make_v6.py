"""Build nb/v6/v6.py + ntl-v6.ipynb from the public 0.563 notebook (vnton37nguyn/ntl-lora-instyle-0-563-please-vote-me,
pulled to research/kernels4/, Apache-2.0) and our v5 Gemma section.
Changes vs 0.563 (and nothing else):
  1. the in-style Qwen2.5-7B LoRA (private kernel ntl-train-lora-qwen25-instyle, not attachable) is replaced by our
     Gemma-4-12B LoRA trained on train + prvsiyan rows (kernel ntl-lora-gemma12b-ext-train), half readout, same weight 0.17
  2. optional SpeechMap first paragraphs in the in-style extra pool (SPEECHMAP['enabled'], set per variant)
  3. GPU guard (stop instead of running on CPU)
Usage: python nb/v6/make_v6.py <variant: a|b> <n_questions>"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VAR = sys.argv[1] if len(sys.argv) > 1 else 'a'
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 300
src = (ROOT / 'research/kernels4/ntl-lora-instyle-0-563-please-vote-me/code.py').read_text(encoding='utf-8')
v5 = (ROOT / 'nb/v5/v5.py').read_text(encoding='utf-8').splitlines()
gemma = '\n'.join(v5[646:771])   # v5 section 3c2 (gemma_lora_proba + add loop)
assert gemma.startswith('# ------------------------------------------------ 3c2)') and 'add(_gk, gemma_lora_proba' in gemma


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:60], s.count(old))
    return s.replace(old, new)


head = f"""# Name That LLM - ours v6{VAR}: public 0.563 notebook (vnton37nguyn v16, built on the 0.552 / 0.549 lineage) with
#   (1) its private in-style Qwen2.5-7B LoRA replaced by our Gemma-4-12B LoRA trained on train + prvsiyan rows
#       (kernel mdmuqtadirfuad/ntl-lora-gemma12b-ext-train; folds 0-2 plain 0.4778, half readout), weight 0.17
#   (2) SpeechMap first paragraphs (xlr8harder/speechmap-data, Apache-2.0) in the in-style extra pool: {'on, ' + str(NQ) + ' questions' if VAR == 'b' else 'off'}
"""
s = head + src
s = sub(s, "W = {'stylegram': 1.516,", "W = {'stylegram': 1.516,")
s = sub(s, "\nLORA = dict(enabled=True", f"""
W['lora_q25'] = 0.0          # v6: private kernel, not attachable
W['gemma_lora_ext'] = 0.17   # v6: our in-style Gemma LoRA takes its slot and weight
GEMMA = dict(enabled=True, half=True)
GEMMA_CKPTS = [('gemma_lora_ext', 'ntl-lora-gemma12b-ext-train')]
SPEECHMAP = dict(enabled={VAR == 'b'}, n_questions={NQ})   # v6b: SpeechMap questions (>= 12 labels) added to the extra pool
LORA = dict(enabled=True""")
s = sub(s, "print('train', train.shape, 'test', test.shape, 'gpus', NGPU, flush=True)\n",
        "print('train', train.shape, 'test', test.shape, 'gpus', NGPU, flush=True)\n"
        "assert NGPU >= 1 or SMOKE, 'no GPU assigned (quota exhausted?) - stopping instead of running ~12 h on CPU'\n")
s = sub(s, """    x = x[x.groupby('prompt_id').model.transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)
""", """    x = x[x.groupby('prompt_id').model.transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)
    _sf = sorted(glob.glob('/kaggle/input/**/speechmap_p0.csv', recursive=True))
    if SPEECHMAP['enabled'] and _sf:   # v6b: same cleaning and >= 12-label rule; one SpeechMap question = one prompt
        sm = pd.read_csv(_sf[0]).rename(columns={'label': 'model'})
        sm = sm.assign(prompt_id='sm_' + sm.question_id.astype(str))[['prompt_id', 'model', 'text']]
        sm = sm[sm.model.isin(LABELS)].copy()
        sm['text'] = sm.text.astype(str).map(lambda t: re.sub(r'\\s+', ' ', re.sub(r'^\\*\\*(.*)\\*\\*$', r'\\1', t.strip())).strip())
        sm = sm[sm.text.str.split().str.len().between(30, 300)].drop_duplicates(subset=['model', 'text'])
        sm = sm.drop_duplicates(subset=['prompt_id', 'model'], keep='first')
        sm = sm[sm.groupby('prompt_id').model.transform('nunique') >= EXTRA_MIN_LABELS]
        _q = sm.prompt_id.drop_duplicates().sample(frac=1, random_state=0).tolist()[:SPEECHMAP['n_questions']]
        x = pd.concat([x, sm[sm.prompt_id.isin(_q)]], ignore_index=True)
        print('speechmap extra:', int(sm.prompt_id.isin(_q).sum()), 'rows,', len(_q), 'questions', flush=True)
    elif SPEECHMAP['enabled']:
        print('WARNING: speechmap_p0.csv not attached, SpeechMap extra rows skipped', flush=True)
""")
s = sub(s, """    run_loras(TE, TR[:3], [('lora_q25', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', 'norms.safetensors',
                            'valid_features.npz')], ('qwen2', 3584, 20), [2, 4, 6, 8, 10, 12, 16, 20])
""", "    # v6: lora_q25 (private kernel) removed; our Gemma LoRA below takes its weight\n\n\n" + gemma + "\n")
(ROOT / 'nb/v6').mkdir(exist_ok=True)
(ROOT / f'nb/v6/v6{VAR}.py').write_text(s, encoding='utf-8', newline='\n')
compile(s, f'v6{VAR}.py', 'exec')
nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.splitlines(True)}],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 4}
d = ROOT / f'nb/v6/{VAR}'
d.mkdir(exist_ok=True)
(d / f'ntl-v6{VAR}.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
m = json.loads((ROOT / 'research/kernels4/ntl-lora-instyle-0-563-please-vote-me/kernel-metadata.json').read_text(encoding='utf-8'))
m.update(id=f'mdmuqtadirfuad/ntl-v6{VAR}', title=f'ntl-v6{VAR}', code_file=f'ntl-v6{VAR}.ipynb', is_private=True)
m.pop('id_no', None)
m.pop('keywords', None)
m['kernel_sources'] = [k for k in m['kernel_sources'] if k] + ['mdmuqtadirfuad/ntl-lora-gemma12b-ext-train']
m['model_sources'] = m['model_sources'] + ['google/gemma-4/Transformers/gemma-4-12b/2']
if VAR == 'b':
    m['dataset_sources'] = m['dataset_sources'] + ['mdmuqtadirfuad/ntl-speechmap-paragraphs']
(d / 'kernel-metadata.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
print('wrote', d, len(s.splitlines()), 'lines')
