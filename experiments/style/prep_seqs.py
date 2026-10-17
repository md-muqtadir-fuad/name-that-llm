"""Precompute token-piece sequences (LLM BPE tokenizers) and spaCy POS/dep sequences for every train text.
Saved as cache/<name>.json (list of space-separated symbol strings), consumed by views.transformed()."""
import json
import sys
import time

from tokenizers import Tokenizer

from common import CACHE, ROOT, TEXTS

TOK = {'tok_qwen3': 'Qwen_Qwen3-8B-Base', 'tok_glm4': 'zai-org_GLM-4-9B-0414', 'tok_gemma3': 'unsloth_gemma-3-1b-pt',
       'tok_o200k': 'Xenova_gpt-4o'}
what = sys.argv[1:] or ['tok', 'spacy']
if 'tok' in what:
    for name, d in TOK.items():
        t0 = time.time()
        tk = Tokenizer.from_file(str(ROOT / 'research/ext_data/tok' / d / 'tokenizer.json'))
        enc = tk.encode_batch(TEXTS, add_special_tokens=False)
        seqs = [' '.join(f't{i}' for i in e.ids) for e in enc]
        (CACHE / f'{name}.json').write_text(json.dumps(seqs), encoding='utf-8')
        print(name, 'mean pieces', sum(len(e.ids) for e in enc) / len(enc), f'{time.time() - t0:.1f}s', flush=True)
if 'spacy' in what:
    import spacy
    t0 = time.time()
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    pos, tag, dep, posw = [], [], [], []
    for doc in nlp.pipe(TEXTS, batch_size=64):
        pos.append(' '.join(t.pos_ for t in doc if not t.is_space))
        tag.append(' '.join(t.tag_ for t in doc if not t.is_space))
        # dependency label + head POS (syntactic frame) per token
        dep.append(' '.join(f'{t.dep_}>{t.head.pos_}' for t in doc if not t.is_space))
        # POS with function words / punctuation kept verbatim (mixed abstraction)
        posw.append(' '.join(t.lower_ if (t.is_stop or t.is_punct) else t.pos_ for t in doc if not t.is_space))
    for name, seq in (('pos_upos', pos), ('pos_tag', tag), ('pos_dep', dep), ('pos_mixed', posw)):
        (CACHE / f'{name}.json').write_text(json.dumps(seq), encoding='utf-8')
    print('spacy', f'{time.time() - t0:.1f}s', flush=True)
