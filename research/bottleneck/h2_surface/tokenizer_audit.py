"""Empirical check of what the members' tokenizers do to surface characters (offline tokenizer.json files only).

Qwen3-8B-Base (proxy for the Qwen3 / Qwen2.5 LoRA, tokattn and frozen heads), GLM-4-9B (glm9b head),
BGE-small-en-v1.5 = BERT-uncased WordPiece (proxy for electra-large-discriminator, which ships the same uncased vocab).
DeBERTa-v3 / desklib / ultrabert tokenizers are not available offline -> not verified here.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from tokenizers import Tokenizer

OUT = Path('research/bottleneck/h2_surface/out'); OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv('data/train.csv')
texts = train.text.tolist()
TOKS = {'qwen3_8b': 'research/ext_data/tok/Qwen_Qwen3-8B-Base/tokenizer.json',
        'glm4_9b': 'research/ext_data/tok/zai-org_GLM-4-9B-0414/tokenizer.json',
        'bert_uncased(bge-small)': next(Path.home().glob('.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*/tokenizer.json'))}
PROBE = 'He said “wait” — then she’d gone… It’s 5–7 km; “Fine,” she\'d say. Café ÉTÉ x² − y'
rows = []
for name, path in TOKS.items():
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    tok = Tokenizer.from_file(str(path))
    enc = tok.encode_batch(texts, add_special_tokens=False)
    lens = np.array([len(e.ids) for e in enc])
    dec = [tok.decode(e.ids) for e in enc]
    unk = tok.token_to_id('[UNK]')
    r = dict(tokenizer=name, normalizer=json.dumps(cfg.get('normalizer'))[:200],
             len_median=int(np.median(lens)), len_max=int(lens.max()),
             frac_over_447=float((lens > 447).mean()), frac_over_512=float((lens > 512).mean()),
             roundtrip_exact=float(np.mean([d == t for d, t in zip(dec, texts)])),
             roundtrip_ws_insensitive=float(np.mean([re.sub(r'\s+', '', d) == re.sub(r'\s+', '', t) for d, t in zip(dec, texts)])),
             frac_with_unk=float(np.mean([unk in e.ids for e in enc])) if unk is not None else 0.0,
             probe_tokens=' | '.join(tok.encode(PROBE, add_special_tokens=False).tokens))
    rows.append(r)
    print(json.dumps(r, ensure_ascii=False, indent=1))
pd.DataFrame(rows).to_csv(OUT / 'tokenizer_audit.csv', index=False)

# byte / word caps of experiments/cnn_lstm (CNN: first 1024 UTF-8 bytes; LSTM: first 256 word/punct tokens)
nbytes = np.array([len(t.encode('utf-8')) for t in texts])
ntoks = np.array([len(re.findall(r"\w+|[^\w\s]", t)) for t in texts])
print(f'cnn_lstm: frac >1024 bytes {np.mean(nbytes > 1024):.3f} (max {nbytes.max()}), frac >256 tokens {np.mean(ntoks > 256):.3f} (max {ntoks.max()})')
