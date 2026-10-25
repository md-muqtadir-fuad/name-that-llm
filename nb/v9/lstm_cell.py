# LSTM member (E050): word-level BiLSTM, two-stage (public pool_v4 passages, then all 2,699 host rows); CPU inference.
# Runs in every mode (also the 3-row preview) so the commit run checks it; the final cell mixes it in at 10% before balancing.
import json as _json, os as _os, re as _re
from pathlib import Path as _Path
import numpy as _np, pandas as _pd, torch as _torch
from torch import nn as _nn
_root = _Path('/kaggle/input') if _Path('/kaggle/input').exists() else _Path(_os.environ['NTL_INPUT'])
_work = _Path('/kaggle/working') if _Path('/kaggle/working').exists() else _Path(_os.environ['NTL_WORK'])
_cfg_path = next(_root.rglob('final_lstm.json'))
_cfg = _json.loads(_cfg_path.read_text(encoding='utf-8'))


class _BiLSTM(_nn.Module):
    def __init__(self, n_vocab, k=18):
        super().__init__()
        self.emb = _nn.Embedding(n_vocab, 128, padding_idx=0)
        self.lstm = _nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.att = _nn.Linear(256, 1)
        self.drop = _nn.Dropout(0.3)
        self.fc = _nn.Linear(512, k)

    def forward(self, x):
        lens = (x > 0).sum(1).clamp(min=1)
        packed = _nn.utils.rnn.pack_padded_sequence(self.emb(x), lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = _nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=x.shape[1])
        m = (x > 0).unsqueeze(2)
        a = _torch.softmax(self.att(h).masked_fill(~m, -1e4), 1)
        mx = h.masked_fill(~m, -1e4).max(1).values
        return self.fc(_torch.cat([(a * h).sum(1), mx], 1))


_dev = 'cuda' if _torch.cuda.is_available() else 'cpu'   # the GPUs are still free at this point; released below
_lstm = _BiLSTM(len(_cfg['vocab']) + 2)
_lstm.load_state_dict(_torch.load(_cfg_path.with_name('final_lstm.pt'), map_location='cpu'))
_lstm.to(_dev).eval()
_tok = _re.compile(_cfg['tok_regex'])
_lstm_test = _pd.read_csv(DATA / 'test.csv', dtype=str)
_seqs = [[_cfg['vocab'].get(w, 1) for w in _tok.findall(t)[:_cfg['hp']['max_len']]] or [1] for t in _lstm_test.text]
_order = _np.argsort([len(s) for s in _seqs])
_lstm_p = _np.zeros((len(_seqs), len(_cfg['labels'])), dtype=_np.float64)
with _torch.no_grad():
    for _s in range(0, len(_order), 128):
        _b = _order[_s:_s + 128]
        _x = _torch.zeros((len(_b), max(len(_seqs[i]) for i in _b)), dtype=_torch.long)
        for _r, _i in enumerate(_b):
            _x[_r, :len(_seqs[_i])] = _torch.tensor(_seqs[_i])
        _lstm_p[_b] = _torch.softmax(_lstm(_x.to(_dev)).double(), 1).cpu().numpy()
del _lstm
if _dev == 'cuda':
    _torch.cuda.empty_cache()
assert _np.isfinite(_lstm_p).all() and _np.allclose(_lstm_p.sum(1), 1)
_np.savez_compressed(_work / 'lstm_probabilities.npz', ids=_lstm_test.id.to_numpy(dtype=str),
                     labels=_np.asarray(_cfg['labels']), probabilities=_lstm_p)
print(f'LSTM_MEMBER {len(_seqs)} rows scored', flush=True)
