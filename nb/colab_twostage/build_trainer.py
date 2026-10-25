"""Builds nb/colab_twostage/twostage_train.py from the proven Kaggle trainer nb/lora_gemma_ext/ntl_lora_gemma12b_ext_train.py
(worker kept verbatim except the checked substitutions below; new header and single-GPU launcher).
Run from the repo root: python nb/colab_twostage/build_trainer.py"""
import ast
from pathlib import Path

SRC = Path('nb/lora_gemma_ext/ntl_lora_gemma12b_ext_train.py').read_text(encoding='utf-8')
HEADER = Path('nb/colab_twostage/parts/header.py').read_text(encoding='utf-8')
LAUNCHER = Path('nb/colab_twostage/parts/launcher.py').read_text(encoding='utf-8')
worker = SRC[SRC.index("WORKER_SRC = r'''"):SRC.index('# ====================================================================== launcher')]


def sub(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


w = worker
w = sub(w, "WORK = Path('/kaggle/working')\nOUT = WORK / 'lora_gemma'", "WORK = Path(os.environ['NTL_WORK'])\nOUT = WORK / 'lora_twostage'")
w = sub(w, "groups = train.prompt_id.to_numpy(dtype=str)\nids = train.id.to_numpy(dtype=str)\n",
        "groups = train.prompt_id.to_numpy(dtype=str)\nids = train.id.to_numpy(dtype=str)\n"
        "# prompt-masked softmax: a pool row may only be one of the labels that answered its pool prompt\n"
        "is_pool = train.id.str.startswith('pool_').to_numpy()\n"
        "ALLOWED = np.ones((len(train), K), dtype=bool)\n"
        "for _g in np.unique(groups[is_pool]):\n"
        "    _i = np.flatnonzero((groups == _g) & is_pool)\n"
        "    _a = np.zeros(K, dtype=bool)\n"
        "    _a[np.unique(y[_i])] = True\n"
        "    ALLOWED[_i] = _a\n")
w = sub(w, "    lab = nn.Linear(D, KL).to(device)\n    drop = nn.Dropout(HP['head_dropout'])\n",
        "    lab = nn.Linear(D, KL).to(device)\n"
        "    init = jp.get('init_from')\n"
        "    if init:  # continue from an earlier job (stage chaining): LoRA tensors + heads\n"
        "        src = OUT / init\n"
        "        assert (src / 'COMPLETE.json').exists(), f'{name} needs {init} first'\n"
        "        st = torch.load(src / 'lora_state.pt', map_location='cpu')\n"
        "        assert len(st) == len(lora_params) and all(a.shape == b.shape for a, b in zip(st, lora_params))\n"
        "        with torch.no_grad():\n"
        "            for p, s in zip(lora_params, st):\n"
        "                p.copy_(s.to(device))\n"
        "        from safetensors.torch import load_file\n"
        "        hs = load_file(str(src / 'heads.safetensors'))\n"
        "        for prefix, module in (('norms', norms), ('plain', plain), ('centered', centered), ('lab', lab)):\n"
        "            module.load_state_dict({k[len(prefix) + 1:]: v.to(device) for k, v in hs.items() if k.startswith(prefix + '.')})\n"
        "        log(f'job {name}: initialised from {init}')\n"
        "    drop = nn.Dropout(HP['head_dropout'])\n")
w = sub(w, "    opt = torch.optim.AdamW([{'params': lora_params, 'lr': HP['lr_adapter']},\n"
           "                             {'params': head_params, 'lr': HP['lr_head']}], weight_decay=HP['weight_decay'])",
        "    lrm = float(jp.get('lr_mult', 1.0))\n"
        "    opt = torch.optim.AdamW([{'params': lora_params, 'lr': HP['lr_adapter'] * lrm},\n"
        "                             {'params': head_params, 'lr': HP['lr_head'] * lrm}], weight_decay=HP['weight_decay'])")
w = sub(w, "        yl = torch.as_tensor(ylab[rows], device=device)\n"
           "        with torch.autocast('cuda', enabled=False):\n"
           "            lp = F.cross_entropy(plain(drop(z)), yt, reduction='sum')\n"
           "            lc = (F.cross_entropy(centered(drop(zc)), yt, reduction='none') * ok).sum()",
        "        yl = torch.as_tensor(ylab[rows], device=device)\n"
        "        am = torch.as_tensor(ALLOWED[rows], device=device)\n"
        "        with torch.autocast('cuda', enabled=False):\n"
        "            lp = F.cross_entropy(plain(drop(z)).masked_fill(~am, -1e4), yt, reduction='sum')\n"
        "            lc = (F.cross_entropy(centered(drop(zc)).masked_fill(~am, -1e4), yt, reduction='none') * ok).sum()")
w = sub(w, "    model.save_pretrained(str(ck / 'adapter'), safe_serialization=True, save_embedding_layers=False)\n",
        "    model.save_pretrained(str(ck / 'adapter'), safe_serialization=True, save_embedding_layers=False)\n"
        "    torch.save([p.detach().cpu() for p in lora_params], ck / 'lora_state.pt')   # exact tensors for stage chaining\n")
w = sub(w, "for job in RUN['worker_jobs'][str(W)]:\n    try:\n",
        "for job in RUN['worker_jobs'][str(W)]:\n"
        "    if (OUT / job / 'COMPLETE.json').exists():\n"
        "        log(f'job {job}: already complete, skipped')\n"
        "        status['completed'].append(job)\n"
        "        continue\n"
        "    try:\n")
w = sub(w, "assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one visible GPU per worker'",
        "assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'no GPU visible'")
out = HEADER + '\n' + w + LAUNCHER
ast.parse(out)
ast.parse(w[w.index("r'''") + 4:w.rindex("'''")], filename='worker.py')
Path('nb/colab_twostage/twostage_train.py').write_text(out, encoding='utf-8', newline='\n')
print('written nb/colab_twostage/twostage_train.py', len(out.splitlines()), 'lines')
