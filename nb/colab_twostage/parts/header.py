# ntl two-stage LoRA trainer for Google Colab (single GPU, resumable). Built by nb/colab_twostage/build_trainer.py from
# nb/lora_gemma_ext's proven trainer (same backbone truncation, embedding trim, 8-depth pooling, plain / centred / lab
# heads, same-prompt packs, OOM fallback).
# Stage 1: train on the public in-style pool only (nishantdahal/ntl-two-stage-lora pool_v1.parquet: 34,940 passages,
#          18 labels, 2,252 prompts, CC-BY-SA-4.0), split by prompt into 3 sequential parts (stage1a -> b -> c, each
#          continuing from the previous one) so a disconnect loses at most one part. Prompt-masked softmax: for a pool
#          row, labels that never answered that pool prompt are excluded from its plain / centred softmax.
# Stage 2: continue from stage1c on host train rows: fold0, fold1, fold2 (GroupKFold(5) by prompt_id, default order)
#          and final (all host rows); 2 epochs, learning rates x stage2_lr_mult.
# Every job writes COMPLETE.json; a rerun skips completed jobs (resume).
# Usage: python twostage_train.py [--smoke] [--dry] [--jobs stage1a,stage1b]
#        env NTL_WORK (outputs), NTL_DATA (train.csv, pool_v1.parquet), NTL_MODEL_DIR (local backbone folder)
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument('--smoke', action='store_true')
_ap.add_argument('--dry', action='store_true')
_ap.add_argument('--jobs', default='')
ARGS = _ap.parse_args()
SMOKE = ARGS.smoke
BACKBONE = 'olmo2-7b'
BACKBONES = {
    'olmo2-7b': dict(slug='allenai/OLMo-2-1124-7B', depth=24, pool_layers=[2, 4, 6, 8, 12, 16, 20, 24]),
    'qwen3-8b-base': dict(slug='Qwen/Qwen3-8B-Base', depth=24, pool_layers=[2, 4, 6, 8, 12, 16, 20, 24]),
}
HP = dict(r=16, alpha=32, lora_dropout=0.05, lr_adapter=1e-4, lr_head=1e-3, weight_decay=0.01, epochs=2,
          warmup=0.1, eff_batch=18, pack=8, length_window=64, max_len=448, head_dropout=0.1, lab_weight=0.5, clip=1.0,
          seed=42, init_scale=256.0, growth_interval=2000, n_folds=5,
          stage1_parts=3, stage1_epochs=1, stage2_epochs=2, stage2_lr_mult=0.5)
LABS = ['claude', 'gemini', 'glm', 'gpt', 'kimi', 'qwen']
JOBS = ['stage1a', 'stage1b', 'stage1c', 'fold0', 'fold1', 'fold2', 'final']
SMOKE_JOBS = ['stage1a', 'fold0']
SMOKE_FIT = (4, 15)    # prompts x passages per prompt
SMOKE_VALID = (2, 10)
