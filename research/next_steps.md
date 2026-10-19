# Next steps (handoff 2026-10-19)
1. Two-stage LoRA (public 0.592 recipe, nishantdahal): add STAGE mode to nb/lora_gemma_ext/ntl_lora_gemma12b_ext_train.py
   - stage 1: fit on external rows only (research/ext_data/unified/ext_all.parquet minus speechmap/oleg, ~10k; + kbench b002/b002r/b003), 1-2 epochs
   - stage 2: continue from stage-1 adapter on host train fold, 2 epochs, lower LR
   - backbone Qwen3-8B 24L (pool layers 2,4,6,8,12,16,20,24); folds 0-2 + final; compare with Gemma-ext E026 (0.4778) via experiments/gemma_oof.py
   - external data must be a public Kaggle dataset before a submission uses it
2. LB facts: v6a 0.556 best; adaptive centering +0.019 on LB (keep); gates neutral (keep).
3. Candidates: v6f = v6a + GLM-ext LoRA + ATLAS 10%; v6g = wider adaptive centering.
4. Running when paused: local seq job (backbone eval -> E046 -> E041 RL); Kaggle ntl-backbone-screen v2 (download nb/backbone_screen/out, run experiments/backbones/screen_eval.py).
