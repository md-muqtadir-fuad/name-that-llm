# Notebooks to rename and make public on the Kaggle website (2026-11-02)

Use the website, not the API: an API quick save creates a new version without outputs. The **Share** button sets
Public without a new version. If changing the title asks you to save a version, check whether it keeps the outputs
before doing the rest.

- Done: `ntl-v1-lora-half` -> "Name That LLM v1: LoRA + half readout (0.544)" (renamed and public by API; its outputs are
  in versions 1-2).
- Already public by API under the old name, latest version without outputs (outputs in earlier versions): rows marked
  "rename only".
- Keep private (not in this list): the 5 most recent, ntl-llama31-8b-weights, new-benchmark-task-*, ntl-honest-oof,
  ntl-honest-cpu, ntl-gemma-parity, ntl-foldcheck3, ntl-eda-and-public-results.

| # | Current notebook | New title | Action |
|---|---|---|---|
| 1 | [ntl-v3-gemma](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v3-gemma) | Name That LLM v3: 0.541 stack + Gemma LoRA (0.547) | rename + Share -> Public |
| 2 | [ntl-v4-gemma-gates](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v4-gemma-gates) | Name That LLM v4: Gemma LoRA + family gates | rename + Share -> Public |
| 3 | [ntl-v5-2gemma-gates](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v5-2gemma-gates) | Name That LLM v5: two Gemma LoRAs + gates (0.552) | rename + Share -> Public |
| 4 | [ntl-v6a](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v6a) | Name That LLM v6a: Gemma-ext LoRA swap (0.556) | rename + Share -> Public |
| 5 | [ntl-v6d](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v6d) | Name That LLM v6d: ablation, gates off (0.554) | rename + Share -> Public |
| 6 | [ntl-v6e](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v6e) | Name That LLM v6e: ablation, centring off (0.537) | rename + Share -> Public |
| 7 | [ntl-v8](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v8) | Name That LLM v8: public 0.639 reproduced | rename + Share -> Public |
| 8 | [ntl-v9](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v9) | Name That LLM v9: 0.639 + word-BiLSTM (0.637) | rename + Share -> Public |
| 9 | [ntl-v10](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v10) | Name That LLM v10: 0.639 + Gemma two-stage (0.635) | rename only |
| 10 | [ntl-v11a](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-v11a) | Name That LLM v11a: 0.651 reproduction (timeout) | rename only |
| 11 | [ntl-block-specialist-cpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-block-specialist-cpu) | Name That LLM: fable/GLM block specialist | rename only |
| 12 | [ntl-gem37-38-lstm-cpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-gem37-38-lstm-cpu) | Name That LLM: BiLSTM for gemini-3.7 vs 3.8 | rename only |
| 13 | [ntl-llama-ts-s1](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-llama-ts-s1) | Name That LLM: Llama-3.1-8B two-stage, stage 1 | rename only |
| 14 | [ntl-llama-twostage](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-llama-twostage) | Name That LLM: Llama-3.1-8B two-stage smoke test | rename + Share -> Public |
| 15 | [ntl-gemma-ts-s1](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-gemma-ts-s1) | Name That LLM: Gemma-4-12B two-stage, stage 1 | rename + Share -> Public |
| 16 | [ntl-gemma-ts-s2](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-gemma-ts-s2) | Name That LLM: Gemma-4-12B two-stage, stage 2 | rename + Share -> Public |
| 17 | [ntl-gemma-twostage](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-gemma-twostage) | Name That LLM: Gemma-4-12B two-stage smoke test | rename + Share -> Public |
| 18 | [ntl-lstm-final-cpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-lstm-final-cpu) | Name That LLM: word-BiLSTM final model (CPU) | rename + Share -> Public |
| 19 | [ntl-lstm-cpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-lstm-cpu) | Name That LLM: word-BiLSTM two-stage (CPU) | rename + Share -> Public |
| 20 | [ntl-cnn-cpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-cnn-cpu) | Name That LLM: byte-CNN two-stage (CPU) | rename + Share -> Public |
| 21 | [ntl-oneshot-logprob](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-oneshot-logprob) | Name That LLM: one-shot style log-probs | rename + Share -> Public |
| 22 | [ntl-pos-affect-cpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-pos-affect-cpu) | Name That LLM: POS and affect features (CPU) | rename + Share -> Public |
| 23 | [ntl-lora-twostage-train](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-lora-twostage-train) | Name That LLM: Qwen3-8B two-stage LoRA trainer | rename + Share -> Public |
| 24 | [ntl-backbone-screen](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-backbone-screen) | Name That LLM: frozen backbone screen | rename + Share -> Public |
| 25 | [ntl-backbone-screen-tpu](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-backbone-screen-tpu) | Name That LLM: frozen backbone screen on TPU | rename + Share -> Public |
| 26 | [ntl-lora-glm9b-ext-train](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-lora-glm9b-ext-train) | Name That LLM: GLM-4-9B LoRA, external data | rename + Share -> Public |
| 27 | [ntl-lora-gemma12b-ext-train](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-lora-gemma12b-ext-train) | Name That LLM: Gemma-4-12B LoRA, external data | rename + Share -> Public |
| 28 | [ntl-lora-gemma12b-train](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-lora-gemma12b-train) | Name That LLM: Gemma-4-12B LoRA trainer | rename + Share -> Public |
| 29 | [ntl-featcache-qwen8](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-featcache-qwen8) | Name That LLM: Qwen3-8B frozen feature cache | rename + Share -> Public |
| 30 | [ntl-featcache-claudeapi](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-featcache-claudeapi) | Name That LLM: Qwen3-8B features, Claude API set | rename + Share -> Public |
| 31 | [ntl-featcache-distractors](https://www.kaggle.com/code/mdmuqtadirfuad/ntl-featcache-distractors) | Name That LLM: Qwen3-8B features, distractors | rename + Share -> Public |
