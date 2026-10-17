## 2. TF-IDF baselines (5-fold GroupKFold by prompt_id, OOF)

Features: char_wb(1-5) + word(1-2, punctuation tokens kept), min_df=2, vectorizers fit on the train fold only.

| variant | balanced acc | fold sd | lab-level bacc | NLL | secs |
|---|---|---|---|---|---|
| svm_sub | 0.3304 | 0.012 | 0.5503 | 2.063 | 106 |

Best: **svm_sub**. Per-class recall:

| model | recall | most-confused-with (share of its passages) |
|---|---|---|
| claude-fable-5 | 0.10 | claude-opus-5 19%, glm-5.3 14%, glm-5.3-flash 13% |
| claude-opus-4-6 | 0.44 | claude-sonnet-5 13%, claude-opus-5 9%, claude-opus-4-8 5% |
| claude-opus-4-8 | 0.23 | claude-opus-5 13%, claude-sonnet-5 12%, claude-opus-4-6 9% |
| claude-opus-5 | 0.65 | glm-5.3 6%, claude-opus-4-8 5%, claude-sonnet-5 5% |
| claude-sonnet-5 | 0.33 | claude-opus-4-6 9%, claude-opus-5 9%, claude-opus-4-8 9% |
| gemini-3.1-pro-preview | 0.76 | gemini-3.6-flash 9%, gemini-3.7-flash 7%, gemini-3.8-flash 7% |
| gemini-3.6-flash | 0.46 | gemini-3.1-pro-preview 19%, gemini-3.7-flash 16%, gemini-3.8-flash 14% |
| gemini-3.7-flash | 0.37 | gemini-3.6-flash 22%, gemini-3.8-flash 21%, gemini-3.1-pro-preview 15% |
| gemini-3.8-flash | 0.30 | gemini-3.7-flash 29%, gemini-3.6-flash 17%, gemini-3.1-pro-preview 16% |
| glm-5.3 | 0.14 | claude-opus-5 26%, glm-5.3-flash 15%, claude-sonnet-5 10% |
| glm-5.3-flash | 0.15 | glm-5.3 17%, claude-opus-5 13%, claude-sonnet-5 9% |
| gpt-5.5 | 0.27 | kimi-k3 16%, gpt-5.6-terra 12%, qwen3.8-max 10% |
| gpt-5.6-luna | 0.30 | gpt-6-astra 18%, gpt-5.6-sol 14%, gpt-5.6-terra 12% |
| gpt-5.6-sol | 0.16 | gpt-5.6-luna 19%, gpt-6-astra 19%, gpt-5.6-terra 15% |
| gpt-5.6-terra | 0.20 | gpt-6-astra 17%, gpt-5.6-luna 17%, gpt-5.5 14% |
| gpt-6-astra | 0.50 | gpt-5.6-luna 13%, gpt-5.6-terra 7%, kimi-k3 7% |
| kimi-k3 | 0.27 | gpt-5.5 16%, claude-opus-5 7%, glm-5.3 7% |
| qwen3.8-max | 0.33 | gpt-5.5 11%, gpt-5.6-luna 11%, kimi-k3 9% |

Top confused pairs (symmetric counts out of ~300):

gemini-3.7-flash <-> gemini-3.8-flash: 76, gemini-3.6-flash <-> gemini-3.7-flash: 57, gpt-5.6-luna <-> gpt-5.6-sol: 49, gpt-5.5 <-> kimi-k3: 48, glm-5.3 <-> glm-5.3-flash: 48, claude-opus-5 <-> glm-5.3: 48, gpt-5.6-luna <-> gpt-6-astra: 47, gemini-3.6-flash <-> gemini-3.8-flash: 47, gpt-5.6-luna <-> gpt-5.6-terra: 43, gemini-3.1-pro-preview <-> gemini-3.6-flash: 43, gpt-5.6-sol <-> gpt-5.6-terra: 41, gpt-5.5 <-> gpt-5.6-terra: 39, gpt-5.6-terra <-> gpt-6-astra: 37, gpt-5.6-sol <-> gpt-6-astra: 36, gemini-3.1-pro-preview <-> gemini-3.8-flash: 34

Share of errors that stay within the true lab: 52%.
