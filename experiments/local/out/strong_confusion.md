### Strong plain blend (public frozen Qwen3.5-9B/Qwen3-14B/Qwen3-8B/DeBERTa/RoBERTa/SVM heads, OOF)


**argmax**: bacc 0.4915, lab-level bacc 0.6926, errors within true lab 58%

| model | recall | top confusions |
|---|---|---|
| claude-fable-5 | 0.17 | claude-opus-5 17%, glm-5.3 17%, claude-opus-4-8 13% |
| claude-opus-4-6 | 0.67 | claude-sonnet-5 7%, claude-opus-4-8 6%, claude-opus-5 5% |
| claude-opus-4-8 | 0.44 | claude-sonnet-5 15%, claude-opus-4-6 9%, claude-fable-5 7% |
| claude-opus-5 | 0.79 | glm-5.3-flash 5%, glm-5.3 4%, claude-fable-5 3% |
| claude-sonnet-5 | 0.53 | claude-opus-4-8 13%, glm-5.3-flash 9%, claude-opus-4-6 8% |
| gemini-3.1-pro-preview | 0.82 | gemini-3.6-flash 7%, gemini-3.7-flash 5%, gemini-3.8-flash 5% |
| gemini-3.6-flash | 0.54 | gemini-3.7-flash 17%, gemini-3.1-pro-preview 14%, gemini-3.8-flash 12% |
| gemini-3.7-flash | 0.44 | gemini-3.8-flash 31%, gemini-3.6-flash 17%, gemini-3.1-pro-preview 6% |
| gemini-3.8-flash | 0.49 | gemini-3.7-flash 30%, gemini-3.6-flash 10%, gemini-3.1-pro-preview 7% |
| glm-5.3 | 0.25 | glm-5.3-flash 19%, claude-opus-5 14%, claude-fable-5 13% |
| glm-5.3-flash | 0.28 | glm-5.3 23%, claude-fable-5 10%, claude-sonnet-5 8% |
| gpt-5.5 | 0.47 | kimi-k3 11%, gpt-5.6-terra 11%, qwen3.8-max 9% |
| gpt-5.6-luna | 0.45 | gpt-6-astra 12%, gpt-5.6-terra 12%, gpt-5.6-sol 11% |
| gpt-5.6-sol | 0.31 | gpt-5.6-luna 18%, gpt-5.6-terra 17%, gpt-6-astra 9% |
| gpt-5.6-terra | 0.39 | gpt-5.5 15%, gpt-5.6-luna 14%, gpt-5.6-sol 14% |
| gpt-6-astra | 0.75 | gpt-5.6-luna 8%, gpt-5.6-terra 5%, gpt-5.6-sol 4% |
| kimi-k3 | 0.49 | gpt-5.5 10%, glm-5.3 7%, gpt-5.6-sol 7% |
| qwen3.8-max | 0.56 | gpt-5.5 9%, gpt-6-astra 8%, gpt-5.6-luna 7% |

Top confused pairs: gemini-3.7-flash<->gemini-3.8-flash 92, glm-5.3<->glm-5.3-flash 64, gemini-3.6-flash<->gemini-3.7-flash 50, gpt-5.6-sol<->gpt-5.6-terra 46, gpt-5.6-luna<->gpt-5.6-sol 44, claude-fable-5<->glm-5.3 44, claude-opus-4-8<->claude-sonnet-5 41, gpt-5.6-luna<->gpt-5.6-terra 39, gpt-5.5<->gpt-5.6-terra 39, gemini-3.6-flash<->gemini-3.8-flash 33

**oracle-prompt Hungarian (full 18)**: bacc 0.6890, lab-level bacc 0.8233, errors within true lab 59%

| model | recall | top confusions |
|---|---|---|
| claude-fable-5 | 0.33 | glm-5.3 27%, glm-5.3-flash 18%, claude-sonnet-5 6% |
| claude-opus-4-6 | 0.89 | claude-sonnet-5 5%, glm-5.3 3%, glm-5.3-flash 1% |
| claude-opus-4-8 | 0.79 | claude-fable-5 7%, claude-sonnet-5 5%, glm-5.3 4% |
| claude-opus-5 | 0.93 | glm-5.3 3%, claude-fable-5 2%, glm-5.3-flash 1% |
| claude-sonnet-5 | 0.73 | claude-opus-4-8 9%, glm-5.3 6%, claude-opus-4-6 3% |
| gemini-3.1-pro-preview | 0.93 | gemini-3.6-flash 5%, gemini-3.8-flash 1%, gemini-3.7-flash 1% |
| gemini-3.6-flash | 0.73 | gemini-3.7-flash 13%, gemini-3.8-flash 10%, gemini-3.1-pro-preview 5% |
| gemini-3.7-flash | 0.61 | gemini-3.8-flash 23%, gemini-3.6-flash 15%, gemini-3.1-pro-preview 1% |
| gemini-3.8-flash | 0.66 | gemini-3.7-flash 26%, gemini-3.6-flash 7%, gemini-3.1-pro-preview 1% |
| glm-5.3 | 0.35 | glm-5.3-flash 28%, claude-fable-5 24%, kimi-k3 7% |
| glm-5.3-flash | 0.41 | claude-fable-5 24%, glm-5.3 17%, claude-opus-4-8 6% |
| gpt-5.5 | 0.69 | gpt-5.6-terra 9%, kimi-k3 9%, gpt-5.6-sol 7% |
| gpt-5.6-luna | 0.67 | gpt-5.6-sol 13%, gpt-5.6-terra 8%, gpt-5.5 3% |
| gpt-5.6-sol | 0.59 | gpt-5.6-terra 15%, gpt-5.6-luna 13%, gpt-5.5 6% |
| gpt-5.6-terra | 0.63 | gpt-5.6-sol 13%, gpt-5.6-luna 13%, gpt-5.5 8% |
| gpt-6-astra | 0.92 | gpt-5.6-luna 3%, gpt-5.5 2%, gpt-5.6-terra 2% |
| kimi-k3 | 0.67 | gpt-5.5 8%, claude-fable-5 6%, glm-5.3-flash 5% |
| qwen3.8-max | 0.88 | kimi-k3 3%, gpt-5.6-luna 2%, gpt-5.5 2% |

Top confused pairs: claude-fable-5<->glm-5.3 76, gemini-3.7-flash<->gemini-3.8-flash 73, glm-5.3<->glm-5.3-flash 67, claude-fable-5<->glm-5.3-flash 63, gpt-5.6-sol<->gpt-5.6-terra 43, gemini-3.6-flash<->gemini-3.7-flash 42, gpt-5.6-luna<->gpt-5.6-sol 40, gpt-5.6-luna<->gpt-5.6-terra 31, gpt-5.5<->gpt-5.6-terra 26, gemini-3.6-flash<->gemini-3.8-flash 26