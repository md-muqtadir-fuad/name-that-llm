# Model-family linkage graph

Two views on prompt-disjoint data (train, 150 prompts):
- **confusion**: symmetrised OOF probability mass a->b (strong frozen-LLM blend + Gemma LoRA, 50/50 log-space);
- **style_centroid**: cosine similarity of per-model centroids of frozen Qwen3-8B L7+22 features after removing each
  passage's TRUE prompt mean (topic removed).
Clusters: average linkage; stability = share of 200 prompt-bootstrap resamples in which two models share a cluster.

## confusion

Top-3 nearest models per model:

- **claude-fable-5** -> glm-5.3 (0.208), glm-5.3-flash (0.204), claude-opus-4-8 (0.149)
- **claude-opus-4-6** -> claude-sonnet-5 (0.183), claude-opus-4-8 (0.155), claude-fable-5 (0.116)
- **claude-opus-4-8** -> claude-sonnet-5 (0.216), claude-opus-4-6 (0.155), claude-fable-5 (0.149)
- **claude-opus-5** -> glm-5.3 (0.183), claude-fable-5 (0.139), glm-5.3-flash (0.133)
- **claude-sonnet-5** -> claude-opus-4-8 (0.216), claude-opus-4-6 (0.183), claude-fable-5 (0.125)
- **gemini-3.1-pro-preview** -> gemini-3.6-flash (0.351), gemini-3.7-flash (0.193), gemini-3.8-flash (0.161)
- **gemini-3.6-flash** -> gemini-3.1-pro-preview (0.351), gemini-3.7-flash (0.342), gemini-3.8-flash (0.257)
- **gemini-3.7-flash** -> gemini-3.8-flash (0.457), gemini-3.6-flash (0.342), gemini-3.1-pro-preview (0.193)
- **gemini-3.8-flash** -> gemini-3.7-flash (0.457), gemini-3.6-flash (0.257), gemini-3.1-pro-preview (0.161)
- **glm-5.3** -> glm-5.3-flash (0.254), claude-fable-5 (0.208), claude-opus-5 (0.183)
- **glm-5.3-flash** -> glm-5.3 (0.254), claude-fable-5 (0.204), claude-opus-5 (0.133)
- **gpt-5.5** -> kimi-k3 (0.216), gpt-5.6-terra (0.205), gpt-5.6-sol (0.165)
- **gpt-5.6-luna** -> gpt-5.6-terra (0.221), gpt-5.6-sol (0.218), gpt-6-astra (0.204)
- **gpt-5.6-sol** -> gpt-5.6-terra (0.223), gpt-5.6-luna (0.218), gpt-6-astra (0.167)
- **gpt-5.6-terra** -> gpt-5.6-sol (0.223), gpt-5.6-luna (0.221), gpt-5.5 (0.205)
- **gpt-6-astra** -> gpt-5.6-luna (0.204), gpt-5.6-sol (0.167), gpt-5.6-terra (0.145)
- **kimi-k3** -> gpt-5.5 (0.216), gpt-5.6-sol (0.108), claude-opus-5 (0.095)
- **qwen3.8-max** -> gpt-5.6-luna (0.143), gpt-5.5 (0.138), gpt-5.6-sol (0.092)

Clusters at k=6 (full data; stability of each pair in brackets):

- {gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash}  [mean pair stability 1.00]
- {gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra}  [mean pair stability 1.00]
- {gpt-5.5, kimi-k3}  [mean pair stability 0.97]
- {qwen3.8-max}
- {claude-opus-4-6, claude-opus-4-8, claude-sonnet-5}  [mean pair stability 1.00]
- {claude-fable-5, claude-opus-5, glm-5.3, glm-5.3-flash}  [mean pair stability 1.00]

Clusters at k=9 (full data; stability of each pair in brackets):

- {gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash}  [mean pair stability 1.00]
- {gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra}  [mean pair stability 0.94]
- {gpt-6-astra}
- {gpt-5.5, kimi-k3}  [mean pair stability 0.97]
- {qwen3.8-max}
- {claude-opus-4-8, claude-sonnet-5}  [mean pair stability 0.98]
- {claude-opus-4-6}
- {claude-fable-5, glm-5.3, glm-5.3-flash}  [mean pair stability 1.00]
- {claude-opus-5}

## style_centroid

Top-3 nearest models per model:

- **claude-fable-5** -> glm-5.3-flash (0.779), glm-5.3 (0.728), claude-opus-5 (0.491)
- **claude-opus-4-6** -> claude-sonnet-5 (0.542), claude-opus-4-8 (0.435), gemini-3.1-pro-preview (0.356)
- **claude-opus-4-8** -> claude-sonnet-5 (0.565), claude-opus-4-6 (0.435), claude-fable-5 (0.354)
- **claude-opus-5** -> glm-5.3 (0.665), kimi-k3 (0.575), gpt-5.5 (0.528)
- **claude-sonnet-5** -> claude-opus-4-8 (0.565), claude-opus-4-6 (0.542), claude-fable-5 (0.241)
- **gemini-3.1-pro-preview** -> gemini-3.6-flash (0.916), gemini-3.7-flash (0.860), gemini-3.8-flash (0.786)
- **gemini-3.6-flash** -> gemini-3.7-flash (0.941), gemini-3.1-pro-preview (0.916), gemini-3.8-flash (0.883)
- **gemini-3.7-flash** -> gemini-3.6-flash (0.941), gemini-3.8-flash (0.935), gemini-3.1-pro-preview (0.860)
- **gemini-3.8-flash** -> gemini-3.7-flash (0.935), gemini-3.6-flash (0.883), gemini-3.1-pro-preview (0.786)
- **glm-5.3** -> claude-fable-5 (0.728), glm-5.3-flash (0.721), claude-opus-5 (0.665)
- **glm-5.3-flash** -> claude-fable-5 (0.779), glm-5.3 (0.721), claude-opus-5 (0.510)
- **gpt-5.5** -> kimi-k3 (0.808), gpt-5.6-sol (0.741), gpt-5.6-terra (0.711)
- **gpt-5.6-luna** -> gpt-5.6-terra (0.778), gpt-5.6-sol (0.731), gpt-6-astra (0.649)
- **gpt-5.6-sol** -> gpt-5.6-terra (0.848), gpt-5.5 (0.741), gpt-5.6-luna (0.731)
- **gpt-5.6-terra** -> gpt-5.6-sol (0.848), gpt-5.6-luna (0.778), gpt-5.5 (0.711)
- **gpt-6-astra** -> gpt-5.6-sol (0.718), gpt-5.6-terra (0.701), gpt-5.6-luna (0.649)
- **kimi-k3** -> gpt-5.5 (0.808), claude-opus-5 (0.575), gpt-5.6-sol (0.557)
- **qwen3.8-max** -> gpt-5.5 (0.464), gpt-5.6-luna (0.406), kimi-k3 (0.392)

Clusters at k=6 (full data; stability of each pair in brackets):

- {claude-fable-5, claude-opus-5, glm-5.3, glm-5.3-flash}  [mean pair stability 0.76]
- {gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3}  [mean pair stability 0.82]
- {qwen3.8-max}
- {gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash}  [mean pair stability 1.00]
- {claude-opus-4-8, claude-sonnet-5}  [mean pair stability 0.69]
- {claude-opus-4-6}

Clusters at k=9 (full data; stability of each pair in brackets):

- {claude-fable-5, glm-5.3, glm-5.3-flash}  [mean pair stability 0.79]
- {claude-opus-5}
- {gpt-5.5, kimi-k3}  [mean pair stability 0.99]
- {gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra}  [mean pair stability 1.00]
- {qwen3.8-max}
- {gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash}  [mean pair stability 1.00]
- {claude-opus-4-8}
- {claude-sonnet-5}
- {claude-opus-4-6}

## Strongest cross-lab links (confusion view)

- gpt-5.5 <-> kimi-k3: 0.216
- claude-fable-5 <-> glm-5.3: 0.208
- claude-fable-5 <-> glm-5.3-flash: 0.204
- claude-opus-5 <-> glm-5.3: 0.183
- gpt-5.6-luna <-> qwen3.8-max: 0.143
- gpt-5.5 <-> qwen3.8-max: 0.138
- claude-opus-5 <-> glm-5.3-flash: 0.133
- claude-sonnet-5 <-> glm-5.3-flash: 0.116