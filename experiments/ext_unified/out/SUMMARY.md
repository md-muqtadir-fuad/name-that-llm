# Unified external corpus

Retained **44,123** passages from 9 sources after shared filters, priority dedup, and train/test 8-gram leakage checks.

## Counts by label
gpt-5.5: 2613; gpt-5.6-luna: 2778
gpt-5.6-sol: 2551; gpt-5.6-terra: 2592
gpt-6-astra: 2180; gemini-3.1-pro-preview: 1798
gemini-3.6-flash: 2471; gemini-3.7-flash: 1950
gemini-3.8-flash: 2450; claude-fable-5: 2659
claude-opus-4-6: 2692; claude-opus-4-8: 2586
claude-opus-5: 2993; claude-sonnet-5: 2563
qwen3.8-max: 4071; glm-5.3: 1759
glm-5.3-flash: 1727; kimi-k3: 1690

## Smallest labels
kimi-k3 (1690), glm-5.3-flash (1727), glm-5.3 (1759), gemini-3.1-pro-preview (1798), gemini-3.7-flash (1950)

## Sources, drops, and licenses
kbench: 520/720 (included); Kaggle Benchmarks proxy output; no separate output license stated | gemini_api: 38/39 (included); Google Gemini API output; provider terms apply | lucas: 918/960 (included); CC BY 4.0 (Kaggle metadata)
abdullah: 816/833 (included); CC BY 4.0 (Kaggle metadata) | dophuchung: 1019/1029 (included); CC BY 4.0 (Kaggle metadata) | prvsiyan: 3210/3216 (included); CC BY 4.0 (Kaggle metadata; upstream provider terms may apply)
extra_api: 4861/4998 (included); CC BY 4.0 (dataset documentation; provider output terms may apply) | oleg: 962/24025 (included); Other (Kaggle metadata; mixed source licenses, consult upstream provenance) | speechmap: 31779/324393 (included); Apache-2.0 source dataset; provider output terms may apply
## Choices and surprises
- Only kbench b002 was used; b001 remains excluded as contaminated. No listed source was wholly dropped; Oleg was reduced by the shared single-paragraph/length/dedup filters. Gemini API has no 3.7 served rows.
- Oleg Kaggle metadata says `Other` and describes mixed licenses. `datasketch` was unavailable; exact 8-gram hash indexing was used. Two-row previews are in ignored `research/ext_data/unified/source_inspection.txt`.
