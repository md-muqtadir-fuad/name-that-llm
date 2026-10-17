# Review: Gemma-4-12B LoRA training kernel (ntl-lora-gemma12b-train)

Reviewer pass before the ~6 GPU-hour full run. Scope: `nb/lora_gemma/ntl_lora_gemma12b_train.py`, `kernel-metadata.json`,
smoke outputs `nb/lora_gemma/smoke_out/v1..v3`, the reference trainer `nb/v1/lora24_src/train_lora24.py` +
`amp_update.py`, the v5 notebook (`research/kernels/name-that-llm-v5`), the inference consumer
`nb/v1/v1.py::lora_plain_proba`, and transformers' `gemma4_unified` modeling code (local 5.20.1; Kaggle runs 5.16.1).

SMOKE_V4_PLACEHOLDER

## Verdict

No blocking bug was found in the training objective, data split, label mapping, tokenization or checkpoint format.
Two gaps in the v3 smoke coverage were real risks for the full run and are now fixed:

1. **The multi-job worker path was never run.** In v3 each worker ran one job. In the full run each worker runs 3 jobs
   in one process (restore the LoRA snapshot, train again after an extraction, skip the memory probe). The extraction
   ran under `torch.inference_mode()`. If any module cached a tensor during extraction, the next job's backward
   would fail with "Inference tensors cannot be saved for backward". That would lose fold2, fold4, fold1 and fold3 (4 of 6 jobs).
2. **There was no OOM recovery in the training loop.** An OOM in any update failed the whole job, losing up to
   about 1.2 h. The worst-case pack (8 rows of 363 tokens) plus AdamW state is estimated at about 12.95 GiB
   allocated and about 13.5 GiB reserved, on a T4 with about 14.7 GiB total minus the CUDA context. The margin was under 1 GiB.

## Fixes applied (in place, `nb/lora_gemma/ntl_lora_gemma12b_train.py`)

| # | Change | Training math changed? |
|---|---|---|
| 1 | `extract()` uses `torch.no_grad()` instead of `torch.inference_mode()`, so no inference tensors exist when the next job trains. | No (identical features) |
| 2 | OOM fallback in the training loop. If an update hits `torch.cuda.OutOfMemoryError`, the cleanup runs outside the `except` block: all gradients are zeroed, `empty_cache` is called, PACK steps down 8 → 6 → 4 → 3 for the rest of the worker, and the whole update is redone. The objective is unchanged because centring stays within each micro-batch. The OOM events are logged in `training_report.json` (`oom_events`, per-update `oom_retries` and `pack`) and in `summary.json`. | Only on the error path |
| 3 | Check on the LoRA restore: the L1 norm of the restored adapter must equal the fresh snapshot's L1 norm (both computed in double precision). This proves each job starts from the identical fresh adapter. | No |
| 4 | Smoke runs 2 jobs per worker (`{0: [fold0, fold2], 1: [final, fold1]}`), which exercises the sequential path and the final → fold transition. GPU0's first job raises one synthetic OOM (`SyntheticOOM`, only when `SMOKE` is set) after a backward, to exercise fallback #2 for real. | No effect on the full run |
| 5 | The summary reads job keys with `.get()` and includes `oom_events`. | No |

The retry control flow was also checked on CPU. The exact loop text from the file was run against mocks for four
cases: no OOM, synthetic OOM, real OOM, and OOM at every pack size (which ends in a clean `RuntimeError` at pack 3).
In each case gradients were zeroed before the redo and every update stepped exactly once.

## Checklist results

### (1) Fold leakage: OK
- `GroupKFold(5).split(X, y, groups=prompt_id)`. folds.csv shows every prompt in exactly one fold, with 540/540/540/540/539 rows and 30 prompts per fold.
- I dry-ran the full plan locally (SMOKE=False, planning only). For every fold job the fit and valid rows are disjoint, the fit and valid prompts are disjoint, fit ∪ valid covers all 2699 rows, and valid equals `folds.csv == k`. final fits all 2699 rows with an empty valid set.
- In the v3 smoke outputs, fold0 fitted prompts {p000..p003} and validated on {p004, p009}, with no overlap.

### (2) Micro-batches and centring: OK
- Every epoch of every job (full plan) contains each fitting row exactly once (asserted in `plan_job`, and re-checked independently). Every micro-batch has 2 or 3 rows from a single prompt. No prompt has fewer than 17 rows, so there are no singletons.
- Centring in `pack_loss`: `index_add` gives the sums per micro-batch, `bincount` the counts, and `zc = z - mean[g]`. This is exact for sizes 2 and 3 and stays per micro-batch when packing is mixed. A lone passage is masked out of the loss. The loss is (ΣCE_plain + ΣCE_centred + 0.5·ΣCE_lab)/rows, which matches the reference (mean per micro-batch / 6 micro-batches = Σ/18).
- Held-out "chunk" centring uses the same 2/3 size rule (`chunk_labels`), and `center_logits_prompt` uses the whole prompt.
- Length grouping is mild. Inside an update the median spread of character lengths is about 860 chars, and the number of distinct classes per update (about 12.1) is the same as for random 19-row draws (about 11.9). So the class mix per update is not biased.

### (3) Labels and class order: OK
- `LABELS = sorted(train.model.unique())`. This matches `v1.py` (`LABELS = sorted([...])`, and `lora_plain_proba` asserts `ro['classes'] == LABELS`). `classes` is saved in readouts.npz and val_outputs.npz.
- The lab map (`startswith` over [claude, gemini, glm, gpt, kimi, qwen]) is correct for all 18 labels, including glm vs gemini: claude ×5, gemini ×4, glm ×2, gpt ×5, kimi, qwen.
- In v3 the heads in heads.safetensors equal the readouts.npz coefficients exactly. Logits recomputed from the saved fp16 z match the saved logits to within 4.1e-4 (plain) and 3.7e-4 (lab).

### (4) Tokens, masking, padding, truncation: OK
- `[BOS=2] + tok(text, add_special_tokens=False, max_length=446) + [EOS=1]`, max 448. The training maximum is 363, so nothing is truncated in training.
- Right padding (`collate`), default position ids (an arange, correct for right padding), and a mask on position 0 (BOS). EOS is included in the pool.
- The anchors are train rows 0..2 in train.csv order. They are extracted as one length-sorted batch of 3 in eval mode, which is the same as `v1.py`'s `embed(pt, 3)` parity path once the Gemma branch prepends BOS.
- `hidden_states[24]` is the output **after** the final RMSNorm (transformers `capture_outputs(tie_last_hidden_states=True)`), while layers 2..20 are raw residual streams. This is consistent only if inference also truncates to exactly 24 layers. Loading all 48 layers would silently change the layer-24 feature, although the anchor parity check would catch it.

### (5) Embedding trim: OK
- `used` = all train token ids ∪ {PAD 0, EOS 1, BOS 2}. Of 262144 ids, 19991 are kept. `remap` sends ids 0, 1, 2 to themselves, and every other remapped id is below 19991, so nothing can collide with special or multimodal ids even if a code path compares ids. No other vocab-sized embedding exists (`hidden_size_per_layer_input: 0`, no per-layer embeddings), and no WARNING appeared.
- `kept_token_ids.npy` is saved, sorted and int64. The embedding is frozen (`requires_grad=False`, not a PEFT target), so inference can re-trim to train ∪ test ids, with its own remap, and get identical features. Inference must also include the test ids, or the remap gives -1 and indexing fails or reads the wrong row.

### (6) Precision: OK
- The base is frozen fp16. All trainable parameters (LoRA, norms, heads) are fp32, which is asserted. The forward runs under fp16 autocast, and pooling, LayerNorm and heads run in fp32 outside autocast.
- `GradScaler(init 256, growth 2000)` follows the standard sequence `scale → backward → unscale_ → clip → step → update`. Overflowing updates are skipped, and `stepped` is detected by the scale shrinking. This differs from the reference, which replays the update with half the scale: a skipped update here loses its 18 rows for that epoch, a negligible cost. v3 had 0 overflows.
- Non-finite packs are skipped and counted. Validation and anchor rows that come out non-finite are retried under bf16.
- Gemma's residual stream stays fp16 (RMSNorm upcasts internally). v5 trained the same backbone in fp16 successfully.

### (7) Runtime and memory: OK, with margin
- Smoke speed is 0.51–0.58 s/row. The smoke passages are representative (mean 766 chars against 755 for the full train set) and had a padded fraction of about 0.68–0.70.
- The full run has 345 updates per fold (6477 row-passes) and 432 for final (8097), so about 1.0 h per fold and 1.25 h for final at 0.55 s/row. Worker0 needs about 3.1 h and worker1 about 3.3 h; budget 3.5 h wall. Kaggle allows 12 h and the monitor deadline is 11.5 h, so the margin is more than 8 h.
- Memory: the base is 10.30 GiB, the probe peak (8 × 363 tokens, forward and backward) is 12.69 GiB, and the training peak is 12.74 GiB allocated / 13.28 GiB reserved. The worst case with Adam state is about 12.95 GiB allocated. With fix #2, an OOM now costs one redone update instead of a job.

### (8) Outputs: OK
Everything is under /kaggle/working. That is 6 adapters × 131 MB, 5 val_outputs files with fp16 z of about 30 MB each, and a 32 MB tokenizer, about 1.1 GB in total and far below 20 GB.

### (9) SMOKE vs full: OK
`SMOKE` changes only which jobs each worker runs, the fit/valid subsets (`take`), the warmup length that follows from those, a per-update log line, and the synthetic OOM (gated on `RUN['smoke']`). `full_plan` is computed identically in both modes, and the dry run with SMOKE=False reproduced the expected counts (345 × 5, 432).

## Non-blocking issues / notes for the inference branch

- **The saved adapter_config.json lists leaf names**, not 164 explicit paths. PEFT ≥ 0.14 collapses more than 20 targets to minimal suffixes, giving `target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`. The builder's report says explicit names, which is inaccurate. This is harmless **only** if `PeftModel.from_pretrained` wraps the 24-layer `.language_model`. Wrapping the full `Gemma4UnifiedModel` would also match vision-tower projections, and the adapter keys would then not line up (`base_model.model.layers...` against `...language_model.layers...`). The LoRA would load as zeros, and only the anchor parity check would notice. The Gemma inference branch should assert that no `lora_` keys are missing when it loads.
- **Plain/lab loss spikes** (10–17) in smoke come from `lr_head = 1e-3` on 30720-dim features. Those features include dims that are large in every row (massive activations), which the centred head cancels. The reference recipe has the same setting, and the full run warms up over 34–43 updates. Watch `training_report.json`. If plain OOF lags centred by a lot, try head lr 3e-4 or warmup 0.2 next time.
- **Pooling multiplies by the mask** (`h * mask`). An fp16 `inf` at a padded or BOS position would become NaN and drop the whole pack, even when the real tokens are finite. `masked_fill` would be more robust. I left it unchanged so the numerics match the reference and v1 inference exactly; the non-finite counters will show whether it ever happens.
- **Folds are not aligned with the Qwen LoRA's (lightningv08) outer folds.** Every row still gets one OOF prediction, which suffices for blending, but per-fold pairing isn't possible.
- **The padding fraction stays about 0.68–0.70.** Same-prompt micro-batches mix lengths and packs pad to the longest row. It is acceptable as is.
- **The summary's time extrapolation uses the median s/row**, which is optimistic. Use the mean.
- **Checkpoint directory name.** `v1.py` globs `**/lora24/{job}/COMPLETE.json`, but this kernel writes `lora_gemma/`. The Gemma inference branch needs its own glob, and it must also: take `.language_model`, set `allow_global_per_layer_attribute_access`, truncate to 24 layers with `layer_types[:24]`, re-trim and remap with train ∪ test ids, prepend BOS 2 and mask position 0, use max_length 446 plus BOS/EOS, hidden size 3840, and pool layers [2,4,8,12,14,16,20,24].
- **The saved base_config's `per_layer_config`** still lists layers 29–47. This is harmless, since it is a reference copy.

## How to launch the full run (unchanged)
Set `SMOKE = False` (line 18 now, since the header comment grew), then run `./.venv/Scripts/kaggle.exe kernels push -p nb/lora_gemma`.
Nothing else changes.
