# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Native NF4 conditioning / teachers | 1 | 24 | 14 | one prompt ≤ 128 tokens; NF4 nested scales/BF16 compute; eager; one GPU |
| Native NF4 response training | 1 | 28 | 14 | batch 2; complete input ≤ 128 tokens; all 52 blocks checkpointed; FP32 routed trainables |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.9.5 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **meta-models/Muse-Glimmer-30B**, revision **`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Native NF4 conditioning / teachers | packed_base: 18.0800; largest_layer_dequant_BF16: 0.2476; prefill: 0.0356; workspace: 4.0000 |
| Native NF4 response training | packed_base: 18.0800; largest_layer_dequant_BF16: 0.2476; trainable_transient: 0.9690; checkpoint_activations: 0.8267; workspace: 6.0000 |

Detailed arithmetic and assumptions:

- **NF4 packed plus islands bytes**: `19413279356`.
- **LoRA elements**: `14696448`.
- **adapted projections**: `208`.
- **generator plus critic allowance elements**: `16257057`.
- **max layer dequant bytes**: `265814016`.
- **host loading**: `two largest2,689,662,976-byte BF16 embedding buffers +one FP32 largest quantized Linear +8 GiB +export copies`.

Scope and limits:

- Glimmer and Meta Glimmer use the same exact Muse-Glimmer model; these are two interfaces, not two independently sized backbones. This is NF4 double quantization, not the no-double FP4 formula.
- Packed storage = Q/2 +ceil(Q/64) uint8 nested scales +4ceil(Q/(64*256)) FP32 second-level scales +2U BF16 islands +52 MiB codebook/alignment allowance. Q=27,082,854,400; U=2,693,772,288; largest quantized matrix132,907,008 elements. pread slices avoid a persistent full dense CPU state_dict.
- Only the source-described native GPU candidate is budgeted. It still requires explicit external GPU ownership and its qualification status is separate from this calculation. Larger/unbounded completions or vision inputs are outside these text-only numbers.
- The glimmer_native core entry point is hidden-response inference, not a finished-text renderer. Do not infer a full Muse vision/generation capability from the training profile.

Source facts (read as text; no model execution):

- [native-candidate/src/glimmer_native/loader.py](native-candidate/src/glimmer_native/loader.py)
- [native-candidate/src/glimmer_native/contract.py](native-candidate/src/glimmer_native/contract.py)
- [native-candidate/src/glimmer_native/native.py](native-candidate/src/glimmer_native/native.py)
- [native-candidate/src/glimmer_native/tensor-headers.json](native-candidate/src/glimmer_native/tensor-headers.json)
- [native-candidate/src/glimmer_native/lifecycle.py](native-candidate/src/glimmer_native/lifecycle.py)
- [Pinned model metadata](https://huggingface.co/meta-models/Muse-Glimmer-30B/tree/a4e59da52a7bc87ae7251dd5545c0dd437c44b68) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.9.5 release `1b8355b184b21bb7d81104ee590813c262f6df0a`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `pg09_audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
