# Native Glimmer NF4 candidate

This independent subproject prepares the actual pinned Muse-Glimmer-30B text
tower for the shared response lifecycle. Its PG0.9.5 mechanical update/resume
and public RAW/OFF evidence remain retained. The completed finished-text RAW32
comparison now keeps **G0.0001 and G0.001 tied best** at mean ON−OFF gain+1.0
across two original cases, with fixed D/bank and horizon128; G0.012/G0.036 each
score0. No unique rate/default change or extra confirmation is required.
[Current decision and exact options](../LEARNING-RATES.md). The older
[preset](presets/pg095-native-nf4-128.json) and
[RAW128 response-SSE measurement](docs/pg095-native-nf4-128-measurement.json)
remain historical diagnostics, not the finished-output selector.

The root `glimmer_modern` PG0.9.5 v5 CPU fixture uses the same new shared base.
Prior root v4 and native v1 PG0.9.4/PG801 packets require their original product
commits and environments; [archived locks](../compatibility/pre-pg095/README.md)
retain those identities. The current package is0.2.0 with a fresh
`glimmer-native6656-nf4-hg-hndl-pg095-continuous-v2` protocol. Existing native
mechanical receipts retain their original source scope; current PG0.9.5 evidence
is recorded in the measurement linked above.

## Exact candidate route

Use Python 3.12 and Torch 2.9.1+cu128 in a new environment. Install this subproject
from an immutable product commit:

```sh
python -m pip install 'git+https://github.com/HyperGAN/glimmer-particle-sliders.git@main#subdirectory=native-candidate'
```

This installs the distinct `glimmer-native` response/inference backend. Its
core is `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` (core0.13.0), with the
published `particlegan==0.9.5` dependency. The explicit
`release-0.9.5` / `continuous` pair is mandatory for the6656-wide native response.
The automatic domain setting retains its width gates. The old PG801 source
profile is not accepted as an alias, and old checkpoints cannot resume here.

Pinned loader dependencies are Transformers 5.22.1, bitsandbytes 0.50.2,
Accelerate 1.15.0, safetensors 0.8.0, tokenizers 0.23.3,
huggingface-hub 2.4.0 and NumPy 2.5.4. Runtime source locks cover their Python
sources and the executed quantization/tokenizer/serialization binaries, plus
selected Torch CUDA binaries. The exact installed core Git origin and all 577
core source/data files are checked independently. Product source/data files
have their own closure. Imports shadowing those distributions are refused.

## CPU readiness

The inspector reads small metadata and safetensors headers, without importing
Torch or constructing a model. Full asset hashing is a separate, explicit CPU
operation:

```sh
CUDA_VISIBLE_DEVICES='' python -m glimmer_native --snapshot /path/to/a4e59da52a7bc87ae7251dd5545c0dd437c44b68 --output metadata-readiness.json
CUDA_VISIBLE_DEVICES='' python -m glimmer_native --snapshot /path/to/a4e59da52a7bc87ae7251dd5545c0dd437c44b68 --verify-assets --output verified-assets.json
CUDA_VISIBLE_DEVICES='' python -m pytest tests -q
```

The locked original snapshot has 29,776,626,688 parameters in 59,553,435,272
weight-file bytes. Its actual text response is the full last hidden 6656-vector,
with 52 text layers and 208 attention projections. Q maps6656→4096, K/V map
6656→256, and O maps4096→6656. No hidden-vector projection, truncation,
field reshape or categorical penalty substitute is allowed.

An optimistic NF4 base-storage bound is 17.629GiB, retaining BF16 embedding,
head and other non-linear parameters. Quantization metadata, HNDL adapters,
gradients/optimizer, checkpointed activations, workspaces, allocator/context
and loading peak are additional costs. The measured native source requires a
live external guardian, one exact GPU UUID, a20GiB process cap,5GiB global free
reserve,25GiB free at admission, and23/16GiB host RAM at admission/runtime.
The actual final teacher/RAW phases peaked at19048MiB GPU memory. This measured
cohort does not guarantee that every data or resource configuration fits.
CPU readiness performs no GPU launch.

## Native contract and measured cohort

The local-only loader disables remote code, uses the original guarded demo's
lazy positioned-read algorithm, BF16 NF4 storage/compute and double
quantization, and requires complete loaded-key accounting. Its new contract
checks the actual native class, whole-host device placement, all quantized
module coverage, packed-weight bytes, actual block 64/nested block 256 state,
all 208 attention targets and runtime sources before any adapter hook.
Offload and multiple-device models are refused by this cohort. A future
offloaded cohort needs its own implementation and qualification.

All attention branches execute the shared rank 8/alpha 8 HNDL projection over one
128×4 learned bank. The critic executes the shared HNDL readout and all GAN
updates, penalties, optimizer/rate derivation and checkpointing remain in the
shared ConditionalPG09. Decoder blocks use scale-captured checkpoint wrappers.
Native-only inference reconstructs the same verified base without training
rows, a critic or a game. The loader/identity contract is implemented, but that
native path has not been executed by the CPU tests.

The qualified native path passed frozen host and unhooked OFF parity, full
native response, real updates, fresh-process exact resume and public RAW
inference before rate selection. Logit comparisons
must include this model's output multiplier and tanh softcap; its raw LM head
alone is not the native logit function. Current LR selection uses only the
equal-case intended-effect ON−OFF mean in finished text; teacher-response error
is diagnostic. Reuse existing searches, RAW checkpoints, outputs and scores;
do not add another confirmation or secondary tiebreak.
The common optional `lr`, `d_lr_mult`, and
`prior_lr_mult` preserve baseline G=.012, D=.018, bank=.03; these are inherited
defaults, not adjusted or recommended native rates.

The native factory and validator require disabled model-weight EMA
(`ema_decay=None` in the resolved SessionConfig). Keep it disabled for training, evaluation, selection, confirmation, exports and
inference. Use raw current weights and do not add EMA or checkpoint averaging;
historical compatibility tests retain their original scope.
The `glimmer-native-eager-bf16-strict-v1` policy inspects caller-owned settings
without changing them: deterministic algorithms with warn-only disabled, highest
FP32 matmul precision, TF32 disabled, FP16/BF16 reduced-precision reductions
disabled, deterministic cuDNN without benchmarking, only math SDPA enabled with
low-precision math reduction disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
TF32 environment overrides are refused. Actual native text/vision attention
must remain eager. The policy is bound into native identity and checked before
load, each update, native forward, and observation; changed source or numerical
inference identities are refused before reconstructing the model.

The actual `from_pretrained` constructor runs in a scoped RNG context with
declared loader seed 41, preserving the external CPU and only owned visible
CUDA generator states on success or failure. It never seeds other GPUs.
Actual PG0.9.5/domain/width admission precedes model or CUDA allocation.

A separate [small CPU software check](docs/small-native-loader-readout-v1.json)
used the actual upstream class with 31,232 random parameters: all 40 tensors
loaded exactly through lazy positioned reads, and native logits matched the
hidden→LM-head→multiplier→softcap path bitwise. This establishes that software
seam only. The current pretrained response evidence is recorded separately
above; the CPU check alone establishes no language effect. The separate
finished-output comparison has only the two-case scope stated above.
