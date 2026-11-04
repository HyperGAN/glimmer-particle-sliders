# Native Glimmer NF4 candidate

This independent subproject prepares the actual pinned Muse-Glimmer-30B text
tower for the shared response lifecycle. It is currently a CPU source and
structural candidate. No model load, native training, exact native resume,
portable native inference, generated language quality or learning rate has been
qualified for this cohort.

The existing root `glimmer_modern` v4 CPU fixture remains on its original core
ddb source. The historical Meta FP32/NF4 products retain their original
installers and evidence. Install each cohort in a separate environment; their
core/source pins are intentionally different. Old packets are never relabeled.

## Exact candidate route

Use Python 3.12 and Torch 2.9.1+cu128 in a new environment. Install this subproject
from an immutable product commit:

```sh
python -m pip install 'git+https://github.com/HyperGAN/glimmer-particle-sliders.git@d1c77d1af91db6dfc0e378fc5361c42c641c6329#subdirectory=native-candidate'
```

This installs the distinct `glimmer-native` response/inference backend. Its
core is 5d68473a098801ee1ab3ebbb12463daef51c6346 and its ParticleGAN source is
801ae1ead8f88b88eca8f5398137e3e1bc01ad36. The explicit
`source-801ae1e-continuous-v1` / `continuous` pair is mandatory. That exact
ParticleGAN source reports 0.9.4 but has different bytes from official 0.9.4;
the official release/default profile continues to refuse this 6656-wide domain.

Pinned loader dependencies are Transformers 5.22.1, bitsandbytes 0.50.2,
Accelerate 1.15.0, safetensors 0.8.0, tokenizers 0.23.3,
huggingface-hub 2.4.0 and NumPy 2.5.4. Runtime source locks cover their Python
sources and the executed quantization/tokenizer/serialization binaries, plus
selected Torch CUDA binaries. The exact installed core Git origin and all 575
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
and loading peak are additional and unmeasured. The initial candidate resource
contract requires a live external guardian, one exact GPU UUID, a 30 GiB process
cap, 5 GiB global free reserve, 35 GiB free at admission, and 23/16 GiB host RAM at
admission/runtime. This is a budget for a future bounded measurement, not a
statement that one-card training fits. No GPU launch is part of CPU readiness.

## Native contract and next proof

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

The next admitted native attempt must prove frozen host and unhooked OFF
parity, full native hidden/logit readout, two real updates, fresh-process exact
resume and raw portable inference before any rate search. Logit comparisons
must include this model's output multiplier and tanh softcap; its raw LM head
alone is not the native logit function. Native text/token outputs must accompany
hidden-fit measurements. The common optional `lr`, `d_lr_mult`, and
`prior_lr_mult` preserve baseline G=.012, D=.018, bank=.03; these are inherited
defaults, not adjusted or recommended native rates.

Host EMA defaults to disabled (`ema_decay=None` in the resolved SessionConfig).
An explicitly requested diagnostic EMA remains separate from raw native results.
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
Actual PG801/domain/width admission precedes model or CUDA allocation.

A separate [small CPU software check](docs/small-native-loader-readout-v1.json)
used the actual upstream class with 31,232 random parameters: all 40 tensors
loaded exactly through lazy positioned reads, and native logits matched the
hidden→LM-head→multiplier→softcap path bitwise. This establishes that software
seam only; pretrained NF4 forward, training, resume and language remain unqualified.
