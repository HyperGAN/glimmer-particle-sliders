# Shared Glimmer and Meta Glimmer explicit fresh95 API

One separate `glimmer_fresh95` addon serves the `glimmer` and `meta-glimmer` interfaces. It uses the existing standalone Glimmer owners and does not duplicate either model or edit the recursively guarded native namespace. The original modern/native packages, registrations and historical Meta package remain unchanged. The existing [standalone LR ties](LEARNING-RATES.md) and separate Meta LR evidence retain their original scope and values; this initialization profile supplies no new LR selection.

## Installation and boundaries

Use Python 3.12 with genuine installed original owners, exact core0.13.0 Git/wheel provenance and Source closure, published PG0.9.5 and the original HNDL/runtime guards. The CPU owner is `meta-glimmer-modern-response==0.2.0`; the native owner is the separate `glimmer-native-response-candidate==0.2.0` subproject in `native-candidate/`. From the repository root, after preparing the appropriate owner environment:

```sh
python -m pip install --no-deps ./fresh95
```

`--no-deps` preserves that environment rather than supplying missing dependencies. The addon metadata exposes `cpu` and `native` extras naming the corresponding original owner. Native NF4 use still requires its original immutable assets, runtime/numerical/packed-model verification and physical GPU admission. No Source overlay or fabricated RECORD is accepted. Do not install the historical Meta PG082 package into this modern environment.

## Authentic cache and two explicit identities

Only already completed original `PreparedAudioResponse`/`AudioRow` data is accepted. The owner authenticates whole saved outputs, base/Source origins and the exact token/template/history provenance before admission. The library validates and binds the records; it does not prove tensor provenance solely from their shapes or hashes. No prepare/read_rows, tokenizer, teacher or generation call is made.

Choose `interface="glimmer"` or `interface="meta-glimmer"` before allocation and keep it fixed through the cache, receipt, FULL and RAW. Glimmer retains its original paired neutral/positive tokens with identical shared history. Meta retains neutral `[user]`, positive `[system,user]`, empty final history, reasoning `low`, generation prompt enabled and the original limit128 without truncation. The completed roles are authenticated, not reconstructed by importing the historical Meta trainer. CPU and native lanes are separate identities; synthetic caches cannot enter the native lane.

```python
from glimmer_fresh95 import open_training, open_inference, save_prepared_cache

# Owner-supplied: exact config and conditioning, original_prepared_rows,
# teacher_origins, component_factory, authenticate_and_admit, full_ids.
cache = save_prepared_cache(
    "completed-cache.pt", config=config, prepared=original_prepared_rows,
    teacher_origins=teacher_origins, cache_id="explicit-new-cache",
    scope="completed-native", owner_admission=authenticate_and_admit,
)
with open_training(
    config=config, cache=cache, directory="new-fresh-run", mode="fresh",
    cohort_id="explicit-new-cohort", component_factory=component_factory,
    owner_admission=authenticate_and_admit, ema_decay=None,
) as owner:
    owner.update()
    full = owner.save("FULL1.pt")
    raw = owner.export("RAW1.pt")

with open_training(
    config=config, cache=cache, directory="new-resume-run", mode="resume",
    resume=full, cohort_id="explicit-new-cohort",
    component_factory=component_factory,
    owner_admission=authenticate_and_admit, ema_decay=None,
) as resumed:
    pass

with open_inference(
    raw, component_factory=component_factory,
    owner_admission=authenticate_and_admit,
    interface=config["interface"], lane=config["lane"],
) as inference:
    hidden = inference.response(ids=full_ids, strength=1.0, seed=7)
```

The callback admits each operation and authenticates the completed cache/base. The factory returns only the pristine original model, without adapters or teacher work. These are owner-integrated API examples, not a standalone model-loading or GPU-admitting CLI. Outputs are original final hidden responses, not a text/audio sampler.

Configuration requires `interface`, `lane`, `conditioning`, `model_identity`, `width`, `model_seed`, `seed`, `steps`, `batch`, `ema_decay`, `pg_version`, `pg_source_profile`, `penalty_input_domain`, `base_precision`, `lr`, `d_lr_mult`, and `prior_lr_mult`. Native additionally requires `snapshot`, `expected_gpu_uuid` and `execution_profile`. Batch is2; CPU is the original width12/FP32 fixture, native is full width6656/52-layer/208-target NF4/BF16 with continuous penalty input. Native loader seed41 is retained. Strength remains [0,1]. Rates are explicit caller inputs; no new default is introduced.

## Initialization and state

Fresh is an explicit new cohort selected before components. The same original down Parameters receive public PG0.9.5 QR at the declared `kaiming_uniform_(a=sqrt(5))` variance `1/(3*fan_in)`. The same 128×4 bank receives R2Normal(0,1) low-discrepancy Gaussian quantiles, not matrix QR. Rank8/alpha8, zero-UP, router/bridge and critic constructor laws remain unchanged. A durable receipt is written before optimizer construction. No base weight or original Parameter identity is replaced.

Strict FULL restore binds the exact new source/configuration/interface/lane/cache/cohort and restores network, bank, critic, optimizer histories, cursors and RNG, without fresh initialization or replayed work. RAW skips cache, critic/game/optimizers and fresh initialization; the receiving caller must state the matching role and lane. Old FULL/RAW is not converted. Use `ema_decay=None`; exports and inference use raw weights, with scalar optimizer statistics unchanged.

Cleanup checks exclusive marker, exact private callbacks/containers, native/checkpoint attributes and captured owned flags/modes before mutation. Foreign state survives refusal. Failed cleanup remains pending and blocks work; restore the reported owned state and retry `close()`. The marker is removed last. The addon never delegates a broad original Slider.close over borrowed state.

## Software evidence and limits

One installed common suite passed four new CPU cases with one tiny update total, including both interface and lane identity boundaries, whole FULL/RAW state, initialization/RNG/Parameter preservation and cleanup retry. This is shared software coverage, not two independent native model qualifications. It does not qualify NF4 kernels/fit, native teachers, finished text, intended effect, LR performance, new defaults or released weights. Existing standalone and Meta measurements remain separate.
