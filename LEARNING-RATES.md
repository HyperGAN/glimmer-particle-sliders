# Learning rates

The current training configuration accepts optional `lr` (base/generator), `d_lr_mult` (critic), and `prior_lr_mult` (actual routed particle bank). Values must be finite positive numbers; booleans are rejected before model allocation. The existing shared ParticleGAN recipe creates all optimizers: D = `lr * d_lr_mult`, bank = `lr * prior_lr_mult`. No product GAN or optimizer math is duplicated.

Omitted controls retain the default `bispf_optimized` recipe: G **0.012**, D multiplier **1.5**, bank multiplier **2.5**. These are inherited defaults, not native calibration results. This factory uses its fixed optimized preset. Use the keys in the shared session configuration; the registered common training dispatcher executes the same recipe. Historical/source-only CLI cohorts retain their original interfaces.

Use `"lr": 0.00002, "d_lr_mult": 1.5, "prior_lr_mult": 2.5` in common session `options`. The numbers are an input example, not a tuned recommendation. The actual resolved shared recipe and configuration bind full-state resume; a changed rate requires a fresh experiment output/cohort. Older packets and closed HNDL source cohorts keep their original source/defaults. Current CPU optimizer/resume tests do not qualify pretrained/native learning rates, audio or semantic quality.

Native/wide-response source gates remain enforced. Native calibration compares identical initialized state, data, stochastic streams, precision, declared schedule horizons and completed-update budgets on the actual pretrained model. Select and independently confirm rates using only held-out raw teacher/student relative SSE: cast student, teacher and neutral to FP64 before subtraction, pool positions within each example, divide by that example's teacher-edit energy, then average examples equally. Keep the teacher-only valid/excluded set fixed across candidates; zero-energy examples remain undefined, and student failures never remove examples. See the [common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).

Preservation/no-edit drift, late/weak-field scores and native media outputs are optional diagnostics; they cannot select, reject or delay a rate. After the native mechanical proof, candidate endpoint net displacement is diagnostic too. Retain exact source/model/data/numerical identity, actual optimizer-rate, finite native-update, checkpoint-continuation and raw-response replay checks. Tiny CPU software checks are not native LR evidence.

## Measured native PG0.9.5 preset

The [native preset](native-candidate/presets/pg095-native-nf4-128.json) supplies
these exact shared-session options for fresh native runs:

```json
{
  "lr": 0.001,
  "d_lr_mult": 18.0,
  "prior_lr_mult": 29.999999999999996,
  "ema_decay": null,
  "steps": 128,
  "batch": 2,
  "seed": 7,
  "pg_source_profile": "release-0.9.5",
  "penalty_input_domain": "continuous"
}
```

The actual rates are G`0.001`, D`0.018000000000000002` and bank
`0.029999999999999995`. Preserve the multiplier literals: rounding the bank
multiplier to30 changes its actual rate. The native factory fixes the source
profile and response domain, so its equivalent call is:

```python
from glimmer_native.lifecycle import config

session = config(
    snapshot, training_tokens, gpu_uuid,
    steps=128, batch=2, seed=7, ema_decay=None,
    lr=0.001, d_lr_mult=18.0, prior_lr_mult=29.999999999999996,
)
```

Run it under the native guardian and numerical policy described in the
[native guide](native-candidate/README.md). Exact checkpoint continuation
requires the original complete session, cohort and source identity.

All four G candidates completed matched32, the two finalists completed
matched64, and the selected G completed128. D and bank stayed fixed. The
selected development error was`0.7172243530127634`; the previously unused
four-example final was **`0.5341646099807267`** versus OFF1.0. All four examples
were valid, and the independent CPU reduction exactly reproduced every saved
SSE, teacher energy and score. Native public RAW/OFF, frozen-base, adapter,
checkpoint and RNG checks passed; final evaluation made zero training updates.

This is the best tested G in the pinned Muse-Glimmer-30B NF4/BF16,
native6656 FP32 response cohort, batch2/common128. It does not establish a joint
G/D/bank optimum or text-generation quality. The root width12 CPU fixture has
its own scope. The [sanitized measurement](native-candidate/docs/pg095-native-nf4-128-measurement.json)
records exact native-source fingerprints and evidence digests. The native
package now matches the measured 20GiB guard and rejects model-weight EMA;
omitted LR controls still retain their inherited values.

The packages remain0.2.0 on sharedcore0.13.0 / publishedPG0.9.5. Historical
source routes and migration receipts retain their original identities.
