# Learning rates

## Current finished-output decision

Reuse the completed four-rate RAW32 comparison on the original native6656
NF4/BF16 Glimmer cohort, batch2, seed7 and common128 schedule. Two fixed neutral
cases (`garden-seed42`, `coast-seed43`) were compared with their matched OFFs.

| G | Mean intended-effect ON−OFF gain | Decision |
| ---: | ---: | --- |
| 0.0001 | +1.0 | Tied best |
| 0.001 | +1.0 | Tied best |
| 0.012 | 0.0 | Retained result |
| 0.036 | 0.0 | Retained result |

Keep **both** best-score rates. There is no unique winner, new default, extra
confirmation, rerating or repeat LR search. Fixed actual D is
`0.018000000000000002`; bank is `0.029999999999999995`. The complete portable
[decision metadata](docs/finished-output-lr-decision.json) retains all candidates,
case gains, evidence hashes and exact tied options. It contains no model weights
or private paths and is not a complete session/checkpoint configuration.

Rank only `mean_cases(effect_intensity(ON) - effect_intensity(OFF))` in finished
outputs, with equal case weights. Keep negative gains, zeros and exact ties.
Technical failures remain missing evidence. Teacher/student relative SSE,
preservation drift, beauty, confidence and completion are not selectors or
tiebreaks. Reuse existing saved RAWs/outputs/scores; do not replay completed work.
This plain-text native BOS/greedy512/noKV scope is separate from Meta/chat;
never pool their scores. The coherent G0.001 RAW128 garden previews and repeated
high-G coast previews are separate scale/budget observations, not a new LR vote.

## Actual common options

The native factory accepts `lr`, `d_lr_mult`, `prior_lr_mult` and
`ema_decay=None`. Shared ParticleGAN computes D=`lr*d_lr_mult` and
bank=`lr*prior_lr_mult`; preserve the saved literal multipliers:

| Tied G | d_lr_mult | prior_lr_mult | ema_decay |
| ---: | ---: | ---: | --- |
| 0.0001 | 180.0 | 299.99999999999994 | null |
| 0.001 | 18.0 | 29.999999999999996 | null |

Both options are retained; the order is not a default choice. For configuration
inspection, choose either complete tied option dictionary explicitly:

```python
from glimmer_native.lifecycle import config

# candidate_options is one of the two tied_rate_overrides in the linked JSON.
# snapshot, paired_tokens_file and admitted_gpu_uuid are caller-owned inputs.
session_config = config(
    snapshot, paired_tokens_file, admitted_gpu_uuid,
    steps=128, batch=2, seed=7, **candidate_options,
)
```

`steps=128` fixes the original schedule horizon; **RAW32** names the measured
completed-update endpoint, not a32-step replacement schedule. This call only
constructs a configuration; use existing saved RAW32 artifacts rather than
retraining them. Exact continuation needs the original complete source/data/
SessionConfig; changing a rate or this documentation does not rebind a checkpoint.
The factory and inherited defaults remain unchanged: optimized G0.012, D
multiplier1.5, bank multiplier2.5. Optional finite positive rates reject booleans;
explicit `ema_decay=None` and RAW weights are required for new work. Do not add
weight EMA, checkpoint averaging or an EMA comparison.

The root width12 FP32 CPU fixture is a separate software route; native6656
requires its original assets, numerical profile and caller-owned admission. No
new native proof or confirmation is requested by this documentation update.

<a id="measured-native-pg095-preset"></a>

## Historical teacher-response SSE screen (RAW128)

The following earlier measurement and preset retain their exact source and
result identities. They describe numerical teacher matching and are **diagnostic
history**, not the current intended-effect LR selector or a preferred rate over
the completed RAW32 tie. The original preset/report files are unchanged.


The [native preset](native-candidate/presets/pg095-native-nf4-128.json) supplies
the exact options used in the earlier 128-update response-error screen:

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

This was the response-SSE-selected G in the pinned Muse-Glimmer-30B NF4/BF16,
native6656 FP32 response cohort, batch2/common128. It does not establish a joint
G/D/bank optimum or text-generation quality. The root width12 CPU fixture has
its own scope. The [sanitized measurement](native-candidate/docs/pg095-native-nf4-128-measurement.json)
records exact native-source fingerprints and evidence digests. The native
package now matches the measured 20GiB guard and rejects model-weight EMA;
omitted LR controls still retain their inherited values.

The packages remain0.2.0 on sharedcore0.13.0 / publishedPG0.9.5. Historical
source routes and migration receipts retain their original identities.
