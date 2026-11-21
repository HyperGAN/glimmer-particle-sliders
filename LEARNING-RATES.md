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

## Earlier teacher-response SSE screen

The earlier response-error screen is diagnostic evidence.
The [archived preset and measurements](https://github.com/HyperGAN/glimmer-particle-sliders/blob/fc2fcfc98d13db0a7f313a613459f2793c89a7a9/LEARNING-RATES.md#historical-teacher-response-sse-screen-raw128) retain their original runtime.
