# Glimmer particle sliders

## Train a text vibe with ParticleGAN 0.12.0

Describe the style in one phrase to train a slider for Muse Glimmer 30B. The native CUDA worker uses Anima's rank-eight routed attention branches, 128 × 4 particle bank, and stock ParticleGAN 0.12.0 EMPF update on frozen text responses. The default run qualifies at least 16 teaching contexts, trains for 32 updates on a 128-step optimizer schedule, and calibrates strength on finished replies. The exported slider includes its selected token-strength schedule.

From this checkout, install the text runtime in its own environment. Set `CUDA_VISIBLE_DEVICES` to one GPU UUID, `NANOGPT_API_KEY` for the Jev reader during training, and `MODEL_SNAPSHOT` to the local snapshot directory ending in `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.

```sh
python3.12 -m venv .venv-text
.venv-text/bin/pip install ./vibe
.venv-text/bin/glimmer-vibe --snapshot "$MODEL_SNAPSHOT" --vibe "short, playful rhymes" --out ./rhyming
.venv-text/bin/glimmer-vibe --snapshot "$MODEL_SNAPSHOT" --adapter ./rhyming/slider.safetensors --prompt "Welcome a friend to a library."
```

Omit `--adapter` to generate with the base model. The run directory contains the teacher verdicts, RAW checkpoints, strength comparisons, and held-out replies. [Shared training implementation](https://github.com/HyperGAN/particle-sliders/tree/0268a29166e228b40b353c3bd8284c9ae1e658dd/packages/particle-sliders-vibe#train-and-sample-a-text-vibe).

In a rhyme qualification on four held-out prompts, the selected constant strength of 2.0 raised mean rhyme score by 0.053 on a 0–1 scale. Most of the gain came from one prompt.

## Prior studies and releases

**Calculated memory:** Native NF4 text-response training: **28 GiB GPU / 14 GiB host**, batch 2/128 tokens, checkpointed. Same Muse-Glimmer backbone as the other Glimmer interface. [Full memory ledger and limits](MEMORY.md). These are planning estimates, not measured peaks.

The earlier CPU fixture and native response study use shared core **0.15.2** at
`ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` and published **ParticleGAN 0.11.1**.
New training uses fresh protocol identities. Earlier checkpoints retain their original environments.

The root `glimmer_modern` package provides the width-12 FP32 CPU response
fixture. The separate [native NF4 package](native-candidate/README.md) keeps the
full pinned 30B model's 6656-wide response and explicit continuous domain.
The completed finished-output comparison reuses four saved RAW32 checkpoints
under the original common128 schedule. **G0.0001 and G0.001 tie best**, each
with mean intended-effect ON−OFF gain **+1.0** across the two matched cases;
G0.012 and G0.036 each score0. There is **no unique rate or default change**.
See [the decision, exact common options and scope](LEARNING-RATES.md).
The older G0.001/RAW128 teacher-response SSE screen is retained as historical
diagnostic evidence, not a current text-effect selector.

```sh
python -m pip install 'git+https://github.com/HyperGAN/glimmer-particle-sliders.git@main'
```

Use a fresh environment. The installed product pins the exact shared-core Git
commit; its source guard checks the complete installed core inventory and
actual immutable VCS origin. Its new CPU protocol is
`glimmer-modern-final-hidden-hg-hndl-engine-v5-cpu`. Native installation uses the
separate subdirectory and its own response/inference backend.

The shared lifecycle owns updates, losses, penalties, optimizers, full-state
continuation and inference export. HyperGAN's verified builder executes HNDL
routed projections and the critic. The real 128 × 4 bank, rank8/alpha8 branches,
frozen base, seed policy and raw hidden-response geometry are unchanged.

Disable model-weight EMA explicitly (`ema_decay=None`) and use raw current
weights for training, selection, confirmation, export and inference. Do not add
EMA or checkpoint averaging. Historical EMA compatibility evidence stays scoped
to its original source and does not guide current runs.

The default root package admits only its synthetic CPU fixture; it does not
load pretrained Glimmer. The native package requires its exact model assets,
BF16/NF4 numerical policy, source locks and explicit device/resource admission.
For its continuous native response use `release-0.11.1` with `continuous`;
no response projection, truncation or categorical reinterpretation is used.

[Learning-rate controls](LEARNING-RATES.md) expose the exact saved tied options;
omitted controls retain inherited defaults. Use one score: the equal-case mean
intended-effect intensity ON minus matched OFF in finished text. Teacher SSE,
preservation, beauty and confidence are diagnostics, never selectors or
tiebreaks. Reuse completed searches, saved RAW checkpoints and existing scores;
no repeated search, rerating or extra confirmation is required. Plain-text and
Meta/chat interfaces share one base but retain separate conditioning and results.

The [historical source locks](https://github.com/HyperGAN/glimmer-particle-sliders/blob/fc2fcfc98d13db0a7f313a613459f2793c89a7a9/compatibility/pre-pg095/README.md),
[original extraction](docs/extraction-provenance.json),
[core migration](docs/core-migration.json) and
[LR migration](docs/lr-migration.json) retain their original bytes.
[Prior migration evidence](https://github.com/HyperGAN/glimmer-particle-sliders/blob/fc2fcfc98d13db0a7f313a613459f2793c89a7a9/docs/pg095-migration.json) describes the earlier source cohort.
The pre-learning-rate cohort retains its [original requirements lock](https://github.com/HyperGAN/glimmer-particle-sliders/blob/fc2fcfc98d13db0a7f313a613459f2793c89a7a9/compatibility/pre-lr-v3/requirements.txt).
The [Meta repository](https://github.com/mikkel/meta-glimmer-particle-slider)
and its 20 published PG82 controls remain a separate reproduction path with
original loaders and recipes. New code does not retrain or relabel those weights.


## Optional shared explicit fresh initialization

See [the separate Glimmer / Meta Glimmer scoped API](SCOPED.md) for one owner-integrated addon with distinct interface/lane identities, authentic completed caches, strict FULL restore and RAW hidden-response inference. Four shared installed synthetic cases do not qualify native NF4 execution or change the measured LR ties.
