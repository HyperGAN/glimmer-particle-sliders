# Glimmer particle sliders

Both current standalone packages use shared core **0.13.0** at
`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423` and published **ParticleGAN 0.9.5**.
New training uses fresh protocol identities. Earlier PG0.9.4/PG801 and PG82
checkpoints retain their original environments and are not converted.

The root `glimmer_modern` package provides the width-12 FP32 CPU response
fixture. The separate [native NF4 package](native-candidate/README.md) keeps the
full pinned 30B model's 6656-wide response and explicit continuous domain.
The new source version has no pretrained LR calibration or native training
qualification yet. Older native proof receipts remain historical evidence.

```sh
python -m pip install 'git+https://github.com/HyperGAN/glimmer-particle-sliders.git@main'
```

Use a fresh environment. The installed product pins the exact shared-core Git
commit; its source guard checks the complete installed core inventory and
actual immutable VCS origin. Its new CPU protocol is
`glimmer-modern-final-hidden-hg-hndl-pg095-v5-cpu`. Native installation uses the
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
For its continuous native response use `release-0.9.5` with `continuous`;
no response projection, truncation or categorical reinterpretation is used.

[Learning-rate controls](LEARNING-RATES.md) retain the inherited numerical
defaults. New-source trials select and independently confirm rates using only
held-out RAW FP64 equal-example pooled relative SSE. Preservation, language,
rendering and listening observations are optional diagnostics.

The historical [PG0.9.4 locks](compatibility/pre-pg095/README.md),
[original extraction](docs/extraction-provenance.json),
[core migration](docs/core-migration.json) and
[LR migration](docs/lr-migration.json) retain their original bytes.
[PG0.9.5 migration](docs/pg095-migration.json) records the new source cohort.
The [Meta repository](https://github.com/mikkel/meta-glimmer-particle-slider)
and its 20 published PG82 controls remain a separate reproduction path with
original loaders and recipes. New code does not retrain or relabel those weights.
