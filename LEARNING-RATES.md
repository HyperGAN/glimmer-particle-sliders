# Learning rates

The current training configuration accepts optional `lr` (base/generator), `d_lr_mult` (critic), and `prior_lr_mult` (actual routed particle bank). Values must be finite positive numbers; booleans are rejected before model allocation. The existing shared ParticleGAN recipe creates all optimizers: D = `lr * d_lr_mult`, bank = `lr * prior_lr_mult`. No product GAN or optimizer math is duplicated.

Omitted controls retain the default `bispf_optimized` recipe: G **0.012**, D multiplier **1.5**, bank multiplier **2.5**. These are inherited defaults, not native calibration results. This factory uses its fixed optimized preset. Use the keys in the shared session configuration; the registered common training dispatcher executes the same recipe. Historical/source-only CLI cohorts retain their original interfaces.

Use `"lr": 0.00002, "d_lr_mult": 1.5, "prior_lr_mult": 2.5` in common session `options`. The numbers are an input example, not a tuned recommendation. The actual resolved shared recipe and configuration bind full-state resume; a changed rate requires a fresh experiment output/cohort. Older packets and closed HNDL source cohorts keep their original source/defaults. Current CPU optimizer/resume tests do not qualify pretrained/native learning rates, audio or semantic quality.

Native/wide-response source gates remain enforced. Native calibration needs the exact pretrained model, an admitted upstream domain/API source, identical fixed data/seeds/update budgets, held-out edit and preservation evaluation, and durable source/recipe/results evidence.

The current package is 0.1.1 / v4 LR CPU protocol. The prior v3 source route remains in `compatibility/pre-lr-v3/requirements.txt`; `docs/core-migration.json` and older validation receipts retain their historical bytes. Current changes have a distinct `docs/lr-migration.json`. Native Glimmer remains unqualified.
