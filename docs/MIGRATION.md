# Phase 1 source-only extraction

Source: `mikkel/meta-glimmer-particle-slider`, commit
`3f0cdc4d17ebc6fa2df36d65b001ecfaf436c577`, product root
`packages/glimmer-modern-response/`.

Destination: `HyperGAN/glimmer-particle-sliders`. This new repository does not
mirror Git history or transfer the source repository. The default branch is a
minimal review scaffold; the executable product is proposed through a draft PR.

At initial extraction commit `c5200a565064d30a15691173c9f432c50fc7fa26`,
all eight `src/glimmer_modern/` runtime files, package configuration, and license
are byte-identical to the reviewed source. Only repository-facing README and
historical-provenance tests are adapted for the independent layout. The module
namespace, distribution name, registry names, cohort, initialization policy,
shared-core pin, HNDL graphs, numerical semantics and artifact identities remain
unchanged. The JSON provenance records original repo/SHA/path/blob and SHA256
against destination path/blob and SHA256, including every adaptation.

## Deferred historical model migration

The original `src/glimmer_slider/` tree and root historical dependency config
remain at their immutable source paths. They are explicitly indexed under
`deferred_historical` in the provenance JSON. This includes the PG82
`upstream.py` loader, historical `training.py`, host adapters, original storage,
native loader, sampler and studio modules. None is imported by the modern
distribution, so none is required for this installed phase 1 product.

Later migration must preserve those source pins, checkpoint formats, published
RAW1024/PEFT identities and original objectives. Original embedded/historical
sources remain authoritative until replacements have been separately verified.
No weights, recordings, private logs, run outputs or unrelated evidence are
copied into this repository. Existing release and training trees remain untouched.

## Qualification and review gates

The only admitted route is the existing width 12 FP32 CPU fixture. Native/wide
training remains unqualified and refused. Core commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214` is the software-qualified target; this migration
does not qualify native models or change shared APIs. Current runtime deltas
from the immutable extraction are recorded in `core-migration.json`; the original
extraction provenance and old preview metadata remain historical evidence. Mandatory Claude RETURN
review and central qualification are required before any landing. No workflow,
access, deployment, package/model release or native GPU job is added.
