# Phase1 source-only extraction

Source: `mikkel/meta-glimmer-particle-slider`, commit
`3f0cdc4d17ebc6fa2df36d65b001ecfaf436c577`, product root
`packages/glimmer-modern-response/`.

Destination: `HyperGAN/glimmer-particle-sliders`. This new repository does not
mirror Git history or transfer the source repository. The default branch is a
minimal review scaffold; the executable product is proposed through a draft PR.

All eight `src/glimmer_modern/` runtime files, package configuration, and license
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
distribution, so none is required for this installed phase1 product.

Later migration must preserve those source pins, checkpoint formats, published
RAW1024/PEFT identities and original objectives. Original embedded/historical
sources remain authoritative until replacements have been separately verified.
No weights, recordings, private logs, run outputs or unrelated evidence are
copied into this repository. Existing release and training trees remain untouched.

## Qualification and review gates

The only admitted route is the existing width12 FP32 CPU fixture. Native/wide
training remains unqualified and refused. Core commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` remains provisional; this extraction
does not admit another core or repair shared APIs. Mandatory Claude RETURN
review and central qualification are required before any landing. No workflow,
access, deployment, package/model release or native GPU job is added.
