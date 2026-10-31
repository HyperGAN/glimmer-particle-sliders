# Historical phase1 receipt

The receipt below applies to extracted head `c5200a565064d30a15691173c9f432c50fc7fa26`
and its original4abc core. It is preserved without relabeling old packet identities.
Current a794/v3 CPU validation is recorded in `docs/core-validation.json`. Old
4abc/v2 previews require their original source; the v3 product refuses them.

# Standalone phase1 validation

This extraction preserves the exact runtime bytes of source commit
`3f0cdc4d17ebc6fa2df36d65b001ecfaf436c577` in
`mikkel/meta-glimmer-particle-slider`, product root
`packages/glimmer-modern-response/`. All eight runtime files, the package
configuration and license retain their original Git blob and SHA256 identities.
README and three historical-origin assertions are adapted for standalone layout;
the assertions now verify the frozen historical lineage without copying or
requiring the umbrella runtime. The four additional extraction tests verify
installed bytes, unique entrypoints, relocated source-lock and deferred lineage.

The installed wheel and sdist are tested from a separate directory with Python
isolated imports, a fresh CPU-only virtual environment and public dependency
resolution. Test modules are in this new repository; no umbrella source is on
the import path. Each artifact passes 73 tests with no skips. `pip check` passes.

Runtime: Python3.14.4, Torch2.15.0+cpu, NumPy2.5.4, ParticleGAN0.9.4,
HNDL0.10.0 and exact HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. The shared-core dependency remains
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, provisional and unrepinned.

## Cross-source primary evidence

The original installed distribution produced an initial state, two-step training
checkpoint, v1 raw/EMA artifacts, v2 native-only raw/EMA artifacts and numerical
responses. The extracted installed distribution reconstructs the identical
initial checkpoint and resumes the source checkpoint to an identical final
checkpoint. It reproduces OFF, half-strength and full-strength responses
byte-for-byte for both raw and EMA v1/v2 artifacts. Native-only inference also
works with the paired-data file absent.

An audit hook forbids file access to the umbrella checkout throughout that
interoperability check. No historical package is installed. CUDA remains
uninitialized. Only tiny CPU fixture tensors are used; they are not included
in the source repository. See [interop receipt](docs/standalone-interop.json).

Initial checkpoint SHA256:
`45fb3bb129423811a234fda2d665c9f6ca535e59e4f88b09fad3c9384806494f`

Final checkpoint SHA256 after four steps:
`1ad209b5085738eb50a16ffad07a6a66dbbdeaeb9b7ed1ec81c99d2cb2e21e24`

The unchanged conformance suite additionally checks exact resume, old-packet
refusal, source/profile/width guards, matched outputs/VJPs/critic second
derivatives, seeded initialization, failure rollback and cleanup. These remain
CPU protocol checks; native/wide Glimmer and quality are unqualified.

## Reproduce from this repository

```sh
python -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
.venv/bin/python -m pip install 'setuptools>=84.2.1' build
.venv/bin/python -m build --no-isolation --outdir dist
.venv/bin/python -m pip install 'dist/meta_glimmer_modern_response-0.1.0-py3-none-any.whl[test]'
.venv/bin/python -I -m pytest tests --import-mode=importlib -q
.venv/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation dist/meta_glimmer_modern_response-0.1.0.tar.gz
.venv/bin/python -I -m pytest tests --import-mode=importlib -q
.venv/bin/python -m pip check
```

No workflows are installed. This draft does not publish a package or weights,
change access, qualify a native model, deploy a service or merge the extraction.
Root owns mandatory Claude RETURN review and final core/merge decisions.
