# Glimmer particle sliders

This separate distribution uses `glimmer_modern` and the separate registry name
`glimmer-modern`. Install it in a fresh environment, never in the historical
PG82 environment. This independent `HyperGAN/glimmer-particle-sliders` repository
extracts the reviewed modern product from `mikkel/meta-glimmer-particle-slider`
at `3f0cdc4d17ebc6fa2df36d65b001ecfaf436c577`, under
`packages/glimmer-modern-response/`. The original repository is not transferred
or modified. See [source provenance](docs/extraction-provenance.json) and
[deferred historical migration](docs/MIGRATION.md).

This is the current standalone source for new Glimmer response-cohort work.
The original PG82 package remains a historical reproduction path. The modern
cohort already used ParticleGAN 0.9.4 before this update; the update adopts the
current shared-core packaging and dependency corrections. It does not retrain
or relabel published weights. The executable product and this upgrade are
proposed for the default branch through review; the default-branch scaffold
alone contains no implementation.

The complete factory delegates updates, optimizers, penalties, sampling, EMA,
checkpoint validation and native-only export to the shared particle-sliders
response lifecycle. HyperGAN's verified builder executes both HNDL routed
projections and the HNDL readout critic. One real 128 × 4 bank participates in all
text attention branches. Q/O and grouped K/V retain their different dimensions;
the response is the genuine final text hidden state, without projection,
pooling, truncation, token reinterpretation or spatial reshaping.

The explicit new objective cohort is the shared released PG0.9.4 conditional
BISPF recipe. It does **not** claim numerical objective equivalence to PG82's
historical winning formulation. Network graph parity is tested separately.
The historical loader, trainer, RAW1024 artifacts and distilled PEFT students
are unchanged and require their original source pins and loaders.

The current cohort is `glimmer-modern-final-hidden-hg-hndl-v3-cpu`, with explicit
initialization policy `cpu-sha256-component-streams-v1`. The declared run `seed`
initializes the bank, every named adapter projection, and the critic using
separate SHA256-derived CPU streams before their random constructors. The
caller CPU RNG is restored on success and failure. CUDA RNG is neither queried
nor seeded by this CPU-only policy, so existing CUDA state is untouched and no
GPU initialization is requested. `model_seed` separately retains the original
frozen base mapping. The policy and run seed are bound in configuration,
backend metadata, graph identities and native-only inference specifications.

Earlier v1/v2 packets require their original source revision and initialization
behavior. They are refused by v3; no checkpoint relabeling or migration occurs.
The shared loss, bank architecture and frozen positive/neutral hidden-response
semantics are unchanged.

## Current admission

Only `mode="cpu-fixture"`, width 12, FP32 CPU is admitted. The bundled causal
model is a synthetic protocol fixture, not pretrained Glimmer. Native model
loading/training, width >= 170, media/token generation and model quality are
unqualified and refused before preparation. No native admission option or
continuous-domain override exists. Passing CPU tests cannot open these gates.

The current pinned shared-core source is `ddb41423bcefc1fe76cd94d0f9b367353b3ea137`
(exact Git pin), official PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0.
`source-lock.json` verifies installed shared sources; the common builder checks
the complete HNDL closure, HyperGAN closure and clean build provenance before
any native attachment. The recursive lock covers all source/data files in both core namespaces, plus
exact installed version and Git origin. Product RETURN review remains pending.
Central synthetic CPU/CUDA qualification does not admit native Glimmer, width
or quality. This product migration runs CPU tests only.
The current package-validation receipt is [merged-base validation](docs/merged-base-validation.json).
The earlier [8ffca receipt](docs/pg094-validation.json) and
[a794 receipt](docs/core-validation.json) remain historical evidence.

Borrowed CPU models with existing local module/parameter hooks, instance call
or serialization overrides, compiled calls or active checkpoint wrappers are
refused before model hashing or attachment. Cleanup restores original instance
attribute presence, including inherited forwards and checkpoint attributes.
These product checks do not provide packet-aware admission: deep packet/source
and tensor checks may still occur after factory preparation in the
shared lifecycle. Universal pre-hook refusal is not claimed.

## CPU use

Create a fresh Python 3.11–3.14 environment and install CPU PyTorch from the
public CPU index, then install this directory. It does not depend on the
historical `meta-glimmer-particle-slider` distribution.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
python -m pip install '.[test]'
python -I -m pytest tests --import-mode=importlib -q
```

The installed shared CLI uses the same registered response backend. After
writing the paired rows described below, create its configuration and run:

```sh
python - <<'PY'
import json
from glimmer_modern.lifecycle import cpu_config
with open('glimmer.json', 'w') as stream:
    json.dump(cpu_config('pairs.json').as_dict(), stream, indent=2)
profile = dict(schema='particle-slider-profile-v1', name='glimmer-cpu-fixture',
    product='glimmer-particle-sliders', config_kind='session', status='fixture',
    interface='response', backend='glimmer-modern', runtime={},
    capabilities=dict(resume=True, periodic_checkpoint=True, check=True),
    default_checkpoint_every=2)
with open('glimmer-profile.json', 'w') as stream:
    json.dump(profile, stream, indent=2)
PY
particle-slider capabilities --interface response --config glimmer.json
particle-slider plan --config glimmer.json --profile glimmer-profile.json --out run --no-reporting
particle-slider train --config glimmer.json --profile glimmer-profile.json --out run --no-reporting --export-format both
```

`run` must be a new empty directory. These commands train only the admitted
synthetic CPU fixture; they cannot load pretrained Glimmer or produce text.
The merged unified runner requires the explicit response fixture profile above.
`cpu_config` supplies its required `dummy=true` marker; false markers are refused.
Adding `--device` would insert an unsupported option into this CPU-only cohort;
its backend already enforces FP32 CPU. The compatibility response commands
(`capabilities`, `evaluate` and `infer`) remain available.

Store a bounded local JSON file with `train` and `evaluation` arrays, each with
2..32 pairs such as `{"neutral":[1,3],"positive":[2,4],"history":[6,7]}`.
IDs belong to the fixture's 32-token vocabulary. Each pair shares exactly the
same continuation history. Every split/target and the frozen base weight bytes
are bound to the continuation identity. Use fixed distinct held-out rows.

```python
from glimmer_modern.lifecycle import cpu_config
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

config = cpu_config('pairs.json')
with open_response(config) as run:
    run.train(stop_after=2)
    run.checkpoint('modern-training.pt')
with open_response(config, resume='modern-training.pt') as run:
    run.train()
    run.evaluate(split='evaluation', strength=.5)
    run.export('modern-raw.pt')
    run.export_inference('modern-ema-v2.pt', selection='ema')
with open_inference('modern-ema-v2.pt', operation='response',
                    native_request={'ids': [1, 3, 6, 7]}) as run:
    response = run.response(ids=[1, 3, 6, 7], strength=.5)
```

The last factory reconstructs only the deterministic CPU base and adapter. It
requires no paired data, teacher cache, critic, optimizer or training factory.
Training and inference use the shared versioned schemas and reject historical
packets, wrong cohorts, sources and tensor contracts. A response is a numerical
hidden-state probe, not a native text quality or rendering result.

`initialization_rng(component=None)` preserves the base `model_seed` mapping;
it is not an unnamed training component. Provenance hash tests check recorded
lineage and self-consistency. External source equality was verified at initial
extraction; those tests do not continuously re-fetch or verify the remote source.

For an existing environment, pip can retain an older wheel with the same 0.12.0
version. Replace that core explicitly, then let the product resolve its extras:

```sh
python -m pip install --force-reinstall --no-deps 'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@ddb41423bcefc1fe76cd94d0f9b367353b3ea137#subdirectory=packages/particle-sliders-core'
python -m pip install '.[test]'
python -m pip check
```

The product verifies the installed Git origin as well as recursive source/data
bytes. Core 0.12.0's version number alone is insufficient for this cohort.

This cohort deliberately requires pip's direct VCS provenance for the exact
repository URL and commit. A core installed from a locally built wheel or an
offline wheelhouse is refused even when its source bytes match, because that
installation does not retain the required direct VCS record. There is no local
wheel or offline provenance exception; use the pinned Git installation above.

Learning-rate controls and preserved defaults: [LEARNING-RATES.md](LEARNING-RATES.md).
