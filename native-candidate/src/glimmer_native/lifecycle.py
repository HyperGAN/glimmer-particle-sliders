"""Explicit native6656 source cohort; the common response lifecycle owns updates."""
import hashlib
import json
import math
from pathlib import Path

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, EvaluationRows, SessionConfig
from particle_sliders.pg09 import ConditionalPG09, PG_VERSION
from particle_sliders.pg09_audio import AudioRow, audio_conditions
from particle_sliders.response import ResponseBackend, ResponseCapabilities

from . import BACKEND_NAME, DOMAIN, EXECUTION, PROFILE, PROTOCOL, WIDTH
from .assets import inspect_snapshot, token_rows
from .contract import validate_ids
from .loader import load_local
from .native import Native, ReadoutCritic
from .provenance import verify_sources

CAPS = ResponseCapabilities("routed_particle_gan", "glimmer_causal_text_hidden",
    "native6656_final_text_hidden_nf4_candidate", "continuous_vector", (1, WIDTH),
    PG_VERSION, PROFILE, penalty_input_domain=DOMAIN, base_precisions=("bfloat16",),
    qualification="CPU source/structural readiness only; native NF4 numerical/resume/language qualification pending")


def rates(options):
    from .learning_rates import recipe_overrides
    return recipe_overrides(**{key: options.get(key) for key in ("lr", "d_lr_mult", "prior_lr_mult")})


def configuration(options):
    if (options.get("execution_profile") != EXECUTION or options.get("pg_source_profile") != PROFILE
            or options.get("penalty_input_domain") != DOMAIN or options.get("base_precision") != "bfloat16"):
        raise ValueError("explicit native NF4/continuous source/numerical profile required")
    seed = options.get("seed", 7)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("bounded integer seed required")
    for key, upper, default in (("steps", 4096, 64), ("batch", 2, 2)):
        value = options.get(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError("bounded positive integer required: " + key)
    ema = options.get("ema_decay")
    if ema is not None and (type(ema) not in (float, int) or not math.isfinite(ema) or not 0 <= ema < 1):
        raise ValueError("finite EMA decay within[0,1) required")
    rates(options)


def validate(options):
    allowed = {"execution_profile", "pg_source_profile", "penalty_input_domain", "base_precision",
               "snapshot", "expected_gpu_uuid", "data_file", "steps", "batch", "seed", "ema_decay",
               "pg_version", "lr", "d_lr_mult", "prior_lr_mult"}
    if set(options) - allowed:
        raise ValueError("unknown native candidate options")
    configuration(options)
    inspect_snapshot(options["snapshot"], full_hashes=False)
    token_rows(options["data_file"])
    verify_sources()


def preflight(options, request, staged):
    if staged or request["native"]:
        raise ValueError("native candidate owns its exact loader and fixed paired-token conditioning")


def _make(snapshot, expected_gpu_uuid, seed):
    model, _, identity = load_local(snapshot, expected_gpu_uuid=expected_gpu_uuid)
    _, source = verify_sources()
    identity["product_source"] = source
    return Native(model, identity, seed=seed)


def prepare(options):
    rows, data_sha = token_rows(options["data_file"])
    native = _make(options["snapshot"], options["expected_gpu_uuid"], options.get("seed", 7))
    try:
        prepared = {split: [native.backend.prepare(AudioRow(
            tuple(row["neutral"] + row["history"]), tuple(row["positive"] + row["history"]), len(row["history"])))
            for row in values] for split, values in rows.items()}
        conditions = audio_conditions(prepared["train"])
        critic = ReadoutCritic(conditions.shape[1], seed=options.get("seed", 7))
        game = ConditionalPG09(native, prepared["train"], critic, conditions=conditions,
            particles=native.particles, response_domain="continuous_vector",
            total_steps=options.get("steps", 64), batch_size=options.get("batch", 2), seed=options.get("seed", 7),
            pg_source_profile=PROFILE, penalty_input_domain=DOMAIN, recipe_overrides=rates(options))
        spec = INFERENCE.specification(native, model_identity=native.identity, protocol=PROTOCOL,
            configuration=dict(execution_profile=EXECUTION, pg_source_profile=PROFILE,
                penalty_input_domain=DOMAIN, base_precision="bfloat16", seed=options.get("seed", 7)))
        evaluation_identity = dict(protocol="glimmer-native-held-out-token-pairs-v1", data_sha256=data_sha,
            evaluation_sha256=hashlib.sha256(json.dumps(rows["evaluation"], sort_keys=True).encode()).hexdigest())
        return Binding(game, native.close, inference=spec, observation=native.observation,
            before_update=lambda _state: native.validate_numerics(),
            evaluation_rows={"evaluation": EvaluationRows(prepared["evaluation"], evaluation_identity)})
    except BaseException:
        native.close()
        raise


def validate_inference(options, spec):
    if set(options) != {"snapshot", "expected_gpu_uuid"}:
        raise ValueError("native-only reload requires exact external snapshot and admitted GPU UUID")
    if set(spec.configuration) != {"execution_profile", "pg_source_profile", "penalty_input_domain", "base_precision", "seed"}:
        raise ValueError("native inference profile fields differ")
    configuration(spec.configuration)
    from particle_sliders.pg09_audio import require_audio_width
    from .numerics import require_policy_identity
    require_audio_width(WIDTH, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
    _, source = verify_sources()
    model = spec.identity.get("model", {})
    if model.get("product_source") != source:
        raise ValueError("inference native product source differs before model allocation")
    require_policy_identity(model.get("numerical"))
    inspect_snapshot(options["snapshot"], full_hashes=False)


def inference_preflight(options, request, spec):
    if set(request["native"]) != {"ids"}:
        raise ValueError("native response requires bounded token IDs")
    validate_ids(request["native"]["ids"])


def prepare_inference(options, spec):
    native = _make(options["snapshot"], options["expected_gpu_uuid"], spec.configuration["seed"])
    try:
        identity = native_identity(native, model_identity=native.identity, protocol=PROTOCOL)
        return InferenceBinding(native.network, identity, native.close,
                                response=native.response, observation=native.observation)
    except BaseException:
        native.close()
        raise


BACKEND = ResponseBackend(BACKEND_NAME, CAPS, prepare, validate, preflight=preflight)
INFERENCE = InferenceBackend(BACKEND_NAME, CAPS, prepare_inference, validate_inference, preflight=inference_preflight)


def config(snapshot, data_file, expected_gpu_uuid, *, steps=64, batch=2, seed=7,
           ema_decay=None, lr=None, d_lr_mult=None, prior_lr_mult=None):
    from .learning_rates import recipe_overrides
    overrides = recipe_overrides(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    return SessionConfig(BACKEND_NAME, PROTOCOL, execution_profile=EXECUTION,
        pg_source_profile=PROFILE, penalty_input_domain=DOMAIN, base_precision="bfloat16",
        snapshot=str(Path(snapshot).resolve()), data_file=str(Path(data_file).resolve()),
        expected_gpu_uuid=expected_gpu_uuid, steps=steps, batch=batch, seed=seed, ema_decay=ema_decay, **overrides)
