"""CPU flag/refusal/RNG tests; no pretrained Glimmer or CUDA allocation."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from glimmer_native.lifecycle import config, configuration
from glimmer_native.numerics import (forward_guards, loader_initialization,
                                     numerical_policy, require_policy_identity)
from glimmer_native.contract import canonical_gpu_uuid, require_admission


def flags():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision=torch.get_float32_matmul_precision(),
        fp16=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash=torch.backends.cuda.flash_sdp_enabled(), efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn=torch.backends.cuda.cudnn_sdp_enabled(), math=torch.backends.cuda.math_sdp_enabled(),
        low=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())


def set_flags(value):
    torch.use_deterministic_algorithms(value["deterministic"], warn_only=value["warn_only"])
    torch.set_float32_matmul_precision(value["precision"])
    torch.backends.cuda.matmul.allow_tf32 = value["tf32"]
    torch.backends.cudnn.allow_tf32 = value["cudnn_tf32"]
    torch.backends.cudnn.benchmark = value["benchmark"]
    torch.backends.cudnn.deterministic = value["cudnn_deterministic"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = value["fp16"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = value["bf16"]
    torch.backends.cuda.enable_flash_sdp(value["flash"])
    torch.backends.cuda.enable_mem_efficient_sdp(value["efficient"])
    torch.backends.cuda.enable_cudnn_sdp(value["cudnn"])
    torch.backends.cuda.enable_math_sdp(value["math"])
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(value["low"])


@pytest.fixture
def strict_numerics(monkeypatch):
    previous = flags()
    set_flags(dict(deterministic=True, warn_only=False, benchmark=False, cudnn_deterministic=True,
        tf32=False, cudnn_tf32=False, precision="highest", fp16=False, bf16=False,
        flash=False, efficient=False, cudnn=False, math=True, low=False))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    try:
        yield numerical_policy()
    finally:
        set_flags(previous)


@pytest.mark.parametrize("key,wrong", [("deterministic", False), ("warn_only", True),
    ("benchmark", True), ("cudnn_deterministic", False), ("tf32", True),
    ("cudnn_tf32", True), ("precision", "high"), ("fp16", True), ("bf16", True),
    ("flash", True), ("efficient", True), ("cudnn", True), ("math", False), ("low", True)])
def test_policy_rejects_drift_without_mutating_flags(strict_numerics, key, wrong):
    changed = dict(flags(), **{key: wrong})
    set_flags(changed)
    if key == "precision":
        torch.set_float32_matmul_precision(wrong)
    before = flags()
    with pytest.raises(ValueError, match="strict numerical policy"):
        require_policy_identity(strict_numerics)
    assert flags() == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("key,value", [("NVIDIA_TF32_OVERRIDE", "1"),
    ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1"), ("CUBLAS_WORKSPACE_CONFIG", ":16:8")])
def test_environment_override_refused_without_flag_mutation(strict_numerics, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    before = flags()
    with pytest.raises(ValueError):
        numerical_policy()
    assert flags() == before


def test_forward_guard_stops_changed_flags_before_native_work(strict_numerics):
    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Linear(1, 1)
            self.model.vision_tower = nn.Linear(1, 1)
            eager = lambda: SimpleNamespace(_attn_implementation="eager")
            self.config = SimpleNamespace(_attn_implementation="eager", text_config=eager(), vision_config=eager())
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return self.model.language_model(value)
    host = Host()
    handles = forward_guards(host, strict_numerics)
    host(torch.ones(1, 1))
    assert host.calls == 1
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    with pytest.raises(ValueError):
        host(torch.ones(1, 1))
    assert host.calls == 1
    for handle in handles:
        handle.remove()
    host(torch.ones(1, 1))
    assert host.calls == 2


@pytest.mark.parametrize("fail", [False, True])
def test_loader_rng_restores_cpu_and_only_owned_device_on_success_or_failure(monkeypatch, fail):
    cpu_before = torch.random.get_rng_state().clone()
    owned, other = torch.Generator().manual_seed(111), torch.Generator().manual_seed(222)
    owned_before, other_before = owned.get_state().clone(), other.get_state().clone()
    monkeypatch.setattr(torch.cuda, "default_generators", (owned, other))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: (owned, other)[device].get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: (owned, other)[device].set_state(state))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: pytest.fail("CUDA-wide seeding forbidden"))
    def run():
        with loader_initialization():
            torch.rand(3)
            torch.rand(3, generator=owned)
            if fail:
                raise RuntimeError("bounded constructor failure")
    if fail:
        with pytest.raises(RuntimeError, match="constructor"):
            run()
    else:
        run()
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    assert torch.equal(owned_before, owned.get_state())
    assert torch.equal(other_before, other.get_state())
    assert not torch.cuda.is_initialized()


def test_native_default_config_disables_ema_explicit_diagnostic_ema_remains():
    default = config("/tmp/model", "/tmp/data", "GPU-72c1b506-891d-b8bc-b353-e020585e1c47")
    assert default.options["ema_decay"] is None
    configuration(default.options)
    diagnostic = config("/tmp/model", "/tmp/data", "GPU-72c1b506-891d-b8bc-b353-e020585e1c47", ema_decay=.9)
    assert diagnostic.options["ema_decay"] == .9
    configuration(diagnostic.options)


def test_observation_and_update_guard_reject_drift_before_body(strict_numerics):
    from glimmer_native.native import Native
    host = nn.Linear(1, 1)
    eager = lambda: SimpleNamespace(_attn_implementation="eager")
    host.config = SimpleNamespace(_attn_implementation="eager", text_config=eager(), vision_config=eager())
    native = object.__new__(Native)
    native.model = host
    native.network = nn.Linear(1, 1)
    native.identity = {"numerical": strict_numerics}
    assert native.validate_numerics() == strict_numerics
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    with pytest.raises(ValueError):
        native.validate_numerics()
    with pytest.raises(ValueError):
        with native.observation():
            pytest.fail("observation entered under changed native numerics")


def test_old_inference_numerics_refused_before_assets_or_model(strict_numerics, monkeypatch):
    import glimmer_native.lifecycle as lifecycle
    from glimmer_native.provenance import verify_sources
    _, source = verify_sources()
    options = config("/tmp/model", "/tmp/data", "GPU-72c1b506-891d-b8bc-b353-e020585e1c47").options
    fields = {k: options[k] for k in ("execution_profile", "pg_source_profile", "penalty_input_domain", "base_precision", "seed")}
    spec = SimpleNamespace(configuration=fields, identity={"model": {"product_source": source, "numerical": {"old": "loose"}}})
    monkeypatch.setattr(lifecycle, "inspect_snapshot", lambda *a, **kw: pytest.fail("assets touched before old identity refusal"))
    with pytest.raises(ValueError, match="numerical identity changed"):
        lifecycle.validate_inference({"snapshot": "/missing", "expected_gpu_uuid": "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"}, spec)
    assert not torch.cuda.is_initialized()


def test_actual_pg_source_refusal_precedes_loader_and_cuda(monkeypatch):
    import glimmer_native
    import glimmer_native.loader as loader
    from particle_sliders.pg09 import PG_SOURCE_PROFILE
    monkeypatch.setattr(glimmer_native, "PROFILE", PG_SOURCE_PROFILE)
    monkeypatch.setattr(loader, "inspect_snapshot", lambda *a, **kw: pytest.fail("assets touched before PG refusal"))
    monkeypatch.setattr(loader, "require_admission", lambda *a, **kw: pytest.fail("CUDA admission reached"))
    with pytest.raises(ValueError):
        loader.load_local("/missing", expected_gpu_uuid="GPU-72c1b506-891d-b8bc-b353-e020585e1c47")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("reported", ["72c1b506-891d-b8bc-b353-e020585e1c47",
    "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"])
def test_admission_accepts_actual_torch_bare_or_nvidia_prefixed_uuid(strict_numerics, monkeypatch, reported):
    from pathlib import Path
    import os
    declared = "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", declared)
    monkeypatch.setenv("GLIMMER_NATIVE_GUARDIAN", str(os.getppid()))
    monkeypatch.setenv("GLIMMER_NATIVE_PROCESS_CAP_GIB", "30")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(uuid=reported))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: (36 * 1024**3, 48 * 1024**3))
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: "MemAvailable: 26214400 kB\n" if str(p) == "/proc/meminfo" else original(p, *a, **kw))
    require_admission(declared)
    require_admission(declared.removeprefix("GPU-"))
    assert not torch.cuda.is_initialized()


def test_admission_refuses_realistic_uuid_mismatch_before_memory_query(strict_numerics, monkeypatch):
    import os
    declared = "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", declared)
    monkeypatch.setenv("GLIMMER_NATIVE_GUARDIAN", str(os.getppid()))
    monkeypatch.setenv("GLIMMER_NATIVE_PROCESS_CAP_GIB", "30")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(uuid="00000000-0000-0000-0000-000000000000"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: pytest.fail("memory queried after UUID mismatch"))
    with pytest.raises(ValueError, match="UUID differs"):
        require_admission(declared)


@pytest.mark.parametrize("bad", [None, True, "", "0", "GPU-invalid", "GPU-GPU-72c1b506-891d-b8bc-b353-e020585e1c47"])
def test_invalid_gpu_uuid_refused_without_cuda(bad):
    with pytest.raises(ValueError):
        canonical_gpu_uuid(bad)
    assert not torch.cuda.is_initialized()
