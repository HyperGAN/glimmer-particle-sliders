import json
import os
from pathlib import Path

import pytest
import torch

from glimmer_native import DOMAIN, PROFILE, WIDTH
from glimmer_native.assets import memory_budget, token_rows
from glimmer_native.contract import (QUANTIZATION, require_admission, tensor_identity,
                                    validate_ids, validate_quantization)
from glimmer_native.lifecycle import CAPS, config, configuration


def test_native_domain_is_explicit_and_fixture_does_not_migrate():
    assert CAPS.response_shape == (1, 6656)
    assert (CAPS.pg_source_profile, CAPS.penalty_input_domain) == (PROFILE, DOMAIN)
    assert CAPS.render is False
    c = config("/tmp/model", "/tmp/pairs.json", "GPU-72c1b506-891d-b8bc-b353-e020585e1c47", lr=.001, d_lr_mult=54, prior_lr_mult=90)
    configuration(c.options)
    for key, wrong in (("execution_profile", "cpu-fixture"), ("pg_source_profile", "release-0.9.4"),
                       ("penalty_input_domain", "auto"), ("base_precision", "float32")):
        options = dict(c.options, **{key: wrong})
        with pytest.raises(ValueError):
            configuration(options)


@pytest.mark.parametrize("bad", [True, False, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("key", ["lr", "d_lr_mult", "prior_lr_mult"])
def test_lr_invalid_before_assets_or_cuda(key, bad):
    with pytest.raises(ValueError):
        config("missing-model", "missing-data", "missing-gpu", **{key: bad})
    assert not torch.cuda.is_initialized()


def test_no_guardian_refuses_before_cuda(monkeypatch):
    monkeypatch.delenv("GLIMMER_NATIVE_GUARDIAN", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CUDA queried before admission"))
    with pytest.raises(ValueError, match="guardian"):
        require_admission("GPU-72c1b506-891d-b8bc-b353-e020585e1c47")


def test_loader_never_reaches_model_construction_without_guardian(monkeypatch):
    import glimmer_native.loader as loader
    import glimmer_native.provenance as provenance
    monkeypatch.delenv("GLIMMER_NATIVE_GUARDIAN", raising=False)
    monkeypatch.setattr(provenance, "verify_sources", lambda: (None, {}))
    monkeypatch.setattr(loader, "inspect_snapshot", lambda *a, **kw: {"full_weight_hashes_verified": True})
    monkeypatch.setattr(loader, "runtime_sources", lambda: {})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("CUDA queried without resource admission"))
    with pytest.raises(ValueError, match="guardian"):
        loader.load_local("/does-not-exist", expected_gpu_uuid="GPU-72c1b506-891d-b8bc-b353-e020585e1c47")
    assert not torch.cuda.is_initialized()


def description():
    return dict(quant_type="nf4", blocksize=64, compress_statistics=True,
                quant_storage="torch.bfloat16", compute_dtype="torch.bfloat16", bnb_quantized=True,
                nested=True, nested_blocksize=256, original_dtype="torch.bfloat16",
                state_quant_type="nf4", state_blocksize=64)


@pytest.mark.parametrize("key,wrong", [("quant_type", "fp4"), ("blocksize", 128),
    ("compress_statistics", False), ("quant_storage", "torch.uint8"),
    ("compute_dtype", "torch.float16"), ("bnb_quantized", False), ("nested", False),
    ("nested_blocksize", 64), ("original_dtype", "torch.float16"),
    ("state_quant_type", "fp4"), ("state_blocksize", 128)])
def test_actual_quantization_contract_rejects_changed_settings(key, wrong):
    validate_quantization(description())
    with pytest.raises(ValueError):
        validate_quantization(dict(description(), **{key: wrong}))


def test_packed_byte_hash_is_exact_and_does_not_dequantize():
    from bitsandbytes.nn import Params4bit
    base = torch.tensor([float("nan"), 1., -2., 0.], dtype=torch.bfloat16)
    packed = Params4bit(base.clone(), requires_grad=False, quant_type="nf4", quant_storage=torch.bfloat16)
    before = packed.detach().view(torch.uint8).clone()
    first = tensor_identity(packed)
    assert first == tensor_identity(packed)
    assert torch.equal(before, packed.detach().view(torch.uint8))
    packed.data[1] = 2.
    assert tensor_identity(packed)["sha256"] != first["sha256"]
    assert not torch.cuda.is_initialized()


def test_native_asset_budget_is_a_bound_with_missing_costs():
    import glimmer_native.assets as assets
    tensors = json.loads(Path(assets.__file__).with_name("tensor-headers.json").read_text())
    budget = memory_budget(tensors)
    assert budget["parameters"] == 29776626688
    assert budget["optimistic_base_storage_bytes"] / 1024**3 == pytest.approx(17.628978729248047)
    assert budget["measured_peak"] is None and budget["estimate_only"]
    assert len(budget["missing"]) == 7
    assert budget["minimum_admission_free_gib"] == 35


@pytest.mark.parametrize("bad", [[], [True], [-1], [202048], [1] * 129, "tokens"])
def test_token_domain_rejects_bad_or_truncated_responses(bad):
    with pytest.raises(ValueError):
        validate_ids(bad)


def test_effective_context_overlap_is_rejected(tmp_path):
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(dict(train=[dict(neutral=[1], positive=[2], history=[3]),
                                        dict(neutral=[4], positive=[5], history=[])],
        evaluation=[dict(neutral=[1, 3], positive=[2, 3], history=[]),
                    dict(neutral=[6], positive=[7], history=[])])))
    with pytest.raises(ValueError, match="overlap"):
        token_rows(p)
