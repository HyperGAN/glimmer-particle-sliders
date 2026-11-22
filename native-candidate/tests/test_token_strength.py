from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    MuseGlimmerConfig, MuseGlimmerForConditionalGeneration, MuseGlimmerTextConfig,
    MuseGlimmerVisionConfig,
)

from glimmer_scheduling import _full_prefix_slider, _token_strengths


@pytest.fixture
def model():
    text = MuseGlimmerTextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        bos_token_id=1, eos_token_id=None, pad_token_id=0,
    )
    vision = MuseGlimmerVisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                                     num_attention_heads=4)
    config = MuseGlimmerConfig(text_config=text, vision_config=vision, projector_hidden_size=16,
                               out_hidden_size=128, image_token_id=30, video_token_id=31)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        return MuseGlimmerForConditionalGeneration(config).eval().requires_grad_(False)


def test_muse_glimmer_uses_one_strength_per_native_cached_token(model):
    prefix = torch.tensor([[1, 7, 8]])
    active = 0.0

    @contextmanager
    def scaled(value):
        nonlocal active
        previous, active = active, value
        try:
            yield
        finally:
            active = previous

    def force_token(_layer, _inputs, logits):
        result = logits.clone()
        result[..., 13] += active * 100
        return result

    hook = model.lm_head.register_forward_hook(force_token)
    options = dict(max_new_tokens=3, do_sample=False, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    try:
        with scaled(0.5):
            baseline = model.generate(prefix, **options)
        with _token_strengths(model, scaled, 3, [0.5, 0.5, 0.5]) as applied:
            constant = model.generate(prefix, **options)
        assert applied == [0.5, 0.5, 0.5]
        assert torch.equal(baseline.sequences, constant.sequences)
        assert all(torch.equal(left, right) for left, right in zip(baseline.scores, constant.scores))
        with _token_strengths(model, scaled, 3, [0.75, 0.25, 0.5]) as applied:
            early_eos = model.generate(prefix, **options, eos_token_id=13)
        assert early_eos.sequences.tolist() == [[1, 7, 8, 13]]
        assert applied == [0.75]
        assert active == 0
    finally:
        hook.remove()


def test_muse_glimmer_preserves_position_strengths_on_native_full_prefix(model):
    projection = model.model.language_model.layers[0].self_attn.q_proj
    base_forward = projection.forward
    branch = SimpleNamespace(
        org_forward=base_forward, scale=1, input_width=projection.in_features,
        output_width=projection.out_features, lora_down=projection, _particles=torch.zeros(1, 1),
        graph=lambda value, particles: value.new_ones((len(value), 1, projection.out_features)),
        multiplier=0.0,
    )
    branch.forward = lambda x: base_forward(x) + branch.multiplier
    projection.forward = branch.forward

    @contextmanager
    def scaled(value):
        previous, branch.multiplier = branch.multiplier, value
        try:
            yield
        finally:
            branch.multiplier = previous

    slider = SimpleNamespace(scaled=scaled, _attached=[(projection, branch)])
    prefix = torch.tensor([[1, 7, 8]])
    options = dict(max_new_tokens=3, do_sample=False, use_cache=False,
                   return_dict_in_generate=True, output_scores=True)
    with scaled(0.5):
        baseline = model.generate(prefix, **options)
    with _full_prefix_slider(slider, prefix.device) as scale:
        with _token_strengths(model, scale, 3, [0.5] * 3, full_prefix=True):
            constant = model.generate(prefix, **options)
    assert torch.equal(baseline.sequences, constant.sequences)
    assert all(torch.equal(left, right) for left, right in zip(baseline.scores, constant.scores))

    observed = []

    def capture(_module, args, output):
        observed.append((output - base_forward(args[0]))[0, :, 0])

    with projection.register_forward_hook(capture):
        with _full_prefix_slider(slider, prefix.device) as scale:
            with _token_strengths(model, scale, 3, [0.5, 0.0, 1.0], full_prefix=True) as applied:
                model.generate(prefix, **options)
    expected = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.0), (0.5, 0.5, 0.5, 0.0, 1.0))
    for actual, target in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, torch.tensor(target))
    assert applied == [0.5, 0.0, 1.0]
    assert projection.forward is branch.forward and branch.multiplier == 0.0


@pytest.fixture
def steering_core():
    import os

    from glimmer_scheduling.steering_core import load_steering_core

    wheel = os.getenv("GLIMMER_STEERING_CORE_WHEEL")
    if not wheel:
        pytest.skip("GLIMMER_STEERING_CORE_WHEEL must name the pinned #269 wheel")
    return load_steering_core(wheel)


@pytest.fixture
def native_session(model):
    from glimmer_native.fibo_hndl import HNDLParticleProjection
    from glimmer_native.native import Native
    from glimmer_native.provenance import verify_sources

    projection = model.model.language_model.layers[0].self_attn.q_proj
    builder, identity = verify_sources()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        branch = HNDLParticleProjection("fixture-q", projection, torch.randn(128, 4), 8, 8,
                                        builder=builder, identity=identity, attach=False)
        torch.nn.init.normal_(branch.lora_up.weight, std=0.1)
    projection.forward = branch.forward

    @contextmanager
    def scaled(value):
        previous, branch.multiplier = branch.multiplier, value
        try:
            yield
        finally:
            branch.multiplier = previous

    @contextmanager
    def observation(_seed, _inputs):
        yield

    native = object.__new__(Native)
    native.model, native.network = model, object()
    native.slider = SimpleNamespace(scaled=scaled, _attached=[(projection, branch)])
    session = SimpleNamespace(selection="raw", _open=lambda: None, _observation=observation,
                              binding=SimpleNamespace(close=native.close, network=native.network))
    return session, projection, branch


def _scores_match(left, right):
    assert torch.equal(left.sequences, right.sequences)
    assert all(torch.equal(a, b) for a, b in zip(left.scores, right.scores, strict=True))


@pytest.mark.parametrize("use_cache", [True, False])
def test_gated_raw_native_replay_and_exact_off(native_session, steering_core, use_cache):
    from glimmer_scheduling import generate
    from glimmer_scheduling.adaptive import ActivationGate

    session, projection, branch = native_session
    ids = torch.tensor([[1, 7, 8]])
    options = dict(input_ids=ids, strength=0.5, use_cache=use_cache, do_sample=False,
                   return_dict_in_generate=True, output_scores=True)

    def policy(parameter, seed):
        return ActivationGate(core=steering_core, layer=0, method="random", parameter=parameter,
                              seed=seed, intervention="raw_adapter", evidence_ref="fixture")

    off, _ = generate(session, step_gains=[0] * 4, **options)
    gated_off, off_receipt = generate(session, steering=policy(0, 2), max_new_tokens=4, **options)
    _scores_match(off, gated_off)
    assert off_receipt["usage"] == {"eligible": 4, "fired": 0}
    adaptive, receipt = generate(session, steering=policy(0.65, 2), max_new_tokens=4, **options)
    assert receipt["effective_strengths"] == [0.0, 0.0, 0.5, 0.5]
    assert [step["input_position"] for step in receipt["steps"]] == [2, 3, 4, 5]
    assert receipt["usage"] == {"eligible": 4, "fired": 2}
    fixed, applied = generate(session, step_gains=[0, 0, 1, 1], **options)
    _scores_match(adaptive, fixed)
    assert applied == receipt["effective_strengths"]
    assert projection.forward == branch.forward and branch.multiplier == 0


@pytest.mark.parametrize("use_cache", [True, False])
def test_live_cfp_last_prompt_position_and_native_eos(model, native_session, steering_core, use_cache):
    from glimmer_scheduling import generate
    from glimmer_scheduling.adaptive import ActivationGate

    session, projection, branch = native_session
    ids = torch.tensor([[1, 7, 8]])
    embeddings = model.get_input_embeddings()(ids)[0, -1].detach().float()
    direction = embeddings / embeddings.norm()
    alignment = steering_core.adaptive.activation_projection(embeddings, direction, strength=0.5).item()
    policy = ActivationGate(core=steering_core, layer=0, method="cfp", parameter=alignment,
                            direction=direction, intervention="raw_adapter", evidence_ref="fixture")
    observed = []

    def stop(_head, _args, out):
        forced = torch.zeros_like(out)
        forced[..., 13] = 10
        return forced

    def capture(_projection, args, result):
        observed.append((result, branch.org_forward(args[0])))

    with model.lm_head.register_forward_hook(stop), projection.register_forward_hook(capture):
        generated, receipt = generate(session, input_ids=ids, steering=policy, max_new_tokens=4,
                                      strength=0.5, use_cache=use_cache, do_sample=False, eos_token_id=13)
    assert generated.tolist() == [[1, 7, 8, 13]]
    assert len(observed) == 1
    assert receipt["usage"] == {"eligible": 1, "fired": 1}
    assert receipt["steps"][0]["alignment"] == pytest.approx(alignment)
    edited, frozen = observed[0]
    assert torch.equal(edited[:, :-1], frozen[:, :-1])
    assert not torch.equal(edited[:, -1], frozen[:, -1])
    assert projection.forward == branch.forward and branch.multiplier == 0


@pytest.mark.parametrize("use_cache", [True, False])
def test_shared_s_psr_and_clabs_are_request_local(native_session, steering_core, use_cache):
    from glimmer_scheduling import generate
    from glimmer_scheduling.adaptive import ActivationGate, FittedPromptSteering

    session, projection, branch = native_session
    options = dict(input_ids=torch.tensor([[1, 7, 8]]), use_cache=use_cache, do_sample=False,
                   return_dict_in_generate=True, output_scores=True)
    fitted = steering_core.prompt.PromptSteering(32)
    with torch.no_grad():
        fitted.probe.weight.zero_()
        fitted.probe.bias.fill_(0.5)
        fitted.direction.weight.fill_(0.2)
    psr = FittedPromptSteering(core=steering_core, fitted=fitted, layer=1, evidence_ref="fixture")
    off, _ = generate(session, step_gains=[0] * 3, **options)
    disabled, receipt = generate(session, steering=psr, strength=0.0, max_new_tokens=3, **options)
    _scores_match(off, disabled)
    assert receipt["steps"] == [] and receipt["effective_strengths"] == [0.0] * 3
    on, receipt = generate(session, steering=psr, strength=-2.0, max_new_tokens=3, **options)
    assert receipt["decoder_input_layer"] == 1 and "decoder_output_layer" not in receipt
    assert receipt["effective_strengths"] == [-1.0] * 3
    assert all(step["coefficient"] == 0.5 for step in receipt["steps"])
    assert not torch.equal(on.scores[0], off.scores[0])
    assert fitted.training and all(p.requires_grad for p in fitted.parameters())

    clabs = steering_core.adaptive.CLABSParameters(mean=0, std=1, epsilon=1e-6, setpoint=0.1,
        proportional_gain=0.3, integral_gain=0.3, beta=0.4)
    policy = ActivationGate(core=steering_core, layer=1, method="clabs", parameter=0.5,
                            direction=torch.ones(32), intervention="residual", seed=21,
                            clabs=clabs, evidence_ref="fixture")
    first, receipt = generate(session, steering=policy, strength=0.5, max_new_tokens=3, **options)
    second, second_receipt = generate(session, steering=policy, strength=0.5, max_new_tokens=3, **options)
    _scores_match(first, second)
    assert receipt == second_receipt and receipt["usage"]["eligible"] == 3
    assert projection.forward == branch.forward and branch.multiplier == 0


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_s_psr_reads_final_native_block_before_norm(model, native_session, steering_core, use_cache, dtype):
    from glimmer_scheduling import generate
    from glimmer_scheduling.adaptive import FittedPromptSteering

    session, projection, branch = native_session
    model.to(dtype=dtype)
    decoder = model.model.language_model
    layer = decoder.layers[-1]
    fitted = steering_core.prompt.PromptSteering(32)
    with torch.no_grad():
        fitted.probe.weight.fill_(0.02)
        fitted.probe.bias.fill_(1.0)
        fitted.direction.weight.copy_(torch.linspace(0.01, 0.2, 32))
    policy = FittedPromptSteering(core=steering_core, fitted=fitted, layer=len(decoder.layers) - 1,
                                 coordinate="decoder_output", evidence_ref="fixture-native-post-block")
    ids = torch.tensor([[1, 7, 8]])
    options = dict(input_ids=ids, use_cache=use_cache, do_sample=False,
                   return_dict_in_generate=True, output_scores=True)
    off, _ = generate(session, step_gains=[0.0] * 3, **options)
    disabled, zero = generate(session, steering=policy, strength=0.0, max_new_tokens=3, **options)
    _scores_match(off, disabled)
    assert zero["decoder_output_layer"] == policy.layer and zero["effective_strengths"] == [0.0] * 3
    assert zero["steps"] == []

    before, during_norm = [], []

    def capture_block(_layer, _args, output):
        before.append(output.detach().clone())

    def capture_norm(_norm, args):
        during_norm.append(args[0].detach().clone())

    with layer.register_forward_hook(capture_block), decoder.norm.register_forward_pre_hook(capture_norm):
        edited, receipt = generate(session, steering=policy, strength=-1.5, max_new_tokens=3, **options)

    assert receipt["decoder_output_layer"] == policy.layer and "decoder_input_layer" not in receipt
    assert [step["input_position"] for step in receipt["steps"]] == [2, 3, 4]
    assert not torch.equal(edited.scores[0], off.scores[0])
    effective = []
    with torch.no_grad():
        for index, (unedited, seen, step) in enumerate(zip(before, during_norm, receipt["steps"], strict=True)):
            leading = 2 if not use_cache or index == 0 else 0
            assert seen.dtype == unedited.dtype == dtype and seen.shape == unedited.shape
            assert seen.shape[1] == (3 + index if not use_cache else (3 if index == 0 else 1))
            assert torch.equal(seen[:, :leading], unedited[:, :leading])
            latest, coefficient = fitted(unedited[:, -1:], -1.5)
            assert torch.equal(seen[:, -1:], latest)
            assert step["coefficient"] == coefficient.item()
            assert step["effective_strength"] == (coefficient * -1.5).item()
            if not use_cache:
                history = torch.tensor(effective, dtype=torch.float32).view(1, -1, 1)
                older = unedited[:, leading:-1]
                reference = (older.float() + history * fitted.direction.weight[0]).to(dtype)
                assert torch.equal(seen[:, leading:-1], reference)
            effective.append(step["effective_strength"])
    assert effective == receipt["effective_strengths"]
    assert fitted.training and all(parameter.requires_grad for parameter in fitted.parameters())
    assert projection.forward == branch.forward and branch.multiplier == 0


def test_post_block_hook_keeps_tuple_tail_and_restores_after_failure(steering_core):
    from glimmer_scheduling.adaptive import FittedPromptSteering, _adaptive_strengths

    metadata = object()

    class Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden, metadata, 7

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Block()
            self.model = SimpleNamespace(language_model=SimpleNamespace(layers=[self.layer]))
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))

        def forward(self, *, input_ids, fail=False):
            result = self.layer(input_ids[..., None].expand(-1, -1, 4).to(torch.bfloat16))
            if fail:
                raise RuntimeError("failed after the post-block hook")
            return result

    @contextmanager
    def scaled(value):
        assert value == 0.0
        yield

    model = Decoder()
    native = SimpleNamespace(model=model, slider=SimpleNamespace(scaled=scaled))
    fitted = steering_core.prompt.PromptSteering(4)
    with torch.no_grad():
        fitted.probe.weight.zero_()
        fitted.probe.bias.fill_(1.0)
        fitted.direction.weight.fill_(0.5)
    policy = FittedPromptSteering(core=steering_core, fitted=fitted, layer=0,
                                 coordinate="decoder_output", evidence_ref="fixture-tuple")
    ids = torch.tensor([[2, 3]])
    expected = torch.tensor([[[2.0] * 4, [3.5] * 4]], dtype=torch.bfloat16)
    options = dict(prefix=2, max_new_tokens=2, strength=1.0, use_cache=False, device=ids.device)
    with _adaptive_strengths(native, policy, **options) as receipt:
        hidden, cache, auxiliary = model(input_ids=ids)
        assert torch.equal(hidden, expected) and cache is metadata and auxiliary == 7
        assert receipt["steps"][0]["effective_strength"] == 1.0
    with pytest.raises(RuntimeError, match="failed after the post-block hook"):
        with _adaptive_strengths(native, policy, **options):
            model(input_ids=ids, fail=True)
    assert not model.layer._forward_hooks and not model._forward_hooks and not model._forward_pre_hooks
    assert torch.equal(model(input_ids=ids)[0], ids[..., None].expand(-1, -1, 4).to(torch.bfloat16))


def test_failed_native_forward_restores_adaptive_scope(model, native_session, steering_core):
    from glimmer_scheduling import generate
    from glimmer_scheduling.adaptive import ActivationGate

    session, projection, branch = native_session
    always = ActivationGate(core=steering_core, layer=0, intervention="raw_adapter", method="random",
                            parameter=1, seed=2, evidence_ref="fixture")
    layer = model.model.language_model.layers[0]
    hooks = (len(model._forward_pre_hooks), len(model._forward_hooks), len(layer._forward_pre_hooks))

    def fail(*_):
        raise RuntimeError("failed native fixture")

    with model.lm_head.register_forward_hook(fail), pytest.raises(RuntimeError, match="failed native fixture"):
        generate(session, input_ids=torch.tensor([[1, 7, 8]]), steering=always, max_new_tokens=3,
                 strength=0.5, use_cache=False)
    assert projection.forward == branch.forward and branch.multiplier == 0
    assert hooks == (len(model._forward_pre_hooks), len(model._forward_hooks), len(layer._forward_pre_hooks))


def test_exact_core_wheel_rejects_altered_source_after_loading(steering_core, tmp_path):
    from zipfile import ZipFile

    from glimmer_scheduling.steering_core import load_steering_core

    wheel = tmp_path / "changed.whl"
    source = b"?" * 10329
    with ZipFile(wheel, "w") as archive:
        archive.writestr("particle_sliders/adaptive_steering.py", source)
    with pytest.raises(ValueError, match="source checksum"):
        load_steering_core(wheel)


def test_independent_steering_loader_imports_stay_isolated(steering_core):
    from concurrent.futures import ThreadPoolExecutor
    import os
    from runpy import run_path
    from threading import Barrier

    import glimmer_scheduling.steering_core as packaged

    ready = Barrier(2)

    def load_separate():
        standalone = run_path(packaged.__file__)
        ready.wait(timeout=5)
        return standalone["load_steering_core"](os.environ["GLIMMER_STEERING_CORE_WHEEL"])

    with ThreadPoolExecutor(2) as threads:
        futures = [threads.submit(load_separate) for _ in range(2)]
        loaders = [future.result(timeout=10) for future in futures]
    assert len({core.adaptive.__name__ for core in [steering_core, *loaders]}) == 3
    assert all(core.adaptive.ActivationGate("cfp", 0).step(-1).fired for core in loaders)


def test_adaptive_request_refuses_an_overlapping_schedule(model, native_session, steering_core):
    from glimmer_scheduling import generate
    from glimmer_scheduling.adaptive import ActivationGate

    session, projection, branch = native_session
    raw = ActivationGate(core=steering_core, layer=0, method="random", parameter=1,
                         seed=2, intervention="raw_adapter", evidence_ref="fixture")
    ids = torch.tensor([[1, 7, 8]])

    def nested(*_):
        with pytest.raises(RuntimeError, match="already has an active scheduled generation"):
            generate(session, input_ids=ids, step_gains=[1])

    with model.lm_head.register_forward_pre_hook(nested):
        output, receipt = generate(session, input_ids=ids, steering=raw, strength=0.5, max_new_tokens=2)
    assert output.shape == (1, 5) and receipt["effective_strengths"] == [0.5, 0.5]
    assert projection.forward == branch.forward and branch.multiplier == 0
    _, applied = generate(session, input_ids=ids, step_gains=[0.5])
    assert applied == [0.5]
