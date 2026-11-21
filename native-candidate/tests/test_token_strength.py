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
