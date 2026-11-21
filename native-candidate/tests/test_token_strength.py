from contextlib import contextmanager

import torch
from transformers import (
    MuseGlimmerConfig, MuseGlimmerForConditionalGeneration, MuseGlimmerTextConfig,
    MuseGlimmerVisionConfig,
)

from glimmer_scheduling import _token_strengths


def test_muse_glimmer_uses_one_strength_per_native_cached_token():
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
        model = MuseGlimmerForConditionalGeneration(config).eval().requires_grad_(False)
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
