"""Real HNDL at native dimensions, without constructing a Glimmer model."""
from pathlib import Path

import torch
from torch import nn

from glimmer_native import DOMAIN, PROFILE, WIDTH
from glimmer_native.native import initialization_rng
from glimmer_native.provenance import verify_sources
from particle_sliders.fibo_hndl import HNDLParticleProjection
from particle_sliders.pg09_audio import require_audio_width


def test_real_native_width_hndl_projection_and_critic_vjp_cpu():
    assert not torch.cuda.is_initialized()
    builder, identity = verify_sources()
    require_audio_width(WIDTH, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
    with initialization_rng(7, "bounded-o-projection-structural-test"):
        base = nn.Linear(4096, 6656, bias=False).requires_grad_(False)
        bank = nn.Parameter(torch.randn(128, 4))
        branch = HNDLParticleProjection("structural-o-projection", base, bank, 8, 8,
                                        builder=builder, identity=identity, attach=False)
        branch.lora_up.weight.data.normal_(0, .001)
        value = torch.randn(1, 1, 4096)
        branch.multiplier = 0.
        assert torch.equal(branch(value), base(value))
        branch.multiplier = 1.
        output = branch(value)
        gradient = torch.autograd.grad(output.square().mean(), bank)[0]
        assert output.shape == (1, 1, 6656)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        source = Path(__import__("glimmer_native.native", fromlist=["x"]).__file__).with_name("readout.hndl").read_text()
        critic = builder(source, input_shape={"response": ["B", WIDTH], "condition": ["B", WIDTH + 4]}, output_shape=["B", 1])
        response = torch.randn(2, WIDTH, requires_grad=True)
        condition = torch.randn(2, WIDTH + 4)
        score = critic(response=response, condition=condition)
        vjp = torch.autograd.grad(score.sum(), response, create_graph=True)[0]
        penalty = vjp.square().mean()
        penalty.backward()
        assert score.shape == (2, 1) and vjp.shape == (2, WIDTH)
        assert torch.isfinite(vjp).all()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert not torch.cuda.is_initialized()
