"""Native placement/scale bindings; HNDL and the shared PG game own the math."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from particle_sliders.fibo_hndl import HNDLParticleProjection
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter
from particle_sliders.lifecycle import native_cleanup

from . import DOMAIN, LAYERS, PROFILE, PROTOCOL, WIDTH
from .assets import TARGET
from .contract import validate_ids
from .provenance import verify_sources


@contextmanager
def initialization_rng(seed, label):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("bounded integer initialization seed required")
    derived = int.from_bytes(hashlib.sha256(f"glimmer-native-v1:{seed}:{label}".encode()).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derived)
        yield


def targets(model, identity):
    from bitsandbytes.nn import Linear4bit
    selected = {n: m for n, m in model.named_modules() if TARGET.fullmatch(n + ".weight")}
    if len(selected) != LAYERS * 4 or set(selected) != set(identity["projections"]):
        raise ValueError("loaded/audited native targets differ")
    for name, module in selected.items():
        if (type(module) is not Linear4bit or module.weight.device != torch.device("cuda:0")
                or [module.out_features, module.in_features] != identity["projections"][name]
                or hasattr(module, "_glimmer_native")):
            raise ValueError("audited NF4 projection changed before attachment")
    return selected


def preserve_native_state(model):
    names = ("gradient_checkpointing", "_gradient_checkpointing_func")
    original = [(m, {k: vars(m)[k] for k in names if k in vars(m)}) for m in model.modules()]
    restore = native_cleanup(model)
    def close():
        try:
            restore()
        finally:
            for module, values in original:
                for key in names:
                    if key in values:
                        setattr(module, key, values[key])
                    elif key in vars(module):
                        delattr(module, key)
    return close


class Slider(nn.Module):
    def __init__(self, model, identity, *, seed):
        super().__init__()
        selected = targets(model, identity)
        builder, source = verify_sources()
        self.source = source
        self._targets = selected
        self._attached = []
        self._close_requested = False
        self._restore = preserve_native_state(model)
        self._modes = [(m, m.training) for m in model.modules()]
        self._flags = [(p, p.requires_grad) for p in model.parameters()]
        with initialization_rng(seed, "bank"):
            self.particles = nn.Parameter(torch.randn(128, 4, device="cpu"))
        self.particles.data = self.particles.data.to("cuda:0")
        self.branches = nn.ModuleDict()
        for index, (name, module) in enumerate(selected.items()):
            with initialization_rng(seed, "projection:" + name):
                branch = HNDLParticleProjection(name, module, self.particles, 8, 8,
                                                builder=builder, identity=source, attach=False)
            branch.to(device="cuda:0", dtype=torch.float32)
            self.branches[str(index)] = branch
        try:
            model.eval().requires_grad_(False)
            for module, branch in zip(selected.values(), self.branches.values()):
                self._attached.append((module, branch))
                module.forward = branch.forward
                object.__setattr__(module, "_glimmer_native", self)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def scaled(self, strength):
        if self._close_requested or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("live native slider and strength within[0,1] required")
        previous = [b.multiplier for b in self.branches.values()]
        try:
            for branch in self.branches.values():
                branch.multiplier = float(strength)
            yield
        finally:
            for branch, value in zip(self.branches.values(), previous):
                branch.multiplier = value

    def close(self):
        self._close_requested = True
        errors, pending = [], []
        for module, branch in reversed(self._attached):
            branch.multiplier = 0.
            try:
                if module.forward != branch.forward or getattr(module, "_glimmer_native", None) is not self:
                    raise RuntimeError("native projection cleanup ownership changed")
                del module.forward
                del module._glimmer_native
            except BaseException as error:
                errors.append(error)
                pending.append((module, branch))
        self._attached = list(reversed(pending))
        try:
            self._restore()
        except BaseException as error:
            errors.append(error)
        finally:
            for module, mode in self._modes:
                module.training = mode
            for parameter, flag in self._flags:
                parameter.requires_grad_(flag)
        if errors:
            raise errors[0]


class _Host(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(hidden_size=WIDTH)


class GlimmerResponse(ARLastHiddenResponseAdapter):
    family = "glimmer_native6656_last_hidden_nf4_v1"

    def _hidden(self, ids):
        tokens = torch.tensor([list(ids)], dtype=torch.long, device="cuda:0")
        return self.model.model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                      use_cache=False, return_dict=True).last_hidden_state[:, -1]


class Native:
    def __init__(self, model, identity, *, seed):
        self.model, self.identity = model, identity
        self.slider = Slider(model, identity, seed=seed)
        try:
            blocks = tuple(model.model.language_model.layers)
            self.backend = GlimmerResponse(_Host(model), self.slider, model_identity=identity,
                checkpoint_modules=blocks, pg_source_profile=PROFILE, penalty_input_domain=DOMAIN)
            self.network = self.backend.network
        except BaseException:
            self.slider.close()
            raise

    @property
    def particles(self):
        return self.slider.particles

    def generated(self, prepared):
        return self.backend.generated(prepared)

    def metadata(self):
        return dict(protocol=PROTOCOL, model_identity=self.identity, response_domain="continuous_vector",
                    response_shape=[1, WIDTH], source_profile=self.slider.source,
                    target="frozen positive final hidden; identical causal continuation history")

    def response(self, *, ids, strength=1., seed=0):
        validate_ids(ids)
        return self.backend._response(tuple(ids), scale=strength).detach().clone()

    @contextmanager
    def observation(self):
        modes = [(m, m.training) for m in self.model.modules()]
        buffers = [(b, b.detach().clone()) for b in self.model.buffers()]
        parameters = list(self.model.parameters()) + list(self.network.parameters())
        gradients = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone()) for p in parameters]
        try:
            self.model.eval()
            yield
        finally:
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)
                for parameter, flag, grad, value in gradients:
                    parameter.requires_grad_(flag)
                    parameter.grad = grad
                    if grad is not None:
                        grad.copy_(value)
            for module, mode in modes:
                module.training = mode

    def close(self):
        try:
            self.backend.close_checkpointing()
        finally:
            self.slider.close()


class ReadoutCritic(nn.Module):
    score_bound = 0

    def __init__(self, condition_width, *, seed):
        super().__init__()
        builder, identity = verify_sources()
        source = Path(__file__).with_name("readout.hndl").read_text()
        with initialization_rng(seed, "critic"):
            self.graph = builder(source, input_shape={"response": ["B", WIDTH],
                "condition": ["B", condition_width]}, output_shape=["B", 1])
        self.graph.to(device="cuda:0", dtype=torch.float32)
        self.identity = identity

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)
