"""Token strength schedules for an open native Glimmer RAW session."""

from contextlib import ExitStack, contextmanager
from math import isfinite
from sys import float_info
from threading import Lock

import torch

_ACTIVE_GENERATIONS = set()
_ACTIVE_LOCK = Lock()


@contextmanager
def _exclusive_generation(model):
    with _ACTIVE_LOCK:
        if model in _ACTIVE_GENERATIONS:
            raise RuntimeError("Glimmer already has an active scheduled generation on this model")
        _ACTIVE_GENERATIONS.add(model)
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_GENERATIONS.remove(model)


def _require_token_position(inputs, prefix, position, budget, full_prefix):
    ids = inputs.get("input_ids")
    cache = inputs.get("past_key_values")
    context_length = 0 if cache is None else int(cache.get_seq_length())
    expected_length = 0 if full_prefix or position == 0 else prefix + position - 1
    width = prefix + position if full_prefix else (prefix if position == 0 else 1)
    if (position >= budget or ids is None or ids.shape != (1, width)
            or context_length != expected_length):
        raise RuntimeError("Glimmer generation diverged from the declared token schedule")
    return width


@contextmanager
def _token_strengths(model, scaled, prefix, strengths, *, full_prefix=False):
    scope = ExitStack()
    applied = []

    def before(_module, _args, inputs):
        position = len(applied)
        _require_token_position(inputs, prefix, position, len(strengths), full_prefix)
        value = strengths[position]
        if full_prefix:
            envelope = (strengths[0],) * prefix + tuple(strengths[1:position + 1])
            scope.enter_context(scaled(envelope))
        else:
            scope.enter_context(scaled(value))
        applied.append(value)

    def after(_module, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(model.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        yield applied


@contextmanager
def _full_prefix_slider(slider, device):
    weights = None
    edited_positions = None

    @contextmanager
    def scaled(envelope):
        nonlocal weights, edited_positions
        if all(value == envelope[0] for value in envelope):
            with slider.scaled(envelope[0]):
                yield
        else:
            with slider.scaled(0.0):
                weights = torch.tensor(envelope, device=device, dtype=torch.float32).view(1, -1, 1)
                edited_positions = weights != 0
                try:
                    yield
                finally:
                    weights = None
                    edited_positions = None

    def forward(branch, x):
        if weights is None:
            return branch.forward(x)
        if x.ndim != 3 or x.shape[:2] != weights.shape[:2]:
            raise RuntimeError("Glimmer projection diverged from the full-prefix envelope")
        # Keep the frozen projection and the HNDL residual on the complete
        # native input. Apply token strengths after the native residual cast.
        value = x.to(branch.lora_down.weight).reshape(-1, 1, branch.input_width)
        bank = branch._particles.unsqueeze(0).expand(value.shape[0], -1, -1)
        delta = branch.graph(value=value, particles=bank).reshape(*x.shape[:-1], branch.output_width)
        frozen = branch.org_forward(x)
        return torch.where(edited_positions, frozen + (delta.to(x) * weights).to(x), frozen)

    with ExitStack() as projections:
        for module, branch in slider._attached:
            if module.forward != branch.forward or branch.scale != 1:
                raise RuntimeError("Glimmer requires its attached rank-8, alpha-8 projections")
            projections.callback(setattr, module, "forward", module.forward)
            module.forward = lambda x, branch=branch: forward(branch, x)
        yield scaled


def generate(session, *, input_ids, step_gains=None, steering=None, max_new_tokens=None,
             attention_mask=None, strength=None, seed=0, use_cache=True, **sampling):
    """Generate native text with a fixed gain list or an explicitly calibrated policy.

    Fixed mode returns the output and its applied strength list. Adaptive mode
    requires the token budget and strength, and returns the output and its receipt.
    """
    from glimmer_native.native import Native

    session._open()
    native = getattr(session.binding.close, "__self__", None)
    if not isinstance(native, Native) or session.selection != "raw" or native.network is not session.binding.network:
        raise ValueError("an open native Glimmer RAW session is required")
    if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or input_ids.shape[1] == 0 or input_ids.dtype != torch.long):
        raise ValueError("one nonempty tokenized text prompt is required")
    if type(use_cache) is not bool:
        raise ValueError("use_cache must be a bool")
    if steering is not None:
        from .adaptive import _adaptive_strengths

        if (step_gains is not None or type(max_new_tokens) is not int or max_new_tokens < 1
                or type(strength) not in (int, float) or not isfinite(strength)):
            raise ValueError("For adaptive mode provide a positive token budget and explicit finite strength without step_gains")
        with (
            _exclusive_generation(native.model),
            session._observation(seed, {"input_ids": input_ids}),
            torch.no_grad(),
            _adaptive_strengths(native, steering, prefix=input_ids.shape[1], max_new_tokens=max_new_tokens,
                                strength=float(strength), use_cache=use_cache, device=input_ids.device) as receipt,
        ):
            output = native.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, use_cache=use_cache, num_beams=1,
                num_return_sequences=1, **sampling)
        sequences = output if isinstance(output, torch.Tensor) else output.sequences
        count = sequences.shape[1] - input_ids.shape[1]
        steps = receipt["steps"]
        if not 0 < count <= max_new_tokens or sequences.shape[0] != 1 or (strength != 0 and len(steps) != count):
            raise RuntimeError("Glimmer did not produce one native output token per adaptive decision")
        receipt.update(generation_seed=seed, max_new_tokens=max_new_tokens)
        receipt["effective_strengths"] = [step["effective_strength"] for step in steps] if strength else [0.0] * count
        return output, receipt

    if max_new_tokens is not None:
        raise ValueError("The fixed step_gains list defines the output token budget")
    strength = 1.0 if strength is None else strength
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("nominal strength must be finite in [0,1]")
    if (not isinstance(step_gains, (list, tuple)) or not step_gains
            or any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains)):
        raise ValueError("provide a nonempty sequence of finite positive or zero token gains")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("effective strengths must be finite in [0,1]")

    model = native.model
    with (
        _exclusive_generation(model),
        session._observation(seed, {"input_ids": input_ids}),
        torch.no_grad(),
        ExitStack() as schedule,
    ):
        scaled = native.slider.scaled
        if not use_cache:
            scaled = schedule.enter_context(_full_prefix_slider(native.slider, input_ids.device))
        applied = schedule.enter_context(
            _token_strengths(model, scaled, input_ids.shape[1], strengths, full_prefix=not use_cache)
        )
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=len(strengths),
            use_cache=use_cache, num_beams=1, num_return_sequences=1, **sampling,
        )
    sequences = output if isinstance(output, torch.Tensor) else output.sequences
    if sequences.shape != (1, input_ids.shape[1] + len(applied)):
        raise RuntimeError("Glimmer output length differs from its applied strengths")
    return output, list(applied)
