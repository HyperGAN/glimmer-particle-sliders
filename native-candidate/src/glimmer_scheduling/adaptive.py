"""Request-local steering on native Muse-Glimmer text generation."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal

import torch

from . import _full_prefix_slider, _require_token_position
from .steering_core import CORE_REVISION, SteeringCore, require_steering_core


@dataclass(frozen=True, kw_only=True)
class ActivationGate:
    """A caller-calibrated shared gate on a native decoder-input residual."""

    core: SteeringCore
    layer: int
    method: str
    parameter: float
    intervention: str
    evidence_ref: str
    direction: torch.Tensor | None = None
    seed: int | None = None
    clabs: object = None
    preserve_norm: bool = False
    projection_bounds: tuple[float, float] | None = None


@dataclass(frozen=True, kw_only=True)
class FittedPromptSteering:
    """A caller-fitted shared S-PSR intervention on one decoder layer."""

    core: SteeringCore
    layer: int
    fitted: torch.nn.Module
    evidence_ref: str
    coordinate: Literal["decoder_input", "decoder_output"] = "decoder_input"


def _policy(steering, layers, width, strength):
    if not isinstance(steering, (ActivationGate, FittedPromptSteering)):
        raise TypeError("Provide ActivationGate or FittedPromptSteering")
    if (type(steering.layer) is not int or not 0 <= steering.layer < len(layers)
            or not isinstance(steering.evidence_ref, str) or not steering.evidence_ref.strip()):
        raise ValueError("Provide a native decoder layer and calibration or fit evidence")
    require_steering_core(steering.core)
    if isinstance(steering, FittedPromptSteering):
        if steering.coordinate not in ("decoder_input", "decoder_output"):
            raise ValueError("Select the fitted Glimmer decoder input or output coordinate")
        fitted = steering.fitted
        if (not isinstance(fitted, steering.core.prompt.PromptSteering)
                or fitted.probe.in_features != width
                or any(p.dtype != torch.float32 or not torch.isfinite(p).all().item()
                       for p in fitted.parameters())):
            raise ValueError("Provide a finite fitted shared FP32 S-PSR module at the native width")
        return None, None, deepcopy(fitted).eval().requires_grad_(False)

    if steering.intervention not in ("residual", "raw_adapter"):
        raise ValueError("Select residual or raw_adapter intervention")
    if type(steering.preserve_norm) is not bool:
        raise ValueError("preserve_norm must be Boolean")
    bounds = steering.projection_bounds
    if bounds is not None:
        low, high = bounds
        if not isfinite(low) or not isfinite(high) or low > high:
            raise ValueError("Projection bounds must be finite and ordered")
    if steering.intervention == "raw_adapter" and (
        steering.layer != 0 or not 0 <= strength <= 1 or steering.preserve_norm or bounds is not None
    ):
        raise ValueError("RAW adapter gates require layer 0, strength in [0,1] and the original edit")
    direction = None
    if steering.method != "random" or steering.intervention == "residual":
        vector = steering.direction
        if not isinstance(vector, torch.Tensor) or not vector.is_floating_point() or vector.shape != (width,):
            raise ValueError("Supply the calibrated direction at the selected native decoder input")
        vector = vector.detach().float()
        length = vector.norm().item()
        if not isfinite(length) or length == 0:
            raise ValueError("The calibrated direction must be finite and nonzero")
        direction = vector / length
    gate = steering.core.adaptive.ActivationGate(steering.method, steering.parameter,
                                                 seed=steering.seed, clabs=steering.clabs)
    return gate, direction, None


@contextmanager
def _adaptive_strengths(native, steering, *, prefix, max_new_tokens, strength, use_cache, device):
    model, slider = native.model, native.slider
    layers = model.model.language_model.layers
    width = model.config.text_config.hidden_size
    gate, direction, fitted = _policy(steering, layers, width, strength)
    intervention = steering.intervention if gate is not None else "residual"
    post_block = fitted is not None and steering.coordinate == "decoder_output"
    layer_key = "decoder_output_layer" if post_block else "decoder_input_layer"
    trace = []
    receipt = dict(core_revision=CORE_REVISION, **{layer_key: steering.layer},
                   evidence_ref=steering.evidence_ref, nominal_strength=strength,
                   method=steering.method if gate is not None else "s_psr", intervention=intervention,
                   use_cache=use_cache, prefill_only_positions=prefix - 1, steps=trace)
    if gate is not None:
        receipt.update(parameter=steering.parameter, seed=steering.seed,
                       clabs=asdict(steering.clabs) if steering.clabs is not None else None,
                       preserve_norm=steering.preserve_norm, projection_bounds=steering.projection_bounds)
    if strength == 0:
        with slider.scaled(0.0):
            yield receipt
        if gate is not None:
            receipt["usage"] = asdict(gate.usage)
        return

    full_prefix = not use_cache
    core = steering.core.adaptive
    expected = None
    evaluation = ExitStack()
    context = prefix - 1

    def before_forward(_model, _args, inputs):
        nonlocal expected
        if expected is not None:
            raise RuntimeError("Glimmer did not visit the selected decoder layer")
        position = len(trace)
        expected = _require_token_position(inputs, prefix, position, max_new_tokens, full_prefix)
        mask = inputs.get("attention_mask")
        if mask is not None and (mask.shape != (1, prefix + position) or mask[0, -1].item() != 1):
            raise RuntimeError("Adaptive Glimmer requires an unmasked final native input token")

    def edit_decoder(hidden):
        nonlocal expected
        if expected is None or not isinstance(hidden, torch.Tensor) or hidden.shape != (1, expected, width):
            raise RuntimeError("Glimmer did not expose one native decoder residual for this evaluation")
        expected = None
        index = len(trace)
        leading = context if full_prefix or index == 0 else 0
        history = trace if full_prefix else ()
        latest = hidden[:, -1:]
        record = dict(output_index=index, input_position=context + index)
        edited = hidden
        if fitted is not None:
            fitted.to(device=hidden.device)
            edited_latest, coefficient = fitted(latest, strength)
            effective = (coefficient * strength).item()
            if not isfinite(effective):
                raise ValueError("Nonfinite Glimmer S-PSR effective strength")
            older = hidden[:, leading:-1]
            if history:
                # Replay the stored coefficients with the frozen direction. Re-reading
                # earlier probe activations would make the schedule depend on replay.
                weights = older.new_tensor([step["effective_strength"] for step in history],
                                           dtype=torch.float32).view(1, -1, 1)
                displaced = older.float() + weights * fitted.direction.weight[0]
                older = torch.where(weights != 0, displaced.to(older.dtype), older)
            edited = torch.cat((hidden[:, :leading], older, edited_latest), dim=1)
            record["coefficient"] = coefficient.item()
        else:
            score = None if steering.method == "random" else core.activation_projection(
                latest, direction, strength=strength).item()
            decision = gate.step(score)
            effective = strength if decision.fired else 0.0
            record.update(alignment=score, probability=decision.probability, fired=decision.fired)
            choices = [False] * leading + [step["fired"] for step in history] + [decision.fired]
            if any(choices):
                if intervention == "raw_adapter":
                    envelope = tuple(strength if on else 0.0 for on in choices)
                    evaluation.enter_context(scaled(envelope))
                else:
                    edited = core.steer_residual(hidden, direction, strength=strength,
                        gate=torch.tensor(choices, device=hidden.device, dtype=torch.bool),
                        preserve_norm=steering.preserve_norm, projection_bounds=steering.projection_bounds)
        if edited is not hidden and not torch.isfinite(edited).all().item():
            raise ValueError("Nonfinite Glimmer residual after steering")
        trace.append(dict(record, effective_strength=effective))
        return edited

    def before_decoder(_layer, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        edited = edit_decoder(hidden)
        if edited is not hidden:
            if args:
                return (edited, *args[1:]), kwargs
            return args, dict(kwargs, hidden_states=edited)
        return None

    def after_decoder(_layer, _args, output):
        hidden = next(iter(output), None) if isinstance(output, tuple) else output
        edited = edit_decoder(hidden)
        if edited is hidden:
            return None
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def after_forward(_model, _args, output):
        evaluation.close()
        if output is not None and expected is not None:
            raise RuntimeError("Glimmer finished a token without visiting the selected layer")

    with slider.scaled(0.0), ExitStack() as hooks:
        if intervention == "raw_adapter":
            scaled = hooks.enter_context(_full_prefix_slider(slider, device))
        hooks.callback(model.register_forward_pre_hook(before_forward, with_kwargs=True).remove)
        layer = layers[steering.layer]
        if post_block:
            hooks.callback(layer.register_forward_hook(after_decoder).remove)
        else:
            hooks.callback(layer.register_forward_pre_hook(before_decoder, with_kwargs=True).remove)
        hooks.callback(model.register_forward_hook(after_forward, always_call=True).remove)
        hooks.callback(evaluation.close)
        yield receipt
    if gate is not None:
        receipt["usage"] = asdict(gate.usage)
