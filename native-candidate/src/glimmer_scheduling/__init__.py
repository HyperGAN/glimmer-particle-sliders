"""Token strength schedules for an open native Glimmer RAW session."""

from contextlib import ExitStack, contextmanager
from sys import float_info

import torch


@contextmanager
def _token_strengths(model, scaled, prefix, strengths):
    scope = ExitStack()
    applied = []

    def before(_module, _args, inputs):
        position = len(applied)
        ids = inputs.get("input_ids")
        cache = inputs.get("past_key_values")
        context_length = 0 if cache is None else int(cache.get_seq_length())
        expected_length = 0 if position == 0 else prefix + position - 1
        width = prefix if position == 0 else 1
        if (position >= len(strengths) or ids is None or ids.shape != (1, width)
                or context_length != expected_length):
            raise RuntimeError("Glimmer generation diverged from the declared cached token schedule")
        value = strengths[position]
        scope.enter_context(scaled(value))
        applied.append(value)

    def after(_module, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(model.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(model.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        yield applied


def generate(session, *, input_ids, step_gains, attention_mask=None, strength=1.0, seed=0, **sampling):
    """Generate one native continuation from an already-opened Glimmer RAW adapter."""
    from glimmer_native.native import Native

    session._open()
    native = getattr(session.binding.close, "__self__", None)
    if not isinstance(native, Native) or session.selection != "raw" or native.network is not session.binding.network:
        raise ValueError("an open native Glimmer RAW session is required")
    if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or input_ids.shape[1] == 0 or input_ids.dtype != torch.long):
        raise ValueError("one nonempty tokenized text prompt is required")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("nominal strength must be finite in [0,1]")
    if (not isinstance(step_gains, (list, tuple)) or not step_gains
            or any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains)):
        raise ValueError("provide a nonempty sequence of finite positive or zero token gains")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("effective strengths must be finite in [0,1]")

    model = native.model
    with session._observation(seed, {"input_ids": input_ids}), torch.no_grad():
        with _token_strengths(model, native.slider.scaled, input_ids.shape[1], strengths) as applied:
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=len(strengths),
                use_cache=True, num_beams=1, num_return_sequences=1, **sampling,
            )
    sequences = output if isinstance(output, torch.Tensor) else output.sequences
    if sequences.shape != (1, input_ids.shape[1] + len(applied)):
        raise RuntimeError("Glimmer output length differs from its applied strengths")
    return output, list(applied)
