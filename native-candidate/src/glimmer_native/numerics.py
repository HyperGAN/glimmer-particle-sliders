"""Caller-owned strict eager/BF16 policy; inspection never changes math flags."""
from contextlib import contextmanager
import os

PROFILE = "glimmer-native-eager-bf16-strict-v1"
LOADER_SEED = 41


def numerical_policy(model=None):
    import torch
    actual = dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        math_sdp_low_precision_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    expected = dict(deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_benchmark=False, cudnn_deterministic=True, tf32_matmul=False,
        tf32_cudnn=False, float32_matmul_precision="highest",
        fp16_reduced_precision_reduction=False, bf16_reduced_precision_reduction=False,
        flash_sdp=False, memory_efficient_sdp=False, cudnn_sdp=False, math_sdp=True,
        math_sdp_low_precision_reduction=False, cublas_workspace_config=":4096:8")
    if actual != expected:
        changed = sorted(k for k in expected if actual[k] != expected[k])
        raise ValueError("native Glimmer strict numerical policy differs: " + ", ".join(changed))
    overrides = {k: os.environ.get(k) for k in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")}
    if any(v not in (None, "0") for v in overrides.values()):
        raise ValueError("native Glimmer strict numerical policy refuses TF32 environment override")
    if model is not None:
        for config in (model.config, model.config.text_config, model.config.vision_config):
            if getattr(config, "_attn_implementation", None) != "eager":
                raise ValueError("loaded Glimmer attention differs from strict eager policy")
    return dict(profile=PROFILE, actual_flags=actual, environment_overrides=overrides,
                attention="eager for native text and vision", base_precision="bfloat16",
                loader_initialization=dict(seed=LOADER_SEED, scope="CPU default generator and only owned visible cuda:0"))


@contextmanager
def loader_initialization(*, device="cuda:0"):
    """Save/restore external RNG on success or failure; never seed other GPUs."""
    import torch
    if device not in ("cpu", "cuda:0"):
        raise ValueError("loader RNG owns CPU and visible cuda:0 only")
    with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
        torch.random.default_generator.manual_seed(LOADER_SEED)
        if device == "cuda:0":
            torch.cuda.default_generators[0].manual_seed(LOADER_SEED)
        yield


def require_policy_identity(expected, model=None):
    current = numerical_policy(model)
    if current != expected:
        raise ValueError("native Glimmer numerical identity changed")
    return current


def forward_guards(model, expected):
    """Native reads, generation and direct tower reads retain the same policy."""
    handles = []
    def check(_module, _args):
        require_policy_identity(expected, model)
    roots = (model, model.model, model.model.language_model, model.model.vision_tower)
    try:
        for module in roots:
            handles.append(module.register_forward_pre_hook(check))
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        raise
    return handles
