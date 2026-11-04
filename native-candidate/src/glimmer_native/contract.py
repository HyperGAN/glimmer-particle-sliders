"""Native NF4 identity and resource contract, checked before HNDL hooks."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys

from . import EXECUTION, LAYERS, VOCAB, WIDTH
from .assets import TARGET, digest

VERSIONS = {"torch": "2.9.1", "transformers": "5.22.1", "bitsandbytes": "0.50.2",
            "accelerate": "1.15.0", "safetensors": "0.8.0", "tokenizers": "0.23.3",
            "huggingface-hub": "2.4.0", "numpy": "2.5.4"}
QUANTIZATION = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16",
                    bnb_4bit_quant_storage="bfloat16")


def validate_ids(ids):
    if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 128
            or any(type(v) is not int or not 0 <= v < VOCAB for v in ids)):
        raise ValueError("native token input requires1..128 IDs in the pinned vocabulary")


def runtime_sources():
    """Bind imported package location and all Python/selected binary loader bytes."""
    result = {}
    lock = json.loads(Path(__file__).with_name("runtime-lock.json").read_text())
    for name, expected in VERSIONS.items():
        dist = metadata.distribution(name)
        if dist.version.split("+")[0] != expected:
            raise ValueError("native dependency version differs: " + name)
        package = name.replace("-", "_")
        imported = sys.modules.get(package)
        if imported is not None and Path(imported.__file__).resolve() != Path(dist.locate_file(package + "/__init__.py")).resolve():
            raise ValueError("native implementation import is shadowed: " + name)
        if package == "torch":
            selected = [f for f in dist.files or () if str(f) in
                        {"torch/__init__.py", "torch/lib/libtorch_cuda.so", "torch/lib/libc10_cuda.so"}]
        else:
            selected = [f for f in dist.files or () if str(f).startswith((package + "/", package + ".libs/"))
                        and (str(f).endswith(".py") or (name in {"safetensors", "tokenizers", "numpy"} and str(f).endswith(".so"))
                             or str(f).endswith("/libbitsandbytes_cuda128.so"))]
        if not selected:
            raise ValueError("native implementation sources absent: " + name)
        files = {str(f): digest(dist.locate_file(f)) for f in selected}
        if files != lock[name]["sources"]:
            raise ValueError("native implementation source bytes differ: " + name)
        result[name] = dict(version=dist.version, sources=files,
                            sources_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    import torch
    if torch.__version__ != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise ValueError("this candidate explicitly requires Torch2.9.1+cu128")
    return result


def require_admission(expected_gpu_uuid):
    """Only the separately admitted parent may enter the model loader."""
    if (not isinstance(expected_gpu_uuid, str) or not re.fullmatch(r"GPU-[0-9a-f-]{36}", expected_gpu_uuid)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid
            or os.environ.get("GLIMMER_NATIVE_GUARDIAN") != str(os.getppid())):
        raise ValueError("native NF4 requires its admitted guardian and one exact visible GPU UUID")
    if os.environ.get("GLIMMER_NATIVE_PROCESS_CAP_GIB") != "30":
        raise ValueError("native NF4 requires the explicit30GiB process guard")
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("one visible CUDA device required")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual != expected_gpu_uuid:
        raise ValueError("visible CUDA UUID differs")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 35 * 1024**3:
        raise MemoryError("native admission requires30GiB cap plus5GiB free reserve")
    available = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")), 0)
    if available < 23 * 1024**3:
        raise MemoryError("native CPU loading admission requires23GiB available RAM")
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
            or not torch.are_deterministic_algorithms_enabled()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("native candidate requires deterministic kernels and TF32 disabled")


def tensor_identity(value):
    """Hash packed bytes in bounded chunks, without dequantization or a full CPU copy."""
    import torch
    tensor = value.detach().contiguous()
    flat = tensor.reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    for start in range(0, flat.numel(), 8 * 1024**2):
        h.update(flat[start:start + 8 * 1024**2].cpu().numpy().tobytes())
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device), sha256=h.hexdigest())


def validate_quantization(description):
    expected = dict(quant_type="nf4", blocksize=64, compress_statistics=True,
                    quant_storage="torch.bfloat16", compute_dtype="torch.bfloat16",
                    bnb_quantized=True, nested=True, nested_blocksize=256, original_dtype="torch.bfloat16",
                    state_quant_type="nf4", state_blocksize=64)
    if any(description.get(k) != v for k, v in expected.items()):
        raise ValueError("loaded NF4 quantization settings differ")


def audit_loaded_model(model, assets, runtime):
    """Inspect actual type/device/settings/packed bytes before installing any hooks."""
    import torch
    from bitsandbytes.nn import Linear4bit, Params4bit
    from transformers import MuseGlimmerForConditionalGeneration
    if type(model) is not MuseGlimmerForConditionalGeneration or not assets["full_weight_hashes_verified"]:
        raise ValueError("exact verified native model and original assets required")
    if (model.config.text_config.hidden_size != WIDTH or model.config.text_config.num_hidden_layers != LAYERS
            or getattr(model.config, "_attn_implementation", None) != "eager"):
        raise ValueError("loaded native architecture/attention differs")
    headers = json.loads(Path(__file__).with_name("tensor-headers.json").read_text())
    projections, quantized = {}, {}
    for name, module in model.named_modules():
        if (getattr(module, "_hf_hook", None) is not None or getattr(module, "gradient_checkpointing", False)
                or getattr(module, "_pg09_scale_checkpoint", False)
                or "forward" in vars(module) or getattr(module, "_compiled_call_impl", None) is not None
                or any(getattr(module, key, None) for key in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))):
            raise ValueError("preexisting native hook/override/offload is unsupported: " + name)
        if isinstance(module, Linear4bit):
            weight = module.weight
            if type(module) is not Linear4bit or type(weight) is not Params4bit or weight.requires_grad:
                raise ValueError("exact frozen Linear4bit/Params4bit required")
            state = weight.quant_state
            if state is None or state.state2 is None:
                raise ValueError("missing actual double-quantized weight state")
            description = dict(quant_type=weight.quant_type, blocksize=weight.blocksize,
                compress_statistics=weight.compress_statistics, quant_storage=str(weight.quant_storage),
                compute_dtype=str(module.compute_dtype), bnb_quantized=weight.bnb_quantized,
                nested=state.nested, nested_blocksize=state.state2.blocksize, original_dtype=str(state.dtype),
                state_quant_type=state.quant_type, state_blocksize=state.blocksize)
            validate_quantization(description)
            if weight.dtype != torch.bfloat16 or any(t.device != torch.device("cuda:0") for t in
                    (state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code)):
                raise ValueError("actual packed weight/storage or quantization device differs")
            if tuple(state.shape) != (module.out_features, module.in_features):
                raise ValueError("quantized original dimensions differ")
            quantized[name] = dict(settings=description, packed_weight=tensor_identity(weight),
                state={k: tensor_identity(v) for k, v in state.as_dict(packed=True).items()})
        match = TARGET.fullmatch(name + ".weight")
        if match:
            if type(module) is not Linear4bit:
                raise ValueError("all native attention targets must actually be NF4")
            expected = headers[name + ".weight"]["shape"]
            if expected != [module.out_features, module.in_features]:
                raise ValueError("native attention projection dimensions differ")
            projections[name] = expected
    if len(projections) != LAYERS * 4:
        raise ValueError("all208 native projections required")
    expected_quantized = {k.removesuffix(".weight") for k, v in headers.items()
        if k.endswith(".weight") and len(v["shape"]) == 2 and k != "lm_head.weight"
        and not any(s in k for s in ("embed_tokens", "position_embedding_table"))}
    if set(quantized) != expected_quantized:
        raise ValueError("actual quantized module coverage differs from this cohort")
    for name, parameter in model.named_parameters():
        if parameter.device != torch.device("cuda:0") or parameter.requires_grad:
            raise ValueError("entire frozen host must reside on the one admitted CUDA device")
        if type(parameter) is not Params4bit and parameter.dtype != torch.bfloat16:
            raise ValueError("unquantized native base must be BF16")
    buffers = {n: tensor_identity(t) for n, t in model.named_buffers()}
    if any(t.device != torch.device("cuda:0") for _, t in model.named_buffers()):
        raise ValueError("offloaded native buffers are unsupported")
    # state_dict includes packed quantization state plus all unquantized host bytes.
    state = {n: tensor_identity(t) for n, t in model.state_dict().items()}
    return dict(execution_profile=EXECUTION, assets=assets, runtime=runtime,
                projections=projections, quantization=dict(QUANTIZATION),
                quantized_modules=quantized, buffers=buffers,
                packed_host_state_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                actual_host_state=state,
                numerical=dict(dtype="bfloat16", attention="eager", tf32=False,
                    deterministic=torch.are_deterministic_algorithms_enabled(), cuda="12.8",
                    device_name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
