"""Independent guarded native loader; no fixture/historical runtime imports."""
from contextlib import ExitStack
from pathlib import Path
import json

from .assets import inspect_snapshot
from .contract import audit_loaded_model, require_admission, runtime_sources


def load_local(snapshot, *, expected_gpu_uuid):
    # Every CPU/source/asset refusal precedes CUDA/model allocation.
    from .provenance import verify_sources
    verify_sources()
    assets = inspect_snapshot(snapshot, full_hashes=True)
    runtime = runtime_sources()
    require_admission(expected_gpu_uuid)
    import torch
    from safetensors import safe_open
    from transformers import (AutoConfig, AutoTokenizer, GenerationConfig,
                              BitsAndBytesConfig, MuseGlimmerForConditionalGeneration)
    root = Path(snapshot).resolve(strict=True)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config._name_or_path = str(root)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    files = sorted(set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values()))
    # Same positioned-read algorithm as the historical guarded demo. Sources and
    # implementation dependencies belong to this independent native cohort.
    with ExitStack() as stack:
        state = {}
        for name in files:
            handle = stack.enter_context(safe_open(root / name, framework="pt", device="cpu", backend="pread"))
            for key in handle.keys():
                if key in state:
                    raise ValueError("duplicate checkpoint tensor")
                state[key] = handle.get_slice(key)
        model, info = MuseGlimmerForConditionalGeneration.from_pretrained(
            None, config=config, state_dict=state, output_loading_info=True,
            local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=True,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16))
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("incomplete native weight loading: " + str(info))
    model.generation_config = GenerationConfig.from_pretrained(root, local_files_only=True)
    model.eval().requires_grad_(False)
    # Imported implementation paths must match the already verified distributions.
    if runtime_sources() != runtime:
        raise ValueError("native implementation identity changed during load")
    identity = audit_loaded_model(model, assets, runtime)
    return model, tokenizer, identity
