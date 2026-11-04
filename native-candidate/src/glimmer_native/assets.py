"""Pinned local assets and bounded tensor headers; no ML imports or downloads."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from . import LAYERS, MODEL_ID, REVISION, VOCAB, WIDTH

TARGET = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def expected_assets():
    return json.loads(Path(__file__).with_name("assets.json").read_text())


def inspect_snapshot(snapshot, *, full_hashes=False):
    """Metadata inspection is distinct from mandatory full pre-load verification."""
    root = Path(snapshot).resolve(strict=True)
    expected = expected_assets()
    allowed = set(expected["files"]) | {".gitattributes", ".cache"}
    if {p.name for p in root.iterdir()} - allowed:
        raise ValueError("unexpected local loader assets")
    actual = {}
    for name, receipt in expected["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != receipt["bytes"]:
            raise ValueError("asset size/path differs: " + name)
        if full_hashes or not name.endswith(".safetensors"):
            if digest(path) != receipt["sha256"]:
                raise ValueError("asset SHA256 differs: " + name)
        actual[name] = receipt
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", {})
    required = dict(model_type="muse_glimmer_text", hidden_size=WIDTH, num_hidden_layers=LAYERS,
                    num_attention_heads=32, num_key_value_heads=2, head_dim=128, vocab_size=VOCAB,
                    intermediate_size=19968, tie_word_embeddings=False)
    if (config.get("model_type") != "muse_glimmer"
            or config.get("architectures") != ["MuseGlimmerForConditionalGeneration"]
            or any(text.get(k) != v for k, v in required.items())):
        raise ValueError("native Glimmer architecture differs")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map", {})
    files = sorted(set(mapping.values()))
    if not files or any(Path(n).name != n or n not in expected["files"] or not n.endswith(".safetensors") for n in files):
        raise ValueError("unsafe or unadmitted weight index")
    tensors = {}
    for name in files:
        with (root / name).open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= header_size <= 8 * 1024**2:
                raise ValueError("unbounded tensor header")
            header = json.loads(stream.read(header_size))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in tensors or mapping.get(key) != name or value["dtype"] != "BF16":
                raise ValueError("tensor header/index/dtype differs")
            shape = value["shape"]
            if not shape or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("invalid tensor dimensions")
            tensors[key] = dict(shape=shape, dtype=value["dtype"], file=name)
    if set(tensors) != set(mapping):
        raise ValueError("incomplete indexed tensors")
    expected_headers = json.loads(Path(__file__).with_name("tensor-headers.json").read_text())
    if tensors != expected_headers:
        raise ValueError("native tensor shape inventory differs")
    targets = {k: v for k, v in tensors.items() if TARGET.fullmatch(k)}
    if len(targets) != LAYERS * 4:
        raise ValueError("native attention targets incomplete")
    return dict(schema="glimmer-native-assets-v1", model_id=MODEL_ID, revision=REVISION,
                files=actual, tensor_headers_sha256=digest(Path(__file__).with_name("tensor-headers.json")),
                full_weight_hashes_verified=full_hashes, response_shape=[1, WIDTH],
                target_count=len(targets), budget=memory_budget(tensors))


def memory_budget(tensors):
    """An optimistic storage bound; explicit headroom remains unmeasured."""
    total = sum(math.prod(v["shape"]) for v in tensors.values())
    quantized = sum(math.prod(v["shape"]) for k, v in tensors.items()
                    if k.endswith(".weight") and len(v["shape"]) == 2 and k != "lm_head.weight"
                    and not any(s in k for s in ("embed_tokens", "position_embedding_table")))
    return dict(parameters=total, candidate_linear_parameters=quantized,
                unquantized_parameters=total - quantized,
                optimistic_base_storage_bytes=quantized // 2 + (total - quantized) * 2,
                estimate_only=True, measured_peak=None,
                missing=["quantization metadata", "HNDL adapters", "gradients and optimizer",
                         "checkpointed activations", "workspaces", "CUDA context/allocator", "loading peak"],
                initial_process_cap_gib=30, global_free_floor_gib=5,
                minimum_admission_free_gib=35, host_admission_free_gib=23, host_runtime_free_gib=16,
                qualification="No model loaded; this budget does not establish one-card training fit")


def token_rows(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("bounded paired-token JSON required")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"train", "evaluation"}:
        raise ValueError("distinct train/evaluation splits required")
    from .contract import validate_ids
    for rows in value.values():
        if not isinstance(rows, list) or not 2 <= len(rows) <= 32:
            raise ValueError("each split requires2..32 rows")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"neutral", "positive", "history"}:
                raise ValueError("paired token contexts and shared history required")
            if any(not isinstance(row[key], list) for key in ("neutral", "positive", "history")):
                raise ValueError("contexts and history must be token ID lists")
            validate_ids(row["neutral"])
            validate_ids(row["positive"])
            validate_ids(row["neutral"] + row["history"])
            validate_ids(row["positive"] + row["history"])
            if row["neutral"] == row["positive"]:
                raise ValueError("teacher context must differ")
    def compiled(row):
        return tuple(row["neutral"] + row["history"]), tuple(row["positive"] + row["history"])
    train = {compiled(r) for r in value["train"]}
    if any(compiled(r) in train for r in value["evaluation"]):
        raise ValueError("evaluation overlaps training")
    return value, digest(path)
