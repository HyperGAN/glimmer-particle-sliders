"""Exact core/product closure; changing either creates a different source cohort."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path


def index(root, prefix="", exclude=()):
    return {prefix + str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and p.suffix != ".pyc" and str(p.relative_to(root)) not in exclude}


def verify_sources():
    import particle_sliders
    from particle_sliders.fibo_hndl import verified_builder
    root = Path(__file__).parent
    lock_path = root / "source-lock.json"
    lock = json.loads(lock_path.read_text())
    dist = metadata.distribution("particle-sliders-core")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if (dist.version != lock["core_version"] or origin.get("url") != lock["core_repository"]
            or origin.get("subdirectory") != lock["core_subdirectory"]
            or origin.get("vcs_info", {}).get("commit_id") != lock["core_commit"]):
        raise ValueError("exact native candidate core Git origin required")
    if Path(particle_sliders.__file__).resolve() != Path(dist.locate_file("particle_sliders/__init__.py")).resolve():
        raise ValueError("shared core import is shadowed")
    actual = {}
    for name in ("particle_sliders", "concept_slider_core"):
        package = Path(dist.locate_file(name)).resolve()
        if any(p.is_symlink() for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            raise ValueError("core closure symlinks are unsupported")
        actual.update(index(package, name + "/"))
    product = index(root, exclude=("source-lock.json",))
    if actual != lock["core_sources"] or product != lock["product_sources"]:
        raise ValueError("native candidate core/product source bytes differ")
    builder, identity = verified_builder()
    return builder, dict(identity, core_commit=lock["core_commit"],
        product_sources_index_sha256=lock["product_sources_index_sha256"],
        source_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest())
