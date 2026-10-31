"""Validate installed source bytes before constructing or attaching native objects."""
import hashlib
import json
import importlib.metadata
from pathlib import Path


def verify_sources():
    import particle_sliders
    from particle_sliders.fibo_hndl import verified_builder
    lock = json.loads(Path(__file__).with_name("source-lock.json").read_text())
    dependency = importlib.metadata.distribution("particle-sliders-core")
    origin = json.loads(dependency.read_text("direct_url.json") or "{}")
    if (dependency.version != lock["core_version"] or origin.get("url") != lock["core_repository"]
            or origin.get("vcs_info", {}).get("commit_id") != lock["core_commit"]):
        raise ValueError("installed shared-core version/Git origin differs; use the original cohort")
    if Path(particle_sliders.__file__).resolve() != Path(dependency.locate_file("particle_sliders/__init__.py")).resolve():
        raise ValueError("shared-core import is shadowed")
    actual = {}
    for package in ("particle_sliders", "concept_slider_core"):
        root = Path(dependency.locate_file(package)).resolve()
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                if path.is_symlink():
                    raise ValueError("shared-core source/data symlink is unsupported")
                actual[package + "/" + str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != lock["core_sources"]:
        raise ValueError("installed shared-core source differs; use the original cohort")
    builder, identity = verified_builder()
    return builder, dict(identity, core_commit=lock["core_commit"],
        core_lock_sha256=hashlib.sha256(Path(__file__).with_name("source-lock.json").read_bytes()).hexdigest())
