"""Maintenance: freeze the new candidate only, never rewrite earlier cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
from importlib.metadata import distribution
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CORE = "ec1161f5d74481be3540cb0dfa63d0e8f8b54e90"

PREFIX = "packages/particle-sliders-core/src/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-source", type=Path, required=True)
    args = parser.parse_args()
    core_repo = args.core_source.resolve()
    if subprocess.check_output(["git", "-C", str(core_repo), "rev-parse", "HEAD"], text=True).strip() != CORE:
        raise ValueError("exact approved PG0.11.1 core checkout required")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(core_repo), *args])
    installed = distribution("particle-sliders-core")
    direct = json.loads(installed.read_text("direct_url.json") or "{}")
    if (direct.get("vcs_info", {}).get("commit_id") != CORE
            or direct.get("subdirectory") != "packages/particle-sliders-core"):
        raise ValueError("install the exact core wheel before generating its closure")
    core = {}
    for package in ("particle_sliders", "concept_slider_core"):
        installed_root = Path(installed.locate_file(package))
        for path in installed_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                name = package + "/" + path.relative_to(installed_root).as_posix()
                core[name] = sha(path.read_bytes())
                if core[name] != sha(git("show", CORE + ":" + PREFIX + name)):
                    raise ValueError("installed core source differs from Git: " + name)
    release = json.loads(git("show", CORE + ":" + PREFIX + "particle_sliders/engine_release.json"))
    root = ROOT / "src/glimmer_native"
    product = {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "source-lock.json"}
    lock = dict(schema="glimmer-native-source-lock-engine-v2", core_commit=CORE, core_version="0.15.2",
        core_repository="https://github.com/HyperGAN/particle-sliders.git", core_subdirectory="packages/particle-sliders-core",
        core_sources=core, product_sources=product,
        product_sources_index_sha256=sha(json.dumps(product, sort_keys=True, separators=(",", ":")).encode()),
        pg_commit=release["source_commit"], profile=release["source_profile"], domain="continuous",
        native_numerical_qualified=False, native_language_qualified=False)
    (root / "source-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(core_files=len(core), product_files=len(product))))


if __name__ == "__main__":
    main()
