"""Maintenance: freeze the new candidate only, never rewrite earlier cohorts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CORE = "5d68473a098801ee1ab3ebbb12463daef51c6346"
CORE_REPO = ROOT.parents[1] / "core"
PREFIX = "packages/particle-sliders-core/src/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(CORE_REPO), *args])
    paths = git("ls-tree", "-r", "--name-only", CORE, "--", PREFIX + "particle_sliders", PREFIX + "concept_slider_core").decode().splitlines()
    core = {p.removeprefix(PREFIX): sha(git("show", CORE + ":" + p)) for p in paths}
    root = ROOT / "src/glimmer_native"
    product = {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "source-lock.json"}
    lock = dict(schema="glimmer-native-source-lock-v1", core_commit=CORE, core_version="0.12.0",
        core_repository="https://github.com/HyperGAN/particle-sliders.git", core_subdirectory="packages/particle-sliders-core",
        core_sources=core, product_sources=product,
        product_sources_index_sha256=sha(json.dumps(product, sort_keys=True, separators=(",", ":")).encode()),
        pg_commit="801ae1ead8f88b88eca8f5398137e3e1bc01ad36", profile="source-801ae1e-continuous-v1", domain="continuous",
        native_numerical_qualified=False, native_language_qualified=False)
    (root / "source-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(core_files=len(core), product_files=len(product))))


if __name__ == "__main__":
    main()
