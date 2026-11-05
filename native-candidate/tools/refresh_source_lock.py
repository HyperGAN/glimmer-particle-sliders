"""Maintenance: freeze the new candidate only, never rewrite earlier cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CORE = "3dd672a7f9b8c169daef90fd63e4bb42b9ddc423"

PREFIX = "packages/particle-sliders-core/src/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-source", type=Path, required=True)
    args = parser.parse_args()
    core_repo = args.core_source.resolve()
    if subprocess.check_output(["git", "-C", str(core_repo), "rev-parse", "HEAD"], text=True).strip() != CORE:
        raise ValueError("exact approved PG0.9.5 core checkout required")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(core_repo), *args])
    paths = git("ls-tree", "-r", "--name-only", CORE, "--", PREFIX + "particle_sliders", PREFIX + "concept_slider_core").decode().splitlines()
    core = {p.removeprefix(PREFIX): sha(git("show", CORE + ":" + p)) for p in paths}
    root = ROOT / "src/glimmer_native"
    product = {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "source-lock.json"}
    lock = dict(schema="glimmer-native-source-lock-pg095-v2", core_commit=CORE, core_version="0.13.0",
        core_repository="https://github.com/HyperGAN/particle-sliders.git", core_subdirectory="packages/particle-sliders-core",
        core_sources=core, product_sources=product,
        product_sources_index_sha256=sha(json.dumps(product, sort_keys=True, separators=(",", ":")).encode()),
        pg_commit="1b8355b184b21bb7d81104ee590813c262f6df0a", profile="release-0.9.5", domain="continuous",
        native_numerical_qualified=False, native_language_qualified=False)
    (root / "source-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(core_files=len(core), product_files=len(product))))


if __name__ == "__main__":
    main()
