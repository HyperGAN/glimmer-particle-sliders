"""CPU asset/shape/budget inspection; never constructs a model."""
import argparse
import json
from pathlib import Path
from .assets import inspect_snapshot

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True)
parser.add_argument("--verify-assets", action="store_true")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
result = inspect_snapshot(args.snapshot, full_hashes=args.verify_assets)
text = json.dumps(result, indent=2) + "\n"
if args.output:
    args.output.write_text(text)
print(text, end="")
