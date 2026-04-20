"""
Merge multiple RLBench ``pick_place_samples.json`` files (each must be a **JSON array** of records)
into one array, then write ``--output``.

Example (your paths):

  cd openpi && uv run examples/rlbench/merge_raw_rlbench.py \\
    /mnt/nas/Starry/sim-datasets/rlbench-18-tasks/data/train/place_wine_at_rack_location/pick_place_samples.json \\
    /mnt/nas/Starry/sim-datasets/rlbench-18-tasks/data/train/place_shape_in_shape_sorter/pick_place_samples.json \\
    --output /mnt/nas/minyangli/place_shape_place_wine.json

By default the output is **pretty-printed** (``indent=2``) for readability in editors. Use ``--compact`` for
minified JSON (smaller file, one long line).

Records are concatenated **in the order of the arguments** (first file, then second, …).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_array(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}, got {type(data).__name__}")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "json_paths",
        nargs="+",
        help="At least two pick_place_samples.json (or compatible) paths.",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output JSON path (parent dirs are created if missing).",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="Pretty-print: spaces per nesting level (default: 2).",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Write minified JSON (no extra whitespace; smaller file, harder to read).",
    )
    args = p.parse_args()
    if len(args.json_paths) < 2:
        p.error("Need at least two json_paths to merge.")

    merged: list = []
    for raw in args.json_paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        chunk = _load_array(path)
        logger.info("%s: %d records", path, len(chunk))
        merged.extend(chunk)

    out: Path = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    indent: int | None = None if args.compact else args.indent
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=indent)
    logger.info("Wrote %d total records to %s", len(merged), out)


if __name__ == "__main__":
    main()
