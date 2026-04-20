#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. Licensed under the Apache License 2.0.
#
# Converts RLBench data to LeRobot format using a pick_place_samples.json manifest
# (same schema as under .../data/train/<task>/pick_place_samples.json).
#
# For each entry with ok=true, the script converts the full episode at episode_dir
# (low_dim_obs.json + per-camera RGB folders), matching convert_rlbench_to_lerobot.py.
# The JSON acts as an episode whitelist / task-specific subset selector.
#
# Usage:
#   PYTHONPATH=src python -m lerobot.scripts.convert_rlbench_pick_place_json_to_lerobot \
#     --pick-place-json /path/to/pick_place_samples.json \
#     --repo-id local/rlbench_place_wine \
#     --root /path/to/output_lerobot_dataset

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.scripts.convert_rlbench_to_lerobot import (
    build_action_vector,
    build_features,
    build_state_vector,
    convert_episode,
    default_depth_cameras,
    default_rgb_cameras,
    load_depth_frame_u8,
    load_json,
    load_rgb_frame,
    lerobot_image_key_from_folder,
    parse_variation_descriptions,
    parse_variation_number,
    sorted_image_paths,
)


def episode_dirs_from_pick_place_json(path: Path) -> list[Path]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("pick_place_samples.json must be a JSON array.")

    out: list[Path] = []
    seen: set[Path] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        if not item.get("ok", False):
            continue
        raw = item.get("episode_dir")
        if not raw:
            continue
        ep = Path(raw).expanduser().resolve()
        if not ep.is_dir():
            logger.warning("Skipping missing episode_dir: %s", ep)
            continue
        if ep in seen:
            continue
        seen.add(ep)
        out.append(ep)

    if not out:
        raise RuntimeError(
            "No episodes to convert: need at least one entry with ok=true and valid episode_dir."
        )
    return out


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    pick_place_path = Path(args.pick_place_json).resolve()
    if not pick_place_path.is_file():
        raise FileNotFoundError(pick_place_path)

    episode_dirs = episode_dirs_from_pick_place_json(pick_place_path)

    rgb_cameras = [x.strip() for x in args.rgb_cameras.split(",") if x.strip()]
    depth_cameras = (
        [x.strip() for x in args.depth_cameras.split(",") if x.strip()] if args.include_depth else []
    )

    ep0 = episode_dirs[0]
    low_dim_path = ep0 / "low_dim_obs.json"
    if not low_dim_path.is_file():
        raise FileNotFoundError(
            f"Missing {low_dim_path}. Export each episode's low_dim_obs.pkl to low_dim_obs.json "
            "(e.g. via pkl_to_json.py) inside the episode folder."
        )

    low_dim_first = load_json(low_dim_path)
    if not isinstance(low_dim_first, list):
        raise ValueError("low_dim_obs.json must be a JSON array of observation dicts.")
    obs0 = low_dim_first[0]
    if "__type__" in obs0:
        obs0 = {k: v for k, v in obs0.items() if k != "__type__"}

    state_dim = int(build_state_vector(obs0).shape[0])
    action_dim = int(build_action_vector(obs0).shape[0])

    rgb_shapes: dict[str, tuple[int, int, int]] = {}
    for folder in rgb_cameras:
        paths = sorted_image_paths(ep0 / folder)
        if not paths:
            raise FileNotFoundError(f"No images under {ep0 / folder} (needed to infer shape).")
        sample = load_rgb_frame(paths[0])
        h, w, c = sample.shape
        rgb_shapes[lerobot_image_key_from_folder(folder)] = (h, w, c)

    depth_shapes: dict[str, tuple[int, int, int]] = {}
    if depth_cameras:
        for folder in depth_cameras:
            paths = sorted_image_paths(ep0 / folder)
            if not paths:
                raise FileNotFoundError(f"No depth images under {ep0 / folder}.")
            sample = load_depth_frame_u8(paths[0])
            h, w, c = sample.shape
            depth_shapes[lerobot_image_key_from_folder(folder)] = (h, w, c)

    features = build_features(state_dim, action_dim, rgb_shapes, depth_shapes if depth_shapes else None)

    out_root = Path(args.root).resolve() if args.root else None
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=out_root,
        robot_type=args.robot_type,
        use_videos=True,
        vcodec=args.vcodec,
    )

    var_desc_path = Path(args.variation_descriptions).resolve() if args.variation_descriptions else None
    var_num_override = args.variation_number

    for ep_dir in episode_dirs:
        p = ep_dir / "low_dim_obs.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")
        low_dim_list = load_json(p)

        descriptions = parse_variation_descriptions(ep_dir, var_desc_path)
        vn = parse_variation_number(ep_dir, var_num_override)
        if vn < 0 or vn >= len(descriptions):
            raise IndexError(
                f"variation_number {vn} out of range for variation_descriptions (len={len(descriptions)})."
            )
        task_text = descriptions[vn]

        convert_episode(
            dataset,
            ep_dir,
            low_dim_list,
            task_text,
            rgb_cameras,
            depth_cameras if depth_cameras else None,
        )

    dataset.finalize()
    print(
        f"Done. Dataset at {dataset.root} repo_id={args.repo_id} "
        f"episodes={len(episode_dirs)} (from {pick_place_path})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pick-place-json",
        type=str,
        required=True,
        help="Path to pick_place_samples.json (array of {ok, episode_dir, ...}).",
    )
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. user/rlbench_task")
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Output dataset directory. Default: $HF_LEROBOT_HOME/<repo-id>",
    )
    p.add_argument("--fps", type=int, default=10, help="RLBench is often 10 Hz.")
    p.add_argument(
        "--rgb-cameras",
        type=str,
        default=",".join(default_rgb_cameras()),
        help="Comma-separated RLBench folder names (e.g. front_rgb,wrist_rgb).",
    )
    p.add_argument(
        "--include-depth",
        action="store_true",
        help="Also load *_depth folders as observation.images.*_depth videos.",
    )
    p.add_argument(
        "--depth-cameras",
        type=str,
        default=",".join(default_depth_cameras()),
        help="Used when --include-depth is set.",
    )
    p.add_argument(
        "--variation-descriptions",
        type=str,
        default=None,
        help="Path to variation_descriptions.json (default: search upward from each episode).",
    )
    p.add_argument(
        "--variation-number",
        type=int,
        default=None,
        help="Override variation index (otherwise read variation_number.json per episode).",
    )
    p.add_argument("--robot-type", type=str, default=None)
    p.add_argument("--vcodec", type=str, default="libsvtav1")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
