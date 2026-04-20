#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. Licensed under the Apache License 2.0.
#
# Converts RLBench demo folders (per-frame RGB/depth in subfolders + low_dim_obs JSON)
# into a local LeRobot v3.0 dataset via LeRobotDataset.create / add_frame / save_episode.

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _natural_key(name: str) -> tuple:
    """Sort 'episode12' before 'episode2'; '2.png' after '1.png'."""
    parts = re.split(r"(\d+)", name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def sorted_image_paths(camera_dir: Path) -> list[Path]:
    if not camera_dir.is_dir():
        return []
    paths = list(camera_dir.glob("*.png")) + list(camera_dir.glob("*.jpg")) + list(camera_dir.glob("*.jpeg"))
    return sorted(paths, key=lambda p: _natural_key(p.stem))


def find_upwards(start: Path, filename: str) -> Path | None:
    for d in [start, *start.parents]:
        p = d / filename
        if p.is_file():
            return p
    return None


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_variation_descriptions(episode_dir: Path, explicit: Path | None) -> list[str]:
    if explicit is not None:
        return load_json(explicit)
    p = find_upwards(episode_dir, "variation_descriptions.json")
    if p is None:
        raise FileNotFoundError(
            "Could not find variation_descriptions.json in episode directory or parents. "
            "Pass --variation-descriptions."
        )
    data = load_json(p)
    if not isinstance(data, list):
        raise ValueError("variation_descriptions must be a JSON array of strings.")
    return data


def parse_variation_number(episode_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    p = find_upwards(episode_dir, "variation_number.json")
    if p is None:
        return 0
    data = load_json(p)
    if isinstance(data, dict) and "variation_number" in data:
        return int(data["variation_number"])
    return int(data)


def build_state_vector(obs: dict) -> np.ndarray:
    """Proprio state: 7 joint positions + 2 gripper joints + 1 gripper_open."""
    jp = np.asarray(obs["joint_positions"], dtype=np.float32)
    gj = np.asarray(obs["gripper_joint_positions"], dtype=np.float32)
    go = np.float32(obs["gripper_open"])
    return np.concatenate([jp, gj, go.reshape(1)], axis=0)


def build_action_vector(obs: dict) -> np.ndarray:
    """Action proxy: 7 joint velocities + 1 gripper_open (common BC setup)."""
    jv = np.asarray(obs["joint_velocities"], dtype=np.float32)
    go = np.float32(obs["gripper_open"])
    return np.concatenate([jv, go.reshape(1)], axis=0)


def load_rgb_frame(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return np.array(rgb, dtype=np.uint8)


def load_depth_frame_u8(path: Path) -> np.ndarray:
    """Normalize RLBench depth to uint8 HxWx3 for video features."""
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 2:
        d = arr.astype(np.float32)
        dmax = float(d.max()) if d.size else 1.0
        if dmax > 0:
            d = d / dmax
        u8 = (d * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3].astype(np.uint8)
    return np.stack([arr.astype(np.uint8)] * 3, axis=-1)


def lerobot_image_key_from_folder(folder: str) -> str:
    """Map RLBench folder names to LeRobot feature keys."""
    if folder.endswith("_rgb"):
        return "observation.images." + folder.removesuffix("_rgb")
    if folder.endswith("_depth"):
        base = folder.removesuffix("_depth")
        return f"observation.images.{base}_depth"
    return "observation.images." + folder


def default_rgb_cameras() -> list[str]:
    return [
        "front_rgb",
        "left_shoulder_rgb",
        "overhead_rgb",
        "right_shoulder_rgb",
        "wrist_rgb",
    ]


def default_depth_cameras() -> list[str]:
    return [
        "front_depth",
        "left_shoulder_depth",
        "overhead_depth",
        "right_shoulder_depth",
        "wrist_depth",
    ]


def discover_episode_dirs(episodes_root: Path) -> list[Path]:
    dirs = [p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode")]
    return sorted(dirs, key=lambda p: _natural_key(p.name))


def build_features(
    state_dim: int,
    action_dim: int,
    rgb_shapes: dict[str, tuple[int, int, int]],
    depth_shapes: dict[str, tuple[int, int, int]] | None,
) -> dict:
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    for name, shape in rgb_shapes.items():
        features[name] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    if depth_shapes:
        for name, shape in depth_shapes.items():
            features[name] = {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
    return features


def convert_episode(
    dataset: LeRobotDataset,
    episode_dir: Path,
    low_dim_list: list[dict],
    task_text: str,
    rgb_cameras: list[str],
    depth_cameras: list[str] | None,
) -> None:
    if not low_dim_list:
        raise ValueError(f"Empty low_dim_obs for {episode_dir}")

    rgb_paths: dict[str, list[Path]] = {c: sorted_image_paths(episode_dir / c) for c in rgb_cameras}
    for c in rgb_cameras:
        if not rgb_paths[c]:
            raise FileNotFoundError(f"No images in {episode_dir / c}")

    n = len(low_dim_list)
    ref_cam = rgb_cameras[0]
    if len(rgb_paths[ref_cam]) != n:
        raise ValueError(
            f"Frame count mismatch in {episode_dir}: low_dim_obs has {n} steps "
            f"but {ref_cam} has {len(rgb_paths[ref_cam])} images."
        )
    for c in rgb_cameras[1:]:
        if len(rgb_paths[c]) != n:
            raise ValueError(
                f"Inconsistent RGB length in {episode_dir}: {ref_cam}={len(rgb_paths[ref_cam])} "
                f"{c}={len(rgb_paths[c])} expected {n}."
            )

    depth_paths: dict[str, list[Path]] | None = None
    if depth_cameras:
        depth_paths = {c: sorted_image_paths(episode_dir / c) for c in depth_cameras}
        for c in depth_cameras:
            if len(depth_paths[c]) != n:
                raise ValueError(
                    f"Depth length mismatch for {c}: got {len(depth_paths[c])} expected {n}."
                )

    for t in range(n):
        obs = low_dim_list[t]
        if "__type__" in obs:
            obs = {k: v for k, v in obs.items() if k != "__type__"}

        frame: dict = {
            "task": task_text,
            "observation.state": build_state_vector(obs),
            "action": build_action_vector(obs),
        }
        for folder in rgb_cameras:
            key = lerobot_image_key_from_folder(folder)
            frame[key] = load_rgb_frame(rgb_paths[folder][t])

        if depth_paths:
            for folder_d in depth_cameras:
                key_d = lerobot_image_key_from_folder(folder_d)
                frame[key_d] = load_depth_frame_u8(depth_paths[folder_d][t])

        dataset.add_frame(frame)

    dataset.save_episode()


def run(args: argparse.Namespace) -> None:
    episodes_root = Path(args.episodes_root).resolve()
    if not episodes_root.is_dir():
        raise NotADirectoryError(episodes_root)

    episode_dirs = discover_episode_dirs(episodes_root)
    if args.episode_glob:
        episode_dirs = [p for p in episode_dirs if fnmatch.fnmatch(p.name, args.episode_glob)]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode* directories under {episodes_root}")

    rgb_cameras = [x.strip() for x in args.rgb_cameras.split(",") if x.strip()]
    depth_cameras = (
        [x.strip() for x in args.depth_cameras.split(",") if x.strip()] if args.include_depth else []
    )

    # Probe first episode for shapes and JSON path
    ep0 = episode_dirs[0]
    low_dim_path = ep0 / "low_dim_obs.json"
    if args.low_dim_json:
        low_dim_path = Path(args.low_dim_json).resolve()

    if not low_dim_path.is_file():
        raise FileNotFoundError(
            f"Missing {low_dim_path}. Export RLBench low_dim_obs.pkl to low_dim_obs.json "
            "inside each episode folder, or pass --low-dim-json for a single shared file (one episode only)."
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

    shared_low_dim: list[dict] | None = None
    if args.low_dim_json:
        if len(episode_dirs) > 1:
            raise ValueError(
                "--low-dim-json only supports a single episode; for multiple episodes, "
                "place low_dim_obs.json inside each episode folder."
            )
        shared_low_dim = load_json(Path(args.low_dim_json).resolve())
        if not isinstance(shared_low_dim, list):
            raise ValueError("--low-dim-json must contain a JSON array.")

    for ep_dir in episode_dirs:
        if shared_low_dim is not None:
            low_dim_list = shared_low_dim
        else:
            p = ep_dir / "low_dim_obs.json"
            if not p.is_file():
                raise FileNotFoundError(f"Missing {p}")
            low_dim_list = load_json(p)

        descriptions = parse_variation_descriptions(ep_dir, var_desc_path)
        vn = parse_variation_number(ep_dir, var_num_override)
        if vn < 0 or vn >= len(descriptions):
            raise IndexError(f"variation_number {vn} out of range for variation_descriptions (len={len(descriptions)}).")
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
    print(f"Done. Dataset at {dataset.root} repo_id={args.repo_id} episodes={len(episode_dirs)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episodes-root",
        type=str,
        required=True,
        help="Path to .../all_variations/episodes (contains episode0, episode1, ...).",
    )
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. user/rlbench_close_jar")
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Output dataset directory. Default: $HF_LEROBOT_HOME/<repo-id>",
    )
    p.add_argument("--fps", type=int, default=10, help="RLBench is often 10 Hz; override if needed.")
    p.add_argument(
        "--rgb-cameras",
        type=str,
        default=",".join(default_rgb_cameras()),
        help="Comma-separated RLBench folder names (e.g. front_rgb,wrist_rgb).",
    )
    p.add_argument(
        "--include-depth",
        action="store_true",
        help="Also load *_depth folders as extra observation.images.*_depth videos.",
    )
    p.add_argument(
        "--depth-cameras",
        type=str,
        default=",".join(default_depth_cameras()),
        help="Used when --include-depth is set.",
    )
    p.add_argument(
        "--low-dim-json",
        type=str,
        default=None,
        help="Optional single low_dim_obs.json for one episode only (skip per-episode file).",
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
        help="Override variation index (otherwise read variation_number.json).",
    )
    p.add_argument("--episode-glob", type=str, default=None, help="Optional glob, e.g. 'episode0' or 'episode*'.")
    p.add_argument("--robot-type", type=str, default=None)
    p.add_argument("--vcodec", type=str, default="libsvtav1")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
