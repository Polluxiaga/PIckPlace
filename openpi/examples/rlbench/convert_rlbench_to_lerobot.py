"""
Convert RLBench `pick_place_samples.json` to a **LeRobot v2.1** dataset (same pipeline as OpenPI examples).

Each JSON record -> **one episode, one frame** (one LeRobot row):
  - **RGB images**: one camera stream per sample, key from ``--ctx-image-key`` (default ``front_rgb``). Two modes:
    - **stack** (default): ``before_pick`` + ``before_place`` concatenated, keys ``ctx_rgb_00`` …
      (default **15** frames = 10 + 5; set ``--num-ctx-frames`` to match your JSON).
    - **random single from before_pick**: ``--random-single-before-pick`` — sample **one** frame uniformly at
      random from ``before_pick`` only; outputs a **single** image ``ctx_rgb_00``. Training config must use
      ``num_ctx_frames=1``.
  - **state**: 8× zeros (no proprio); OpenPI can ignore or mask at train time.
  - **actions**: 18 = concat(pick_9, place_9), each 9 = (x,y,z) + 6D rotation (Zhou et al.: first two columns of R).
  - **task**: from ``variation_descriptions.pkl`` + ``variation_number.pkl`` under ``episode_dir``, else task folder name.

**Gripper pose source** (``--gripper-pose-source``):

  - ``in_camera`` (default): reads ``pick`` / ``place`` → ``gripper_pose_in_cameras[<name>]`` (7 = xyz + quat in **that
    camera frame**). Set ``--ctx-image-key`` and ``--gripper-pose-camera`` so images and poses match (e.g. ``front_rgb``
    + ``front``, or ``overhead_rgb`` + ``overhead``). If ``--gripper-pose-camera`` is omitted, it defaults to the image
    key with the ``_rgb`` suffix stripped (``front_rgb`` → ``front``).
  - ``world``: legacy ``gripper_pose_world``.

Quaternion order is **xyzw** by default; use ``--quat-order wxyz`` if your data are ``(w,x,y,z)``.

Uses `LeRobotDataset.create(..., use_videos=False)` so frames stay as **PNG** (no v3-only paths).

Usage:
  cd openpi && uv run examples/rlbench/convert_rlbench_to_lerobot.py \\
    --json-path /path/to/pick_place_samples.json

  # Align overhead images with overhead camera-frame poses:
  uv run examples/rlbench/convert_rlbench_to_lerobot.py \\
    --json-path /path/to/pick_place_samples.json \\
    --ctx-image-key overhead_rgb --gripper-pose-camera overhead

  # Legacy world-frame poses:
  uv run examples/rlbench/convert_rlbench_to_lerobot.py \\
    --json-path /path/to/pick_place_samples.json --gripper-pose-source world

  # Single random frame from before_pick only (new dataset; set OpenPI ``num_ctx_frames=1``):
  uv run examples/rlbench/convert_rlbench_to_lerobot.py \\
    --json-path /path/to/pick_place_samples.json \\
    --random-single-before-pick --repo-id your_hf/repo_rand1
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import tyro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_SIZE = (128, 128)
STATE_DIM = 8
ACTION_DIM = 18
DEFAULT_REPO = "local/rlbench_pick_place_ctx15"
DEFAULT_REPO_RANDOM_BEFORE_PICK = "local/rlbench_pick_rand1_before_pick"
DEFAULT_CTX_IMAGE_KEY = "front_rgb"


def _episode_rng(sample_seed: int, episode_dir: Path) -> np.random.Generator:
    """Stable per-episode RNG so re-running conversion picks the same frame for the same episode."""
    h = int(hashlib.md5(str(episode_dir.resolve()).encode(), usedforsecurity=False).hexdigest()[:8], 16)
    seed = (int(sample_seed) ^ h) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _folder_name_to_prompt(name: str) -> str:
    s = name.replace("_", " ").strip()
    return re.sub(r"\s+", " ", s)


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [qx,qy,qz,qw] -> 3x3 rotation matrix."""
    x, y, z, w = [float(t) for t in q]
    n = (x * x + y * y + z * z + w * w) ** 0.5 + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_to_rot6d(q_xyzw: np.ndarray) -> np.ndarray:
    """Zhou et al.: flatten first two columns of R -> 6D (continuous rotation)."""
    r = _quat_to_rotmat_xyzw(q_xyzw)
    return r[:, :2].reshape(6).astype(np.float32)


def pose7_to_dof9(pose7: np.ndarray, *, quat_order: str) -> np.ndarray:
    """
    pose7: [x,y,z, q(4)] in world or camera frame (same encoding).
    quat_order: 'xyzw' or 'wxyz' for the last 4 components.
    """
    pos = np.asarray(pose7[:3], dtype=np.float32)
    q = np.asarray(pose7[3:7], dtype=np.float64)
    if quat_order == "wxyz":
        q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    elif quat_order != "xyzw":
        raise ValueError(quat_order)
    rot6 = _quat_to_rot6d(q)
    return np.concatenate([pos, rot6], axis=0)


def _default_gripper_pose_camera_from_image_key(ctx_image_key: str) -> str:
    """``front_rgb`` -> ``front``; ``overhead_rgb`` -> ``overhead``."""
    if ctx_image_key.endswith("_rgb"):
        return ctx_image_key[: -len("_rgb")]
    return ctx_image_key


def _get_stage_pose7(
    stage: dict | None,
    *,
    gripper_pose_source: str,
    pose_camera: str,
    episode_dir: Path,
    stage_name: str,
) -> np.ndarray | None:
    if not isinstance(stage, dict):
        logger.warning(
            "Missing or invalid %r (expected object, got %s) in %s; skipping sample fields.",
            stage_name,
            type(stage).__name__,
            episode_dir,
        )
        return None
    if gripper_pose_source == "world":
        raw = stage.get("gripper_pose_world")
        if raw is None:
            logger.warning("Missing gripper_pose_world for %s in %s", stage_name, episode_dir)
            return None
        return np.asarray(raw, dtype=np.float32)
    if gripper_pose_source == "in_camera":
        gpic = stage.get("gripper_pose_in_cameras") or {}
        if pose_camera not in gpic:
            logger.warning(
                "Missing gripper_pose_in_cameras[%r] for %s in %s (keys: %s)",
                pose_camera,
                stage_name,
                episode_dir,
                sorted(gpic.keys()),
            )
            return None
        return np.asarray(gpic[pose_camera], dtype=np.float32)
    raise ValueError(f"Unknown gripper_pose_source: {gripper_pose_source!r}")


def load_episode_prompt(episode_dir: Path, fallback_prompt: str) -> str:
    desc_path = episode_dir / "variation_descriptions.pkl"
    idx_path = episode_dir / "variation_number.pkl"
    if not desc_path.is_file() or not idx_path.is_file():
        return fallback_prompt
    try:
        with desc_path.open("rb") as f:
            descs = pickle.load(f)
        with idx_path.open("rb") as f:
            vi = pickle.load(f)
    except Exception as e:
        logger.warning("Could not read variation pickles in %s: %s", episode_dir, e)
        return fallback_prompt

    if isinstance(descs, str):
        return descs
    if isinstance(descs, (list, tuple)) and len(descs) > 0:
        i = int(np.asarray(vi).item()) if vi is not None else 0
        i = max(0, min(i, len(descs) - 1))
        t = descs[i]
        return t if isinstance(t, str) else str(t)
    if isinstance(descs, dict):
        i = int(np.asarray(vi).item()) if vi is not None else 0
        keys = sorted(descs.keys())
        if 0 <= i < len(keys):
            t = descs[keys[i]]
            return t if isinstance(t, str) else str(t)
    return fallback_prompt


def build_features(image_size: tuple[int, int], num_ctx: int) -> dict:
    ft: dict = {
        "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
    }
    h, w = image_size[0], image_size[1]
    for i in range(num_ctx):
        ft[f"ctx_rgb_{i:02d}"] = {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        }
    return ft


def main(
    json_path: str,
    *,
    repo_id: str | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
    num_ctx_frames: int = 15,
    random_single_before_pick: bool = False,
    sample_seed: int = 2026,
    quat_order: str = "xyzw",
    overwrite: bool = True,
    ctx_image_key: str = DEFAULT_CTX_IMAGE_KEY,
    gripper_pose_source: str = "in_camera",
    gripper_pose_camera: str | None = None,
) -> None:
    json_path = Path(json_path).resolve()
    task_folder = json_path.parent.name
    fallback_prompt = _folder_name_to_prompt(task_folder)

    if gripper_pose_source not in ("world", "in_camera"):
        raise ValueError("gripper_pose_source must be 'world' or 'in_camera'")
    pose_camera = gripper_pose_camera or _default_gripper_pose_camera_from_image_key(ctx_image_key)
    logger.info(
        "Images: images[%r]; actions: %s (%r)",
        ctx_image_key,
        gripper_pose_source,
        (
            "gripper_pose_world"
            if gripper_pose_source == "world"
            else f"gripper_pose_in_cameras[{pose_camera!r}]"
        ),
    )

    if random_single_before_pick:
        num_ctx_frames = 1
        if repo_id is None:
            repo_id = DEFAULT_REPO_RANDOM_BEFORE_PICK
        logger.info("random_single_before_pick: num_ctx_frames forced to 1; sampling one frame from before_pick only.")
    elif repo_id is None:
        repo_id = DEFAULT_REPO

    out_root = HF_LEROBOT_HOME / repo_id
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)

    with json_path.open() as f:
        samples = json.load(f)

    features = build_features(image_size, num_ctx_frames)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features=features,
        use_videos=False,
        image_writer_threads=4,
        image_writer_processes=2,
    )

    n_ok = 0
    n_skip_pick_null = 0
    n_skip_place_null = 0
    for rec in samples:
        if not rec.get("ok", False):
            continue
        episode_dir = Path(rec["episode_dir"])
        init_frames = rec.get("initial_state_frames") or {}
        before_pick = init_frames.get("before_pick") or []
        before_place = init_frames.get("before_place") or []

        if random_single_before_pick:
            if len(before_pick) < 1:
                logger.warning("random_single_before_pick: empty before_pick (%s); skipping.", episode_dir)
                continue
            rng = _episode_rng(sample_seed, episode_dir)
            pick_idx = int(rng.integers(0, len(before_pick)))
            frames_order = [before_pick[pick_idx]]
        else:
            frames_order = list(before_pick) + list(before_place)
            if len(frames_order) != num_ctx_frames:
                logger.warning(
                    "Expected %d context frames, got %d (episode %s); skipping.",
                    num_ctx_frames,
                    len(frames_order),
                    episode_dir,
                )
                continue

        ctx_images: list[np.ndarray] = []
        ok = True
        for fr in frames_order:
            paths = fr.get("images") or {}
            if ctx_image_key not in paths:
                logger.warning("Missing images[%r] in %s", ctx_image_key, episode_dir)
                ok = False
                break
            p = Path(paths[ctx_image_key])
            if not p.is_file():
                logger.warning("Missing file %s", p)
                ok = False
                break
            ctx_images.append(_load_rgb(p, image_size))
        if not ok:
            continue

        pick7 = _get_stage_pose7(
            rec["pick"],
            gripper_pose_source=gripper_pose_source,
            pose_camera=pose_camera,
            episode_dir=episode_dir,
            stage_name="pick",
        )
        place7 = _get_stage_pose7(
            rec["place"],
            gripper_pose_source=gripper_pose_source,
            pose_camera=pose_camera,
            episode_dir=episode_dir,
            stage_name="place",
        )
        if pick7 is None or place7 is None:
            bad_pick = not isinstance(rec.get("pick"), dict)
            bad_place = not isinstance(rec.get("place"), dict)
            if bad_pick:
                n_skip_pick_null += 1
            if bad_place:
                n_skip_place_null += 1
            if bad_pick or bad_place:
                reasons = []
                if bad_pick:
                    reasons.append("pick is null or not an object")
                if bad_place:
                    reasons.append("place is null or not an object")
                logger.warning(
                    "Skip JSON row (%s) — episode_dir=%s",
                    "; ".join(reasons),
                    episode_dir.resolve(),
                )
            continue
        p9 = pose7_to_dof9(pick7, quat_order=quat_order)
        pl9 = pose7_to_dof9(place7, quat_order=quat_order)
        actions = np.concatenate([p9, pl9], axis=0).astype(np.float32)

        frame_row: dict = {
            "task": load_episode_prompt(episode_dir, fallback_prompt),
            "state": np.zeros((STATE_DIM,), dtype=np.float32),
            "actions": actions,
        }
        for i, img in enumerate(ctx_images):
            frame_row[f"ctx_rgb_{i:02d}"] = img

        dataset.add_frame(frame_row)
        dataset.save_episode()
        n_ok += 1

    logger.info("Wrote %d episodes (1 frame each) under %s", n_ok, out_root)
    logger.info(
        "Skipped at pose step (after images loaded): pick was null/not-a-dict %d times; "
        "place was null/not-a-dict %d times (one JSON row can count both). "
        "Does not include rows that failed earlier (not ok, frame count, missing files, missing pose keys).",
        n_skip_pick_null,
        n_skip_place_null,
    )
    if random_single_before_pick:
        logger.info(
            "Mode: random_single_before_pick (one ctx_rgb_00 per episode). "
            "Set OpenPI LeRobotRLBenchPickPlaceDataConfig num_ctx_frames=1."
        )
    logger.info("LeRobot codebase_version is v2.1 (default for this lerobot). Task fallback: %r", fallback_prompt)


if __name__ == "__main__":
    tyro.cli(main)
