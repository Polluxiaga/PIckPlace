"""Checkpoint visualization utilities for RLBench pick-place configs."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import sys
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from scripts.evaluation import eval as _checkpoint_eval


FRONT_CAMERA_INTRINSICS = np.array(
    [
        [-175.83856040078922, 0.0, 64.0],
        [0.0, -175.83856040078922, 64.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _rgb_to_uint8_hwc(rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(rgb)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        if img.min() >= -1.01 and img.max() <= 1.01:
            img = (img + 1.0) / 2.0 * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def _rot6d_to_rotation_matrix(rot6: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6, dtype=np.float64).reshape(6)[:3]
    a2 = np.asarray(rot6, dtype=np.float64).reshape(6)[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float64)


def _dof9_to_gripper_wireframe(
    dof9: np.ndarray,
    *,
    half_opening_m: float = 0.022,
    finger_extent_m: float = 0.04,
    handle_length_m: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dof9 = np.asarray(dof9, dtype=np.float64).reshape(-1)
    pos = dof9[:3]
    rot = _rot6d_to_rotation_matrix(dof9[3:9])

    def to_camera_frame(p_local: np.ndarray) -> np.ndarray:
        return pos + rot @ np.asarray(p_local, dtype=np.float64).reshape(3)

    width = float(half_opening_m)
    extent = float(finger_extent_m)
    segments = [
        (to_camera_frame([width, -extent, 0.0]), to_camera_frame([width, extent, 0.0])),
        (to_camera_frame([-width, -extent, 0.0]), to_camera_frame([-width, extent, 0.0])),
        (to_camera_frame([-width, -extent, 0.0]), to_camera_frame([width, -extent, 0.0])),
    ]
    handle = float(handle_length_m)
    if handle > 1e-9:
        segments.append((to_camera_frame([0.0, -extent, 0.0]), to_camera_frame([0.0, -extent - handle, 0.0])))
    return segments


def _project_camera_points_to_uv(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    z = pc[:, 2]
    valid = z > 1e-6
    uv_h = (np.asarray(intrinsics, dtype=np.float64).reshape(3, 3) @ pc.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
    uv = np.stack([u, v], axis=-1)
    valid = valid & np.isfinite(uv).all(axis=1)
    return uv, valid


def _draw_dof9_wireframe_on_bgr(
    img_bgr: np.ndarray,
    dof9: np.ndarray,
    *,
    jaw_color_bgr: tuple[int, int, int],
    tcp_color_bgr: tuple[int, int, int],
    line_thickness: int = 2,
    tcp_dot_radius: int = 2,
) -> None:
    for p0, p1 in _dof9_to_gripper_wireframe(dof9):
        uv, valid = _project_camera_points_to_uv(np.stack([p0, p1], axis=0), FRONT_CAMERA_INTRINSICS)
        if not (bool(valid[0]) and bool(valid[1]) and np.isfinite(uv).all()):
            continue
        a = (int(round(float(uv[0, 0]))), int(round(float(uv[0, 1]))))
        b = (int(round(float(uv[1, 0]))), int(round(float(uv[1, 1]))))
        cv2.line(img_bgr, a, b, jaw_color_bgr, line_thickness, cv2.LINE_AA)

    tcp = np.asarray(dof9, dtype=np.float64).reshape(-1)[:3]
    uv_tcp, valid_tcp = _project_camera_points_to_uv(tcp.reshape(1, 3), FRONT_CAMERA_INTRINSICS)
    if bool(valid_tcp[0]) and np.isfinite(uv_tcp).all():
        c = (int(round(float(uv_tcp[0, 0]))), int(round(float(uv_tcp[0, 1]))))
        cv2.circle(img_bgr, c, tcp_dot_radius, tcp_color_bgr, -1, cv2.LINE_AA)


def _render_gt_pred_overlay(rgb: np.ndarray, gt_dof9: np.ndarray, pred_dof9: np.ndarray) -> np.ndarray:
    img_bgr = cv2.cvtColor(_rgb_to_uint8_hwc(rgb), cv2.COLOR_RGB2BGR)
    _draw_dof9_wireframe_on_bgr(
        img_bgr,
        gt_dof9,
        jaw_color_bgr=(0, 140, 0),
        tcp_color_bgr=(255, 0, 255),
    )
    _draw_dof9_wireframe_on_bgr(
        img_bgr,
        pred_dof9,
        jaw_color_bgr=(0, 0, 255),
        tcp_color_bgr=(255, 255, 0),
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _make_image_grid(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("Cannot create image grid from empty image list.")
    imgs = [_rgb_to_uint8_hwc(img) for img in images]
    if len(imgs) == 1:
        return imgs[0]
    cols = min(2, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    pad = 6
    grid = np.full(
        (rows * h + pad * (rows - 1), cols * w + pad * (cols - 1), 3),
        255,
        dtype=np.uint8,
    )
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        grid[y : y + h, x : x + w] = img
    return grid


def _save_all_case_visualizations(
    vis_dir: epath.Path,
    *,
    eval_repo_id: str,
    batch_pre: dict[str, Any],
    gt_flat: np.ndarray,
    pred_flat: np.ndarray,
    start_index: int,
    action_mse: np.ndarray,
    pick_action_mse: np.ndarray,
    place_action_mse: np.ndarray,
    pick_rot6d_err: np.ndarray,
    place_rot6d_err: np.ndarray,
    pick_trans_err: np.ndarray,
    place_trans_err: np.ndarray,
) -> list[dict[str, Any]]:
    all_vis_dir = vis_dir / "all_cases"
    all_vis_dir.mkdir(parents=True, exist_ok=True)
    first_camera = sorted(batch_pre["image"].keys())[0]
    rows: list[dict[str, Any]] = []
    for i in range(pred_flat.shape[0]):
        case_index = start_index + i
        scene = _checkpoint_eval._scene_name_for_eval_index(eval_repo_id, case_index)
        rgb = np.asarray(batch_pre["image"][first_camera][i])
        pick_overlay = _render_gt_pred_overlay(rgb, gt_flat[i, :9], pred_flat[i, :9])
        place_overlay = _render_gt_pred_overlay(rgb, gt_flat[i, 9:18], pred_flat[i, 9:18])
        overlay_grid = _make_image_grid([pick_overlay, place_overlay])
        rel_path = f"case_{case_index:05d}.png"
        out_path = all_vis_dir / rel_path
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay_grid, cv2.COLOR_RGB2BGR))
        scene_dir = all_vis_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_path = scene_dir / rel_path
        if scene_path.exists():
            scene_path.unlink()
        try:
            os.link(str(out_path), str(scene_path))
        except OSError:
            cv2.imwrite(str(scene_path), cv2.cvtColor(overlay_grid, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "case_index": case_index,
                "eval_repo_id": eval_repo_id,
                "scene": scene,
                "image": rel_path,
                "scene_image": f"{scene}/{rel_path}",
                "action_mse": float(action_mse[i]),
                "pick_action_mse": float(pick_action_mse[i]),
                "place_action_mse": float(place_action_mse[i]),
                "pick_rot6d_l2": float(pick_rot6d_err[i]),
                "place_rot6d_l2": float(place_rot6d_err[i]),
                "pick_trans_l2": float(pick_trans_err[i]),
                "place_trans_l2": float(place_trans_err[i]),
            }
        )
    return rows


def _write_all_case_visualization_index(vis_dir: epath.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_vis_dir = vis_dir / "all_cases"
    metrics_path = all_vis_dir / "metrics.jsonl"
    _checkpoint_eval.write_jsonl(metrics_path, rows)
    scene_rows = _checkpoint_eval.build_scene_metric_rows(rows, eval_repo_id=str(rows[0].get("eval_repo_id", "")))
    _checkpoint_eval.write_jsonl(all_vis_dir / "scene_metrics.jsonl", scene_rows)
    summary_path = all_vis_dir / "README.txt"
    summary_path.write_text(
        "Each image shows pick on the left and place on the right.\n"
        "GT gripper: green/magenta. Predicted gripper: red/cyan.\n"
        "Use metrics.jsonl to sort cases by action_mse, translation, or rotation error.\n"
        "Scene-grouped hardlinks/copies are under all_cases/<scene>/.\n"
    )


def _largest_divisor_at_most(n: int, limit: int) -> int:
    for value in range(min(n, limit), 0, -1):
        if n % value == 0:
            return value
    return 1


def generate_checkpoint_visualizations(
    config: _config.TrainConfig,
    *,
    checkpoint_dir: epath.Path,
    eval_repo_id: str,
    batch_size: int,
    seed: int,
    model: Any | None = None,
    max_decoding_steps: int | None = None,
) -> dict[str, float]:
    params_path = checkpoint_dir / "params"
    assets_root = checkpoint_dir / "assets"
    if model is None and not params_path.is_dir():
        raise FileNotFoundError(f"Missing params directory: {params_path}")

    data_config_train = config.data.create(config.assets_dirs, config.model)
    train_norm_key = data_config_train.asset_id or data_config_train.repo_id
    if not train_norm_key:
        raise ValueError("Could not determine training repo/asset id for norm stats.")
    norm_stats = _checkpoints.load_norm_stats(assets_root, train_norm_key)
    if norm_stats is None:
        raise FileNotFoundError(
            f"Could not load norm stats from {assets_root / train_norm_key}. "
            "Ensure the checkpoint was saved with assets."
        )

    data_config = dataclasses.replace(
        data_config_train,
        repo_id=eval_repo_id,
        norm_stats=norm_stats,
    )
    data_config_pre = dataclasses.replace(
        data_config,
        model_transforms=_checkpoint_eval._pre_tokenize_model_transforms(config.model),
    )
    dataset = _data_loader.create_torch_dataset(
        data_config_pre,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    dataset = _data_loader.transform_dataset(dataset, data_config_pre)
    n = len(dataset)
    local_batch_size = _largest_divisor_at_most(n, batch_size)
    if local_batch_size <= 0:
        raise ValueError("Visualization batch size must be positive.")
    num_batches = n // local_batch_size

    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None,
        shuffle=False,
        num_batches=num_batches,
        num_workers=0,
        seed=seed,
        framework="jax",
    )

    tok = _checkpoint_eval._fast_tokenizer_from_config(
        config.model,
        checkpoint_assets_dir=checkpoint_dir / "assets",
        asset_id=data_config.asset_id,
    )
    tokenize = _transforms.TokenizeFASTInputs(tok)
    extract = _transforms.ExtractFASTActions(
        tok,
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
    )
    if model is None:
        params = _model.restore_params(params_path, dtype=jnp.bfloat16)
        model = config.model.load(params)
    model.eval()

    if max_decoding_steps is None:
        max_decoding_steps = _checkpoint_eval._eval_max_decoding_steps(config)

    rng = jax.random.key(seed)
    total_examples = 0
    rows: list[dict[str, Any]] = []
    vis_dir = checkpoint_dir / "test_vis"
    for batch_idx, batch_pre in enumerate(torch_loader.torch_loader):
        if batch_idx >= num_batches:
            break
        gt_actions_norm = np.asarray(batch_pre["actions"])
        if config.oracle_action_eval:
            pred_actions_norm = _checkpoint_eval._oracle_actions_from_tokenizer(tok, gt_actions_norm)
        else:
            batch_inf = _checkpoint_eval._tokenize_collated_batch(batch_pre, tokenize, include_actions=False)
            obs_inf = _model.Observation.from_dict(batch_inf)
            rng, sample_rng = jax.random.split(rng)
            token_ids = model.sample_actions(
                sample_rng,
                obs_inf,
                max_decoding_steps=max_decoding_steps,
            )
            token_np = np.asarray(jax.device_get(token_ids)).astype(np.int32)
            pred_chunks = [extract({"actions": token_np[i]})["actions"] for i in range(token_np.shape[0])]
            pred_actions_norm = np.stack(pred_chunks, axis=0)
        pred_actions_raw = _checkpoint_eval._unnormalize_actions(
            pred_actions_norm,
            data_config.norm_stats["actions"],
            use_quantiles=data_config.use_quantile_norm,
        )
        gt_actions_raw = _checkpoint_eval._unnormalize_actions(
            gt_actions_norm,
            data_config.norm_stats["actions"],
            use_quantiles=data_config.use_quantile_norm,
        )

        pred_flat = np.asarray(pred_actions_raw[:, 0, :18], dtype=np.float64)
        gt_flat = np.asarray(gt_actions_raw[:, 0, :18], dtype=np.float64)
        pick_rot6d_err = np.linalg.norm(pred_flat[:, 3:9] - gt_flat[:, 3:9], axis=-1)
        place_rot6d_err = np.linalg.norm(pred_flat[:, 12:18] - gt_flat[:, 12:18], axis=-1)
        pick_trans_err = np.linalg.norm(pred_flat[:, :3] - gt_flat[:, :3], axis=-1)
        place_trans_err = np.linalg.norm(pred_flat[:, 9:12] - gt_flat[:, 9:12], axis=-1)
        action_mse = np.mean((pred_flat - gt_flat) ** 2, axis=-1)
        pick_action_mse = np.mean((pred_flat[:, :9] - gt_flat[:, :9]) ** 2, axis=-1)
        place_action_mse = np.mean((pred_flat[:, 9:18] - gt_flat[:, 9:18]) ** 2, axis=-1)
        batch_examples = int(pred_flat.shape[0])
        rows.extend(
                _save_all_case_visualizations(
                    vis_dir,
                    eval_repo_id=eval_repo_id,
                    batch_pre=batch_pre,
                gt_flat=gt_flat,
                pred_flat=pred_flat,
                start_index=total_examples,
                action_mse=action_mse,
                pick_action_mse=pick_action_mse,
                place_action_mse=place_action_mse,
                pick_rot6d_err=pick_rot6d_err,
                place_rot6d_err=place_rot6d_err,
                pick_trans_err=pick_trans_err,
                place_trans_err=place_trans_err,
            )
        )
        total_examples += batch_examples

    _write_all_case_visualization_index(vis_dir, rows)
    return {"num_examples_visualized": float(total_examples)}


def generate_full_test_vis_for_checkpoint(
    config: _config.TrainConfig,
    *,
    step: int,
    model: Any | None = None,
) -> dict[str, float] | None:
    if not _checkpoint_eval.checkpoint_eval_enabled():
        logging.info("Full test visualization skipped: checkpoint eval disabled.")
        return None

    eval_repo_id, _, batch_size, seed = _checkpoint_eval._eval_runtime_options(config)
    checkpoint_dir = config.checkpoint_dir / str(step)
    if not checkpoint_dir.exists():
        logging.warning("Full test visualization skipped: %s does not exist", checkpoint_dir)
        return None

    logging.info("Generating full test visualizations for best checkpoint %s", checkpoint_dir)
    summary = generate_checkpoint_visualizations(
        config,
        checkpoint_dir=checkpoint_dir,
        eval_repo_id=eval_repo_id,
        batch_size=batch_size,
        seed=seed,
        model=model,
        max_decoding_steps=_checkpoint_eval._eval_max_decoding_steps(config),
    )
    logging.info(
        "Saved %.0f full test visualizations for step %d under %s",
        summary["num_examples_visualized"],
        step,
        checkpoint_dir / "test_vis" / "all_cases",
    )
    return summary


def main(
    *,
    config_name: str = "pickplace_all_qbin64",
    checkpoint_dir: pathlib.Path,
    eval_repo_id: str,
    batch_size: int = 16,
    seed: int = 0,
) -> None:
    config = _config.get_config(config_name)
    summary = generate_checkpoint_visualizations(
        config,
        checkpoint_dir=epath.Path(checkpoint_dir).expanduser().resolve(),
        eval_repo_id=eval_repo_id,
        batch_size=batch_size,
        seed=seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
