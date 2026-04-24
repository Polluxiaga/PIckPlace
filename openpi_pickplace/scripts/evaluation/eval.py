"""Checkpoint evaluation utilities for RLBench pick-place configs."""

from __future__ import annotations

import csv
import dataclasses
import functools
import json
import logging
import os
import pathlib
import sys
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import cv2
import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


FRONT_CAMERA_INTRINSICS = np.array(
    [
        [-175.83856040078922, 0.0, 64.0],
        [0.0, -175.83856040078922, 64.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _write_local_sweep_metrics(
    metrics_path: epath.Path,
    latest_metrics_path: epath.Path,
    csv_path: epath.Path,
    *,
    payload: dict[str, Any],
) -> None:
    with metrics_path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    latest_metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)


def _loss_token_labels(action_dim: int, action_horizon: int) -> list[str]:
    labels: list[str] = []
    n_action_tokens = action_dim * action_horizon
    for i in range(n_action_tokens):
        dim = i % action_dim
        horizon = i // action_dim
        if action_horizon == 1 and action_dim == 18:
            prefix = "pick" if dim < 9 else "place"
            labels.append(f"{prefix}_{dim % 9:02d}")
        elif action_horizon == 1:
            labels.append(f"action_{dim:02d}")
        else:
            labels.append(f"h{horizon:02d}_action_{dim:02d}")
    labels.extend(["pipe", "eos"])
    return labels


def _token_ce_metrics(prefix: str, ce_by_loss_pos: np.ndarray, *, action_dim: int, action_horizon: int) -> dict[str, float]:
    values = np.asarray(ce_by_loss_pos, dtype=np.float64)
    labels = _loss_token_labels(action_dim, action_horizon)
    base = f"{prefix}/token_ce" if prefix else "token_ce"
    ce_base = f"{prefix}/ce" if prefix else "ce"
    token_mean_base = f"{prefix}/token_ce_mean" if prefix else "token_ce_mean"
    n_action_tokens = min(action_dim * action_horizon, len(values))
    metrics = {
        f"{base}/{label}": float(values[i])
        for i, label in enumerate(labels[:n_action_tokens])
    }
    if n_action_tokens:
        metrics[f"{base}/action_mean"] = float(np.mean(values[:n_action_tokens]))
    if action_horizon == 1 and action_dim == 18 and len(values) >= 18:
        pick_mean = float(np.mean(values[:9]))
        place_mean = float(np.mean(values[9:18]))
        metrics[f"{base}/pick_mean"] = pick_mean
        metrics[f"{base}/place_mean"] = place_mean
        metrics[f"{ce_base}/pick"] = pick_mean
        metrics[f"{ce_base}/place"] = place_mean
        metrics[f"{token_mean_base}/pick"] = pick_mean
        metrics[f"{token_mean_base}/place"] = place_mean
    return metrics


def _token_ce_sums_by_loss_position(
    token_nll: np.ndarray,
    loss_mask: np.ndarray,
    *,
    max_loss_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    token_nll = np.asarray(token_nll, dtype=np.float64)
    loss_mask = np.asarray(loss_mask, dtype=bool)
    loss_pos = np.cumsum(loss_mask.astype(np.int32), axis=-1) - 1
    valid = loss_mask & (loss_pos >= 0) & (loss_pos < max_loss_tokens)
    sums = np.zeros(max_loss_tokens, dtype=np.float64)
    counts = np.zeros(max_loss_tokens, dtype=np.float64)
    for pos in range(max_loss_tokens):
        m = valid & (loss_pos == pos)
        sums[pos] = float(np.sum(token_nll[m]))
        counts[pos] = float(np.sum(m))
    return sums, counts


def _eval_wandb_payload(payload: dict[str, Any], *, prefix: str, checkpoint_step_key: str) -> dict[str, float]:
    keep = (
        f"{prefix}/token_ce_avg",
        f"{prefix}/action_mse_avg",
        f"{prefix}/pick_rot6d_l2",
        f"{prefix}/place_rot6d_l2",
        f"{prefix}/pick_trans_l2",
        f"{prefix}/place_trans_l2",
    )
    out = {checkpoint_step_key: float(payload[checkpoint_step_key])}
    out.update({key: float(payload[key]) for key in keep if key in payload})
    return out


def _token_ce_values_from_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        label: float(metrics[f"token_ce/{label}"])
        for label in _loss_token_labels(18, 1)[:18]
        if f"token_ce/{label}" in metrics
    }


def _token_ce_values_from_payload(payload: dict[str, Any], *, metric_prefix: str) -> dict[str, float]:
    return {
        label: float(payload[f"{metric_prefix}/token_ce/{label}"])
        for label in _loss_token_labels(18, 1)[:18]
        if f"{metric_prefix}/token_ce/{label}" in payload
    }


def _save_token_ce_chart(
    vis_dir: epath.Path,
    *,
    step: int,
    history: dict[int, dict[str, float]],
) -> epath.Path | None:
    if not history:
        return None
    steps = sorted(history)
    if not any(history[s] for s in steps):
        return None

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"step_{step:06d}_token_ce.png"
    colors = plt.get_cmap("tab10").colors[:9]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    for dof_idx in range(9):
        color = colors[dof_idx]
        pick_label = f"pick_{dof_idx:02d}"
        place_label = f"place_{dof_idx:02d}"
        pick_y = [history[s].get(pick_label, np.nan) for s in steps]
        place_y = [history[s].get(place_label, np.nan) for s in steps]
        ax.plot(steps, pick_y, color=color, linestyle="-", marker="o", linewidth=1.8, markersize=3.5)
        ax.plot(steps, place_y, color=color, linestyle="--", marker="s", linewidth=1.8, markersize=3.5)

    from matplotlib.lines import Line2D

    dof_handles = [
        Line2D([0], [0], color=colors[i], lw=2, label=f"dof_{i:02d}")
        for i in range(9)
    ]
    stage_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="pick"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="place"),
    ]
    legend1 = ax.legend(handles=dof_handles, title="dimension", ncol=3, loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=stage_handles, title="stage", loc="upper right", fontsize=8)
    ax.set_title("Per-token CE by checkpoint")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("cross entropy")
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return out_path


def _fast_tokenizer_from_config(
    model_config: _model.BaseModelConfig,
    *,
    checkpoint_assets_dir: epath.Path | str | None = None,
    asset_id: str | None = None,
) -> _tokenizer.FASTTokenizer:
    tokenizer_cls = (
        _tokenizer.FASTTokenizer
        if model_config.fast_model_tokenizer is None
        else model_config.fast_model_tokenizer
    )
    kwargs = dict(model_config.fast_model_tokenizer_kwargs or {})
    for key, fname in (
        ("bin_edges_path", "bin_edges.npy"),
        ("codebook_path", "codebook.npy"),
        ("vq_params_path", "vq_params.npz"),
    ):
        if key in kwargs and not epath.Path(kwargs[key]).exists():
            if checkpoint_assets_dir and asset_id:
                alt = epath.Path(checkpoint_assets_dir) / asset_id / fname
                if alt.exists():
                    logging.info("%s fallback: using checkpoint assets %s", key, alt)
                    kwargs[key] = str(alt)
    return tokenizer_cls(model_config.max_token_len, **kwargs)


def _pre_tokenize_model_transforms(model_config: _model.BaseModelConfig) -> _transforms.Group:
    return _transforms.Group(
        inputs=[
            _transforms.InjectDefaultPrompt(None),
            _transforms.ResizeImages(224, 224),
        ],
    )


def _batch_copy(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = _batch_copy(value)
        elif isinstance(value, np.ndarray):
            out[key] = np.array(value, copy=True)
        else:
            out[key] = value.copy() if hasattr(value, "copy") else value
    return out


def _take_batch_index(x, *, index: int, bsz: int):
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == bsz:
        return np.asarray(x[index])
    return x


def _tokenize_collated_batch(
    batch_pre: dict,
    tokenize: _transforms.TokenizeFASTInputs,
    *,
    include_actions: bool,
) -> dict:
    bsz = int(batch_pre["state"].shape[0])
    outs = []
    for i in range(bsz):
        sample = jax.tree.map(functools.partial(_take_batch_index, index=i, bsz=bsz), batch_pre)
        if not include_actions:
            sample.pop("actions", None)
        prompt = sample.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            sample["prompt"] = str(prompt)
        outs.append(tokenize(_batch_copy(sample)))
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *outs)


def _unnormalize_actions(x: np.ndarray, stats: Any, *, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float64)
        q99 = np.asarray(stats.q99, dtype=np.float64)
        dim = q01.shape[-1]
        head = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        mean = np.asarray(stats.mean, dtype=np.float64)
        std = np.asarray(stats.std, dtype=np.float64)
        dim = mean.shape[-1]
        head = x[..., :dim] * (std + 1e-6) + mean
    if dim < x.shape[-1]:
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return head


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


def _build_overlay_grid(
    batch_pre: dict[str, Any],
    gt_flat: np.ndarray,
    pred_flat: np.ndarray,
    *,
    max_samples: int = 4,
) -> np.ndarray:
    overlays = []
    first_camera = sorted(batch_pre["image"].keys())[0]
    for i in range(min(max_samples, pred_flat.shape[0])):
        rgb = np.asarray(batch_pre["image"][first_camera][i])
        overlays.append(_render_gt_pred_overlay(rgb, gt_flat[i, :9], pred_flat[i, :9]))
        overlays.append(_render_gt_pred_overlay(rgb, gt_flat[i, 9:18], pred_flat[i, 9:18]))
    return _make_image_grid(overlays)


def _save_overlay_grid(
    vis_dir: epath.Path,
    *,
    step: int,
    overlay_grid: np.ndarray,
) -> epath.Path:
    vis_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = vis_dir / f"step_{step:06d}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_grid, cv2.COLOR_RGB2BGR))
    return overlay_path


def list_checkpoint_steps(checkpoint_root: epath.Path) -> list[int]:
    steps = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if not (child / "params").is_dir():
            continue
        steps.append(int(child.name))
    return sorted(steps)


def evaluate_checkpoint(
    config: _config.TrainConfig,
    *,
    checkpoint_dir: epath.Path,
    eval_repo_id: str,
    batch_size: int,
    seed: int,
    model: Any | None = None,
    max_decoding_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
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
        model_transforms=_pre_tokenize_model_transforms(config.model),
    )

    dataset = _data_loader.create_torch_dataset(
        data_config_pre,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    dataset = _data_loader.transform_dataset(dataset, data_config_pre)
    n = len(dataset)
    if n < batch_size:
        batch_size = n
    if batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive.")

    local_batch_size = batch_size // jax.process_count()
    if local_batch_size <= 0:
        raise ValueError(f"Invalid batch size {batch_size} for process_count={jax.process_count()}.")

    num_batches = n // local_batch_size
    if num_batches == 0:
        raise ValueError("No full batches; reduce batch_size.")

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

    tok = _fast_tokenizer_from_config(
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

    rng = jax.random.key(seed)
    total_examples = 0
    total_ce_sum = 0.0
    total_action_mse_sum = 0.0
    total_pick_action_mse_sum = 0.0
    total_place_action_mse_sum = 0.0
    total_pick_rot6d_l2_sum = 0.0
    total_place_rot6d_l2_sum = 0.0
    total_pick_trans_l2_sum = 0.0
    total_place_trans_l2_sum = 0.0
    max_loss_tokens = config.model.action_horizon * config.model.action_dim + 2
    if max_decoding_steps is None:
        max_decoding_steps = max_loss_tokens
    total_token_ce_sums = np.zeros(max_loss_tokens, dtype=np.float64)
    total_token_ce_counts = np.zeros(max_loss_tokens, dtype=np.float64)
    media: dict[str, Any] = {}
    media_written = False

    for batch_idx, batch_pre in enumerate(torch_loader.torch_loader):
        if batch_idx >= num_batches:
            break

        batch_ce = _tokenize_collated_batch(batch_pre, tokenize, include_actions=True)
        observation = _model.Observation.from_dict(batch_ce)
        actions = batch_ce["actions"]
        rng, step_rng = jax.random.split(rng)
        token_nll, loss_mask = model.compute_loss_per_token(step_rng, observation, actions, train=False)
        token_nll = np.asarray(jax.device_get(token_nll))
        loss_mask = np.asarray(jax.device_get(loss_mask))
        per_example_ce = np.sum(token_nll * loss_mask, axis=-1) / np.clip(np.sum(loss_mask, axis=-1), 1, None)
        token_ce_sums, token_ce_counts = _token_ce_sums_by_loss_position(
            token_nll,
            loss_mask,
            max_loss_tokens=max_loss_tokens,
        )
        total_token_ce_sums += token_ce_sums
        total_token_ce_counts += token_ce_counts

        batch_inf = _tokenize_collated_batch(batch_pre, tokenize, include_actions=False)
        obs_inf = _model.Observation.from_dict(batch_inf)
        rng, sample_rng = jax.random.split(rng)
        token_ids = model.sample_actions(sample_rng, obs_inf, max_decoding_steps=max_decoding_steps)
        token_np = np.asarray(jax.device_get(token_ids)).astype(np.int32)

        pred_chunks = [extract({"actions": token_np[i]})["actions"] for i in range(token_np.shape[0])]
        pred_actions_norm = np.stack(pred_chunks, axis=0)
        gt_actions_norm = np.asarray(batch_pre["actions"])

        pred_actions_raw = _unnormalize_actions(
            pred_actions_norm,
            data_config.norm_stats["actions"],
            use_quantiles=data_config.use_quantile_norm,
        )
        gt_actions_raw = _unnormalize_actions(
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
        total_examples += batch_examples
        total_ce_sum += float(np.sum(per_example_ce))
        total_action_mse_sum += float(np.sum(action_mse))
        total_pick_action_mse_sum += float(np.sum(pick_action_mse))
        total_place_action_mse_sum += float(np.sum(place_action_mse))
        total_pick_rot6d_l2_sum += float(np.sum(pick_rot6d_err))
        total_place_rot6d_l2_sum += float(np.sum(place_rot6d_err))
        total_pick_trans_l2_sum += float(np.sum(pick_trans_err))
        total_place_trans_l2_sum += float(np.sum(place_trans_err))

        if not media_written:
            checkpoint_step = int(checkpoint_dir.name)
            overlay_grid = _build_overlay_grid(batch_pre, gt_flat, pred_flat)
            _save_overlay_grid(
                checkpoint_dir / "test_vis",
                step=checkpoint_step,
                overlay_grid=overlay_grid,
            )
            media = {
                "test_vis/overlay": wandb.Image(
                    overlay_grid,
                    caption=f"checkpoint_step={checkpoint_step}; GT: green/magenta, Pred: red/cyan",
                ),
            }
            media_written = True

    if total_examples == 0:
        raise ValueError("No evaluation examples were processed.")

    token_ce = total_token_ce_sums / np.clip(total_token_ce_counts, 1, None)
    metrics = {
        "mean_cross_entropy": total_ce_sum / total_examples,
        "action_mse": total_action_mse_sum / total_examples,
        "action_mse_avg": total_action_mse_sum / total_examples,
        "pick_action_mse": total_pick_action_mse_sum / total_examples,
        "place_action_mse": total_place_action_mse_sum / total_examples,
        "pick_rot6d_l2": total_pick_rot6d_l2_sum / total_examples,
        "place_rot6d_l2": total_place_rot6d_l2_sum / total_examples,
        "pick_trans_l2": total_pick_trans_l2_sum / total_examples,
        "place_trans_l2": total_place_trans_l2_sum / total_examples,
        "action_mse/pick": total_pick_action_mse_sum / total_examples,
        "action_mse/place": total_place_action_mse_sum / total_examples,
        "rot6d_l2/pick": total_pick_rot6d_l2_sum / total_examples,
        "rot6d_l2/place": total_place_rot6d_l2_sum / total_examples,
        "trans_l2/pick": total_pick_trans_l2_sum / total_examples,
        "trans_l2/place": total_place_trans_l2_sum / total_examples,
        "num_examples_used": float(total_examples),
    }
    metrics.update(
        _token_ce_metrics(
            "",
            token_ce,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
        )
    )
    if "token_ce/action_mean" in metrics:
        metrics["token_ce_avg"] = metrics["token_ce/action_mean"]
    return metrics, media


def checkpoint_eval_enabled() -> bool:
    return os.environ.get("POST_TRAIN_EVAL", "1").lower() not in {"0", "false", "no", "off"}


def _eval_runtime_options(config: _config.TrainConfig) -> tuple[str, str, int, int]:
    eval_repo_id = os.environ.get("EVAL_REPO_ID", "minyangli/pick_place_all_test")
    metric_prefix = os.environ.get("EVAL_METRIC_PREFIX", "test")
    batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
    seed = int(os.environ.get("EVAL_SEED", str(config.seed)))
    return eval_repo_id, metric_prefix, batch_size, seed


def _eval_max_decoding_steps(config: _config.TrainConfig) -> int:
    default = config.model.action_horizon * config.model.action_dim + 2
    return int(os.environ.get("EVAL_MAX_DECODING_STEPS", str(default)))


def _sweep_paths(checkpoint_root: epath.Path, metric_prefix: str) -> tuple[epath.Path, epath.Path, epath.Path]:
    return (
        checkpoint_root / f"{metric_prefix}_sweep.jsonl",
        checkpoint_root / f"{metric_prefix}_sweep.latest.json",
        checkpoint_root / f"{metric_prefix}_sweep.csv",
    )


def reset_eval_sweep_files(config: _config.TrainConfig) -> None:
    _, metric_prefix, _, _ = _eval_runtime_options(config)
    metrics_path, latest_metrics_path, csv_path = _sweep_paths(config.checkpoint_dir, metric_prefix)
    for path in (metrics_path, latest_metrics_path, csv_path):
        if path.exists():
            path.unlink()


def load_eval_token_ce_history(config: _config.TrainConfig) -> dict[int, dict[str, float]]:
    _, metric_prefix, _, _ = _eval_runtime_options(config)
    metrics_path, _, _ = _sweep_paths(config.checkpoint_dir, metric_prefix)
    history: dict[int, dict[str, float]] = {}
    if not metrics_path.exists():
        return history

    with metrics_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed eval sweep line in %s", metrics_path)
                continue
            step = payload.get("step")
            if step is None:
                continue
            values = _token_ce_values_from_payload(payload, metric_prefix=metric_prefix)
            if values:
                history[int(step)] = values
    return history


def _define_eval_wandb_metrics(metric_prefix: str) -> str:
    checkpoint_step_key = f"{metric_prefix}/checkpoint_step"
    wandb.define_metric(f"{metric_prefix}/*", step_metric=checkpoint_step_key, overwrite=True)
    wandb.define_metric("test_vis/*", step_metric=checkpoint_step_key, overwrite=True)
    wandb.define_metric(checkpoint_step_key, hidden=True, overwrite=True)
    return checkpoint_step_key


def evaluate_and_log_checkpoint(
    config: _config.TrainConfig,
    *,
    step: int,
    token_ce_history: dict[int, dict[str, float]] | None = None,
    model: Any | None = None,
) -> dict[str, float] | None:
    if not checkpoint_eval_enabled():
        logging.info("Checkpoint eval disabled.")
        return None

    eval_repo_id, metric_prefix, batch_size, seed = _eval_runtime_options(config)
    checkpoint_root = config.checkpoint_dir
    checkpoint_dir = checkpoint_root / str(step)
    if not checkpoint_dir.exists():
        logging.warning("Checkpoint eval skipped: %s does not exist", checkpoint_dir)
        return None

    metrics_path, latest_metrics_path, csv_path = _sweep_paths(checkpoint_root, metric_prefix)
    checkpoint_step_key = _define_eval_wandb_metrics(metric_prefix)

    logging.info(
        "Evaluating checkpoint %s on %s with batch_size=%d",
        checkpoint_dir,
        eval_repo_id,
        batch_size,
    )
    metrics, media = evaluate_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        eval_repo_id=eval_repo_id,
        batch_size=batch_size,
        seed=seed,
        model=model,
        max_decoding_steps=_eval_max_decoding_steps(config),
    )
    payload = {
        "eval_repo_id": eval_repo_id,
        "step": step,
        checkpoint_step_key: float(step),
        **{f"{metric_prefix}/{k}": float(v) for k, v in metrics.items()},
    }
    _write_local_sweep_metrics(metrics_path, latest_metrics_path, csv_path, payload=payload)

    if token_ce_history is None:
        token_ce_history = {}
    token_ce_history[step] = _token_ce_values_from_metrics(metrics)
    token_ce_chart = _save_token_ce_chart(
        checkpoint_dir / "test_vis",
        step=step,
        history=token_ce_history,
    )
    if token_ce_chart is not None:
        media["test_vis/token_ce"] = wandb.Image(
            str(token_ce_chart),
            caption=f"checkpoint_step={step}; pick=solid, place=dashed",
        )

    wandb_payload = _eval_wandb_payload(
        payload,
        prefix=metric_prefix,
        checkpoint_step_key=checkpoint_step_key,
    )
    wandb.log({**wandb_payload, **media})
    logging.info(
        "Eval step %d: ce=%.4f action_mse=%.4f pick_trans=%.4f place_trans=%.4f",
        step,
        payload[f"{metric_prefix}/mean_cross_entropy"],
        payload[f"{metric_prefix}/action_mse"],
        payload[f"{metric_prefix}/pick_trans_l2"],
        payload[f"{metric_prefix}/place_trans_l2"],
    )
    return {k: float(v) for k, v in metrics.items()}


def run_post_train_eval_sweep(config: _config.TrainConfig) -> None:
    if not checkpoint_eval_enabled():
        logging.info("Post-train eval sweep disabled.")
        return

    eval_repo_id, _, batch_size, _ = _eval_runtime_options(config)
    checkpoint_root = config.checkpoint_dir
    steps = list_checkpoint_steps(checkpoint_root)
    if not steps:
        logging.warning("Post-train eval skipped: no checkpoints found under %s", checkpoint_root)
        return

    reset_eval_sweep_files(config)
    logging.info(
        "Starting post-train eval sweep: repo=%s, checkpoints=%s, batch_size=%d",
        eval_repo_id,
        steps,
        batch_size,
    )
    token_ce_history: dict[int, dict[str, float]] = {}
    for step in steps:
        evaluate_and_log_checkpoint(config, step=step, token_ce_history=token_ce_history)


def main(
    *,
    config_name: str = "pickplace_all_qbin64",
    checkpoint_dir: pathlib.Path,
    eval_repo_id: str,
    batch_size: int = 16,
    seed: int = 0,
    compute_action_mse: bool = True,
) -> None:
    del compute_action_mse  # Kept for compatibility with the older eval script CLI.
    config = _config.get_config(config_name)
    metrics, _ = evaluate_checkpoint(
        config,
        checkpoint_dir=epath.Path(checkpoint_dir).expanduser().resolve(),
        eval_repo_id=eval_repo_id,
        batch_size=batch_size,
        seed=seed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
