import dataclasses
import functools
import json
import logging
import copy
import os
import pathlib
import platform
import sys
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

from scripts.evaluation import eval as _checkpoint_eval


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _write_local_metrics(
    metrics_path: epath.Path,
    latest_metrics_path: epath.Path,
    *,
    step: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "step": step,
        **{key: float(value) for key, value in metrics.items()},
    }
    with metrics_path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    latest_metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _mean_ce_by_loss_position(
    token_nll: jnp.ndarray,
    loss_mask: jnp.ndarray,
    *,
    max_loss_tokens: int,
) -> jnp.ndarray:
    loss_mask = loss_mask.astype(jnp.bool_)
    loss_pos = jnp.cumsum(loss_mask.astype(jnp.int32), axis=-1) - 1
    valid = loss_mask & (loss_pos >= 0) & (loss_pos < max_loss_tokens)
    one_hot = jax.nn.one_hot(
        jnp.clip(loss_pos, 0, max_loss_tokens - 1),
        max_loss_tokens,
        dtype=token_nll.dtype,
    )
    valid_f = valid.astype(token_nll.dtype)
    sums = jnp.sum(one_hot * token_nll[..., None] * valid_f[..., None], axis=(0, 1))
    counts = jnp.sum(one_hot * valid_f[..., None], axis=(0, 1))
    return sums / jnp.clip(counts, 1)


def _train_wandb_payload(metrics: dict[str, Any]) -> dict[str, float]:
    keep = ("param_norm", "loss", "grad_norm")
    return {key: float(metrics[key]) for key in keep if key in metrics}


def _probe_wandb_payload(metrics: dict[str, Any]) -> dict[str, float]:
    keep = (
        "probe/token_ce_avg",
        "probe/action_mse_avg",
        "probe/pick_rot6d_l2",
        "probe/place_rot6d_l2",
        "probe/pick_trans_l2",
        "probe/place_trans_l2",
    )
    return {key: float(metrics[key]) for key in keep if key in metrics}


FRONT_CAMERA_INTRINSICS = np.array(
    [
        [-175.83856040078922, 0.0, 64.0],
        [0.0, -175.83856040078922, 64.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


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
            out[key] = copy.copy(value) if hasattr(value, "copy") else value
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


def _token_ce_values_by_dof(ce_by_loss_pos: np.ndarray) -> dict[str, float]:
    values = np.asarray(ce_by_loss_pos, dtype=np.float64)
    labels = _loss_token_labels(18, 1)[:18]
    return {
        label: float(values[i])
        for i, label in enumerate(labels)
        if i < len(values)
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
    from matplotlib.lines import Line2D

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
    ax.set_title("Probe per-token CE by step")
    ax.set_xlabel("train step")
    ax.set_ylabel("cross entropy")
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return out_path


def _init_probe_context(config: _config.TrainConfig) -> dict[str, Any]:
    data_config = config.data.create(config.assets_dirs, config.model)
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
    probe_batch_size = max(1, min(4, len(dataset)))
    probe_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=probe_batch_size,
        sharding=None,
        shuffle=False,
        num_batches=1,
        num_workers=0,
        seed=config.seed,
        framework="jax",
    )
    probe_batch = next(iter(probe_loader.torch_loader))
    tokenizer_cls = config.model.fast_model_tokenizer
    tokenizer_kwargs = dict(config.model.fast_model_tokenizer_kwargs or {})
    if tokenizer_cls is None:
        raise ValueError("Probe evaluation currently requires a FAST tokenizer.")
    tokenizer = tokenizer_cls(config.model.max_token_len, **tokenizer_kwargs)
    return {
        "batch": probe_batch,
        "tokenize": _transforms.TokenizeFASTInputs(tokenizer),
        "extract": _transforms.ExtractFASTActions(
            tokenizer,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
        ),
        "action_stats": data_config.norm_stats["actions"],
        "use_quantiles": data_config.use_quantile_norm,
    }


def _evaluate_probe_batch(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    probe_context: dict[str, Any],
    psample_actions,
    *,
    step: int,
    token_ce_history: dict[int, dict[str, float]] | None = None,
    save_token_ce_chart: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    batch_pre = probe_context["batch"]
    batch_ce = _tokenize_collated_batch(batch_pre, probe_context["tokenize"], include_actions=True)
    obs_ce = _model.Observation.from_dict(batch_ce)
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    ce_rng = jax.random.fold_in(jax.random.key(config.seed), int(step))
    token_nll, loss_mask = model.compute_loss_per_token(ce_rng, obs_ce, batch_ce["actions"], train=False)
    token_nll = np.asarray(jax.device_get(token_nll), dtype=np.float64)
    loss_mask = np.asarray(jax.device_get(loss_mask), dtype=np.float64)
    per_example_ce = np.sum(token_nll * loss_mask, axis=-1) / np.clip(np.sum(loss_mask, axis=-1), 1, None)
    ce_by_loss_pos = np.asarray(
        jax.device_get(
            _mean_ce_by_loss_position(
                jnp.asarray(token_nll),
                jnp.asarray(loss_mask),
                max_loss_tokens=config.model.action_horizon * config.model.action_dim + 2,
            )
        ),
        dtype=np.float64,
    )
    n_action_tokens = min(config.model.action_horizon * config.model.action_dim, len(ce_by_loss_pos))
    token_ce_avg = float(np.mean(ce_by_loss_pos[:n_action_tokens])) if n_action_tokens else float(np.mean(per_example_ce))

    batch_inf = _tokenize_collated_batch(batch_pre, probe_context["tokenize"], include_actions=False)
    obs_inf = _model.Observation.from_dict(batch_inf)
    eval_rng = jax.random.fold_in(jax.random.key(config.seed), int(step))
    token_ids = psample_actions(eval_rng, state, obs_inf)
    token_np = np.asarray(jax.device_get(token_ids)).astype(np.int32)

    pred_chunks = [
        probe_context["extract"]({"actions": token_np[i]})["actions"]
        for i in range(token_np.shape[0])
    ]
    pred_actions_norm = np.stack(pred_chunks, axis=0)
    gt_actions_norm = np.asarray(batch_pre["actions"])

    pred_actions_raw = _unnormalize_actions(
        pred_actions_norm,
        probe_context["action_stats"],
        use_quantiles=probe_context["use_quantiles"],
    )
    gt_actions_raw = _unnormalize_actions(
        gt_actions_norm,
        probe_context["action_stats"],
        use_quantiles=probe_context["use_quantiles"],
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

    overlay_grid = _build_overlay_grid(batch_pre, gt_flat, pred_flat)
    vis_dir = config.checkpoint_dir / "probe_vis"
    _save_overlay_grid(vis_dir, step=step, overlay_grid=overlay_grid)
    token_ce_chart = None
    if token_ce_history is not None:
        token_ce_history[int(step)] = _token_ce_values_by_dof(ce_by_loss_pos)
        if save_token_ce_chart:
            token_ce_chart = _save_token_ce_chart(
                vis_dir,
                step=step,
                history=token_ce_history,
            )
    summary_path = vis_dir / f"step_{step:06d}_metrics.json"

    metrics = {
        "probe/pick_rot6d_l2": float(np.mean(pick_rot6d_err)),
        "probe/place_rot6d_l2": float(np.mean(place_rot6d_err)),
        "probe/pick_trans_l2": float(np.mean(pick_trans_err)),
        "probe/place_trans_l2": float(np.mean(place_trans_err)),
        "probe/action_mse": float(np.mean(action_mse)),
        "probe/ce": float(np.mean(per_example_ce)),
        "probe/token_ce_avg": token_ce_avg,
        "probe/action_mse_avg": float(np.mean(action_mse)),
        "probe/rot6d_l2/pick": float(np.mean(pick_rot6d_err)),
        "probe/rot6d_l2/place": float(np.mean(place_rot6d_err)),
        "probe/trans_l2/pick": float(np.mean(pick_trans_err)),
        "probe/trans_l2/place": float(np.mean(place_trans_err)),
        "probe/action_mse/pick": float(np.mean(pick_action_mse)),
        "probe/action_mse/place": float(np.mean(place_action_mse)),
    }
    summary_path.write_text(json.dumps({"step": step, **metrics}, indent=2, sort_keys=True) + "\n")
    media = {
        "probe_vis/overlay": wandb.Image(
            overlay_grid,
            caption="GT: green/magenta, Pred: red/cyan",
        ),
    }
    if token_ce_chart is not None:
        media["probe_vis/token_ce"] = wandb.Image(
            str(token_ce_chart),
            caption=f"step={step}; pick=solid, place=dashed",
        )
    return metrics, media


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def scheduled_num_teacher_tokens(step: at.ArrayLike) -> at.Array:
        action_token_count = config.model.action_horizon * config.model.action_dim + 2
        final_teacher_tokens = int(np.clip(config.num_teacher_tokens, 0, action_token_count))
        if config.self_forcing_warmup_steps <= 0 and config.self_forcing_ramp_steps <= 0:
            return jnp.asarray(final_teacher_tokens, dtype=jnp.int32)

        if config.self_forcing_ramp_steps <= 0:
            return jnp.where(
                step < config.self_forcing_warmup_steps,
                jnp.asarray(action_token_count, dtype=jnp.int32),
                jnp.asarray(final_teacher_tokens, dtype=jnp.int32),
            )

        progress = (step - config.self_forcing_warmup_steps) / config.self_forcing_ramp_steps
        progress = jnp.clip(progress, 0.0, 1.0)
        teacher_tokens = action_token_count + progress * (final_teacher_tokens - action_token_count)
        return jnp.ceil(teacher_tokens).astype(jnp.int32)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if config.paradigm == "self_forcing":
            teacher_tokens = scheduled_num_teacher_tokens(state.step)
            chunked_loss = model.compute_self_forcing_loss(
                rng, observation, actions, train=True,
                num_teacher_tokens=teacher_tokens,
            )
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    if config.paradigm == "self_forcing":
        info["num_teacher_tokens"] = scheduled_num_teacher_tokens(state.step)
    return new_state, info


def token_ce_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Float[at.Array, "l"]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    eval_rng = jax.random.fold_in(rng, state.step)
    token_nll, loss_mask = model.compute_loss_per_token(eval_rng, observation, actions, train=False)
    return _mean_ce_by_loss_position(
        token_nll,
        loss_mask,
        max_loss_tokens=config.model.action_horizon * config.model.action_dim + 2,
    )


@at.typecheck
def sample_actions_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation: _model.Observation,
) -> _model.Actions:
    model = nnx.merge(state.model_def, state.params)
    return model.sample_actions(rng, observation)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    enable_probe = os.environ.get("OPENPI_ENABLE_PROBE", "0").lower() in {"1", "true", "yes", "on"}
    eval_on_checkpoint = _checkpoint_eval.checkpoint_eval_enabled()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    metrics_path = config.checkpoint_dir / "metrics.jsonl"
    latest_metrics_path = config.checkpoint_dir / "metrics.latest.json"
    if eval_on_checkpoint and not resuming:
        _checkpoint_eval.reset_eval_sweep_files(config)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    probe_context = None
    probe_token_ce_history: dict[int, dict[str, float]] = {}
    if enable_probe:
        try:
            probe_context = _init_probe_context(config)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Probe evaluation disabled: %s", exc)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    ptrain_step_warmup = None
    if config.paradigm == "self_forcing" and config.self_forcing_warmup_steps > 0:
        teacher_config = dataclasses.replace(config, paradigm="teacher_forcing")
        ptrain_step_warmup = jax.jit(
            functools.partial(train_step, teacher_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    psample_actions = jax.jit(
        sample_actions_step,
        in_shardings=(replicated_sharding, train_state_sharding, replicated_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    if probe_context is not None and start_step == 0:
        probe_metrics, probe_media = _evaluate_probe_batch(
            config,
            train_state,
            probe_context,
            psample_actions,
            step=0,
        )
        _write_local_metrics(metrics_path, latest_metrics_path, step=0, metrics=probe_metrics)
        wandb.log(
            {
                **_probe_wandb_payload(probe_metrics),
                **probe_media,
            },
            step=0,
        )

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    eval_token_ce_history = _checkpoint_eval.load_eval_token_ce_history(config) if resuming else {}
    for step in pbar:
        with sharding.set_mesh(mesh):
            active_train_step = ptrain_step
            if ptrain_step_warmup is not None and step < config.self_forcing_warmup_steps:
                active_train_step = ptrain_step_warmup
            train_state, info = active_train_step(train_rng, train_state, batch)
            if config.paradigm == "self_forcing" and step < config.self_forcing_warmup_steps:
                info["num_teacher_tokens"] = jnp.asarray(
                    config.model.action_horizon * config.model.action_dim + 2,
                    dtype=jnp.int32,
                )
        infos.append(info)
        numeric_log_payload: dict[str, float] = {}
        media_log_payload: dict[str, Any] = {}
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            numeric_log_payload.update({k: float(v) for k, v in reduced_info.items()})
            infos = []
        should_probe = probe_context is not None and (
            (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        )
        if should_probe:
            probe_metrics, probe_media = _evaluate_probe_batch(
                config,
                train_state,
                probe_context,
                psample_actions,
                step=step,
                token_ce_history=probe_token_ce_history,
                save_token_ce_chart=step == config.num_train_steps - 1,
            )
            numeric_log_payload.update(probe_metrics)
            media_log_payload.update(probe_media)
        if numeric_log_payload:
            _write_local_metrics(metrics_path, latest_metrics_path, step=step, metrics=numeric_log_payload)
            wandb.log(
                {
                    **_train_wandb_payload(numeric_log_payload),
                    **_probe_wandb_payload(numeric_log_payload),
                    **media_log_payload,
                },
                step=step,
            )
        batch = next(data_iter)

        should_save = (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        if should_save:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            if eval_on_checkpoint:
                logging.info("Waiting for checkpoint %d to finish before eval", step)
                checkpoint_manager.wait_until_finished()
                eval_model = nnx.merge(train_state.model_def, train_state.params)
                eval_model.eval()
                _checkpoint_eval.evaluate_and_log_checkpoint(
                    config,
                    step=step,
                    token_ce_history=eval_token_ce_history,
                    model=eval_model,
                )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
