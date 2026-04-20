"""
LeRobot **RLBench pick+place** 上的 **pi0-FAST** 单条样本推理（与训练同款 policy 栈）。

- **Policy 构建**：对齐 ``examples/inference_uniform_pickplace.py`` —— ``get_config`` +
  ``policy_config.create_trained_policy`` + ``policy.infer``。
- **数据与归一化**：对齐 ``examples/eval_uniform_pickplace_rlbench.py`` —— checkpoint 下
  ``assets/<train_repo_id>/`` 的 norm stats；``--eval-repo-id`` 指向要读的 LeRobot 仓库
  （在 ``HF_LEROBOT_HOME`` 下）。

**9-DOF**：``(x,y,z)`` + Zhou rot-6D。可视化 **默认** 认为 pose 已在 **相机坐标系**（与 ``ctx_rgb_00`` 同一相机），
仅用 **内参 K** 投影；若仍为 **世界系**，加 ``--no-vis-dof-in-camera-frame`` 并配合 ``FRONT_CAMERA_EXTRINSICS``。
可选 ``--vis-output``；加 ``--vis-compare-gt-pred`` 时同图叠画 **GT**（LeRobot ``actions``）与 **Predicted**（GT：深绿夹爪 + 品红 TCP；Pred：红夹爪 + 青黄 TCP；低分辨率不写文字）。

**推理是否走 ``Pi0FAST.sample_actions``**：checkpoint 为 ``params/``（非 ``model.safetensors``）时，``create_trained_policy`` 加载的是 **JAX ``Pi0FAST``**；``Policy`` 里 ``_sample_actions = module_jit(model.sample_actions)``，``infer`` 里对其调用并传入 ``sample_kwargs``（含 ``temperature``）。终端会打印 ``sample_actions kwargs`` 供核对。

**语言指令**：推理时 **会** 使用任务描述。数据经 ``create_torch_dataset(..., prompt_from_task=True)`` 时，``PromptFromLeRobotTask`` 按每条样本的 ``task_index`` 写入 ``prompt``；``_lerobot_sample_to_policy_observation`` 把它放进 ``obs["prompt"]``，再经 ``RLBenchPickPlaceInputs`` 与图像、state 一并进 Pi0-FAST。若你导出数据里 **多条 episode 的 ``task`` 字符串相同**（例如都退回同一 ``fallback_prompt``），则 **task_index 与 prompt 在样本间也完全相同**，此时仅靠换图区分场景，Pred 仍可能一直一样。跑推理时会打印 ``task_index`` 与 ``prompt`` 摘要便于核对。

**``--sample-index`` 与「底图 / Pred 不变、GT 在变」**：LeRobot 每行对应 **一整条 episode**（见 ``convert_rlbench_to_lerobot.py``）。
磁盘上的 ``actions`` 是该 episode **真实的 pick+place 目标位姿**；而 ``ctx_rgb_00``（尤其 ``--random-single-before-pick`` 导出的单帧集）
只是 **抓取前某一帧的随机快照**。不同 ``sample_index`` = 不同 episode：GT 随 episode 的 pick/place 标注而变；若多条 episode 任务与场景很像，
这些快照在像素上可能几乎一样，模型输入（图 + 同一类 prompt）也接近，**Pred 甚至会因离散分箱而与相邻样本完全相同**。
这不表示 ``sample_index`` 未生效——可对比终端打印的 ``pick_dof`` / ``place_dof`` 与 ``actions``，或对 ``observation/ctx_rgb_00`` 做 ``.mean()`` / 哈希自查。

**Pred 换 sample 仍完全一样**：常见有三类原因。（1）**贪心**：``--infer-temperature 0`` 时每步 ``argmax``；图略有差别但每步最可能 token 仍相同 → dof 可完全一致。（2）**logits 极尖**：即使 ``temperature≈0.7``，若模型对动作 token 几乎 one-hot，``categorical`` 仍几乎总抽到同一 token，表现像贪心；可试 **更大温度**（如 ``1.5``～``2.5``）看 dof 是否开始分散。（3）**旧版 Policy RNG**：JAX ``Policy`` 默认 ``jax.random.key(0)``，每次 **单独起进程** 跑一条样本时，第一次 ``infer`` 的随机子 key 总从同一把主钥匙切出来；本脚本已对 **``sample_index`` 做 ``jax.random.fold_in``**，使不同 index 对应不同解码随机流（在温度>0 时才有意义）。用 ``--debug-obs`` 看 ``ctx_rgb_00`` 的 md5；不设 ``--infer-temperature`` 时不传 kwargs，与 ``Pi0FAST.sample_actions`` 签名默认一致（当前为 ``temperature=0``，贪心）。

Usage::

    export HF_LEROBOT_HOME=/path/to/parent/of/repo

    cd openpi && uv run examples/inference_vis_rlbench.py \\
      --checkpoint-dir /path/to/checkpoint/5000 \\
      --eval-repo-id minyangli/place_wine_rlbench_v2_eval \\
      --sample-index 0 \\
      --vis-output /tmp/vis.png \\
      --vis-compare-gt-pred   # 可选：同图 GT + Pred（夹爪与 TCP 分色）
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import shutil
import warnings
from collections.abc import Callable

import cv2
import jax
import numpy as np
import tyro

from openpi import transforms as _transforms
from openpi.models import model as openpi_model
from openpi.policies import policy as openpi_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download as openpi_download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# 与 LeRobot v2.1 ``lerobot.common.datasets.utils.DEFAULT_IMAGE_PATH`` 一致（PNG 存盘路径）
LEROBOT_V21_IMAGE_RELPATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


def _params_tree_fingerprint(params_dir: pathlib.Path) -> str:
    """根据 params 目录下文件路径+大小+mtime 做摘要，用于确认是否换了不同 checkpoint。"""
    if not params_dir.is_dir():
        return "missing"
    h = hashlib.sha256()
    for p in sorted(x for x in params_dir.rglob("*") if x.is_file()):
        try:
            rel = p.relative_to(params_dir).as_posix()
        except ValueError:
            rel = str(p)
        st = p.stat()
        h.update(rel.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
    return h.hexdigest()[:24]


def _nnx_weight_partial_sum(model: object) -> tuple[float, int]:
    """前若干叶子参数求和；不同 checkpoint 应明显不同。"""
    import flax.nnx as nnx
    import jax.numpy as jnp

    _, state = nnx.split(model)
    pure = state.to_pure_dict()
    total = 0.0
    count = 0
    max_leaves = 64
    for leaf in jax.tree.leaves(pure):
        if hasattr(leaf, "shape") and getattr(leaf, "size", 0) > 0:
            if getattr(leaf, "dtype", None) == jnp.bool_:
                continue
            total += float(jnp.sum(jnp.asarray(leaf, dtype=jnp.float32)))
            count += 1
            if count >= max_leaves:
                break
    return total, count


def _maybe_restore_tokenizer_assets_from_checkpoint(
    train_config: _config.TrainConfig,
    checkpoint_assets_dir: pathlib.Path,
) -> None:
    """Restore tokenizer asset files (``bin_edges.npy`` / ``codebook.npy`` / ``vq_params.npz``)
    from the checkpoint's ``assets/<asset_id>/`` if they are missing at the paths referenced
    by ``fast_model_tokenizer_kwargs``.

    Mirrors the fallback used by ``eval_uniform_pickplace_rlbench.py``. Useful when a
    checkpoint is moved to a new machine without the training-time ``assets/``. On the
    original training machine this is a no-op (the configured path already exists).
    """
    model_config = train_config.model
    kwargs = model_config.fast_model_tokenizer_kwargs or {}
    if not kwargs:
        return
    targets = [
        ("bin_edges_path", "bin_edges.npy"),
        ("codebook_path", "codebook.npy"),
        ("vq_params_path", "vq_params.npz"),
    ]
    missing = [(k, f) for k, f in targets if kwargs.get(k) and not os.path.exists(kwargs[k])]
    if not missing:
        return
    if not checkpoint_assets_dir.is_dir():
        return
    for key, fname in missing:
        cfg_path = pathlib.Path(kwargs[key])
        candidates = sorted(checkpoint_assets_dir.rglob(fname))
        if not candidates:
            warnings.warn(
                f"infer: {key}={cfg_path} missing and no {fname} found under {checkpoint_assets_dir}. "
                "Tokenizer instantiation will likely fail.",
                UserWarning,
                stacklevel=1,
            )
            continue
        src = candidates[0]
        if len(candidates) > 1:
            print(f"infer: multiple {fname} found in checkpoint assets, using {src}")
        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src), str(cfg_path))
            print(f"infer: restored {key} {src} -> {cfg_path} (fallback from checkpoint assets)")
        except OSError as e:
            warnings.warn(
                f"infer: failed to restore {key} from {src}: {e}", UserWarning, stacklevel=1
            )


def _normalize_obs_up_to_tokenize(
    policy: openpi_policy.Policy,
    obs_with_actions: dict,
) -> dict:
    """Apply ``policy._input_transform`` step-by-step but stop **before** ``TokenizeFASTInputs``.

    Used to recover the normalized GT actions in float space (before tokenization) so we can
    pass them to the VQ encoder to compute an "oracle" code for diagnostics.
    """
    from openpi.transforms import TokenizeFASTInputs

    composite = policy._input_transform
    steps = getattr(composite, "transforms", None)
    if steps is None:
        return composite(obs_with_actions)
    data = obs_with_actions
    for tr in steps:
        if isinstance(tr, TokenizeFASTInputs):
            break
        data = tr(data)
    return data


def _debug_pi0_fast_decode_probe(
    policy: openpi_policy.Policy,
    obs: dict,
    *,
    model_config: openpi_model.BaseModelConfig,
    sample_kwargs: dict,
    actions_gt: np.ndarray | None = None,
    quiet: bool = False,
) -> dict | None:
    """验证 tokenizer 改动效果：prefill 注入 "Action: " 后 output_tokens 是否为有效 bin tokens。

    ``actions_gt``: optional raw (un-normalized) GT actions from the dataset row. When provided
    together with a ``VQVAEActionTokenizer``, the probe additionally prints an **oracle code**
    analysis: what code the VQ encoder assigns to the true action, whether the model's predicted
    code matches it, and how far the two codebook entries / decoded actions are apart.

    ``quiet``: suppress per-horizon oracle prints (only returns stats). Used by the aggregation
    path in ``--sample-range`` mode so each sample only emits a single compact one-liner outside
    instead of a multi-line dump.

    Returns a dict with per-horizon oracle-code stats when the VQVAE + GT path runs; ``None``
    otherwise. Stats keys: ``oracle_codes``, ``pred_codes`` (-1 if invalid token), ``match``,
    ``codebook_L2`` (per-h or None), ``cos_sim`` (per-h or None), ``oracle_recon_mse`` (float,
    normalized space), ``pred_recon_mse`` (float, normalized space).
    """
    import jax.numpy as jnp

    from openpi.models import tokenizer as tokmod

    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = openpi_model.Observation.from_dict(inputs)
    sub = jax.random.key(42424242)
    toks = policy._sample_actions(sub, observation, **sample_kwargs)
    arr = np.asarray(toks)
    tok_row = np.asarray(arr[0] if arr.ndim > 1 else arr, dtype=np.int32)
    seq = tok_row.reshape(-1)

    cls = model_config.fast_model_tokenizer
    kwargs = model_config.fast_model_tokenizer_kwargs or {}
    if cls is None:
        tok = tokmod.FASTTokenizer(model_config.max_token_len)
    else:
        tok = cls(model_config.max_token_len, **kwargs)

    pm = getattr(tok, "_paligemma_tokenizer", None)
    if pm is None:
        return None

    # Minimal setup reused by the oracle-code section below:
    #   is_vq    — whether this is a Joint VQ-VAE tokenizer (the only case we diagnose)
    #   n_bins   — codebook_size for VQ tokenizer
    #   fast_skip— number of trailing vocab slots reserved for FAST skip tokens
    n_bins = getattr(tok, "_n_bins", None)
    fast_skip = getattr(tok, "_fast_skip_tokens", 128)
    is_vq = hasattr(tok, "_vq") and getattr(tok, "_vq", None) is not None

    # --- VQVAE oracle code 诊断（仅 VQVAE tokenizer + 提供了 GT actions 时）---
    # 解释：VQ encode(normalize(GT)) = "oracle_code"（如果 VLM 完美拟合应该预测的 code）。
    # 比较 oracle vs 模型 argmax 的 code：若不一致，用 codebook L2 距离和 decode 后 MSE
    # 量化"飞多远"，并把两者 decode 出来的 18-D action 都打印出来。
    if not (is_vq and actions_gt is not None):
        return None
    vq = getattr(tok, "_vq", None)
    if vq is None:
        if not quiet:
            print("  [oracle] tokenizer 有 is_vq 标记但缺 _vq，跳过 oracle 诊断")
        return None
    try:
        gt_raw = np.asarray(actions_gt)
        if gt_raw.ndim == 1:
            gt_raw = gt_raw[np.newaxis, :]
        # 只取前 action_horizon 帧的前 action_dim 维（与 tokenize 时一致）
        H, D = model_config.action_horizon, model_config.action_dim
        gt_slice = gt_raw[:H, :D].astype(np.float32)
        # 用 policy 的 input transforms（到 Normalize 为止）得到 normalized GT actions
        side_obs = {k: v for k, v in obs.items()}
        side_obs["actions"] = gt_slice
        side_data = _normalize_obs_up_to_tokenize(policy, side_obs)
        gt_norm = np.asarray(side_data.get("actions"))
        if gt_norm is None or gt_norm.ndim == 0:
            if not quiet:
                print("  [oracle] 无法从 transforms 中拿到 normalized actions，跳过")
            return None
        if gt_norm.ndim == 1:
            gt_norm = gt_norm[np.newaxis, :]
        oracle_codes = np.asarray(vq.encode_to_code(gt_norm)).reshape(-1).astype(np.int64)
        # 预测的 code（与 extract_actions 内部一致：取前 H 个 token，按 PaliGemma → code 映射）
        v = pm.vocab_size()
        pred_codes_all = (v - 1 - fast_skip - tok_row[:H].astype(np.int64))
        K = int(tok._codebook_size) if hasattr(tok, "_codebook_size") else int(n_bins)
        pred_valid_mask = (pred_codes_all >= 0) & (pred_codes_all < K)
        pred_codes = pred_codes_all.copy()
        pred_codes[~pred_valid_mask] = -1  # invalid marker

        cb = vq._codebook  # (K, latent_dim)，已在 JointVQVAEInfer 里按需 l2-normalize
        if not quiet:
            print(
                f"  [oracle] GT actions (raw, 前 {D} 维): "
                f"{np.array2string(gt_slice[0], precision=4, separator=', ', suppress_small=True)}"
            )
            print(
                f"  [oracle] GT actions (normalized): "
                f"{np.array2string(gt_norm[0], precision=4, separator=', ', suppress_small=True)}"
            )
        # decode oracle & pred 后比较 18-D action
        oracle_dec = np.asarray(vq.decode_from_code(oracle_codes))  # (H, D)
        pred_codes_for_decode = np.where(pred_valid_mask, pred_codes, 0)
        pred_dec = np.asarray(vq.decode_from_code(pred_codes_for_decode))  # (H, D)
        per_h_cb_l2: list[float | None] = []
        per_h_cos: list[float | None] = []
        matches: list[bool] = []
        for h in range(H):
            oc = int(oracle_codes[h])
            pc = int(pred_codes[h])
            match = oc == pc
            matches.append(match)
            if pc >= 0:
                cb_dist = float(np.linalg.norm(cb[oc] - cb[pc]))
                a, b = cb[oc], cb[pc]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                cos_sim = float(np.dot(a, b) / denom)
                per_h_cb_l2.append(cb_dist)
                per_h_cos.append(cos_sim)
                if not quiet:
                    act_mse = float(np.mean((oracle_dec[h] - pred_dec[h]) ** 2))
                    act_l2 = float(np.linalg.norm(oracle_dec[h] - pred_dec[h]))
                    tag = "✓" if match else "✗"
                    print(
                        f"  [oracle] h={h}: oracle_code={oc} pred_code={pc} {tag}  "
                        f"codebook_L2={cb_dist:.4f} cos={cos_sim:+.4f}  "
                        f"decoded_action MSE={act_mse:.5f} L2={act_l2:.4f}"
                    )
            else:
                per_h_cb_l2.append(None)
                per_h_cos.append(None)
                if not quiet:
                    print(
                        f"  [oracle] h={h}: oracle_code={oc} pred_code=<invalid token>  "
                        "(模型没有生成落在 codebook 范围内的 code token)"
                    )
        # 归一化空间里的重构误差（区分 "decode 本身就烂" vs "code 挑错了"）
        oracle_recon_mse = float(np.mean((gt_norm[:H] - oracle_dec) ** 2))
        pred_recon_mse = float(np.mean((gt_norm[:H] - pred_dec) ** 2))
        if not quiet:
            print(
                f"  [oracle] MSE(normalized): oracle_decode vs GT={oracle_recon_mse:.5f}  "
                f"pred_decode vs GT={pred_recon_mse:.5f}  "
            )
        return {
            "oracle_codes": [int(c) for c in oracle_codes.tolist()],
            "pred_codes": [int(c) for c in pred_codes.tolist()],
            "match": matches,
            "codebook_L2": per_h_cb_l2,
            "cos_sim": per_h_cos,
            "oracle_recon_mse": oracle_recon_mse,
            "pred_recon_mse": pred_recon_mse,
        }
    except Exception as e:  # noqa: BLE001
        if not quiet:
            print(f"  [oracle] 诊断失败: {type(e).__name__}: {e}")
        return None

# --- RLBench front camera (与数据里 front_rgb / ctx_rgb 分辨率一致时使用；cx=cy=64 对应 128×128) ---
FRONT_CAMERA_INTRINSICS: np.ndarray = np.array(
    [
        [-175.83856040078922, 0.0, 64.0],
        [0.0, -175.83856040078922, 64.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# 以下为 **camera→world**（常见 RLBench / 仿真导出）：p_h_w = T_cw @ p_h_c。投影时需 world→camera，故对 T 求逆。
FRONT_CAMERA_EXTRINSICS: np.ndarray = np.array(
    [
        [1.1920928955078125e-07, -0.42261794209480286, -0.9063079357147217, 1.349999189376831],
        [-1.0, -5.960464477539062e-07, 1.4901161193847656e-07, 3.715465624054559e-08],
        [-5.662441253662109e-07, 0.9063079357147217, -0.42261791229248047, 1.579999327659607],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# --- 可视化原理 ---
# **相机系 dof（默认）**：dof9 的前 3 维为相机系 TCP，rot6d 的 R 列为夹爪轴在 **相机系** 的朝向；
# 线框点 p_cam = p + R @ p_local，再仅用 K 做针孔投影（不再乘外参）。
# **世界系 dof**（``--no-vis-dof-in-camera-frame``）：同上但 p、R 在世界系，先 world→camera（4×4 + ``--vis-extrinsics-are-cam2world``），再 K。
# 投影均要求相机系下 z>0。
# ---------------------------------------------------------------------------


def rot6d_to_rotation_matrix(rot6: np.ndarray) -> np.ndarray:
    """Zhou et al.: 前两列 -> 正交 ``3×3``（与 ``convert_rlbench_to_lerobot._quat_to_rot6d`` 互逆）。"""
    a1 = np.asarray(rot6, dtype=np.float64).reshape(6)[:3]
    a2 = np.asarray(rot6, dtype=np.float64).reshape(6)[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float64)


def dof9_to_gripper_wireframe(
    dof9: np.ndarray,
    *,
    half_opening_m: float = 0.022,
    finger_extent_m: float = 0.04,
    handle_length_m: float = 0.05,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """9-DOF → 夹爪线框顶点（**与 dof 同一参考系**：相机系或世界系），``U`` 形 + **柄**。

    **局部系约定**：列为 ``R`` 的 ``x,y,z`` 为夹爪在 **该参考系** 下的姿态；``±X`` 为张合方向，``Y`` 沿手指长度。
    - 两指：``x=±half_opening``，``y`` 从 ``-L`` 到 ``+L``（``L=finger_extent``）。
    - **底边**（靠机器人端）：在 ``y=-L`` 连接 ``(-w,-L,0)`` 与 ``(w,-L,0)``；开口朝 ``+Y``。
    - **柄**：从底边中点 ``(0,-L,0)`` 沿局部 **-Y** 延伸 ``handle_length_m``（与开口相反，像叉子柄）。
      ``handle_length_m<=0`` 时不画柄。
    """
    d = np.asarray(dof9, dtype=np.float64).reshape(-1)
    if d.size < 9:
        raise ValueError(f"dof9 must have length >= 9, got {d.size}")
    pos = d[:3]
    r = rot6d_to_rotation_matrix(d[3:9])

    def to_frame(p_local: np.ndarray) -> np.ndarray:
        return pos + r @ np.asarray(p_local, dtype=np.float64).reshape(3)

    w = float(half_opening_m)
    L = float(finger_extent_m)
    segs: list[tuple[np.ndarray, np.ndarray]] = [
        (to_frame([w, -L, 0.0]), to_frame([w, +L, 0.0])),
        (to_frame([-w, -L, 0.0]), to_frame([-w, +L, 0.0])),
        (to_frame([-w, -L, 0.0]), to_frame([w, -L, 0.0])),
    ]
    hlen = float(handle_length_m)
    if hlen > 1e-9:
        segs.append((to_frame([0.0, -L, 0.0]), to_frame([0.0, -L - hlen, 0.0])))
    return segs


def project_camera_points_to_uv(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """相机系 3D 点 → 像素 ``(u,v)``（仅乘 ``K``，**不再**做 world→camera）。"""
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


def project_world_points_to_uv(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_4x4: np.ndarray,
    *,
    extrinsics_are_cam2world: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """世界系点 → 像素 ``(u,v)``（**输入始终在 world**）。

    ``extrinsics_4x4``：若 ``extrinsics_are_cam2world=True``（默认），则为 **camera→world** 的 ``T_cw``，
    内部先 ``inv(T)`` 得到 world→camera，再 ``p_c = R p_w + t``。若为 **world→camera** 则设
    ``extrinsics_are_cam2world=False``，不再求逆。
    """
    t_wc = np.asarray(extrinsics_4x4, dtype=np.float64).reshape(4, 4)
    if extrinsics_are_cam2world:
        t_wc = np.linalg.inv(t_wc)
    r = t_wc[:3, :3]
    t = t_wc[:3, 3]
    pw = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pc = (r @ pw.T + t.reshape(3, 1)).T
    z = pc[:, 2]
    valid = z > 1e-6
    uv_h = (np.asarray(intrinsics, dtype=np.float64).reshape(3, 3) @ pc.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
    uv = np.stack([u, v], axis=-1)
    valid = valid & np.isfinite(uv).all(axis=1)
    return uv, valid


def _gt_pick_place_dof9_from_sample(sample: object) -> tuple[np.ndarray, np.ndarray] | None:
    """从 **未走 transform_dataset** 的 LeRobot ``__getitem__`` 行取 18 维 action，拆成 pick/place 各 9 维。

    与 ``convert_rlbench_to_lerobot`` 一致：磁盘上一般为 **物理空间**（与 policy 反归一化后的 pred 同尺度）。
    ``actions`` 形状可为 ``(18,)``、``(1,18)`` 或 ``(H,18)``（取当前帧 ``[0]``）。
    """
    try:
        sample_np = _to_numpy_tree(sample)
        base = _base_dict_from_lerobot_row(sample_np)
    except Exception:
        return None
    if "actions" not in base:
        return None
    a = np.asarray(base["actions"], dtype=np.float64)
    if a.ndim == 0:
        return None
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a0 = np.asarray(a[0], dtype=np.float64).reshape(-1)
    if a0.size < 18:
        return None
    return a0[:9].copy(), a0[9:18].copy()


def _rgb_to_uint8_hwc(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return x


def _build_project_pts_fn(
    intrinsics: np.ndarray,
    *,
    dof_in_camera_frame: bool,
    extrinsics_4x4: np.ndarray | None,
    extrinsics_are_cam2world: bool,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def project_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if dof_in_camera_frame:
            return project_camera_points_to_uv(pts, intrinsics)
        assert extrinsics_4x4 is not None
        return project_world_points_to_uv(
            pts,
            intrinsics,
            extrinsics_4x4,
            extrinsics_are_cam2world=extrinsics_are_cam2world,
        )

    return project_pts


def _draw_dof9_wireframe_on_bgr(
    img_bgr: np.ndarray,
    dof9: np.ndarray,
    *,
    project_pts: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    half_opening_m: float,
    finger_extent_m: float,
    handle_length_m: float,
    jaw_color_bgr: tuple[int, int, int],
    tcp_color_bgr: tuple[int, int, int],
    line_thickness: int = 2,
    tcp_dot_radius: int = 2,
    warn_if_empty: bool = False,
) -> bool:
    """在已有 BGR 图上画一组 9-DOF 线框；返回是否至少画出一段线或 TCP 点。"""
    segments = dof9_to_gripper_wireframe(
        dof9,
        half_opening_m=half_opening_m,
        finger_extent_m=finger_extent_m,
        handle_length_m=handle_length_m,
    )
    drew_any = False

    def draw_seg(p0: np.ndarray, p1: np.ndarray, color: tuple[int, int, int]) -> None:
        nonlocal drew_any
        uv, valid = project_pts(np.stack([p0, p1], axis=0))
        if not (bool(valid[0]) and bool(valid[1])):
            return
        a = (int(round(float(uv[0, 0]))), int(round(float(uv[0, 1]))))
        b = (int(round(float(uv[1, 0]))), int(round(float(uv[1, 1]))))
        if not (np.isfinite(uv).all()):
            return
        cv2.line(img_bgr, a, b, color, line_thickness, cv2.LINE_AA)
        drew_any = True

    for s0, s1 in segments:
        draw_seg(s0, s1, jaw_color_bgr)

    tcp = np.asarray(dof9, dtype=np.float64).reshape(-1)[:3]
    uv_tcp, vtcp = project_pts(tcp.reshape(1, 3))
    if bool(vtcp[0]) and np.isfinite(uv_tcp).all():
        c = (int(round(float(uv_tcp[0, 0]))), int(round(float(uv_tcp[0, 1]))))
        cv2.circle(img_bgr, c, tcp_dot_radius, tcp_color_bgr, -1, cv2.LINE_AA)
        drew_any = True

    if not drew_any and warn_if_empty:
        warnings.warn(
            "可视化未画出任何线段/点：相机系 dof 请确认 xyz 与 K 同单位、z>0；"
            "世界系 dof 请检查外参与 --vis-extrinsics-are-cam2world。",
            UserWarning,
            stacklevel=3,
        )
    return drew_any


def save_dof9_pose_visualization(
    rgb: np.ndarray,
    dof9: np.ndarray,
    intrinsics: np.ndarray,
    out_path: pathlib.Path | str,
    *,
    dof_in_camera_frame: bool = True,
    extrinsics_4x4: np.ndarray | None = None,
    extrinsics_are_cam2world: bool = True,
    half_opening_m: float = 0.022,
    finger_extent_m: float = 0.04,
    handle_length_m: float = 0.05,
    jaw_color_bgr: tuple[int, int, int] = (0, 0, 255),
    tcp_color_bgr: tuple[int, int, int] = (0, 255, 128),
) -> None:
    """在推理用 RGB 上绘制 9-DOF 的 ``U`` 形夹爪 + 柄。

    - ``dof_in_camera_frame=True``（默认）：``dof9`` 已在 **相机系**，仅用 ``intrinsics`` 投影。
    - ``dof_in_camera_frame=False``：``dof9`` 在 **世界系**，须提供 ``extrinsics_4x4`` 做 world→camera。

    **颜色（BGR）：** ``jaw_color_bgr`` 手指、底边与柄；``tcp_color_bgr`` 为 TCP 小圆点。
    """
    if not dof_in_camera_frame and extrinsics_4x4 is None:
        raise ValueError("世界系 dof 可视化需要 extrinsics_4x4（或改用 dof_in_camera_frame=True）。")

    img_bgr = cv2.cvtColor(_rgb_to_uint8_hwc(rgb), cv2.COLOR_RGB2BGR)
    project_pts = _build_project_pts_fn(
        intrinsics,
        dof_in_camera_frame=dof_in_camera_frame,
        extrinsics_4x4=extrinsics_4x4,
        extrinsics_are_cam2world=extrinsics_are_cam2world,
    )
    _draw_dof9_wireframe_on_bgr(
        img_bgr,
        dof9,
        project_pts=project_pts,
        half_opening_m=half_opening_m,
        finger_extent_m=finger_extent_m,
        handle_length_m=handle_length_m,
        jaw_color_bgr=jaw_color_bgr,
        tcp_color_bgr=tcp_color_bgr,
        warn_if_empty=True,
    )

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img_bgr)


def save_dof9_gt_pred_visualization(
    rgb: np.ndarray,
    gt_dof9: np.ndarray,
    pred_dof9: np.ndarray,
    intrinsics: np.ndarray,
    out_path: pathlib.Path | str,
    *,
    dof_in_camera_frame: bool = True,
    extrinsics_4x4: np.ndarray | None = None,
    extrinsics_are_cam2world: bool = True,
    half_opening_m: float = 0.022,
    finger_extent_m: float = 0.04,
    handle_length_m: float = 0.05,
    # 夹爪线 vs TCP 点用互补色，避免低分辨率下糊成一团（均为 BGR）
    gt_jaw_bgr: tuple[int, int, int] = (0, 140, 0),
    gt_tcp_bgr: tuple[int, int, int] = (255, 0, 255),
    pred_jaw_bgr: tuple[int, int, int] = (0, 0, 255),
    pred_tcp_bgr: tuple[int, int, int] = (255, 255, 0),
) -> None:
    """同一张底图上画 **GT**（深绿线 + 品红 TCP）与 **Pred**（红线 + 青黄 TCP）；不写文字。"""
    if not dof_in_camera_frame and extrinsics_4x4 is None:
        raise ValueError("世界系 dof 可视化需要 extrinsics_4x4。")

    img_bgr = cv2.cvtColor(_rgb_to_uint8_hwc(rgb), cv2.COLOR_RGB2BGR)
    project_pts = _build_project_pts_fn(
        intrinsics,
        dof_in_camera_frame=dof_in_camera_frame,
        extrinsics_4x4=extrinsics_4x4,
        extrinsics_are_cam2world=extrinsics_are_cam2world,
    )
    # 先画 GT，再画 Pred（重叠处上层为预测）
    _draw_dof9_wireframe_on_bgr(
        img_bgr,
        gt_dof9,
        project_pts=project_pts,
        half_opening_m=half_opening_m,
        finger_extent_m=finger_extent_m,
        handle_length_m=handle_length_m,
        jaw_color_bgr=gt_jaw_bgr,
        tcp_color_bgr=gt_tcp_bgr,
        warn_if_empty=False,
    )
    _draw_dof9_wireframe_on_bgr(
        img_bgr,
        pred_dof9,
        project_pts=project_pts,
        half_opening_m=half_opening_m,
        finger_extent_m=finger_extent_m,
        handle_length_m=handle_length_m,
        jaw_color_bgr=pred_jaw_bgr,
        tcp_color_bgr=pred_tcp_bgr,
        warn_if_empty=False,
    )

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img_bgr)


def _vis_pick_place_paths(base: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """``vis.png`` -> ``vis_pick.png`` 与 ``vis_place.png``（无后缀时 ``vis_pick`` / ``vis_place``）。"""
    stem = base.stem
    suf = base.suffix
    parent = base.parent
    if stem.endswith("_pick") or stem.endswith("_place"):
        stem = stem.rsplit("_", 1)[0]
    pick = parent / f"{stem}_pick{suf}"
    place = parent / f"{stem}_place{suf}"
    return pick, place


def _to_numpy_tree(obj: object) -> object:
    import torch

    if isinstance(obj, dict):
        return {k: _to_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy_tree(x) for x in obj)
    if isinstance(obj, torch.Tensor):
        return np.asarray(obj.detach().cpu().numpy())
    return obj


def _coerce_lerobot_flat_keys(row: dict) -> dict:
    """把常见 LeRobot 嵌套/点号键收成 repack 期望的顶层 ``state`` / ``ctx_rgb_*``。"""
    out = dict(row)
    obs = out.pop("observation", None)
    if isinstance(obs, dict):
        for ok, ov in obs.items():
            sk = str(ok)
            if sk == "state":
                out.setdefault("state", ov)
            elif sk.startswith("ctx_rgb_"):
                out.setdefault(sk, ov)
    renames: list[tuple[object, str]] = []
    for k in list(out.keys()):
        ks = str(k)
        if ks == "observation.state":
            renames.append((k, "state"))
        elif ks.startswith("observation.ctx_rgb_"):
            renames.append((k, ks.removeprefix("observation.")))
    for old, new in renames:
        if old not in out:
            continue
        if new not in out:
            out[new] = out.pop(old)
        else:
            del out[old]
    return out


def _base_dict_from_lerobot_row(row: dict) -> dict:
    """``flatten_dict`` 后抽出 ``state``、``ctx_rgb_*``、``prompt``、可选 ``actions``。"""
    row = _coerce_lerobot_flat_keys(row)
    flat = _transforms.flatten_dict(row)
    base: dict = {}

    if "state" in flat:
        base["state"] = flat["state"]
    elif "observation/state" in flat:
        base["state"] = flat["observation/state"]
    elif "observation.state" in flat:
        base["state"] = flat["observation.state"]
    else:
        raise KeyError(
            "Cannot find state. First flatten keys: " + str(list(flat.keys())[:40])
        )

    for k, v in flat.items():
        sk = str(k)
        if sk.startswith("ctx_rgb_"):
            base[sk] = v
        elif "ctx_rgb_" in sk:
            last = sk.split("/")[-1]
            if last.startswith("ctx_rgb_"):
                base[last] = v
            elif sk.startswith("observation.ctx_rgb_"):
                base[sk.removeprefix("observation.")] = v
    if not any(isinstance(k, str) and k.startswith("ctx_rgb_") for k in base):
        for k, v in flat.items():
            sk = str(k)
            if sk.startswith("observation.ctx_rgb_"):
                base[sk.removeprefix("observation.")] = v
    if not any(isinstance(k, str) and k.startswith("ctx_rgb_") for k in base):
        raise KeyError("Cannot find ctx_rgb_* cameras. Keys: " + str(list(flat.keys())[:40]))

    for meta in ("prompt", "task_index", "task"):
        if meta in flat:
            base[meta] = flat[meta]
    if "actions" in flat:
        base["actions"] = flat["actions"]
    elif "action" in flat:
        base["actions"] = flat["action"]
    return base


def _post_repack_rlbench_obs(base: dict, *, num_ctx_frames: int) -> dict:
    """``RLBenchPickPlaceInputs`` 入口：``observation/state``、``observation/ctx_rgb_*``、``prompt``。"""
    out: dict = {"observation/state": np.asarray(base["state"], dtype=np.float32)}
    for i in range(num_ctx_frames):
        k = f"ctx_rgb_{i:02d}"
        if k not in base:
            raise KeyError(f"Missing {k} (num_ctx_frames={num_ctx_frames})")
        out[f"observation/{k}"] = base[k]
    p = base.get("prompt", "")
    if isinstance(p, bytes):
        out["prompt"] = p.decode("utf-8")
    elif hasattr(p, "item") and not isinstance(p, str):
        out["prompt"] = str(np.asarray(p).item())
    else:
        out["prompt"] = str(p) if p is not None else ""
    if "actions" in base:
        out["actions"] = base["actions"]
    return out


def _rlbench_num_ctx_frames(train_config: _config.TrainConfig) -> int:
    d = train_config.data
    n = getattr(d, "num_ctx_frames", None)
    return int(n) if n is not None else 1


def _lerobot_sample_to_policy_observation(
    sample: dict, *, train_config: _config.TrainConfig
) -> dict:
    """单条 LeRobot 样本 -> ``policy.infer`` 所需字典（关闭 Repack，直接给 ``observation/*``）。"""
    num_ctx = _rlbench_num_ctx_frames(train_config)
    sample_np = _to_numpy_tree(sample)
    base = _base_dict_from_lerobot_row(sample_np)
    return _to_numpy_tree(_post_repack_rlbench_obs(base, num_ctx_frames=num_ctx))


def _unwrap_inner_lerobot_dataset(dataset: object) -> object:
    """``TransformedDataset`` 等外层剥掉，得到 ``LeRobotDataset``（若有）。"""
    cur = dataset
    for _ in range(16):
        inner = getattr(cur, "_dataset", None)
        if inner is None:
            break
        cur = inner
    return cur


def _scalar_int_from_row(val: object) -> int | None:
    if val is None:
        return None
    return int(np.asarray(val).reshape(-1)[0])


def _print_lerobot_ctx_rgb_disk_paths(
    dataset: object,
    row: dict,
    *,
    train_config: _config.TrainConfig,
    sample_index: int,
) -> None:
    """打印当前样本在 LeRobot 仓库根目录下的 ``ctx_rgb_*`` PNG 路径（转换时写入的副本，非原始 RLBench 路径）。"""
    base = _unwrap_inner_lerobot_dataset(dataset)
    root = getattr(base, "root", None)
    repo_id = getattr(base, "repo_id", None)
    row_np = _to_numpy_tree(row)
    ep = _scalar_int_from_row(row_np.get("episode_index"))
    fr = _scalar_int_from_row(row_np.get("frame_index"))
    abs_idx = _scalar_int_from_row(row_np.get("index"))
    if root is None:
        print(
            "infer: ctx_rgb 磁盘路径：无法从 dataset 解析 root（非预期 LeRobot 封装）。"
            " 原始 RLBench 文件路径未写入本数据集。"
        )
        return
    root_p = pathlib.Path(root).resolve()
    if ep is None or fr is None:
        print(
            f"infer: ctx_rgb 磁盘路径：LeRobot root={root_p!s} repo_id={repo_id!r} "
            f"sample_index={sample_index} global_index={abs_idx} "
            f"episode_index={ep} frame_index={fr}（缺 episode/frame，无法拼 PNG 路径）。"
            " 原始 RLBench 路径未写入 LeRobot。"
        )
        return
    nctx = _rlbench_num_ctx_frames(train_config)
    # print(
    #     f"infer: LeRobot 内 PNG（非原始 RLBench 路径）root={root_p!s} repo_id={repo_id!r} "
    #     f"sample_index={sample_index} episode_index={ep} frame_index={fr}"
    #     + (f" parquet_index={abs_idx}" if abs_idx is not None else "")
    #     + "  （frame_index=该 episode 内第几帧；本转换每 episode 仅 1 帧，故恒为 0）"
    # )
    for i in range(nctx):
        key = f"ctx_rgb_{i:02d}"
        rel = LEROBOT_V21_IMAGE_RELPATH.format(
            image_key=key, episode_index=ep, frame_index=fr
        )
        full = (pathlib.Path(root) / rel).resolve()
        ex = full.is_file()
        # print(f"infer:   {key} → {full!s}  exists={ex}")


def _prompt_and_task_index_for_print(obs: dict, dataset_row: dict) -> tuple[str, int | None]:
    """终端摘要：进入 policy 的文本指令与 LeRobot task_index。"""
    pr = obs.get("prompt", "")
    if isinstance(pr, bytes):
        pr = pr.decode("utf-8")
    elif not isinstance(pr, str):
        pr = str(np.asarray(pr).item()) if getattr(pr, "shape", None) == () else str(pr)
    ti = dataset_row.get("task_index")
    task_i: int | None = None
    if ti is not None:
        task_i = int(np.asarray(ti).reshape(-1)[0])
    return pr, task_i


def _print_debug_obs(*, sample_index: int, obs: dict, dataset_row: dict) -> None:
    """打印当前样本观测指纹，用于核对「换 sample_index 是否换图 / GT 是否变」。"""
    rgb = np.asarray(obs["observation/ctx_rgb_00"])
    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb
    digest = hashlib.md5(rgb_u8.tobytes(), usedforsecurity=False).hexdigest()
    pr = obs.get("prompt", "")
    if isinstance(pr, bytes):
        pr = pr.decode("utf-8")
    elif not isinstance(pr, str):
        pr = str(np.asarray(pr).item()) if getattr(pr, "shape", None) == () else str(pr)
    act = dataset_row.get("actions")
    act_s = ""
    if act is not None:
        a = np.asarray(_to_numpy_tree(act)).reshape(-1)
        act_s = f" disk_actions[:6]={np.array2string(a[:6], precision=4, separator=',')}"
    ep = dataset_row.get("episode_index")
    ep_s = ""
    if ep is not None:
        e = ep.item() if hasattr(ep, "item") else ep
        ep_s = f" episode_index={int(e)}"
    print(
        f"[debug_obs] sample_index={sample_index}{ep_s} ctx_rgb_00 md5={digest} "
        f"mean={float(np.mean(rgb)):.6g} std={float(np.std(rgb)):.6g} prompt={pr!r}{act_s}"
    )


def _resolve_sample_indices(
    sample_index: int, sample_range: str | None, n: int
) -> list[int]:
    """Return the list of sample indices to process.

    ``sample_range`` overrides ``sample_index`` when provided:
      - ``"all"``             → ``[0, n)``
      - ``"A:B"`` (inclusive) → ``[A, A+1, ..., B]``
      - ``"A"``               → ``[A]`` (single)
    Single sample path uses ``sample_index`` and disables the final aggregated summary.
    """
    if sample_range is None:
        if sample_index < 0 or sample_index >= n:
            raise IndexError(f"sample_index={sample_index} not in [0, {n})")
        return [sample_index]
    r = sample_range.strip().lower()
    if r == "all":
        return list(range(n))
    if ":" in r:
        a_s, b_s = r.split(":", 1)
        a, b = int(a_s), int(b_s)
    else:
        a = b = int(r)
    if a < 0 or b >= n or a > b:
        raise IndexError(
            f"sample_range={sample_range!r} invalid for dataset of size {n} "
            f"(expected 0 <= start <= end < {n})"
        )
    return list(range(a, b + 1))


def _vis_sample_output(
    template: pathlib.Path | None, idx: int, *, multi_sample: bool
) -> pathlib.Path | None:
    """Derive a per-sample vis output path from a template.

    - If ``template`` is ``None``, return ``None`` (vis disabled).
    - If ``template`` contains ``{i}`` or ``{idx}``, format with the sample index.
    - In multi-sample mode, if no placeholder is present, auto-insert ``_{idx}``
      before the extension so each sample gets a distinct file.
    """
    if template is None:
        return None
    s = str(template)
    if "{i}" in s or "{idx}" in s:
        return pathlib.Path(s.format(i=idx, idx=idx))
    if not multi_sample:
        return pathlib.Path(s)
    p = pathlib.Path(s)
    return p.with_name(f"{p.stem}_{idx}{p.suffix}") if p.suffix else p / f"sample_{idx}.png"


def _percentiles(xs: list[float]) -> tuple[float, float, float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.mean(a)),
        float(np.percentile(a, 50)),
        float(np.percentile(a, 75)),
        float(np.percentile(a, 95)),
    )


def _summarize_match_rate(
    stats_list: list[dict],
    *,
    codebook_size: int | None,
    neighbor_cos_threshold: float = 0.7,
) -> None:
    """Print aggregated match-rate + action-error stats over multiple samples."""
    n = len(stats_list)
    if n == 0:
        print("[summary] no samples to aggregate.")
        return

    has_oracle = [s for s in stats_list if s.get("oracle_codes") is not None]
    n_oracle = len(has_oracle)

    perfect = neighbor = far = invalid = 0
    oracle_mses: list[float] = []
    pred_mses: list[float] = []
    for s in has_oracle:
        ocs = s["oracle_codes"]
        pcs = s["pred_codes"]
        cos_list = s.get("cos_sim") or []
        for h, (oc, pc) in enumerate(zip(ocs, pcs)):
            if pc < 0:
                invalid += 1
            elif oc == pc:
                perfect += 1
            else:
                cs = cos_list[h] if h < len(cos_list) else None
                if cs is not None and cs >= neighbor_cos_threshold:
                    neighbor += 1
                else:
                    far += 1
        if s.get("oracle_recon_mse") is not None:
            oracle_mses.append(float(s["oracle_recon_mse"]))
        if s.get("pred_recon_mse") is not None:
            pred_mses.append(float(s["pred_recon_mse"]))

    total_h = perfect + neighbor + far + invalid

    pick_l2 = [s["pick_dof_l2"] for s in stats_list if s.get("pick_dof_l2") is not None]
    place_l2 = [s["place_dof_l2"] for s in stats_list if s.get("place_dof_l2") is not None]
    pick_tr = [s["pick_trans_l2"] for s in stats_list if s.get("pick_trans_l2") is not None]
    place_tr = [s["place_trans_l2"] for s in stats_list if s.get("place_trans_l2") is not None]

    print()
    print("=" * 72)
    print(f"Aggregate summary over N={n} samples")
    print("=" * 72)

    if total_h > 0:
        k_str = f"K={codebook_size}" if codebook_size is not None else "K=?"
        print(f"\nVQ oracle-code analysis ({k_str}, over {total_h} horizon-steps from {n_oracle} samples):")
        print(
            f"  perfect match (oracle == pred)                 : {perfect:4d}/{total_h} "
            f"({100.0 * perfect / total_h:.1f}%)"
        )
        print(
            f"  neighbor mismatch (cos >= {neighbor_cos_threshold:.2f})              : {neighbor:4d}/{total_h} "
            f"({100.0 * neighbor / total_h:.1f}%)"
        )
        print(
            f"  far mismatch (cos < {neighbor_cos_threshold:.2f})                    : {far:4d}/{total_h} "
            f"({100.0 * far / total_h:.1f}%)"
        )
        print(
            f"  invalid pred (not a valid code token)          : {invalid:4d}/{total_h} "
            f"({100.0 * invalid / total_h:.1f}%)"
        )
        acceptable = perfect + neighbor
        print(
            f"  visually acceptable (perfect + neighbor)       : {acceptable:4d}/{total_h} "
            f"({100.0 * acceptable / total_h:.1f}%)"
        )
    else:
        print("\n(No oracle-code stats available: non-VQ tokenizer or GT actions missing.)")

    if oracle_mses and pred_mses:
        o_mean, o_med, _, _ = _percentiles(oracle_mses)
        p_mean, p_med, _, _ = _percentiles(pred_mses)
        ratio = (p_mean / o_mean) if o_mean > 1e-12 else float("inf")
        print("\nDecoded-action MSE vs GT (normalized space):")
        print(f"  oracle_decode : mean={o_mean:.5f}  median={o_med:.5f}")
        print(f"  pred_decode   : mean={p_mean:.5f}  median={p_med:.5f}  (ratio {ratio:.2f}× over oracle)")

    if pick_l2 or place_l2:
        print("\nWorld-space dof L2 (unnormalized, all 9 dims):")
        if pick_l2:
            m, med, p75, p95 = _percentiles(pick_l2)
            print(f"  pick_dof  (N={len(pick_l2):3d}) : mean={m:.4f}  median={med:.4f}  p75={p75:.4f}  p95={p95:.4f}")
        if place_l2:
            m, med, p75, p95 = _percentiles(place_l2)
            print(f"  place_dof (N={len(place_l2):3d}) : mean={m:.4f}  median={med:.4f}  p75={p75:.4f}  p95={p95:.4f}")
    if pick_tr or place_tr:
        print("\nTranslation-only L2 (first 3 dims, meters):")
        if pick_tr:
            m, med, p75, p95 = _percentiles(pick_tr)
            print(f"  pick_trans  (N={len(pick_tr):3d}) : mean={m:.4f}  median={med:.4f}  p75={p75:.4f}  p95={p95:.4f}")
        if place_tr:
            m, med, p75, p95 = _percentiles(place_tr)
            print(f"  place_trans (N={len(place_tr):3d}) : mean={m:.4f}  median={med:.4f}  p75={p75:.4f}  p95={p95:.4f}")

    if has_oracle:
        ranked = sorted(
            has_oracle,
            key=lambda s: float(s.get("pred_recon_mse") or 0.0),
            reverse=True,
        )[:10]
        print("\nWorst 10 samples by pred_decode-vs-GT MSE:")
        for s in ranked:
            oc = s["oracle_codes"][0] if s["oracle_codes"] else -1
            pc = s["pred_codes"][0] if s["pred_codes"] else -1
            cos_list = s.get("cos_sim") or [None]
            cs = cos_list[0] if cos_list else None
            cs_str = f"{cs:+.2f}" if cs is not None else "  n/a"
            pmse = float(s.get("pred_recon_mse") or 0.0)
            pick_s = (
                f"pick_L2={s['pick_dof_l2']:.3f}"
                if s.get("pick_dof_l2") is not None
                else "pick_L2=  n/a"
            )
            place_s = (
                f"place_L2={s['place_dof_l2']:.3f}"
                if s.get("place_dof_l2") is not None
                else "place_L2=  n/a"
            )
            print(
                f"  sample_idx={s['sample_idx']:3d}  oracle={oc:3d}  pred={pc:3d}  "
                f"cos={cs_str}  pred_mse={pmse:.4f}  {pick_s}  {place_s}"
            )
    print("=" * 72)


def main(
    *,
    config_name: str = "pi0_fast_rlbench_pickplace_rand1_lora_cam",
    checkpoint_dir: pathlib.Path,
    eval_repo_id: str,
    sample_index: int = 0,
    sample_range: str | None = None,
    vis_output: pathlib.Path | None = None,
    vis_dof_in_camera_frame: bool = True,
    vis_extrinsics_are_cam2world: bool = True,
    vis_half_opening_m: float = 0.022,
    vis_finger_extent_m: float = 0.04,
    vis_handle_length_m: float = 0.1,
    vis_compare_gt_pred: bool = False,
    infer_temperature: float | None = None,
    infer_rng_seed: int = 0,
    debug_obs: bool = False,
    debug_fast_decode: bool = False,
    print_image_paths: bool = True,
) -> None:
    """加载 checkpoint，从 ``eval_repo_id`` 取第 ``sample_index`` 条，跑一次 ``policy.infer`` 并打印。

    ``infer_temperature``：传给 Pi0-FAST ``sample_actions``。``None`` 表示不传该参数，使用 ``pi0_fast.py`` 里签名默认（当前为 ``0``，贪心）；显式传 ``2.0`` 等即覆盖该默认。
    ``infer_rng_seed``：JAX 主钥匙；与 ``sample_index`` 组合为 ``fold_in(key(seed), index)``，避免「每条样本单独起进程时总从同一把 key(0) 切出同一段随机流」。
    ``debug_obs``：打印 ``ctx_rgb_00`` md5、磁盘 ``actions`` 前几维等，便于核对样本是否真切换。
    ``debug_fast_decode``：在 ``infer`` 前多跑一次 ``sample_actions``，分析 output_tokens 中有多少落在有效 bin 范围 [0, n_bins) 内、extract_actions 归一化结果是否非零等，用于验证 tokenizer 改动（"Action: " prefill 注入 + loss_mask 排除）是否生效。
    ``print_image_paths``：打印当前样本在 LeRobot 数据集根目录下的 ``ctx_rgb_*`` PNG 路径（由转换脚本写入；**不是** JSON 里原始 RLBench 图像路径）。

    若给定 ``vis_output``，在 ``ctx_rgb_00`` 上保存 pick/place 可视化。默认 **dof 已在相机系**（仅用 K）；
    世界系 dof 用 ``--no-vis-dof-in-camera-frame`` 并配合 ``FRONT_CAMERA_EXTRINSICS`` 与
    ``--vis-extrinsics-are-cam2world``。

    ``--vis-compare-gt-pred``：同图叠画 GT 与预测（GT 深绿夹爪+品红 TCP，Pred 红夹爪+青黄 TCP）；文件名仍为 ``*_pick`` / ``*_place``。若无 ``actions`` 则回退为仅预测。

    ``--sample-range``：批量模式。格式 ``"A:B"``（闭区间）或 ``"all"``。启用后遍历 dataset，
    **只加载一次模型**，每个样本打印一行 compact 日志，结束时输出 match-rate / MSE / L2 聚合报告。
    ``--vis-output`` 在此模式下可用 ``{i}`` 占位符（如 ``results/sample{i}.png``）；若不带占位符，
    自动在扩展名前插入 ``_{i}``。单样本模式下 ``--sample-range`` 不传即可。
    """
    ckpt = pathlib.Path(checkpoint_dir).expanduser().resolve()
    params_path = ckpt / "params"
    assets_root = ckpt / "assets"
    if not params_path.is_dir():
        raise FileNotFoundError(f"Missing params: {params_path}")
    ckpt_resolved = pathlib.Path(openpi_download.maybe_download(str(ckpt))).resolve()
    if ckpt_resolved != ckpt:
        warnings.warn(
            f"checkpoint 路径经 maybe_download 后与输入不同，请确认是否指向预期目录。",
            UserWarning,
            stacklevel=1,
        )

    train_config = _config.get_config(config_name)
    model_config = train_config.model

    # If the configured tokenizer asset (bin_edges.npy / codebook.npy / vq_params.npz)
    # is missing at the path in the config, restore it from the checkpoint's assets dir.
    # On the original training machine this is a no-op.
    _maybe_restore_tokenizer_assets_from_checkpoint(train_config, assets_root)

    data_train = train_config.data.create(train_config.assets_dirs, model_config)
    train_norm_key = data_train.asset_id or data_train.repo_id
    if not train_norm_key:
        raise ValueError("Could not determine asset_id / repo_id for norm stats.")
    norm_stats = _checkpoints.load_norm_stats(assets_root, train_norm_key)
    if norm_stats is None:
        raise FileNotFoundError(
            f"Missing norm stats under {assets_root / train_norm_key}. "
            "Checkpoint must contain training assets."
        )

    data_for_loader = dataclasses.replace(
        data_train,
        repo_id=eval_repo_id,
        norm_stats=norm_stats,
    )

    dataset = _data_loader.create_torch_dataset(
        data_for_loader,
        action_horizon=model_config.action_horizon,
        model_config=model_config,
    )
    n = len(dataset)
    indices = _resolve_sample_indices(sample_index, sample_range, n)
    multi_sample = len(indices) > 1

    sample_kwargs = (
        {"temperature": float(infer_temperature)} if infer_temperature is not None else {}
    )
    # Single policy creation across all samples (heavy step: compiling sample_actions).
    # In multi-sample mode we can only bake one rng into the policy; this only matters
    # when temperature > 0. Debug probe has its own fixed key so it stays deterministic.
    base_rng_seed = int(infer_rng_seed) & 0xFFFFFFFF
    first_idx = indices[0]
    policy_rng = jax.random.fold_in(jax.random.key(base_rng_seed), first_idx)
    policy = _policy_config.create_trained_policy(
        train_config,
        ckpt,
        repack_transforms=_transforms.Group(inputs=[]),
        sample_kwargs=sample_kwargs,
        rng=policy_rng,
    )

    # Codebook size (for summary header); best-effort, falls back to None.
    codebook_size: int | None = None
    try:
        _tok = model_config.fast_model_tokenizer
        _kw = model_config.fast_model_tokenizer_kwargs or {}
        if _tok is not None:
            _ti = _tok(model_config.max_token_len, **_kw)
            codebook_size = getattr(_ti, "_codebook_size", None) or getattr(_ti, "_n_bins", None)
            if codebook_size is not None:
                codebook_size = int(codebook_size)
    except Exception:  # noqa: BLE001
        codebook_size = None

    stats_list: list[dict] = []
    try:
        for idx in indices:
            row = dataset[idx]
            if print_image_paths and not multi_sample:
                _print_lerobot_ctx_rgb_disk_paths(
                    dataset, row, train_config=train_config, sample_index=idx
                )
            obs = _lerobot_sample_to_policy_observation(row, train_config=train_config)
            if debug_obs and not multi_sample:
                _print_debug_obs(sample_index=idx, obs=obs, dataset_row=_to_numpy_tree(row))
            obs_no_gt = {k: v for k, v in obs.items() if k != "actions"}

            oracle_stats: dict | None = None
            if debug_fast_decode:
                gt_actions_raw = obs.get("actions")
                oracle_stats = _debug_pi0_fast_decode_probe(
                    policy,
                    obs_no_gt,
                    model_config=model_config,
                    sample_kwargs=sample_kwargs,
                    actions_gt=np.asarray(gt_actions_raw) if gt_actions_raw is not None else None,
                    quiet=multi_sample,
                )

            result = policy.infer(obs_no_gt)

            pk = result.get("pick_dof")
            pl = result.get("place_dof")
            gt_dof = _gt_pick_place_dof9_from_sample(row)

            pick_dof_l2: float | None = None
            place_dof_l2: float | None = None
            pick_trans_l2: float | None = None
            place_trans_l2: float | None = None
            if gt_dof is not None and pk is not None and pl is not None:
                gtp, gtl = gt_dof
                pk9 = np.asarray(pk).reshape(-1)[:9]
                pl9 = np.asarray(pl).reshape(-1)[:9]
                pick_dof_l2 = float(np.linalg.norm(pk9 - gtp))
                place_dof_l2 = float(np.linalg.norm(pl9 - gtl))
                pick_trans_l2 = float(np.linalg.norm(pk9[:3] - gtp[:3]))
                place_trans_l2 = float(np.linalg.norm(pl9[:3] - gtl[:3]))

            sample_stats: dict = {
                "sample_idx": idx,
                "pick_dof_l2": pick_dof_l2,
                "place_dof_l2": place_dof_l2,
                "pick_trans_l2": pick_trans_l2,
                "place_trans_l2": place_trans_l2,
            }
            if oracle_stats is not None:
                sample_stats.update(oracle_stats)
            stats_list.append(sample_stats)

            # --- Printing ---
            if not multi_sample:
                # Single-sample mode: preserve original verbose prints.
                print("actions:", result.get("actions"))
                print("pick_dof (pred):", result.get("pick_dof"))
                print("place_dof (pred):", result.get("place_dof"))
                if gt_dof is not None:
                    gtp, gtl = gt_dof
                    print(
                        "gt_pick_dof (disk actions[:9]):",
                        np.array2string(gtp, precision=6, separator=", ", suppress_small=True),
                    )
                    print(
                        "gt_place_dof (disk actions[9:18]):",
                        np.array2string(gtl, precision=6, separator=", ", suppress_small=True),
                    )
                else:
                    print("gt_pick_dof / gt_place_dof: (无 ≥18 维 actions，跳过)")
            else:
                # Multi-sample mode: compact one-liner per sample.
                if oracle_stats is not None and oracle_stats.get("oracle_codes"):
                    oc = oracle_stats["oracle_codes"][0]
                    pc = oracle_stats["pred_codes"][0]
                    cos_list = oracle_stats.get("cos_sim") or []
                    cs = cos_list[0] if cos_list else None
                    cs_str = f"{cs:+.3f}" if cs is not None else "  n/a"
                    tag = "✓" if oc == pc and pc >= 0 else "✗"
                    omse = oracle_stats.get("oracle_recon_mse") or 0.0
                    pmse = oracle_stats.get("pred_recon_mse") or 0.0
                    pick_s = f"{pick_dof_l2:.3f}" if pick_dof_l2 is not None else "  n/a"
                    place_s = f"{place_dof_l2:.3f}" if place_dof_l2 is not None else "  n/a"
                    print(
                        f"sample={idx:3d}  oracle={oc:3d} pred={pc:3d} {tag}  "
                        f"cos={cs_str}  oracle_mse={omse:.4f} pred_mse={pmse:.4f}  "
                        f"pick_L2={pick_s}  place_L2={place_s}"
                    )
                else:
                    pick_s = f"{pick_dof_l2:.3f}" if pick_dof_l2 is not None else "  n/a"
                    place_s = f"{place_dof_l2:.3f}" if place_dof_l2 is not None else "  n/a"
                    print(f"sample={idx:3d}  pick_L2={pick_s}  place_L2={place_s}")

            # --- Visualization (per-sample file path) ---
            per_sample_vis = _vis_sample_output(vis_output, idx, multi_sample=multi_sample)
            if per_sample_vis is not None:
                rgb = np.asarray(obs["observation/ctx_rgb_00"])
                if pk is None or pl is None:
                    raise ValueError("vis_output 需要模型输出 pick_dof 与 place_dof；当前缺少其一。")
                pk9 = np.asarray(pk).reshape(-1)[:9]
                pl9 = np.asarray(pl).reshape(-1)[:9]
                path_pick, path_place = _vis_pick_place_paths(per_sample_vis)
                _ext = None if vis_dof_in_camera_frame else FRONT_CAMERA_EXTRINSICS

                do_compare = bool(vis_compare_gt_pred)
                gt_pair: tuple[np.ndarray, np.ndarray] | None = None
                if do_compare:
                    gt_pair = _gt_pick_place_dof9_from_sample(dataset[idx])
                    if gt_pair is None:
                        if not multi_sample:
                            warnings.warn(
                                "未从当前样本解析到 actions（≥18 维），回退为仅绘制 Predicted。",
                                UserWarning,
                                stacklevel=1,
                            )
                        do_compare = False

                if do_compare:
                    assert gt_pair is not None
                    gt_pk9, gt_pl9 = gt_pair
                    save_dof9_gt_pred_visualization(
                        rgb, gt_pk9, pk9, FRONT_CAMERA_INTRINSICS, path_pick,
                        dof_in_camera_frame=vis_dof_in_camera_frame,
                        extrinsics_4x4=_ext,
                        extrinsics_are_cam2world=vis_extrinsics_are_cam2world,
                        half_opening_m=vis_half_opening_m,
                        finger_extent_m=vis_finger_extent_m,
                        handle_length_m=vis_handle_length_m,
                    )
                    save_dof9_gt_pred_visualization(
                        rgb, gt_pl9, pl9, FRONT_CAMERA_INTRINSICS, path_place,
                        dof_in_camera_frame=vis_dof_in_camera_frame,
                        extrinsics_4x4=_ext,
                        extrinsics_are_cam2world=vis_extrinsics_are_cam2world,
                        half_opening_m=vis_half_opening_m,
                        finger_extent_m=vis_finger_extent_m,
                        handle_length_m=vis_handle_length_m,
                    )
                else:
                    save_dof9_pose_visualization(
                        rgb, pk9, FRONT_CAMERA_INTRINSICS, path_pick,
                        dof_in_camera_frame=vis_dof_in_camera_frame,
                        extrinsics_4x4=_ext,
                        extrinsics_are_cam2world=vis_extrinsics_are_cam2world,
                        half_opening_m=vis_half_opening_m,
                        finger_extent_m=vis_finger_extent_m,
                        handle_length_m=vis_handle_length_m,
                        tcp_color_bgr=(255, 255, 0),
                    )
                    save_dof9_pose_visualization(
                        rgb, pl9, FRONT_CAMERA_INTRINSICS, path_place,
                        dof_in_camera_frame=vis_dof_in_camera_frame,
                        extrinsics_4x4=_ext,
                        extrinsics_are_cam2world=vis_extrinsics_are_cam2world,
                        half_opening_m=vis_half_opening_m,
                        finger_extent_m=vis_finger_extent_m,
                        handle_length_m=vis_handle_length_m,
                        tcp_color_bgr=(255, 0, 255),
                    )
    finally:
        del policy

    if multi_sample:
        _summarize_match_rate(stats_list, codebook_size=codebook_size)


if __name__ == "__main__":
    tyro.cli(main)
