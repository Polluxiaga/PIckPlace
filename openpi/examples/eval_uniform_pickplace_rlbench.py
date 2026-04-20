"""
Evaluate a trained **pi0-FAST** RLBench pick+place checkpoint on a LeRobot dataset using **token cross-entropy**
(same objective as ``scripts/train.py`` / ``Pi0FAST.compute_loss_per_token`` (masked mean) with ``train=False``).

Optionally decodes predicted action tokens with the FAST tokenizer and reports **MSE** vs ground-truth actions
(both in **normalized** space, matching the dataloader after ``Normalize``).

**Norm stats:** Always loaded from the checkpoint under ``<checkpoint>/assets/<train_repo_id>/`` (same as training).
The **eval** dataset only needs a different ``--eval-repo-id`` and must live under ``HF_LEROBOT_HOME`` like training.

**Example:**

.. code-block:: bash

    export HF_LEROBOT_HOME=/mnt/nas/minyangli   # parent of minyangli/<repo_id>
    export CUDA_VISIBLE_DEVICES=0

    uv run examples/eval_uniform_pickplace_rlbench.py \\
      --config-name pi0_fast_rlbench_pickplace_rand1_lora \\
      --checkpoint-dir /root/minyangli/openpi/checkpoints/pi0_fast_rlbench_pickplace_rand1_lora/my_experiment/5000 \\
      --eval-repo-id minyangli/place_wine_rlbench_v2_eval
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import logging
import os
import pathlib

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fast_tokenizer_from_config(
    model_config: _model.BaseModelConfig,
    *,
    checkpoint_assets_dir: pathlib.Path | str | None = None,
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
        if key in kwargs and not os.path.exists(kwargs[key]):
            if checkpoint_assets_dir and asset_id:
                alt = os.path.join(str(checkpoint_assets_dir), asset_id, fname)
                if os.path.exists(alt):
                    logger.info("%s fallback: using checkpoint assets %s", key, alt)
                    kwargs[key] = alt
    return tokenizer_cls(model_config.max_token_len, **kwargs)


def _pre_tokenize_model_transforms(model_config: _model.BaseModelConfig) -> _transforms.Group:
    """Same as ``ModelTransformFactory`` for PI0_FAST but without ``TokenizeFASTInputs`` (applied in-loop)."""
    return _transforms.Group(
        inputs=[
            _transforms.InjectDefaultPrompt(None),
            _transforms.ResizeImages(224, 224),
        ],
    )


def _batch_copy(batch: dict) -> dict:
    """Shallow copy with numpy array copies so ``TokenizeFASTInputs`` can pop ``prompt`` twice."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = np.array(v, copy=True)
        else:
            out[k] = copy.copy(v) if hasattr(v, "copy") else v
    return out


def _take_batch_index(x, *, index: int, bsz: int):
    """Slice batch dimension 0 for collated numpy leaves (matches training: tokenize per sample, then stack)."""
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == bsz:
        return np.asarray(x[index])
    return x


def _tokenize_collated_batch(
    batch_pre: dict,
    tokenize: _transforms.TokenizeFASTInputs,
    *,
    include_actions: bool,
) -> dict:
    """Apply ``TokenizeFASTInputs`` per example (prompt is str per row), then stack like the training collate."""
    bsz = int(batch_pre["state"].shape[0])
    outs: list[dict] = []
    for i in range(bsz):
        s = jax.tree.map(functools.partial(_take_batch_index, index=i, bsz=bsz), batch_pre)
        if not include_actions:
            s.pop("actions", None)
        p = s.get("prompt")
        if p is not None and not isinstance(p, str):
            s["prompt"] = str(p)
        outs.append(tokenize(_batch_copy(s)))
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *outs)


def main(
    *,
    config_name: str = "pi0_fast_rlbench_pickplace_rand1_lora",
    checkpoint_dir: pathlib.Path,
    eval_repo_id: str,
    batch_size: int = 8,
    seed: int = 0,
    compute_action_mse: bool = True,
) -> None:
    """Run CE eval on an eval LeRobot repo; optionally MSE(pred_actions, gt) after tokenizer decode.

    Args:
        config_name: Registered ``TrainConfig`` name (must match how the checkpoint was trained).
        checkpoint_dir: Step directory containing ``params/`` and ``assets/`` (e.g. ``.../my_experiment/5000``).
        eval_repo_id: LeRobot dataset id for **evaluation** (under ``HF_LEROBOT_HOME``).
        batch_size: Per-device batch size (must be <= dataset size; remainder batches are dropped).
        seed: RNG seed for ``compute_loss_per_token`` / ``sample_actions`` (data order is not shuffled).
        compute_action_mse: If True, run autoregressive decode + ``ExtractFASTActions`` and MSE vs GT actions.
    """
    ckpt = epath.Path(checkpoint_dir).expanduser().resolve()
    params_path = ckpt / "params"
    assets_root = ckpt / "assets"
    if not params_path.is_dir():
        raise FileNotFoundError(f"Missing params directory: {params_path}")

    train_config = _config.get_config(config_name)
    model_config = train_config.model

    # Build DataConfig like training, then point loader at eval repo and norms at checkpoint.
    data_config_train = train_config.data.create(train_config.assets_dirs, model_config)
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

    action_q01 = np.asarray(norm_stats["actions"].q01, dtype=np.float64)
    action_q99 = np.asarray(norm_stats["actions"].q99, dtype=np.float64)

    def _unnorm_actions(x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float64) + 1.0) / 2.0 * (action_q99 - action_q01 + 1e-6) + action_q01
    logger.info(
        "Eval samples: repo_id=%s (under HF_LEROBOT_HOME=%s)",
        eval_repo_id,
        os.environ.get("HF_LEROBOT_HOME", "(default ~/.cache/huggingface/lerobot)"),
    )

    # Loader stops before TokenizeFASTInputs; we tokenize in-loop (with/without GT actions).
    data_config_pre = dataclasses.replace(
        data_config,
        model_transforms=_pre_tokenize_model_transforms(model_config),
    )

    dataset = _data_loader.create_torch_dataset(
        data_config_pre,
        action_horizon=model_config.action_horizon,
        model_config=model_config,
    )
    dataset = _data_loader.transform_dataset(dataset, data_config_pre)
    n = len(dataset)
    if n < batch_size:
        raise ValueError(f"Dataset size {n} is smaller than batch_size {batch_size}.")

    local_batch_size = batch_size // jax.process_count()
    num_batches = n // local_batch_size
    if num_batches == 0:
        raise ValueError("No full batches; reduce batch_size.")

    # Do not iterate ``TorchDataLoader.__iter__`` with JAX: batched ``prompt`` is numpy unicode (<U*) and cannot
    # be device_put. Use the inner PyTorch loader for numpy batches; tokenize per sample then stack (as in training).
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
        model_config,
        checkpoint_assets_dir=ckpt / "assets",
        asset_id=data_config.asset_id,
    )
    tokenize = _transforms.TokenizeFASTInputs(tok)
    extract = _transforms.ExtractFASTActions(
        tok,
        action_horizon=model_config.action_horizon,
        action_dim=model_config.action_dim,
    )

    logger.info("Loaded checkpoint params from %s", params_path)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = train_config.model.load(params)

    rng = jax.random.key(seed)
    total_loss = 0.0
    total_mse = 0.0
    per_pos_ce_accum: np.ndarray | None = None
    per_pos_ce_count = 0
    per_pos_mse_accum: np.ndarray | None = None
    per_pos_mse_count = 0
    printed_examples = 0
    max_print_examples = 3

    for batch_idx, batch_pre in enumerate(torch_loader.torch_loader):
        if batch_idx >= num_batches:
            break
        # CE: same tokens as training (prefix + action postfix).
        batch_ce = _tokenize_collated_batch(batch_pre, tokenize, include_actions=True)
        observation = _model.Observation.from_dict(batch_ce)
        actions = batch_ce["actions"]
        rng, step_rng = jax.random.split(rng)
        token_nll, loss_mask = model.compute_loss_per_token(step_rng, observation, actions, train=False)
        tn = np.asarray(jax.device_get(token_nll))
        lm = np.asarray(jax.device_get(loss_mask))
        masked_nll = tn * lm
        per_example_loss = np.sum(masked_nll, axis=-1) / np.clip(np.sum(lm, axis=-1), 1, None)
        step_loss = float(np.mean(per_example_loss))
        total_loss += step_loss
        for b in range(tn.shape[0]):
            b_idxs = np.where(lm[b])[0]
            b_nll = tn[b, b_idxs].astype(np.float64)
            if per_pos_ce_accum is None:
                per_pos_ce_accum = b_nll.copy()
            else:
                per_pos_ce_accum += b_nll
            per_pos_ce_count += 1

        if compute_action_mse:
            # Inference: prefix only, then decode tokens -> continuous actions; MSE vs normalized GT.
            batch_inf = _tokenize_collated_batch(batch_pre, tokenize, include_actions=False)
            obs_inf = _model.Observation.from_dict(batch_inf)
            rng, sample_rng = jax.random.split(rng)
            token_ids = model.sample_actions(sample_rng, obs_inf)
            # token_ids = model.sample_actions(sample_rng, obs_inf, max_decoding_steps=22)
            token_np = np.asarray(jax.device_get(token_ids)).astype(np.int32)
            # UniformBinningPickPlaceTokenizer.extract_actions is single-sequence; batch over B.
            pred_chunks = [
                extract({"actions": token_np[i]})["actions"] for i in range(token_np.shape[0])
            ]
            pred_actions = np.stack(pred_chunks, axis=0)
            gt_actions = np.asarray(batch_pre["actions"])

            flat_pred_norm = pred_actions.reshape(pred_actions.shape[0], -1)
            flat_gt_norm = gt_actions.reshape(gt_actions.shape[0], -1)
            flat_pred_raw = _unnorm_actions(flat_pred_norm)
            flat_gt_raw = _unnorm_actions(flat_gt_norm)
            flat_sq = (flat_pred_raw - flat_gt_raw) ** 2
            step_mse = float(np.mean(flat_sq))
            total_mse += step_mse

            if per_pos_mse_accum is None:
                per_pos_mse_accum = np.sum(flat_sq, axis=0)
            else:
                per_pos_mse_accum += np.sum(flat_sq, axis=0)
            per_pos_mse_count += flat_sq.shape[0]

            if printed_examples < max_print_examples:
                n_print = min(max_print_examples - printed_examples, flat_pred_raw.shape[0])
                for b in range(n_print):
                    idx_global = batch_idx * local_batch_size + b
                    print(f"\n--- Example {idx_global} (autoregressive, unnormalized) ---")
                    print(f"  pred: {np.array2string(flat_pred_raw[b], precision=4, suppress_small=True, max_line_width=200)}")
                    print(f"  gt  : {np.array2string(flat_gt_raw[b], precision=4, suppress_small=True, max_line_width=200)}")
                    print(f"  err²: {np.array2string(flat_sq[b], precision=4, suppress_small=True, max_line_width=200)}")
                printed_examples += n_print

        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == num_batches:
            msg = (
                f"batch {batch_idx + 1} / {num_batches}  batch_mean_ce={step_loss:.6f}  running_mean_ce={total_loss / (batch_idx + 1):.6f}"
            )
            if compute_action_mse:
                msg += f"  batch_mse={step_mse:.6f}  running_mean_mse={total_mse / (batch_idx + 1):.6f}"
            logger.info(msg)

    mean_ce = total_loss / max(num_batches, 1)
    logger.info(
        "Done. eval_repo_id=%s  num_examples_used=%d  mean_ce=%.6f",
        eval_repo_id,
        num_batches * local_batch_size,
        mean_ce,
    )
    print(f"mean_cross_entropy={mean_ce:.6f}  (over {num_batches} batches, batch_size={local_batch_size})")
    if compute_action_mse:
        mean_mse = total_mse / max(num_batches, 1)
        logger.info("mean_action_mse (unnormalized/real space)=%.6f", mean_mse)
        print(f"mean_action_mse_unnormalized={mean_mse:.6f}")

    if per_pos_ce_accum is not None and per_pos_ce_count > 0:
        mean_per_pos = per_pos_ce_accum / per_pos_ce_count
        model.print_mean_token_cross_entropy(mean_per_pos, num_examples=per_pos_ce_count)

    if per_pos_mse_accum is not None and per_pos_mse_count > 0:
        mean_pos_mse = per_pos_mse_accum / per_pos_mse_count
        n_action = int(model.action_horizon * model.action_dim)
        pick_dim = n_action // 2
        print(f"\n--- Per-dimension mean MSE (autoregressive, {per_pos_mse_count} examples) ---")
        for i, v in enumerate(mean_pos_mse):
            if i < pick_dim:
                label = f"pick[{i}]"
            else:
                label = f"place[{i - pick_dim}]"
            print(f"  dim {i:>2d} ({label:>10s}): MSE = {v:.6f}")
        print(f"  mean over all dims: {float(np.mean(mean_pos_mse)):.6f}")


if __name__ == "__main__":
    tyro.cli(main)
