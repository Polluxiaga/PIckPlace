"""Train phase VQ-VAE assets for ``PhaseVQVAEActionTokenizer``.

The produced tokenizer keeps two action tokens per sample:

* pick 9D -> pick VQ-VAE code
* place 9D -> place VQ-VAE code

Run after norm stats are available. The default config writes:
``assets/pickplace_all_phase_vqvae128/pick_place_all/{pick,place}_vq_params.npz``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _init_hf_env() -> None:
    os.environ.setdefault("HF_LEROBOT_HOME", "/mnt/nas/minyangli")
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


_init_hf_env()

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tyro

import openpi.models.action_vq as _action_vq
import openpi.policies.rlbench_policy as _rlbench_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


class _RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _collect_normalized_actions(config_name: str, max_frames: int | None) -> np.ndarray:
    config = _config.get_config(config_name)
    data_config_factory = config.data
    base = data_config_factory.create_base_config(config.assets_dirs, config.model)
    if base.norm_stats is None:
        raise RuntimeError("norm_stats not found. Reuse or compute norm stats before training VQ-VAE assets.")

    num_ctx = getattr(data_config_factory, "num_ctx_frames", 1)
    data_config_no_tok = dataclasses.replace(
        base,
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform(_config._rlbench_pick_place_repack(num_ctx))]
        ),
        data_transforms=_transforms.Group(
            inputs=[
                _rlbench_policy.RLBenchPickPlaceInputs(
                    model_type=config.model.model_type,
                    num_ctx_frames=num_ctx,
                ),
            ],
        ),
        model_transforms=_transforms.Group(),
    )

    dataset = _data_loader.create_torch_dataset(
        data_config_no_tok,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config_no_tok.repack_transforms.inputs,
            *data_config_no_tok.data_transforms.inputs,
            _transforms.Normalize(data_config_no_tok.norm_stats, use_quantiles=data_config_no_tok.use_quantile_norm),
            _RemoveStrings(),
        ],
    )

    batch_size = min(config.batch_size, len(dataset))
    num_batches = len(dataset) // batch_size
    if max_frames is not None:
        num_batches = min(num_batches, max_frames // batch_size)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        num_batches=num_batches,
    )

    actions: list[np.ndarray] = []
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Collecting normalized actions"):
        a = np.asarray(batch["actions"])
        actions.append(a.reshape(-1, a.shape[-1]))
    return np.concatenate(actions, axis=0)


def _train_one_phase(
    name: str,
    actions: np.ndarray,
    *,
    output_path: pathlib.Path,
    codebook_size: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    n_steps: int,
    batch_size: int,
    lr: float,
    commit_beta: float,
    translation_weight: float,
    rotation_weight: float,
    log_every: int,
    seed: int,
    l2_normalize: bool,
    dead_reset_every: int,
    dead_reset_threshold: int,
) -> None:
    rngs = nnx.Rngs(params=jax.random.key(seed))
    model = _action_vq.JointVQVAE(
        action_dim=actions.shape[-1],
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        hidden_dims=tuple(hidden_dims),
        l2_normalize=l2_normalize,
        rngs=rngs,
    )
    model.init_codebook_from_actions(actions, seed=seed)
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    dim_weights = jnp.asarray([translation_weight] * 3 + [rotation_weight] * (actions.shape[-1] - 3), dtype=jnp.float32)

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            out = model(batch, dim_weights=dim_weights)
            loss = out["recon_loss"] + out["codebook_loss"] + commit_beta * out["commit_loss"]
            return loss, out

        (loss, out), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(grads)
        return loss, out

    rng = np.random.default_rng(seed)
    usage_window = np.zeros(codebook_size, dtype=np.int64)
    usage_total = np.zeros(codebook_size, dtype=np.int64)
    replace = batch_size > len(actions)
    print(
        f"\nTraining {name} phase VQ-VAE: K={codebook_size}, latent_dim={latent_dim}, "
        f"steps={n_steps}, batch={batch_size}, translation_weight={translation_weight}"
    )
    for step in range(n_steps):
        idx = rng.choice(len(actions), size=batch_size, replace=replace)
        batch = jnp.asarray(actions[idx], dtype=jnp.float32)
        loss, out = train_step(model, optimizer, batch)
        codes = np.asarray(out["code_idx"])
        np.add.at(usage_window, codes, 1)
        np.add.at(usage_total, codes, 1)
        if step % log_every == 0 or step == n_steps - 1:
            print(
                f"{name} step {step:>6d}: loss={float(loss):.5f} "
                f"recon={float(out['recon_loss']):.5f} active={(usage_total > 0).sum()}/{codebook_size}"
            )
        if dead_reset_every > 0 and (step + 1) % dead_reset_every == 0 and step < n_steps - 1:
            reset = model.reset_dead_codes(
                usage_window,
                actions,
                seed=seed + step + 1,
                threshold=dead_reset_threshold,
            )
            if reset:
                print(f"{name} dead-code reset @ {step + 1}: {reset}/{codebook_size}")
            usage_window[:] = 0

    @nnx.jit
    def eval_step(model, batch):
        return model(batch, dim_weights=dim_weights)

    sq_err: list[np.ndarray] = []
    usage = np.zeros(codebook_size, dtype=np.int64)
    for start in range(0, len(actions), 4096):
        batch = jnp.asarray(actions[start : start + 4096], dtype=jnp.float32)
        out = eval_step(model, batch)
        pred = np.asarray(out["action_hat"])
        sq_err.append((np.asarray(batch) - pred) ** 2)
        np.add.at(usage, np.asarray(out["code_idx"]), 1)
    err = np.concatenate(sq_err, axis=0)
    print(f"{name} final mse={float(err.mean()):.6f}, trans_mse={float(err[:, :3].mean()):.6f}, active={(usage > 0).sum()}/{codebook_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _action_vq.save_vq_params(model, str(output_path))
    print(f"Saved {name} VQ params to {output_path}")


def main(
    config_name: str = "pickplace_all_phase_vqvae128",
    codebook_size: int = 128,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] = (64, 128),
    n_steps: int = 10_000,
    batch_size: int = 64,
    lr: float = 3e-4,
    commit_beta: float = 0.25,
    translation_weight: float = 2.0,
    rotation_weight: float = 1.0,
    log_every: int = 500,
    seed: int = 0,
    max_frames: int | None = None,
    pick_dim: int = 9,
    place_dim: int = 9,
    l2_normalize: bool = True,
    dead_reset_every: int = 500,
    dead_reset_threshold: int = 0,
) -> None:
    config = _config.get_config(config_name)
    actions = _collect_normalized_actions(config_name, max_frames)
    expected_dim = pick_dim + place_dim
    if actions.shape[-1] != expected_dim:
        raise ValueError(f"expected {expected_dim}D actions, got {actions.shape[-1]}")

    base = config.data.create_base_config(config.assets_dirs, config.model)
    output_dir = pathlib.Path(config.assets_dirs) / (base.asset_id or base.repo_id)
    _train_one_phase(
        "pick",
        actions[:, :pick_dim],
        output_path=output_dir / "pick_vq_params.npz",
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        commit_beta=commit_beta,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
        log_every=log_every,
        seed=seed,
        l2_normalize=l2_normalize,
        dead_reset_every=dead_reset_every,
        dead_reset_threshold=dead_reset_threshold,
    )
    _train_one_phase(
        "place",
        actions[:, pick_dim:expected_dim],
        output_path=output_dir / "place_vq_params.npz",
        codebook_size=codebook_size,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        commit_beta=commit_beta,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
        log_every=log_every,
        seed=seed + 1,
        l2_normalize=l2_normalize,
        dead_reset_every=dead_reset_every,
        dead_reset_threshold=dead_reset_threshold,
    )


if __name__ == "__main__":
    tyro.cli(main)
