"""Compute scalar per-dimension VQ codebooks for ``VQActionTokenizer``.

This script runs a simple 1-D K-means independently on each normalized action
dimension and saves the resulting centers as ``codebook.npy`` with shape
``[action_dim, codebook_size]``.

Run ``scripts/data/compute_norm_stats.py`` first. The VQ config introduced for
pick-place reuses the existing normalization stats from ``pickplace_all_qbin64``,
so you usually only need to compute the codebook asset.

Example:

.. code-block:: bash

    uv run scripts/tokenizer/compute_codebook.py \
      --config-name pickplace_all_vq64 \
      --codebook-size 64
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _init_hf_env() -> None:
    """Mirror the training launcher defaults before importing lerobot/openpi."""
    os.environ.setdefault("HF_LEROBOT_HOME", "/mnt/nas/minyangli")
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


_init_hf_env()

import numpy as np
import tqdm
import tyro

import openpi.policies.rlbench_policy as _rlbench_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


class _RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _run_1d_kmeans(
    values: np.ndarray,
    *,
    k: int,
    max_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lloyd's algorithm in 1-D with quantile initialization."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot fit codebook on an empty array")

    unique = np.unique(x)
    if unique.size <= k:
        padded = np.pad(unique, (0, k - unique.size), mode="edge")
        return np.sort(padded.astype(np.float32))

    init_q = np.linspace(0.0, 1.0, k + 2, dtype=np.float64)[1:-1]
    centers = np.quantile(x, init_q)
    centers = np.asarray(centers, dtype=np.float64)

    # Break ties if repeated quantiles appear for low-variance dimensions.
    if np.unique(centers).size < k:
        jitter = rng.normal(scale=1e-6, size=k)
        centers = centers + jitter

    for _ in range(max_iters):
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        assignments = np.searchsorted(boundaries, x, side="right")
        new_centers = centers.copy()
        for idx in range(k):
            mask = assignments == idx
            if np.any(mask):
                new_centers[idx] = x[mask].mean()
        new_centers.sort()
        if np.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
            centers = new_centers
            break
        centers = new_centers

    return centers.astype(np.float32)


def main(
    config_name: str,
    codebook_size: int = 64,
    max_frames: int | None = None,
    max_iters: int = 50,
    seed: int = 42,
) -> None:
    config = _config.get_config(config_name)

    data_config_factory = config.data
    base = data_config_factory.create_base_config(config.assets_dirs, config.model)
    if base.norm_stats is None:
        raise RuntimeError("norm_stats not found. Run scripts/data/compute_norm_stats.py first.")

    num_ctx = getattr(data_config_factory, "num_ctx_frames", 1)
    data_config_no_tok = dataclasses.replace(
        base,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(_config._rlbench_pick_place_repack(num_ctx)),
            ]
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
            _transforms.Normalize(
                data_config_no_tok.norm_stats,
                use_quantiles=data_config_no_tok.use_quantile_norm,
            ),
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

    all_actions: list[np.ndarray] = []
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Collecting normalized actions"):
        a = np.asarray(batch["actions"])
        all_actions.append(a.reshape(-1, a.shape[-1]))

    all_actions_arr = np.concatenate(all_actions, axis=0)
    action_dim = all_actions_arr.shape[1]
    print(f"Collected {all_actions_arr.shape[0]} samples x {action_dim} dims")

    rng = np.random.default_rng(seed)
    codebook = np.zeros((action_dim, codebook_size), dtype=np.float32)
    for d in range(action_dim):
        codebook[d] = _run_1d_kmeans(
            all_actions_arr[:, d],
            k=codebook_size,
            max_iters=max_iters,
            rng=rng,
        )

    output_dir = config.assets_dirs / (data_config_no_tok.asset_id or data_config_no_tok.repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "codebook.npy"
    np.save(str(output_path), codebook)
    print(f"Saved codebook shape={codebook.shape} to {output_path}")

    for d in range(min(action_dim, 6)):
        print(
            f"  dim {d}: center_min={codebook[d, 0]:.6f} "
            f"center_max={codebook[d, -1]:.6f}"
        )


if __name__ == "__main__":
    tyro.cli(main)
