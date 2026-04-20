"""Compute quantile bin edges for QuantileBinningPickPlaceTokenizer.

Run this AFTER ``compute_norm_stats.py`` (needs norm_stats for quantile normalization).

Saves ``bin_edges.npy`` (shape ``[action_dim, n_bins + 1]``) to the config assets directory,
alongside the existing ``norm_stats.json``.

Example:

.. code-block:: bash

    uv run scripts/compute_bin_edges.py \
      --config-name pi0_fast_rlbench_pickplace_rand1_lora_cam_qbin128 \
      --n-bins 128
"""

from __future__ import annotations

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.policies.rlbench_policy as _rlbench_policy
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


class _RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main(config_name: str, n_bins: int = 128, max_frames: int | None = None) -> None:
    config = _config.get_config(config_name)

    # Build a data config WITHOUT model_transforms (which would try to load bin_edges).
    data_config_factory = config.data
    base = data_config_factory.create_base_config(config.assets_dirs, config.model)
    if base.norm_stats is None:
        raise RuntimeError(
            "norm_stats not found. Run scripts/compute_norm_stats.py first."
        )

    # Reconstruct just the repack + data transforms (no tokenizer).
    num_ctx = getattr(data_config_factory, "num_ctx_frames", 1)
    data_config_no_tok = dataclasses.replace(
        base,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    _config._rlbench_pick_place_repack(num_ctx)
                ),
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
    print(f"Collected {all_actions_arr.shape[0]} samples × {action_dim} dims")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.zeros((action_dim, n_bins + 1), dtype=np.float64)
    for d in range(action_dim):
        bin_edges[d] = np.quantile(all_actions_arr[:, d], quantiles)

    for d in range(action_dim):
        unique_edges = np.unique(bin_edges[d])
        if len(unique_edges) < n_bins + 1:
            print(
                f"  WARNING: dim {d} has only {len(unique_edges)} unique edges "
                f"(data may be near-constant). Adding tiny jitter."
            )
            bin_edges[d] = np.linspace(
                bin_edges[d, 0] - 1e-9, bin_edges[d, -1] + 1e-9, n_bins + 1
            )

    output_dir = config.assets_dirs / (data_config_no_tok.asset_id or data_config_no_tok.repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bin_edges.npy"
    np.save(str(output_path), bin_edges)
    print(f"Saved bin_edges shape={bin_edges.shape} to {output_path}")

    for d in range(min(action_dim, 6)):
        print(
            f"  dim {d}: edges[0]={bin_edges[d, 0]:.6f}  edges[-1]={bin_edges[d, -1]:.6f}  "
            f"range={bin_edges[d, -1] - bin_edges[d, 0]:.6f}"
        )


if __name__ == "__main__":
    tyro.cli(main)
