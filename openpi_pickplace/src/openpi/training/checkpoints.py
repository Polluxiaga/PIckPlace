from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
from typing import Protocol

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


BEST_CHECKPOINT_METADATA = "best_checkpoint.json"


@dataclasses.dataclass(frozen=True)
class BestCheckpointRecord:
    step: int
    metric: float
    metric_key: str
    mode: str


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    retention_mode: str = "latest_and_periodic",
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_to_keep = 1
    effective_keep_period = keep_period
    if retention_mode == "latest_only":
        effective_keep_period = None
    elif retention_mode == "best_only":
        max_to_keep = None
        effective_keep_period = None

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=effective_keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
            # Copy tokenizer-specific asset files (bin_edges.npy / codebook.npy) if present.
            import numpy as np
            for tok_transform in data_config.model_transforms.inputs:
                tok_obj = getattr(tok_transform, "_tokenizer", None)
                if tok_obj is None:
                    continue
                bin_edges = getattr(tok_obj, "_bin_edges", None)
                if bin_edges is not None:
                    dst = epath.Path(directory / data_config.asset_id / "bin_edges.npy")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(dst), bin_edges)
                    logging.info("Saved bin_edges to %s", dst)
                codebook = getattr(tok_obj, "_codebook", None)
                if codebook is not None:
                    dst = epath.Path(directory / data_config.asset_id / "codebook.npy")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(dst), codebook)
                    logging.info("Saved codebook to %s", dst)
                vq_params_path = getattr(tok_obj, "_vq_params_path", None)
                if vq_params_path is not None and epath.Path(vq_params_path).exists():
                    import shutil
                    dst = epath.Path(directory / data_config.asset_id / "vq_params.npz")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(vq_params_path), str(dst))
                    logging.info("Saved vq_params to %s", dst)
                vq_params_paths = getattr(tok_obj, "_vq_params_paths", None)
                if vq_params_paths is not None:
                    import shutil
                    for filename, src in vq_params_paths.items():
                        if src is not None and epath.Path(src).exists():
                            dst = epath.Path(directory / data_config.asset_id / filename)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(str(src), str(dst))
                            logging.info("Saved %s to %s", filename, dst)
                if bin_edges is not None or codebook is not None or vq_params_path is not None or vq_params_paths is not None:
                    break

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
    *,
    state_sharding=None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        if state_sharding is None:
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
        else:
            train_state_sharding, params_sharding = _split_params(state_sharding)
            logging.info("Restoring checkpoint with explicit target shardings.")
            restored = checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    train_state=ocp.args.PyTreeRestore(
                        item=train_state,
                        restore_args=_build_restore_args(train_state, train_state_sharding),
                    ),
                    params=ocp.args.PyTreeRestore(
                        item={"params": params},
                        restore_args={"params": _build_restore_args(params, params_sharding)},
                    ),
                ),
            )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


def load_best_checkpoint_record(checkpoint_dir: epath.Path | str) -> BestCheckpointRecord | None:
    path = epath.Path(checkpoint_dir) / BEST_CHECKPOINT_METADATA
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        logging.warning("Ignoring malformed best checkpoint metadata at %s", path)
        return None
    try:
        return BestCheckpointRecord(
            step=int(payload["step"]),
            metric=float(payload["metric"]),
            metric_key=str(payload["metric_key"]),
            mode=str(payload["mode"]),
        )
    except (KeyError, TypeError, ValueError):
        logging.warning("Ignoring incomplete best checkpoint metadata at %s", path)
        return None


def save_best_checkpoint_record(checkpoint_dir: epath.Path | str, record: BestCheckpointRecord) -> None:
    path = epath.Path(checkpoint_dir) / BEST_CHECKPOINT_METADATA
    path.write_text(json.dumps(dataclasses.asdict(record), indent=2, sort_keys=True) + "\n")


def is_better_checkpoint(metric: float, best: BestCheckpointRecord | None, *, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return metric < best.metric
    if mode == "max":
        return metric > best.metric
    raise ValueError(f"Unsupported best checkpoint mode: {mode}")


def prune_checkpoints(checkpoint_manager: ocp.CheckpointManager, *, keep_steps: set[int]) -> None:
    for saved_step in tuple(checkpoint_manager.all_steps()):
        if saved_step not in keep_steps:
            checkpoint_manager.delete(saved_step)


def _build_restore_args(item, shardings):
    def _restore_arg(leaf, leaf_sharding):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=leaf.dtype,
                sharding=leaf.sharding or leaf_sharding,
                global_shape=leaf.shape,
            )
        if isinstance(leaf, jax.Array):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=leaf.dtype,
                sharding=leaf.sharding,
                global_shape=leaf.shape,
            )
        if isinstance(leaf, np.ndarray):
            return ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=leaf.dtype, global_shape=leaf.shape)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=leaf.dtype,
                sharding=leaf_sharding,
                global_shape=leaf.shape,
            )
        return None

    return jax.tree.map(_restore_arg, item, shardings)
