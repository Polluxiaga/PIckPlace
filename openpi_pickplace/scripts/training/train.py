import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform
import sys
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

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

from scripts.evaluation import eval as _checkpoint_eval
from scripts.evaluation import visualize as _checkpoint_vis


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


def _train_wandb_payload(metrics: dict[str, Any]) -> dict[str, float]:
    keep = ("param_norm", "loss", "grad_norm")
    return {key: float(metrics[key]) for key in keep if key in metrics}


def _resolve_best_checkpoint_metric(metrics: dict[str, Any], metric_name: str) -> float | None:
    candidates = [metric_name]
    if "/" not in metric_name:
        candidates.append(f"test/{metric_name}")
    metric_value = next((metrics[key] for key in candidates if key in metrics), None)
    if metric_value is None:
        return None
    return float(metric_value)


def _best_checkpoint_full_test_vis_enabled() -> bool:
    return os.environ.get("BEST_CHECKPOINT_FULL_TEST_VIS", "1").lower() not in {"0", "false", "no", "off"}


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


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
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
        retention_mode=config.checkpoint_retention,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    metrics_path = config.checkpoint_dir / "metrics.jsonl"
    if eval_on_checkpoint and not resuming:
        _checkpoint_eval.reset_eval_sweep_files(config)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            data_loader,
            state_sharding=train_state_sharding,
        )

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

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    eval_token_ce_history = _checkpoint_eval.load_eval_token_ce_history(config) if resuming else {}
    best_checkpoint = (
        _checkpoints.load_best_checkpoint_record(config.checkpoint_dir)
        if config.checkpoint_retention == "best_only"
        else None
    )
    if best_checkpoint is not None and best_checkpoint.step not in set(checkpoint_manager.all_steps()):
        logging.warning(
            "Ignoring stale best checkpoint metadata for missing step %d.",
            best_checkpoint.step,
        )
        best_checkpoint = None
    if config.checkpoint_retention == "best_only" and best_checkpoint is not None:
        _checkpoints.prune_checkpoints(checkpoint_manager, keep_steps={best_checkpoint.step})
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
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            numeric_log_payload.update({k: float(v) for k, v in reduced_info.items()})
            infos = []
        if numeric_log_payload:
            _write_local_metrics(metrics_path, step=step, metrics=numeric_log_payload)
            wandb.log(
                _train_wandb_payload(numeric_log_payload),
                step=step,
            )
        batch = next(data_iter)

        should_save = (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
        if should_save:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            best_metric_source: dict[str, Any] | None = None
            if eval_on_checkpoint:
                logging.info("Waiting for checkpoint %d to finish before eval", step)
                checkpoint_manager.wait_until_finished()
                eval_model = nnx.merge(train_state.model_def, train_state.params)
                eval_model.eval()
                best_metric_source = _checkpoint_eval.evaluate_and_log_checkpoint(
                    config,
                    step=step,
                    token_ce_history=eval_token_ce_history,
                    model=eval_model,
                )
            elif config.checkpoint_retention == "best_only":
                logging.info("Waiting for checkpoint %d to finish before best-checkpoint retention", step)
                checkpoint_manager.wait_until_finished()
                best_metric_source = numeric_log_payload

            if config.checkpoint_retention == "best_only":
                metric_value = None if best_metric_source is None else _resolve_best_checkpoint_metric(
                    best_metric_source,
                    config.best_checkpoint_metric,
                )
                if metric_value is None:
                    logging.warning(
                        "Best checkpoint retention skipped for step %d: metric %s not found.",
                        step,
                        config.best_checkpoint_metric,
                    )
                else:
                    is_best = _checkpoints.is_better_checkpoint(
                        metric_value,
                        best_checkpoint,
                        mode=config.best_checkpoint_mode,
                    )
                    if is_best:
                        best_checkpoint = _checkpoints.BestCheckpointRecord(
                            step=step,
                            metric=metric_value,
                            metric_key=config.best_checkpoint_metric,
                            mode=config.best_checkpoint_mode,
                        )
                        _checkpoints.save_best_checkpoint_record(config.checkpoint_dir, best_checkpoint)
                        logging.info(
                            "Checkpoint %d is the new best: %s=%.6f",
                            step,
                            config.best_checkpoint_metric,
                            metric_value,
                        )
                    else:
                        logging.info(
                            "Checkpoint %d is not better than current best step %d: %s=%.6f vs %.6f",
                            step,
                            best_checkpoint.step,
                            config.best_checkpoint_metric,
                            metric_value,
                            best_checkpoint.metric,
                        )

                    keep_steps = {best_checkpoint.step} if best_checkpoint is not None else {step}
                    _checkpoints.prune_checkpoints(checkpoint_manager, keep_steps=keep_steps)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    if config.checkpoint_retention == "best_only" and _best_checkpoint_full_test_vis_enabled():
        best_record = _checkpoints.load_best_checkpoint_record(config.checkpoint_dir)
        if best_record is None:
            logging.warning("Full test visualization skipped: no best checkpoint record found.")
        elif best_record.step not in set(checkpoint_manager.all_steps()):
            logging.warning(
                "Full test visualization skipped: best checkpoint step %d is not available.",
                best_record.step,
            )
        else:
            _checkpoint_vis.generate_full_test_vis_for_checkpoint(config, step=best_record.step)


if __name__ == "__main__":
    main(_config.cli())
