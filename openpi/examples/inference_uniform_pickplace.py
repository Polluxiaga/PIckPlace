"""
Inference with ``pi0_fast_pickplace_uniform`` (UniformBinningPickPlaceTokenizer, 9+9 DOF, horizon=1).

**Smoke run (default):** if ``OPENPI_CHECKPOINT_DIR`` is unset, uses the public ``pi0_fast_droid`` checkpoint
so the pipeline runs end-to-end. The VLM weights load, but **actions are not meaningful** (different tokenizer /
action_dim vs training). Use this only to verify code paths; finetune with this TrainConfig for real behavior.

**Your checkpoint:** ``export OPENPI_CHECKPOINT_DIR=/path/to/checkpoint`` then run again.

Usage::

    cd openpi && uv run examples/inference_uniform_pickplace.py
    # or
    export OPENPI_CHECKPOINT_DIR=/path/to/your/trained/checkpoint
    uv run examples/inference_uniform_pickplace.py
"""

import os
import warnings

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

CONFIG_NAME = "pi0_fast_pickplace_uniform"
# Public JAX checkpoint (same backbone as pi0_fast_droid; OK for load + smoke inference only).
_DEFAULT_SMOKE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_fast_droid"


def main() -> None:
    checkpoint_dir = os.environ.get("OPENPI_CHECKPOINT_DIR", _DEFAULT_SMOKE_CHECKPOINT)
    if checkpoint_dir == _DEFAULT_SMOKE_CHECKPOINT:
        warnings.warn(
            "Using pi0_fast_droid weights only to smoke-test the pipeline. "
            "pick_dof / place_dof / actions are NOT semantically valid until you finetune with "
            "pi0_fast_pickplace_uniform.",
            UserWarning,
            stacklevel=1,
        )

    config = _config.get_config(CONFIG_NAME)
    checkpoint_dir = download.maybe_download(checkpoint_dir)

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    example = droid_policy.make_droid_example()
    result = policy.infer(example)
    del policy

    print("actions:", result.get("actions"))
    print("pick_dof:", result.get("pick_dof"))
    print("place_dof:", result.get("place_dof"))
    if result.get("actions") is not None:
        print("actions shape:", result["actions"].shape)


if __name__ == "__main__":
    main()
