"""Policy transforms for RLBench → LeRobot pick+place (one or many ``ctx_rgb_*`` views + 18-DOF actions)."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RLBenchPickPlaceInputs(transforms.DataTransformFn):
    """Maps LeRobot ``ctx_rgb_00`` … ``ctx_rgb_{N-1}`` + ``state`` + ``prompt`` into the FAST ``image`` dict.

    ``N`` must match the dataset (``num_ctx_frames`` in :class:`openpi.training.config.LeRobotRLBenchPickPlaceDataConfig`).
    Single-image datasets use ``N=1`` (only ``ctx_rgb_00``).
    """

    model_type: _model.ModelType
    num_ctx_frames: int = 1

    def __call__(self, data: dict) -> dict:
        if self.model_type != _model.ModelType.PI0_FAST:
            raise ValueError("RLBenchPickPlaceInputs only supports PI0_FAST.")

        state = np.asarray(data["observation/state"], dtype=np.float32)
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, bool | np.bool_] = {}
        for i in range(self.num_ctx_frames):
            name = f"ctx_rgb_{i:02d}"
            key = f"observation/{name}"
            images[name] = _parse_image(data[key])
            image_masks[name] = np.True_

        out: dict = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            p = data["prompt"]
            if isinstance(p, bytes):
                p = p.decode("utf-8")
            out["prompt"] = p
        return out


@dataclasses.dataclass(frozen=True)
class RLBenchPickPlaceOutputs(transforms.DataTransformFn):
    """Return the first 18 action dimensions (9 pick + 9 place)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :18])}
