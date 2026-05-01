"""Small VQ-VAE modules used by action tokenizers."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


class _MLP(nnx.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, *, rngs: nnx.Rngs):
        dims = [in_dim, *hidden_dims, out_dim]
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.gelu(x)
        return x


class JointVQVAE(nnx.Module):
    """MLP VQ-VAE for a single action vector."""

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 64,
        codebook_size: int = 128,
        hidden_dims: tuple[int, ...] = (64, 128),
        *,
        l2_normalize: bool = True,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.hidden_dims = tuple(hidden_dims)
        self.l2_normalize = bool(l2_normalize)
        self.encoder = _MLP(action_dim, tuple(hidden_dims), latent_dim, rngs=rngs)
        self.decoder = _MLP(latent_dim, tuple(reversed(hidden_dims)), action_dim, rngs=rngs)

        codebook = jax.random.normal(rngs.params(), (codebook_size, latent_dim)) * 0.1
        if self.l2_normalize:
            codebook = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)
        self.codebook = nnx.Param(codebook)

    def _maybe_normalize(self, x):
        if self.l2_normalize:
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x

    def encode(self, action):
        return self._maybe_normalize(self.encoder(action))

    def quantize(self, z_e):
        codebook = self._maybe_normalize(self.codebook.value)
        dists = (
            jnp.sum(z_e * z_e, axis=-1, keepdims=True)
            - 2.0 * z_e @ codebook.T
            + jnp.sum(codebook * codebook, axis=-1)
        )
        code_idx = jnp.argmin(dists, axis=-1)
        return codebook[code_idx], code_idx

    def __call__(self, action, dim_weights=None):
        z_e = self.encode(action)
        z_q, code_idx = self.quantize(z_e)
        z_q_ste = z_e + jax.lax.stop_gradient(z_q - z_e)
        action_hat = self.decoder(z_q_ste)
        sq_err = (action - action_hat) ** 2
        if dim_weights is not None:
            sq_err = sq_err * dim_weights
        recon_loss = jnp.mean(sq_err)
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)
        commit_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)
        return {
            "action_hat": action_hat,
            "code_idx": code_idx,
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
        }

    def init_codebook_from_actions(self, actions: np.ndarray, seed: int = 0, noise_std: float = 0.02) -> None:
        z_e = np.asarray(self.encode(jnp.asarray(actions, dtype=jnp.float32)))
        rng = np.random.default_rng(seed)
        if z_e.shape[0] >= self.codebook_size:
            idx = rng.choice(z_e.shape[0], size=self.codebook_size, replace=False)
            codebook = z_e[idx].astype(np.float32, copy=True)
        else:
            idx = rng.integers(0, z_e.shape[0], size=self.codebook_size)
            codebook = z_e[idx].astype(np.float32, copy=True)
            codebook += rng.normal(size=codebook.shape).astype(np.float32) * noise_std
        if self.l2_normalize:
            codebook = codebook / (np.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)
        self.codebook.value = jnp.asarray(codebook, dtype=jnp.float32)

    def reset_dead_codes(
        self,
        usage: np.ndarray,
        actions: np.ndarray,
        *,
        seed: int,
        threshold: int = 0,
        noise_std: float = 0.01,
    ) -> int:
        dead = np.where(usage <= threshold)[0]
        if len(dead) == 0:
            return 0
        z_e = np.asarray(self.encode(jnp.asarray(actions, dtype=jnp.float32)))
        rng = np.random.default_rng(seed)
        codebook = np.asarray(self.codebook.value).copy()
        codebook[dead] = z_e[rng.integers(0, z_e.shape[0], size=len(dead))]
        codebook[dead] += rng.normal(size=codebook[dead].shape).astype(np.float32) * noise_std
        if self.l2_normalize:
            codebook = codebook / (np.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)
        self.codebook.value = jnp.asarray(codebook, dtype=jnp.float32)
        return int(len(dead))


def save_vq_params(model: JointVQVAE, out_path: str) -> None:
    data: dict[str, np.ndarray] = {}
    for i, layer in enumerate(model.encoder.layers):
        data[f"enc_W_{i}"] = np.asarray(layer.kernel.value)
        data[f"enc_b_{i}"] = np.asarray(layer.bias.value)
    for i, layer in enumerate(model.decoder.layers):
        data[f"dec_W_{i}"] = np.asarray(layer.kernel.value)
        data[f"dec_b_{i}"] = np.asarray(layer.bias.value)
    data["codebook"] = np.asarray(model.codebook.value)
    data["action_dim"] = np.array(model.action_dim, dtype=np.int64)
    data["codebook_size"] = np.array(model.codebook_size, dtype=np.int64)
    data["latent_dim"] = np.array(model.latent_dim, dtype=np.int64)
    data["enc_n_layers"] = np.array(len(model.encoder.layers), dtype=np.int64)
    data["dec_n_layers"] = np.array(len(model.decoder.layers), dtype=np.int64)
    data["l2_normalize"] = np.array(int(bool(model.l2_normalize)), dtype=np.int64)
    np.savez(out_path, **data)


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class JointVQVAEInfer:
    """Pure-numpy VQ-VAE encoder/decoder loaded from ``vq_params.npz``."""

    def __init__(self, params_npz_path: str):
        with np.load(params_npz_path) as z:
            self.action_dim = int(z["action_dim"])
            self.codebook_size = int(z["codebook_size"])
            self.latent_dim = int(z["latent_dim"])
            self.l2_normalize = bool(int(z["l2_normalize"])) if "l2_normalize" in z.files else False
            enc_n = int(z["enc_n_layers"])
            dec_n = int(z["dec_n_layers"])
            self._encoder = [
                (z[f"enc_W_{i}"].astype(np.float32), z[f"enc_b_{i}"].astype(np.float32)) for i in range(enc_n)
            ]
            self._decoder = [
                (z[f"dec_W_{i}"].astype(np.float32), z[f"dec_b_{i}"].astype(np.float32)) for i in range(dec_n)
            ]
            codebook = z["codebook"].astype(np.float32)
            if self.l2_normalize:
                codebook = codebook / (np.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)
            self._codebook = codebook
            self._codebook_sq = np.sum(codebook * codebook, axis=-1)

    def _mlp(self, x: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if i < len(layers) - 1:
                x = _gelu_np(x)
        return x

    def _normalize_np(self, x: np.ndarray) -> np.ndarray:
        if self.l2_normalize:
            return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x

    def encode_to_code(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        orig_shape = action.shape[:-1]
        flat = action.reshape(-1, self.action_dim)
        z = self._normalize_np(self._mlp(flat, self._encoder))
        dists = np.sum(z * z, axis=-1, keepdims=True) - 2.0 * z @ self._codebook.T + self._codebook_sq
        codes = np.argmin(dists, axis=-1).astype(np.int64)
        return codes.reshape(orig_shape)

    def decode_from_code(self, code_idx: np.ndarray) -> np.ndarray:
        code_idx = np.asarray(code_idx, dtype=np.int64)
        orig_shape = code_idx.shape
        flat_idx = np.clip(code_idx.reshape(-1), 0, self.codebook_size - 1)
        action = self._mlp(self._codebook[flat_idx], self._decoder)
        return action.reshape(*orig_shape, self.action_dim) if orig_shape else action[0]
