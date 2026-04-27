import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

######## TODO ##################################################
class UniformBinningPickPlaceTokenizer:
    """
    Pi0-FAST-compatible tokenizer: uniform binning of continuous DOFs into PaliGemma token ids.

    - Splits each timestep into ``pick_dim`` + ``place_dim`` scalars (default 9+9).
    - Values are assumed **normalized** to ``[value_min, value_max]`` (typically [-1, 1]) before
      discretization, matching ``transforms.Normalize`` + DROID assets.
    - Training / inference contract: ``actions`` have shape ``(action_horizon, pick_dim + place_dim)``.

    Wire-up: set ``Pi0FASTConfig(action_dim=pick_dim + place_dim, fast_model_tokenizer=UniformBinningPickPlaceTokenizer,
    fast_model_tokenizer_kwargs={...})``. Use ``transforms.SplitPickPlaceActions`` if you want separate
    ``pick_dof`` / ``place_dof`` keys after ``ExtractFASTActions``.

    Note: Pretrained FAST checkpoints expect a different action codebook; retrain or finetune for this tokenizer.
    """

    def __init__(
        self,
        max_len: int = 256,
        *,
        pick_dim: int = 9,
        place_dim: int = 9,
        n_bins: int = 256,
        fast_skip_tokens: int = 128,
        value_min: float = -1.0,
        value_max: float = 1.0,
    ):
        self._max_len = max_len
        self.pick_dim = pick_dim
        self.place_dim = place_dim
        self.dof_dim = pick_dim + place_dim
        self._n_bins = n_bins
        self._fast_skip_tokens = fast_skip_tokens
        self.value_min = value_min
        self.value_max = value_max

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._bin_edges = np.linspace(value_min, value_max, n_bins + 1)

    def _continuous_to_bins(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), self.value_min, self.value_max)
        bins = np.digitize(x, self._bin_edges[:-1]) - 1
        return np.clip(bins, 0, self._n_bins - 1).astype(np.int64)

    def _bins_to_paligemma_ids(self, bins: np.ndarray) -> np.ndarray:
        """Map each bin index to a distinct token id in the PaliGemma tail (same layout as FAST)."""
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - bins

    def _paligemma_ids_to_bins(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        """Inverse of ``_bins_to_paligemma_ids``."""
        t = np.asarray(token_ids, dtype=np.int64)
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - t

    def _bins_to_continuous(self, bins: np.ndarray) -> np.ndarray:
        """Map bin indices back to bin centers in ``[value_min, value_max]`` (inverse of ``_continuous_to_bins``)."""
        b = bins.astype(np.float64)
        centers = (b + 0.5) / float(self._n_bins) * (self.value_max - self.value_min) + self.value_min
        return centers.astype(np.float32)

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        action_prefix_tokens = self._paligemma_tokenizer.encode("Action: ")
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim == 1:
                a = a[np.newaxis, :]
            if a.shape[-1] != self.dof_dim:
                raise ValueError(
                    f"actions last dim must be pick_dim+place_dim={self.dof_dim}, got {a.shape[-1]}"
                )
            flat = a.reshape(-1)
            expected = a.shape[0] * self.dof_dim
            if flat.size != expected:
                raise ValueError(f"actions shape {a.shape} inconsistent with dof_dim={self.dof_dim}")
            bins = self._continuous_to_bins(flat)
            pg_ids = self._bins_to_paligemma_ids(bins)
            action_content_tokens = pg_ids.astype(int).tolist() + self._paligemma_tokenizer.encode("|", add_eos=True)
            postfix_tokens = action_prefix_tokens + action_content_tokens
        else:
            action_content_tokens = []
            postfix_tokens = action_prefix_tokens

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [False] * len(action_prefix_tokens) + [True] * len(action_content_tokens)

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing `max_token_len` in Pi0FASTConfig."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        if action_dim != self.dof_dim:
            logging.warning(
                "UniformBinningPickPlaceTokenizer: action_dim=%s differs from dof_dim=%s; "
                "set Pi0FASTConfig.action_dim=%s.",
                action_dim,
                self.dof_dim,
                self.dof_dim,
            )
        need = action_horizon * self.dof_dim
        all_bin_ids = self._paligemma_ids_to_bins(tokens[:need])

        out = np.zeros(need, dtype=np.float32)
        for pos in range(min(need, len(all_bin_ids))):
            b = int(all_bin_ids[pos])
            if 0 <= b < self._n_bins:
                out[pos] = self._bins_to_continuous(np.array([b]))[0]
            elif 0 <= b < 1024:
                clamped = np.clip(b, 0, self._n_bins - 1)
                out[pos] = self._bins_to_continuous(np.array([clamped]))[0]
        return out.reshape(action_horizon, self.dof_dim)

class QuantileBinningPickPlaceTokenizer(FASTTokenizer):
    """Pick+Place tokenizer using **per-dimension quantile bin edges**.

    Unlike :class:`UniformBinningPickPlaceTokenizer` (equal-width bins over ``[-1, 1]``),
    this tokenizer uses data-driven bin boundaries so that each bin contains roughly the
    same number of training samples, giving higher resolution where the data is dense.

    ``bin_edges`` must be pre-computed (see ``scripts/tokenizer/compute_bin_edges.py``) and stored as
    a ``.npy`` file.  Shape: ``(action_dim, n_bins + 1)`` — one row of sorted edge values
    per action dimension.

    Everything else (prompt encoding, loss mask, PaliGemma ID mapping, ``extract_actions``)
    is identical to :class:`UniformBinningPickPlaceTokenizer`.
    """

    def __init__(
        self,
        max_len: int = 256,
        *,
        pick_dim: int = 9,
        place_dim: int = 9,
        n_bins: int = 128,
        bin_edges_path: str,
        fast_skip_tokens: int = 128,
    ):
        self._max_len = max_len
        self.pick_dim = pick_dim
        self.place_dim = place_dim
        self.dof_dim = pick_dim + place_dim
        self._n_bins = n_bins
        self._fast_skip_tokens = fast_skip_tokens

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        raw_edges = np.load(bin_edges_path)
        if raw_edges.shape != (self.dof_dim, n_bins + 1):
            raise ValueError(
                f"bin_edges shape mismatch: expected ({self.dof_dim}, {n_bins + 1}), got {raw_edges.shape}"
            )
        self._bin_edges = raw_edges.astype(np.float64)
        logging.info("QuantileBinningPickPlaceTokenizer: loaded bin_edges from %s", bin_edges_path)

    # ------------------------------------------------------------------
    # Binning helpers (per-dimension, using quantile edges)
    # ------------------------------------------------------------------

    def _continuous_to_bins(self, flat_values: np.ndarray) -> np.ndarray:
        """Digitize *flat_values* (length ``dof_dim * horizon``) using per-dimension edges."""
        x = np.asarray(flat_values, dtype=np.float64)
        bins = np.empty_like(x, dtype=np.int64)
        for i, val in enumerate(x):
            dim = i % self.dof_dim
            edges = self._bin_edges[dim]
            val_clipped = np.clip(val, edges[0], edges[-1])
            b = int(np.searchsorted(edges, val_clipped, side="right")) - 1
            bins[i] = np.clip(b, 0, self._n_bins - 1)
        return bins

    def _bins_to_continuous(self, flat_bins: np.ndarray, *, offset: int = 0) -> np.ndarray:
        """Convert bin indices back to bin-center values using per-dimension edges.

        ``offset``: index of the first bin in the flat sequence relative to action dim 0.
        """
        out = np.empty(len(flat_bins), dtype=np.float32)
        for i, b in enumerate(flat_bins):
            dim = (offset + i) % self.dof_dim
            b = int(np.clip(b, 0, self._n_bins - 1))
            center = (self._bin_edges[dim, b] + self._bin_edges[dim, b + 1]) / 2.0
            out[i] = float(center)
        return out

    # ------------------------------------------------------------------
    # PaliGemma ID mapping (same as UniformBinning)
    # ------------------------------------------------------------------

    def _bins_to_paligemma_ids(self, bins: np.ndarray) -> np.ndarray:
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - bins

    def _paligemma_ids_to_bins(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        t = np.asarray(token_ids, dtype=np.int64)
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - t

    # ------------------------------------------------------------------
    # tokenize  (identical structure to UniformBinning)
    # ------------------------------------------------------------------

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        action_prefix_tokens = self._paligemma_tokenizer.encode("Action: ")
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim == 1:
                a = a[np.newaxis, :]
            if a.shape[-1] != self.dof_dim:
                raise ValueError(
                    f"actions last dim must be pick_dim+place_dim={self.dof_dim}, got {a.shape[-1]}"
                )
            flat = a.reshape(-1)
            bins = self._continuous_to_bins(flat)
            pg_ids = self._bins_to_paligemma_ids(bins)
            action_content_tokens = pg_ids.astype(int).tolist() + self._paligemma_tokenizer.encode("|", add_eos=True)
            postfix_tokens = action_prefix_tokens + action_content_tokens
        else:
            action_content_tokens = []
            postfix_tokens = action_prefix_tokens

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [False] * len(action_prefix_tokens) + [True] * len(action_content_tokens)

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    # ------------------------------------------------------------------
    # extract_actions  (identical logic to UniformBinning, but uses per-dim decode)
    # ------------------------------------------------------------------

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        if action_dim != self.dof_dim:
            logging.warning(
                "QuantileBinningPickPlaceTokenizer: action_dim=%s differs from dof_dim=%s",
                action_dim, self.dof_dim,
            )
        need = action_horizon * self.dof_dim
        all_bin_ids = self._paligemma_ids_to_bins(tokens[:need])

        out = np.zeros(need, dtype=np.float32)
        for pos in range(min(need, len(all_bin_ids))):
            b = int(all_bin_ids[pos])
            if 0 <= b < self._n_bins:
                out[pos] = self._bins_to_continuous(np.array([b]), offset=pos)[0]
            elif 0 <= b < 1024:
                clamped = np.clip(b, 0, self._n_bins - 1)
                out[pos] = self._bins_to_continuous(np.array([clamped]), offset=pos)[0]
        return out.reshape(action_horizon, self.dof_dim)


class VQActionTokenizer(FASTTokenizer):
    """Pick+Place tokenizer using a **per-dimension learnable scalar codebook**.

    The codebook ``(action_dim, codebook_size)`` is pre-trained offline (currently via
    K-means on normalized actions — see ``scripts/train_vq.py``) and then **frozen** when
    used inside the Pi0-FAST training pipeline.  For each action value we look up the
    nearest codebook entry along its own dimension; ``codebook_size`` acts like ``n_bins``
    from :class:`QuantileBinningPickPlaceTokenizer`.

    This is the minimal VQ-VAE variant: no encoder / no decoder, just learnable bin
    centers.  The prompt format, PaliGemma ID mapping, ``tokenize`` and
    ``extract_actions`` logic are identical to :class:`QuantileBinningPickPlaceTokenizer`
    so the rest of the pipeline (train / eval / inference) is unchanged.
    """

    def __init__(
        self,
        max_len: int = 256,
        *,
        pick_dim: int = 9,
        place_dim: int = 9,
        codebook_size: int = 128,
        codebook_path: str,
        fast_skip_tokens: int = 128,
    ):
        self._max_len = max_len
        self.pick_dim = pick_dim
        self.place_dim = place_dim
        self.dof_dim = pick_dim + place_dim
        self._n_bins = codebook_size  # reused name for PaliGemma mapping / validity checks
        self._codebook_size = codebook_size
        self._fast_skip_tokens = fast_skip_tokens

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        raw_codebook = np.load(codebook_path)
        if raw_codebook.shape != (self.dof_dim, codebook_size):
            raise ValueError(
                f"codebook shape mismatch: expected ({self.dof_dim}, {codebook_size}), "
                f"got {raw_codebook.shape}"
            )
        # Sort each row so that argmin-by-midpoints yields a monotonic mapping (useful for
        # debugging and consistent with quantile-bin ordering).
        sorted_codebook = np.sort(raw_codebook.astype(np.float64), axis=-1)
        self._codebook = sorted_codebook
        # Midpoints between consecutive centers — used by ``searchsorted`` for fast 1-D NN.
        self._midpoints = (sorted_codebook[:, :-1] + sorted_codebook[:, 1:]) / 2.0
        logging.info(
            "VQActionTokenizer: loaded codebook shape=%s from %s", self._codebook.shape, codebook_path
        )

    # ------------------------------------------------------------------
    # Binning helpers (per-dimension, using the learned codebook)
    # ------------------------------------------------------------------

    def _continuous_to_bins(self, flat_values: np.ndarray) -> np.ndarray:
        """Nearest-codebook assignment per-dimension (length ``dof_dim * horizon``)."""
        x = np.asarray(flat_values, dtype=np.float64)
        bins = np.empty_like(x, dtype=np.int64)
        for i, val in enumerate(x):
            dim = i % self.dof_dim
            # 1-D nearest neighbour via midpoint search (codebook is sorted).
            b = int(np.searchsorted(self._midpoints[dim], val, side="right"))
            bins[i] = np.clip(b, 0, self._codebook_size - 1)
        return bins

    def _bins_to_continuous(self, flat_bins: np.ndarray, *, offset: int = 0) -> np.ndarray:
        """Look up the codebook entry corresponding to each bin (per-dimension)."""
        out = np.empty(len(flat_bins), dtype=np.float32)
        for i, b in enumerate(flat_bins):
            dim = (offset + i) % self.dof_dim
            b = int(np.clip(b, 0, self._codebook_size - 1))
            out[i] = float(self._codebook[dim, b])
        return out

    # ------------------------------------------------------------------
    # PaliGemma ID mapping (same as UniformBinning / QuantileBinning)
    # ------------------------------------------------------------------

    def _bins_to_paligemma_ids(self, bins: np.ndarray) -> np.ndarray:
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - bins

    def _paligemma_ids_to_bins(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        t = np.asarray(token_ids, dtype=np.int64)
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - t

    # ------------------------------------------------------------------
    # tokenize  (identical structure to QuantileBinning)
    # ------------------------------------------------------------------

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        action_prefix_tokens = self._paligemma_tokenizer.encode("Action: ")
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim == 1:
                a = a[np.newaxis, :]
            if a.shape[-1] != self.dof_dim:
                raise ValueError(
                    f"actions last dim must be pick_dim+place_dim={self.dof_dim}, got {a.shape[-1]}"
                )
            flat = a.reshape(-1)
            bins = self._continuous_to_bins(flat)
            pg_ids = self._bins_to_paligemma_ids(bins)
            action_content_tokens = pg_ids.astype(int).tolist() + self._paligemma_tokenizer.encode("|", add_eos=True)
            postfix_tokens = action_prefix_tokens + action_content_tokens
        else:
            action_content_tokens = []
            postfix_tokens = action_prefix_tokens

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = (
            [False] * len(prefix_tokens)
            + [False] * len(action_prefix_tokens)
            + [True] * len(action_content_tokens)
        )

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    # ------------------------------------------------------------------
    # extract_actions  (position-aligned decode; identical to QuantileBinning)
    # ------------------------------------------------------------------

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        if action_dim != self.dof_dim:
            logging.warning(
                "VQActionTokenizer: action_dim=%s differs from dof_dim=%s",
                action_dim, self.dof_dim,
            )
        need = action_horizon * self.dof_dim
        all_bin_ids = self._paligemma_ids_to_bins(tokens[:need])

        out = np.zeros(need, dtype=np.float32)
        for pos in range(min(need, len(all_bin_ids))):
            b = int(all_bin_ids[pos])
            if 0 <= b < self._codebook_size:
                out[pos] = self._bins_to_continuous(np.array([b]), offset=pos)[0]
            elif 0 <= b < 1024:
                clamped = np.clip(b, 0, self._codebook_size - 1)
                out[pos] = self._bins_to_continuous(np.array([clamped]), offset=pos)[0]
        return out.reshape(action_horizon, self.dof_dim)


class PhaseVQActionTokenizer(FASTTokenizer):
    """Pick/place phase-conditioned vector VQ tokenizer.

    Each timestep is represented by two tokens instead of 18 scalar tokens:

    * one token for the full 9-D pick pose
    * one token for the full 9-D place pose

    The decoded action is still the usual 18-D ``[pick, place]`` vector, so eval and
    policy code can stay on the same action contract. Use this with
    ``Pi0FASTConfig(action_dim=2, action_horizon=1)`` because ``action_dim`` is the
    number of generated action-code tokens for FAST training.
    """

    def __init__(
        self,
        max_len: int = 256,
        *,
        pick_dim: int = 9,
        place_dim: int = 9,
        codebook_size: int = 64,
        pick_codebook_path: str,
        place_codebook_path: str,
        fast_skip_tokens: int = 128,
    ):
        self._max_len = max_len
        self.pick_dim = pick_dim
        self.place_dim = place_dim
        self.dof_dim = pick_dim + place_dim
        self._n_bins = codebook_size
        self._codebook_size = codebook_size
        self._fast_skip_tokens = fast_skip_tokens

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._pick_codebook = np.load(pick_codebook_path).astype(np.float32)
        self._place_codebook = np.load(place_codebook_path).astype(np.float32)
        expected_pick = (codebook_size, pick_dim)
        expected_place = (codebook_size, place_dim)
        if self._pick_codebook.shape != expected_pick:
            raise ValueError(
                f"pick_codebook shape mismatch: expected {expected_pick}, got {self._pick_codebook.shape}"
            )
        if self._place_codebook.shape != expected_place:
            raise ValueError(
                f"place_codebook shape mismatch: expected {expected_place}, got {self._place_codebook.shape}"
            )
        logging.info(
            "PhaseVQActionTokenizer: loaded pick codebook %s from %s and place codebook %s from %s",
            self._pick_codebook.shape,
            pick_codebook_path,
            self._place_codebook.shape,
            place_codebook_path,
        )

    def _codes_to_paligemma_ids(self, codes: np.ndarray) -> np.ndarray:
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - codes

    def _paligemma_ids_to_codes(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        t = np.asarray(token_ids, dtype=np.int64)
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - t

    def _nearest_codes(self, vectors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        diff = vectors[:, None, :] - codebook[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=-1), axis=-1).astype(np.int64)

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        action_prefix_tokens = self._paligemma_tokenizer.encode("Action: ")
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim == 1:
                a = a[np.newaxis, :]
            if a.shape[-1] != self.dof_dim:
                raise ValueError(
                    f"actions last dim must be pick_dim+place_dim={self.dof_dim}, got {a.shape[-1]}"
                )

            pick_codes = self._nearest_codes(a[:, : self.pick_dim], self._pick_codebook)
            place_codes = self._nearest_codes(a[:, self.pick_dim : self.dof_dim], self._place_codebook)
            codes = np.stack([pick_codes, place_codes], axis=-1).reshape(-1)
            pg_ids = self._codes_to_paligemma_ids(codes)
            action_content_tokens = pg_ids.astype(int).tolist() + self._paligemma_tokenizer.encode("|", add_eos=True)
            postfix_tokens = action_prefix_tokens + action_content_tokens
        else:
            action_content_tokens = []
            postfix_tokens = action_prefix_tokens

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = (
            [False] * len(prefix_tokens)
            + [False] * len(action_prefix_tokens)
            + [True] * len(action_content_tokens)
        )

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating.")
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        if action_dim != 2:
            logging.warning(
                "PhaseVQActionTokenizer expects FAST action_dim=2 code tokens, got %s",
                action_dim,
            )
        need = action_horizon * 2
        all_codes = self._paligemma_ids_to_codes(tokens[:need])

        out = np.zeros((action_horizon, self.dof_dim), dtype=np.float32)
        for h in range(action_horizon):
            pick_pos = 2 * h
            place_pos = pick_pos + 1
            if pick_pos < len(all_codes):
                pick_code = int(all_codes[pick_pos])
                if 0 <= pick_code < self._codebook_size:
                    out[h, : self.pick_dim] = self._pick_codebook[pick_code]
                elif 0 <= pick_code < 1024:
                    out[h, : self.pick_dim] = self._pick_codebook[int(np.clip(pick_code, 0, self._codebook_size - 1))]
            if place_pos < len(all_codes):
                place_code = int(all_codes[place_pos])
                if 0 <= place_code < self._codebook_size:
                    out[h, self.pick_dim : self.dof_dim] = self._place_codebook[place_code]
                elif 0 <= place_code < 1024:
                    out[h, self.pick_dim : self.dof_dim] = self._place_codebook[
                        int(np.clip(place_code, 0, self._codebook_size - 1))
                    ]
        return out


class VQVAEActionTokenizer(FASTTokenizer):
    """Joint VQ-VAE action tokenizer: each 18-D action uses **one** code token.

    Unlike the per-dimension binning tokenizers
    (:class:`UniformBinningPickPlaceTokenizer`, :class:`QuantileBinningPickPlaceTokenizer`,
    :class:`VQActionTokenizer`) which emit ``action_dim`` tokens per action vector, this
    tokenizer runs a pre-trained VQ-VAE encoder (see ``scripts/train_vq.py``) to map the
    whole 18-D vector to a **single** code index and embeds that code as a single
    PaliGemma ``<loc>`` token.  Conceptually:

        raw 18-D action ──► encoder (MLP) ──► nearest codebook entry ──► code_idx ∈ [0, K)

    The VQ parameters (encoder / decoder / codebook) are loaded from ``vq_params.npz``
    and run in pure numpy, so the tokenizer remains JAX-free at data-prep / inference
    time.  The sequence layout is::

        [Task: ... , State: ... ] Action: <loc_code> | EOS

    For ``action_horizon > 1`` the action tokens are placed back-to-back so the sequence
    is ``action_horizon`` code tokens long instead of ``action_horizon * action_dim``.
    """

    def __init__(
        self,
        max_len: int = 256,
        *,
        action_dim: int = 18,
        codebook_size: int = 1024,
        vq_params_path: str,
        fast_skip_tokens: int = 128,
    ):
        # Defer heavy import until needed (avoids a hard JAX dependency at import time).
        from openpi.models.action_vq import JointVQVAEInfer

        self._max_len = max_len
        self.action_dim = action_dim
        self.dof_dim = action_dim  # alias kept for symmetry with bin tokenizers
        self._n_bins = codebook_size  # reused name for PaliGemma ID mapping checks
        self._codebook_size = codebook_size
        self._fast_skip_tokens = fast_skip_tokens

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._vq = JointVQVAEInfer(vq_params_path)
        if self._vq.action_dim != action_dim:
            raise ValueError(
                f"vq action_dim mismatch: expected {action_dim}, got {self._vq.action_dim}"
            )
        if self._vq.codebook_size != codebook_size:
            raise ValueError(
                f"vq codebook_size mismatch: expected {codebook_size}, got {self._vq.codebook_size}"
            )
        # Expose the asset path so ``checkpoints.save_assets`` can copy it.
        self._vq_params_path = vq_params_path
        logging.info(
            "VQVAEActionTokenizer: loaded vq_params from %s (K=%d, d=%d)",
            vq_params_path, self._codebook_size, self._vq.latent_dim,
        )

    # ------------------------------------------------------------------
    # PaliGemma ID mapping (same formula as the bin tokenizers)
    # ------------------------------------------------------------------

    def _codes_to_paligemma_ids(self, codes: np.ndarray) -> np.ndarray:
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - codes

    def _paligemma_ids_to_codes(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        t = np.asarray(token_ids, dtype=np.int64)
        v = self._paligemma_tokenizer.vocab_size()
        return v - 1 - self._fast_skip_tokens - t

    # Aliases so that existing debug helpers (``_bins_to_paligemma_ids`` /
    # ``_paligemma_ids_to_bins``) keep working transparently.
    def _bins_to_paligemma_ids(self, bins: np.ndarray) -> np.ndarray:
        return self._codes_to_paligemma_ids(bins)

    def _paligemma_ids_to_bins(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        return self._paligemma_ids_to_codes(token_ids)

    # ------------------------------------------------------------------
    # tokenize  (action part is ``action_horizon`` tokens long, not action_horizon*action_dim)
    # ------------------------------------------------------------------

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        action_prefix_tokens = self._paligemma_tokenizer.encode("Action: ")
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim == 1:
                a = a[np.newaxis, :]
            if a.shape[-1] != self.action_dim:
                raise ValueError(
                    f"actions last dim must be action_dim={self.action_dim}, got {a.shape[-1]}"
                )
            # One code per action vector (shape: (action_horizon,)).
            codes = self._vq.encode_to_code(a)
            codes = np.asarray(codes, dtype=np.int64).reshape(-1)
            pg_ids = self._codes_to_paligemma_ids(codes)
            action_content_tokens = (
                pg_ids.astype(int).tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
            postfix_tokens = action_prefix_tokens + action_content_tokens
        else:
            action_content_tokens = []
            postfix_tokens = action_prefix_tokens

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = (
            [False] * len(prefix_tokens)
            + [False] * len(action_prefix_tokens)
            + [True] * len(action_content_tokens)
        )

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    # ------------------------------------------------------------------
    # extract_actions  (one code token per action; position-aligned fallback)
    # ------------------------------------------------------------------

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        if action_dim != self.action_dim:
            logging.warning(
                "VQVAEActionTokenizer: action_dim=%s differs from configured %s",
                action_dim, self.action_dim,
            )
        # Take the first ``action_horizon`` tokens as potential code tokens.
        need = action_horizon
        all_codes = self._paligemma_ids_to_codes(tokens[:need])

        out = np.zeros((action_horizon, action_dim), dtype=np.float32)
        for h in range(min(action_horizon, len(all_codes))):
            c = int(all_codes[h])
            if 0 <= c < self._codebook_size:
                out[h] = self._vq.decode_from_code(np.array([c]))[0]
            elif 0 <= c < 1024:
                # Out-of-range <loc> token: clamp to the nearest valid code.
                clamped = int(np.clip(c, 0, self._codebook_size - 1))
                out[h] = self._vq.decode_from_code(np.array([clamped]))[0]
            # otherwise (non-<loc> token, e.g. text / '|' / EOS / padding): leave zeros
        return out


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        # Decode predicted output tokens
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
