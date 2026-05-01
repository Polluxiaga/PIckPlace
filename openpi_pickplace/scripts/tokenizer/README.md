# Tokenizer Assets

Scripts in this directory build offline assets used by action tokenizers.

- `compute_bin_edges.py`: computes quantile bin boundaries for `QuantileBinningPickPlaceTokenizer`.
- `compute_codebook.py`: computes per-dimension scalar VQ codebooks for `VQActionTokenizer`.
- `train_phase_vqvae.py`: trains pick/place VQ-VAE assets for `PhaseVQVAEActionTokenizer`.

Run `scripts/data/compute_norm_stats.py` first because quantile binning is computed over normalized actions.

For `pickplace_all_phase_vqvae128`, the tokenizer assets can be regenerated with:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python scripts/tokenizer/train_phase_vqvae.py \
  --config-name pickplace_all_phase_vqvae128 \
  --codebook-size 128 \
  --n-steps 10000 \
  --batch-size 64 \
  --latent-dim 32 \
  --translation-weight 2.0 \
  --rotation-weight 1.0
```
