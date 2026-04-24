# Tokenizer Assets

Scripts in this directory build offline assets used by action tokenizers.

- `compute_bin_edges.py`: computes quantile bin boundaries for `QuantileBinningPickPlaceTokenizer`.
- `compute_codebook.py`: computes per-dimension scalar VQ codebooks for `VQActionTokenizer`.

Run `scripts/data/compute_norm_stats.py` first because quantile binning is computed over normalized actions.
