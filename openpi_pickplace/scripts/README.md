# Script Layout

- `data/`: dataset conversion and data-derived preprocessing assets such as norm stats.
  - `data/rlbench/`: RLBench conversion and merge helpers.
  - `data/libero/`: LIBERO conversion and benchmark helpers kept for future use.
- `tokenizer/`: offline assets used by action tokenizers, such as quantile bin edges.
- `training/`: training entrypoints and training tests.
- `evaluation/`: metric evaluation over datasets/checkpoints.
- `visualization/`: inference visualization and plotting utilities.
- `remote/`: optional WebSocket serving and Docker deployment helpers.
- `run/`: shell wrappers that tie the pieces together for common workflows.
