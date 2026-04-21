# Run Entrypoints

Shell wrappers for the current pick-place workflow live here.

- `train_pick_place_all.sh`: train pick-place configs and optionally run the post-train eval sweep. Defaults to `pickplace_all_qbin64`; override with `CONFIG_NAME=pickplace_all_uniform` or another config.
- `infer_vis.sh`: render GT-vs-prediction visualizations from a checkpoint.

The repository-root shell files forward to these scripts for convenience.
