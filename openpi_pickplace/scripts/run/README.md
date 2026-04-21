# Run Entrypoints

Shell wrappers for the current pick-place workflow live here.

- `train_pick_place_all_qbin64.sh`: train `pickplace_all_qbin64` and optionally run the post-train eval sweep.
- `infer_vis.sh`: render GT-vs-prediction visualizations from a checkpoint.

The repository-root shell files forward to these scripts for convenience.
