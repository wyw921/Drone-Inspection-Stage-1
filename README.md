# 安中 Stage1 3D Baseline Recover

这个目录是从 Git 保留基线提交 `preserve-anzhong-stage1-3d-shot8-v1` 原样恢复出来的对照副本。

用途：

- 作为 `98.8%` 覆盖率基线的可回退源码副本
- 作为后续“小剂量 hard-region rescue”注入的实验底座
- 与当前活跃目录 `anzhong_stage1_3d` 做并行对照

关键位置：

- `src/uav_coarse_3d_viewpoint_rl.py`
- `src/uav_coarse_3d_planner.py`
- `src/uav_coarse_3d_full_pipeline.py`
- `src/run_anzhong_stage1_3d.py`
- `src/run_anzhong_stage1_3d_baseline_recover.py`
- `src/run_registered_full_pipeline.py`
- `coarse_models/`

保留基线结果：

- `outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8`

说明：

- 这里先保持基线源码原貌，尽量不改动核心逻辑。
- 后续新的 hard-region 增强实验，会优先在这个目录中单独进行。
- 仓库内已经整理了可直接使用的粗模型，见 `coarse_models/README.md`。
- 统一入口：`python3 src/run_registered_full_pipeline.py --list-models`
- 当前推荐模型 key：`anzhong_tower_single`、`anzhong_tower_plus_north_teaching`、`anzhong_surrounding_buildings`
