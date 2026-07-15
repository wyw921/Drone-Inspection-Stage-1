# 安中 Stage1 3D 工作目录说明

这个目录保存当前安中大楼粗三维巡检主线的代码、结果和相关说明。

## 目录内容

- [`src`](./src/)
  当前安中主线源码
- [`outputs`](./outputs/)
  训练结果、路径图、CSV 与 Markdown 汇总
- [`src_backups`](./src_backups/)
  关键阶段源码备份
- [`docs`](./docs/)
  补充材料

## 推荐入口

- [`src/run_anzhong_stage1_3d.py`](./src/run_anzhong_stage1_3d.py)
- [`src/uav_coarse_3d_viewpoint_rl.py`](./src/uav_coarse_3d_viewpoint_rl.py)
- [`src/uav_coarse_3d_full_pipeline.py`](./src/uav_coarse_3d_full_pipeline.py)
- [`src/uav_coarse_3d_planner.py`](./src/uav_coarse_3d_planner.py)
- [`src/uav_anzhong_graph_path_agent.py`](./src/uav_anzhong_graph_path_agent.py)

## 当前保留基线

核心基线输出目录：

- [`outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8`](./outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8/)

基线摘要：

- 视点数：48
- 认证覆盖率：98.8%
- 前向重叠：89.2%
- 旁向重叠：72.4%
- 最弱结构段覆盖率：97.8%
- 平均质量：0.824
- 拍照质量：0.768
- 总张数：344
- 最优分层路径长度：1179.0 m

关键结果文件包括：

- `coarse_3d_full_pipeline_results.md`
- `coarse_3d_full_pipeline_results.csv`
- `baseline_path_3d.png`
- `hierarchical_path_3d.png`
- `baseline_path_waypoints.csv`
- `hierarchical_path_waypoints.csv`
- `hierarchical_selected_captures.csv`

## GitHub 对应存档

- 分支：`preserve-anzhong-stage1-3d-shot8-v1`
- Release：
  [Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

