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

- [`src/uav_coarse_3d_full_pipeline.py`](./src/uav_coarse_3d_full_pipeline.py)
  当前主链路总入口，适合直接看完整结果生成流程。
- [`src/uav_coarse_3d_viewpoint_rl.py`](./src/uav_coarse_3d_viewpoint_rl.py)
  选点训练入口，核心是 `点索引 + shot_count + 距离层` 的三层动作设计。
- [`src/uav_coarse_3d_planner.py`](./src/uav_coarse_3d_planner.py)
  候选点、覆盖关系和几何质量的底层生成模块。
- [`src/run_anzhong_stage1_3d.py`](./src/run_anzhong_stage1_3d.py)
  便捷启动脚本，适合快速跑通一版基础规划。
- [`src/uav_anzhong_graph_path_agent.py`](./src/uav_anzhong_graph_path_agent.py)
  图路径规划实验支线，不是当前默认主链路入口。

## 当前主线思路

当前保留的安中 Stage1 主线，核心流程是：

1. 从粗三维模型出发，生成更大的候选视点池。
2. 用 Maskable PPO 以三层结构化动作选择拍摄方案：
   点索引 + shot_count + 距离层。
3. 奖励设计同时考虑：
   认证覆盖率、同站前向重叠、临站旁向重叠、质量指标、路径长度与安全性。
4. 质量部分使用更细的联合指标：
   `mean_quality_normalized`、`quality_good_fraction`、`weakest_quality_normalized`。
5. 目标是尽量逼近三维重建友好的明确阈值：
   覆盖率 100%、前向重叠 80%、旁向重叠 60%。
6. 选点完成后，路径按 building、sector / cluster 分层组织。
7. 连线时优先使用安全的直接连接；若可能穿模或距离建筑过近不安全，则回退到避障搜索。
8. 路径后处理只保留转弯半径处的局部平滑，普通路段保持直线飞行。

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
