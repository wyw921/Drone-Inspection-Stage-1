# 无人机巡检与三维重建项目

这是一个面向无人机自主巡检与三维重建的研究型代码仓库。  
当前公开的主线工作聚焦于安中大楼 LOD1 粗模场景下的三维视点规划与路径组织。

## 项目概览

当前主链路为：

`候选视点生成 -> Maskable PPO 选点 -> 分层路径规划 -> 结果导出与可视化`

重点优化目标包括：

- 认证覆盖率
- 同站前向重叠率
- 临站旁向重叠率
- 结构质量指标
- 路径长度与飞行安全性

## 当前主要目录

- [`anzhong_stage1_3d`](./anzhong_stage1_3d/)
  安中大楼三维主线工作目录
- [`src`](./src/)
  仓库级算法原型与实验脚本
- [`outputs`](./outputs/)
  历史实验结果、对比图与训练输出

## 当前基线结果

当前保留的稳定主线为安中大楼 coarse-3D `mainline` 版本。

| 方法 | 视点数 | 认证覆盖率 | 前向重叠 | 旁向重叠 | 最弱结构段覆盖率 | 平均质量 | 拍照质量 | 总张数 | 路径长度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Maskable PPO best-of-N path mainline | 41 | 99.9% | 91.8% | 69.1% | 99.7% | 0.828 | 0.802 | 369 | 1086.0 m |

对应结果目录：

- [`anzhong_stage1_3d_baseline_recover/outputs/anzhong_capture_full_pipeline_mainline_seed17_nospiral`](./anzhong_stage1_3d_baseline_recover/outputs/anzhong_capture_full_pipeline_mainline_seed17_nospiral/)

## 关键入口

如果你只关心当前安中三维主线，建议按职责查看：

- [`anzhong_stage1_3d/src/uav_coarse_3d_full_pipeline.py`](./anzhong_stage1_3d/src/uav_coarse_3d_full_pipeline.py)
  当前主链路总入口，负责把候选生成、Maskable PPO 选点、路径规划、结果导出串起来。
- [`anzhong_stage1_3d/src/uav_coarse_3d_viewpoint_rl.py`](./anzhong_stage1_3d/src/uav_coarse_3d_viewpoint_rl.py)
  选点训练入口，重点是 `点索引 + shot_count + 距离层` 的三层动作设计，以及覆盖率、前向重叠、旁向重叠、质量、路径代价的联合优化。
- [`anzhong_stage1_3d/src/uav_coarse_3d_planner.py`](./anzhong_stage1_3d/src/uav_coarse_3d_planner.py)
  几何底座，负责读入粗模、表面采样、候选视点生成、覆盖关系与质量矩阵计算。
- [`anzhong_stage1_3d/src/run_anzhong_stage1_3d.py`](./anzhong_stage1_3d/src/run_anzhong_stage1_3d.py)
  便捷启动脚本，适合快速用默认参数跑一版基础规划结果。
- [`anzhong_stage1_3d/src/uav_anzhong_graph_path_agent.py`](./anzhong_stage1_3d/src/uav_anzhong_graph_path_agent.py)
  图路径规划实验支线，不是当前 coarse-3D 主线的默认入口。

## 当前主线思路

当前这条安中 Stage1 主线，从选点到结尾的核心思路是：

1. 先基于粗三维模型生成更大的候选视点池，而不是固定卡死在少量站点。
2. Maskable PPO 不再只决定“去哪个点”，而是用三层结构化动作同时决定：
   点索引 + shot_count + 距离层。
3. 选点评价不只看覆盖率，而是同时推动：
   认证覆盖率、同站前向重叠、临站旁向重叠、质量指标、路径代价。
4. 质量部分拆成更细的结构指标联合约束：
   `mean_quality_normalized`、`quality_good_fraction`、`weakest_quality_normalized`。
5. 停止目标不是“差不多够了”，而是尽量朝三维重建友好的明确目标逼近：
   覆盖率 100%、前向重叠 80%、旁向重叠 60%。
6. 选点完成后，路径规划使用 `mainline` 顺序优化组织访问顺序。
7. 每段连接时优先尝试可直连的安全 Dubins-like 路径；如果直连会穿模，或者离建筑过近不安全，再回退到 3D A* 避障连接。
8. 路径后处理不做整条全局平滑，只保留连接段内部必要的转弯半径处理，非急转弯段保持直线飞行。

## GitHub 保留版本

为了方便对照与下载，本仓库保留了：

- 基线分支：`preserve-anzhong-stage1-3d-shot8-v1`
- Release：
  [Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

Release 中包含大文件结果包：

- `selection_maskable_ppo.zip`

## 说明

- 历史结果目录中保留了多次实验输出，命名通常包含日期与版本号。
- 当前活跃主线目录名称为 `anzhong_stage1_3d`。
