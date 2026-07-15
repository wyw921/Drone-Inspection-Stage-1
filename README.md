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

当前保留的稳定基线为安中大楼 fullscale、shot8、多 seed 训练版本。

| 方法 | 视点数 | 认证覆盖率 | 前向重叠 | 旁向重叠 | 最弱结构段覆盖率 | 平均质量 | 拍照质量 | 总张数 | 路径长度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Maskable PPO best-of-N path baseline | 48 | 98.8% | 89.2% | 72.4% | 97.8% | 0.824 | 0.768 | 344 | 1183.9 m |
| Maskable PPO best-of-N + hierarchical path agent | 48 | 98.8% | 89.2% | 72.4% | 97.8% | 0.824 | 0.768 | 344 | 1179.0 m |

对应结果目录：

- [`anzhong_stage1_3d/outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8`](./anzhong_stage1_3d/outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8/)

## 关键入口

如果你只关心当前安中三维主线，建议优先查看：

- [`anzhong_stage1_3d/src/run_anzhong_stage1_3d.py`](./anzhong_stage1_3d/src/run_anzhong_stage1_3d.py)
- [`anzhong_stage1_3d/src/uav_coarse_3d_viewpoint_rl.py`](./anzhong_stage1_3d/src/uav_coarse_3d_viewpoint_rl.py)
- [`anzhong_stage1_3d/src/uav_coarse_3d_full_pipeline.py`](./anzhong_stage1_3d/src/uav_coarse_3d_full_pipeline.py)
- [`anzhong_stage1_3d/src/uav_coarse_3d_planner.py`](./anzhong_stage1_3d/src/uav_coarse_3d_planner.py)
- [`anzhong_stage1_3d/src/uav_anzhong_graph_path_agent.py`](./anzhong_stage1_3d/src/uav_anzhong_graph_path_agent.py)

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

