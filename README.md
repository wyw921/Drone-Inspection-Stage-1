# Drone Inspection Stage 1

这是一个面向无人机巡检与三维重建的研究型代码仓库。
当前公开整理的主线工作，聚焦于安中大楼及其周边建筑粗模型上的三维视点规划、拍摄组织与安全路径生成。

## 推荐入口

如果你第一次打开这个仓库，建议直接看下面两个分支：

- `stable/anzhong-stage1-3d-current`
  当前整理后的稳定快照，适合作为阅读、复现实验和后续继续开发的主入口。
- `preserve-anzhong-stage1-3d-shot8-v1`
  保留的早期稳定基线，用来做可回退、可对照版本。

`main` 现在主要作为仓库首页导航，不再假设它一定承载最新可运行主线。

## 当前稳定快照包含什么

稳定快照分支：
[`stable/anzhong-stage1-3d-current`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/stable/anzhong-stage1-3d-current)

其中已经整理好：

- 当前稳定版 `src/` 代码
- 已注册的 3 套可直接使用的粗模型
- 统一模型注册与运行入口
- 一组代表性的结果图、CSV 和 Markdown 汇总

## 可直接使用的粗模型

稳定快照里目前整理了 3 套粗模型：

- `anzhong_tower_single`
  安中大楼单体建筑
- `anzhong_tower_plus_north_teaching`
  安中大楼 + 北教学楼
- `anzhong_surrounding_buildings`
  安中大楼建筑群

模型说明见：
[`coarse_models/README.md`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/coarse_models/README.md)

## 运行入口

如果你想直接列出模型并开始跑：

```bash
python3 src/run_registered_full_pipeline.py --list-models
```

当前稳定快照里最推荐看的几个文件：

- [`src/run_registered_full_pipeline.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/run_registered_full_pipeline.py)
- [`src/uav_coarse_3d_full_pipeline.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/uav_coarse_3d_full_pipeline.py)
- [`src/uav_coarse_3d_viewpoint_rl.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/uav_coarse_3d_viewpoint_rl.py)
- [`src/coarse_model_registry.py`](https://github.com/wyw921/Drone-Inspection-Stage-1/blob/stable/anzhong-stage1-3d-current/src/coarse_model_registry.py)

## 当前主线方法概览

当前这条 Stage 1 主线，从候选生成到路径输出，核心思路是：

1. 基于粗三维模型生成较大的候选视点池，而不是固定少量站点。
2. 用三层结构化动作做选点决策：`点索引 + shot_count + 距离层`。
3. 选点评价不只看覆盖率，还联合考虑：
   认证覆盖率、同站前向重叠、临站旁向重叠、质量指标、路径代价。
4. 质量部分细化为：
   `mean_quality_normalized`、`quality_good_fraction`、`weakest_quality_normalized`。
5. 路径连接优先尝试安全直连或 Dubins-like 连接，不安全时回退到 3D A* 避障。
6. 路径后处理不做整条全局平滑，只在需要满足转弯半径的位置做局部圆弧化。

## 基线与下载

保留基线分支：
[`preserve-anzhong-stage1-3d-shot8-v1`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/preserve-anzhong-stage1-3d-shot8-v1)

对应 Release：
[Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

如果你需要我整理时生成的原始备份快照，也保留了这个分支：

- `codex/stable-snapshot-20260717`

## 说明

- 历史实验目录和较大的训练产物仍然保留在本地工作区，不全部放到当前稳定快照分支里。
- GitHub 上优先保留“可阅读、可运行、可回退”的版本结构。
