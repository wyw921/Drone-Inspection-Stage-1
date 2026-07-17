# Anzhong Stage 1 3D Stable Snapshot

这是当前整理后的稳定快照分支，适合作为：

- GitHub 上的主阅读入口
- 后续继续开发和回退的稳定代码底座
- 单建筑与多建筑粗模型对照测试的统一运行版本

如果你是从仓库首页进来的，`main` 负责导航；
如果你已经切到这个分支，这里就是“当前稳定版本体”。

## 这份稳定版包含什么

当前分支已经整理好：

- 当前稳定版 `src/` 代码
- 3 套可直接运行的粗模型
- 统一模型注册与运行入口
- 一组代表性的结果图、CSV 和 Markdown 汇总

## 可直接使用的粗模型

当前注册好的模型如下：

| model_key | 场景 | 用途 |
|---|---|---|
| `anzhong_tower_single` | 安中大楼单体建筑 | 单建筑主基线 |
| `anzhong_tower_plus_north_teaching` | 安中大楼 + 北教学楼 | 多建筑泛化测试 |
| `anzhong_surrounding_buildings` | 安中大楼建筑群 | 周边建筑群对照测试 |

模型文件统一放在：

```text
coarse_models/<model_key>/coarse_model/
```

更多模型说明见：
[`coarse_models/README.md`](./coarse_models/README.md)

## 快速开始

先列出当前可用模型：

```bash
python3 src/run_registered_full_pipeline.py --list-models
```

直接跑一套模型：

```bash
python3 src/run_registered_full_pipeline.py --model-key anzhong_tower_single
```

## 推荐先看的文件

- [`src/run_registered_full_pipeline.py`](./src/run_registered_full_pipeline.py)
  统一入口，按 `model_key` 选择粗模型并启动整条链路。
- [`src/coarse_model_registry.py`](./src/coarse_model_registry.py)
  模型注册表，定义每套粗模型的 key、路径和说明。
- [`src/uav_coarse_3d_full_pipeline.py`](./src/uav_coarse_3d_full_pipeline.py)
  全链路入口，负责把候选生成、视点选择、路径规划、导出串起来。
- [`src/uav_coarse_3d_viewpoint_rl.py`](./src/uav_coarse_3d_viewpoint_rl.py)
  Maskable PPO 选点主逻辑。
- [`src/uav_coarse_3d_planner.py`](./src/uav_coarse_3d_planner.py)
  几何与候选生成底座。

## 当前主线方法

当前稳定版的主线思路是：

1. 先基于粗三维模型生成较大的候选视点池，而不是固定少量站点。
2. Maskable PPO 使用三层结构化动作：
   `点索引 + shot_count + 距离层`。
3. 选点评价联合考虑：
   覆盖率、同站前向重叠、临站旁向重叠、质量指标、路径代价。
4. 质量部分细化为：
   `mean_quality_normalized`、`quality_good_fraction`、`weakest_quality_normalized`。
5. 路径连接优先尝试安全直连或 Dubins-like 连接；
   不安全时回退到 3D A* 避障。
6. 路径后处理不做整条全局平滑，只在需要满足转弯半径的位置做局部圆弧化。

## 分支关系

- `main`
  仓库首页导航页
- `stable/anzhong-stage1-3d-current`
  当前稳定快照
- `preserve-anzhong-stage1-3d-shot8-v1`
  保留的早期稳定基线
- `codex/stable-snapshot-20260717`
  整理稳定分支时保留的原始备份快照

## 基线与下载

保留基线分支：
[`preserve-anzhong-stage1-3d-shot8-v1`](https://github.com/wyw921/Drone-Inspection-Stage-1/tree/preserve-anzhong-stage1-3d-shot8-v1)

对应 Release：
[Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

## 结果文件

当前分支里保留了一组代表性输出，方便直接查看：

- `outputs/anzhong_capture_full_pipeline_mainline_seed17_nospiral/`
- `outputs/anzhong_selection_stabilized_formal/`

其中包括：

- 路径图
- 选中视点 CSV
- 路径航点 CSV
- PPO 训练历史
- Markdown / CSV 结果汇总

## 说明

- GitHub 上优先保留“可阅读、可运行、可回退”的版本结构。
- 较大的历史训练产物和完整实验工作区仍然保留在本地，不全部推到这个稳定分支里。
