# 无人机巡检与三维重建项目

这是一个围绕“无人机自主巡检 + 三维重建友好视点规划”持续迭代的研究型代码仓库。  
当前主线已经从早期的 2D / 2.5D 原型，推进到安中大楼场景下的粗三维全链路：

`候选视点生成 -> Maskable PPO 选点 -> 分层路径规划 -> 结果导出与可视化`

## 当前重点

当前最重要的工作目录是：

- [`anzhong_stage1_3d`](./anzhong_stage1_3d/)

这套主线面向安中 LOD1 粗模，目标不是单纯把覆盖率做高，而是让结果对三维重建更友好，重点优化：

- 认证覆盖率
- 同站前向重叠率
- 临站旁向重叠率
- 结构质量指标
- 路径长度与飞行安全性

## 当前基线

目前保留下来的稳定基线是安中大楼 fullscale、shot8、多 seed 训练版本。

关键结果如下：

| 方法 | 视点数 | 认证覆盖率 | 前向重叠 | 旁向重叠 | 最弱结构段覆盖率 | 平均质量 | 拍照质量 | 总张数 | 路径长度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Maskable PPO best-of-N path baseline | 48 | 98.8% | 89.2% | 72.4% | 97.8% | 0.824 | 0.768 | 344 | 1183.9 m |
| Maskable PPO best-of-N + hierarchical path agent | 48 | 98.8% | 89.2% | 72.4% | 97.8% | 0.824 | 0.768 | 344 | 1179.0 m |

对应结果文件在：

- [`anzhong_stage1_3d/outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8`](./anzhong_stage1_3d/outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8/)

## GitHub 保留版本

为了方便回退、对照和直接下载，本仓库在 GitHub 上保留了两类“基线存档”：

- 保留分支：`preserve-anzhong-stage1-3d-shot8-v1`
- Release：
  [Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

其中 Release 里挂了大文件结果包：

- `selection_maskable_ppo.zip`


## 仓库结构

### 1. 安中三维主线

- [`anzhong_stage1_3d/src`](./anzhong_stage1_3d/src/)
  当前安中场景的核心代码
- [`anzhong_stage1_3d/outputs`](./anzhong_stage1_3d/outputs/)
  安中主线下的训练结果、路径图、CSV、模型包
- [`anzhong_stage1_3d/src_backups`](./anzhong_stage1_3d/src_backups/)
  重要阶段的源码备份

### 2. 工作区级 src

- [`src`](./src/)
  仓库级别的算法原型与实验脚本，包含 2D、2.5D、多建筑 RL、粗三维 RL 等历史与并行研究代码

### 3. 工作区级 outputs

- [`outputs`](./outputs/)
  早期实验结果、对比图、训练输出、路径复现实验等

## 推荐入口

如果你现在主要是继续安中三维主线，优先看这几个文件：

- [`anzhong_stage1_3d/src/run_anzhong_stage1_3d.py`](./anzhong_stage1_3d/src/run_anzhong_stage1_3d.py)
  安中 stage1 三维入口封装
- [`anzhong_stage1_3d/src/uav_coarse_3d_viewpoint_rl.py`](./anzhong_stage1_3d/src/uav_coarse_3d_viewpoint_rl.py)
  Maskable PPO 视点选取环境、奖励、动作设计
- [`anzhong_stage1_3d/src/uav_coarse_3d_full_pipeline.py`](./anzhong_stage1_3d/src/uav_coarse_3d_full_pipeline.py)
  视点生成、选点、路径组织、结果导出的全链路
- [`anzhong_stage1_3d/src/uav_coarse_3d_planner.py`](./anzhong_stage1_3d/src/uav_coarse_3d_planner.py)
  粗三维候选生成与几何约束
- [`anzhong_stage1_3d/src/uav_anzhong_graph_path_agent.py`](./anzhong_stage1_3d/src/uav_anzhong_graph_path_agent.py)
  分层路径组织与图结构路径优化

如果要看更早的研究脉络，可以再看：

- [`src/uav_2d_planner.py`](./src/uav_2d_planner.py)
- [`src/uav_25d_planner.py`](./src/uav_25d_planner.py)
- [`src/uav_rl_inspection.py`](./src/uav_rl_inspection.py)

## 当前主线方法说明

### 视点候选生成

- 以粗三维 mesh / scene 为输入
- 使用表面法向生成外侧候选
- 结合 building、sector、cluster 做组织
- 避免候选点过度堆在建筑内侧

### 视点选取

- 使用 Maskable PPO
- 动作是结构化三层动作，而不是把所有组合完全摊平成一个大动作表
- 当前重点约束：
  - 认证覆盖率
  - 同站前向重叠
  - 临站旁向重叠
  - `mean_quality_normalized`
  - `quality_good_fraction`
  - `weakest_quality_normalized`

### 路径规划

- 先按 building 组织
- 再按 sector / cluster 组织
- 做 2-opt 局部重排
- 优先尝试安全直连 / Dubins 风格连接
- 如果发现穿模或离建筑过近，则回退到 3D A*
- 不再保留“全局平滑”，只保留转弯半径处的局部平滑

## 如何继续这条主线

建议按下面顺序继续：

1. 先在 `anzhong_stage1_3d` 下改代码和跑实验，不要一上来就动 GitHub 存档。
2. 新实验优先保持：
   - 训练可复现
   - 输出目录命名清楚
   - CSV / Markdown / 路径图齐全
3. 优先对照以下指标：
   - 认证覆盖率
   - 前向重叠率
   - 旁向重叠率
   - 最弱结构段覆盖率
   - 平均质量
   - 拍照质量
   - 路径长度
4. 如果你要继续冲 100% 覆盖率，优先检查：
   - 候选池是否偏内侧
   - 外侧立面候选是否不足
   - 站点 shot 数是否偏少
   - 奖励是否过早偏向“短路径”而牺牲末端补洞

## 命名说明

之前目录里出现过 `handoff_partner_stage1_3d` 这样的临时命名。  
目前活跃主线已经统一改成：

- `anzhong_stage1_3d`

GitHub 上对应的分支、release 和目录也已经同步统一。

## 备注

- 这个仓库里保留了不少历史实验结果，很多目录名带日期和版本号，这是故意保留的实验记录，不建议随便清理。
- 如果后面需要“再保存一版”，建议同时保留：
  - 一个清晰命名的 Git 分支
  - 一个 GitHub Release
  - 一个本地结果目录

