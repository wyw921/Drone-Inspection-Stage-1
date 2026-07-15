# 安中 Stage1 3D 工作目录说明

这个目录是当前安中大楼粗三维巡检主线的独立工作区。  
如果只关心“安中场景怎么训练、怎么跑全链路、怎么找基线结果”，从这里开始就可以。

## 这条主线在做什么

目标是面向三维重建友好的无人机离线规划：

- 输入：安中大楼 LOD1 粗模
- 输出：
  - 一组可用于拍摄的候选 / 选中视点
  - 对应 shot 数与 standoff 配置
  - 一条满足安全性约束的分层飞行路径
  - 可视化图、CSV 结果、Markdown 汇总、模型缓存

当前主链路是：

`候选视点生成 -> Maskable PPO 选点 -> 分层路径规划 -> 路径图 / 结果导出`

## 目录结构

- [`src`](./src/)
  当前安中主线源码
- [`outputs`](./outputs/)
  安中主线下的训练结果和全链路输出
- [`src_backups`](./src_backups/)
  若干关键版本的源码备份
- [`docs`](./docs/)
  补充说明材料

## 推荐先看

- [`src/run_anzhong_stage1_3d.py`](./src/run_anzhong_stage1_3d.py)
  安中 stage1 三维入口
- [`src/uav_coarse_3d_viewpoint_rl.py`](./src/uav_coarse_3d_viewpoint_rl.py)
  视点选取 PPO 的核心逻辑
- [`src/uav_coarse_3d_full_pipeline.py`](./src/uav_coarse_3d_full_pipeline.py)
  从候选到路径的完整主链路
- [`src/uav_coarse_3d_planner.py`](./src/uav_coarse_3d_planner.py)
  粗三维候选生成与几何约束
- [`src/uav_anzhong_graph_path_agent.py`](./src/uav_anzhong_graph_path_agent.py)
  分层路径组织与图路径优化

## 当前保留基线

当前最重要的基线输出目录是：

- [`outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8`](./outputs/anzhong_capture_full_pipeline_fullscale_rewrite_v3_noglobalsmooth_8k_multiseed_shot8/)

其中可以重点看：

- `coarse_3d_full_pipeline_results.md`
- `coarse_3d_full_pipeline_results.csv`
- `baseline_path_3d.png`
- `hierarchical_path_3d.png`
- `baseline_path_waypoints.csv`
- `hierarchical_path_waypoints.csv`
- `hierarchical_selected_captures.csv`

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

## 关键方法约束

### 覆盖与重叠

这里的 overlap 不是简单全局统计，而是更偏向重建友好的局部连接约束：

- 同站前向重叠
- 临站旁向重叠

目标不是“随便盖住”，而是让相邻拍摄序列更容易连成可重建的照片网络。

### 质量

当前质量奖励不只看覆盖率，还联合推动：

- `mean_quality_normalized`
- `quality_good_fraction`
- `weakest_quality_normalized`

completion bonus 也要求覆盖、overlap、quality 共同达线。

### 路径

当前路径组织逻辑是：

1. 先按 building 分组
2. 再按 sector / cluster 分组
3. 做 2-opt 局部排序
4. 优先尝试安全直连
5. 如果穿模或距离建筑过近，则回退到 3D A*
6. 仅保留转弯半径处的局部平滑，不保留全局平滑

## 结果包与 GitHub 存档

GitHub 上已经为这条主线保留了 release：

- [Anzhong Stage1 3D Preserve Fullscale Shot8 V1](https://github.com/wyw921/Drone-Inspection-Stage-1/releases/tag/anzhong-stage1-3d-preserve-fullscale-shot8-v1)

其中大文件结果包：

- `selection_maskable_ppo.zip`

由于这个文件较大，放在 release asset 里而不是普通仓库文件里。

## 后续建议

如果继续沿这条线往前推，建议优先做：

1. 扩大候选池但保持候选外侧分布合理。
2. 继续提升末端覆盖空洞，冲击 100% 认证覆盖率。
3. 当 7 张不够时，系统性比较 8 张 / 9 张 shot 配置。
4. 对比不同 seed 下的 coverage、overlap、quality、path length 稳定性。
5. 所有重要新版本都保留：
   - 一个命名清楚的输出目录
   - 一个 Git 分支
   - 一个 release 或结果包

