# Coarse Models

当前仓库内已经整理好的可直接用粗模型有 3 套：

- `anzhong_tower_single`
- `anzhong_tower_plus_north_teaching`
- `anzhong_surrounding_buildings`

目录约定：

- 每套模型都放在 `coarse_models/<model_key>/coarse_model/`
- 统一包含 `coarse_scene.ply`
- 如果原始数据提供了结构化场景，也一并保留 `coarse_scene.json`
- 如果原始数据提供了几何交换格式，也一并保留 `coarse_scene.obj`

推荐入口：

- 跑全链路：`python3 src/run_registered_full_pipeline.py --model-key <model_key>`
- 查看可用模型：`python3 src/run_registered_full_pipeline.py --list-models`

当前模型说明：

| model_key | 用途 | 说明 |
|---|---|---|
| `anzhong_tower_single` | 安中大楼单体建筑 | 安中大楼基础粗模型，适合作为单建筑主基线 |
| `anzhong_tower_plus_north_teaching` | 安中大楼 + 北教学楼 | 21 个原始建筑块、归并后约 4 个主建筑，适合作为 campus 泛化测试 |
| `anzhong_surrounding_buildings` | 安中大楼建筑群（周边版） | 小型多建筑对照，包含 `Construction lab / Anzhong gate / Computing center / Clock tower`，更接近“安中周边建筑群”而不是“安中 + 北教学楼” |
