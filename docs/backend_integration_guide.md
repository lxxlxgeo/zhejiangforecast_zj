# 在线建模业务后端对接说明

本文档面向业务后端联调使用，依据《浙江省调在线建模_最小化实现路径技术文档.docx》的最小闭环整理。当前算法服务采用 FastAPI + SQLite，本阶段后端只需要围绕 `task_id` 串联数据接入、建模、评估、发布和推理。

## 1. 服务地址与运行方式

本地 WSL 启动：

```bash
cd /mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj
source /home/lxce/miniconda3/etc/profile.d/conda.sh
conda activate ml_sc
export PYTHONPATH=src
export ZJ_FORECAST_HOME=$PWD/runtime_api
bash scripts/start_api_wsl.sh 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{
  "status": "ok",
  "db_path": ".../runtime_api/zj_forecast.db"
}
```

## 2. 后端调用主流程

业务后端建议按以下顺序调用：

1. `POST /api/v1/online-modeling/ingest` 创建任务并执行数据接入、清洗、NWP 对齐。
2. `GET /api/v1/online-modeling/data/status` 查询数据质量检查结果。
3. `GET /api/v1/online-modeling/data/preview` 预览清洗后训练集或评估集。
4. `POST /api/v1/online-modeling/data/edit` 保存人工编辑点位记录。
5. `POST /api/v1/online-modeling/train` 触发建模。
6. `GET /api/v1/online-modeling/train/status` 轮询训练状态。
7. `POST /api/v1/online-modeling/evaluate` 触发统一评估。
8. `GET /api/v1/online-modeling/evaluate/result` 获取曲线、日准确率、平均准确率和最优模型。
9. `POST /api/v1/online-modeling/publish` 发布选定模型。
10. `POST /api/v1/online-modeling/infer` 使用已发布模型推理。
11. `GET /api/v1/online-modeling/infer/status` 查询推理状态。

`task_id` 是全链路主键。业务系统需要保存 `task_id`、最终 `selected_model_id/published_model_id`，并把它们与场站、方案、用户操作记录关联。

## 3. 状态流

任务状态按 docx 约定实现：

| 状态 | 含义 | 常见触发 |
| --- | --- | --- |
| `CREATED` | 任务已创建，参数已登记 | `ingest` 且 `run_etl=false` |
| `DATA_READY` | 实发、NWP、场站信息已完成读取或索引 | 数据读取完成 |
| `CLEANED` | 清洗与样本构造完成 | `ingest` 默认完成后 |
| `TRAINING` | 候选模型训练中 | `train` |
| `TRAINED` | 候选模型产物已生成 | `train` 完成 |
| `EVALUATED` | 评估结果已生成 | `evaluate` 完成 |
| `PUBLISHED` | 模型已发布，可用于推理 | `publish` 完成 |
| `FAILED` | 任务失败 | 任一阶段异常 |

本地联调可使用 `sync=true` 简化调用；生产联调建议使用 `sync=false`，由业务后端根据 `job_id` 或 `task_id` 轮询状态。

## 4. 接口说明

### 4.1 创建任务与数据接入

`POST /api/v1/online-modeling/ingest`

请求关键字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `task_id` | 否 | 不传则算法服务生成；建议业务后端传入可追溯 ID |
| `station_id`/`region_id` | 二选一 | 单站或区域对象 |
| `object_type` | 是 | `station` 或 `region` |
| `station_type` | 是 | `wind` 或 `solar` |
| `train_start/train_end` | 是 | 训练时间范围 |
| `eval_start/eval_end` | 是 | 评估时间范围 |
| `model_candidates` | 否 | 候选模型列表 |
| `feature_set` | 否 | 方案特征集标识 |
| `station.capacity_mw` | 建议 | 装机容量 MW |
| `station.longitude/latitude` | NWP 必填 | 场站经纬度，支持十进制度或度分秒 |
| `data_paths.power` | 当前测试必填 | 实发功率文件路径 |
| `data_paths.nwp_root` | NWP 必填 | EC NetCDF 根目录 |
| `etl_options` | 否 | NWP 对齐参数，如 `grid_size`、`sequence_steps` |
| `train_options` | 否 | DL 训练参数，如 `device`、`dl_epochs` |
| `run_etl` | 否 | 默认 `true`，建议保持默认 |

风电示例：

```json
{
  "task_id": "backend_flow_demo",
  "station_id": "js_yancheng_h3",
  "object_type": "station",
  "station_type": "wind",
  "train_start": "2025-11-02T00:00:00",
  "train_end": "2025-11-02T23:45:00",
  "eval_start": "2025-11-03T00:00:00",
  "eval_end": "2025-11-03T23:45:00",
  "feature_set": "ec_hres_wind_n1",
  "model_candidates": ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"],
  "station": {
    "capacity_mw": 300.0,
    "longitude": "120°36'08.736477\"",
    "latitude": "034°18'51.964916\""
  },
  "data_paths": {
    "power": "../测试场站数据/js_yanchengH3_wind/station_power/station_power_original_2502-2601.csv",
    "nwp_root": "/mnt/d/data/netcdf/ecmwf/jiangsu"
  },
  "etl_options": {
    "max_nwp_samples": 192,
    "sequence_steps": 9,
    "grid_size": 16,
    "horizon_codes": ["N1"]
  },
  "train_options": {
    "device": "cpu",
    "dl_epochs": 1,
    "dl_batch_size": 4
  }
}
```

返回重点字段：

```json
{
  "task_id": "backend_flow_demo",
  "status": "CLEANED",
  "work_dir": ".../tasks/backend_flow_demo",
  "request_json": {
    "artifacts": {
      "train_dataset": ".../train_dataset_nwp_ml.csv",
      "eval_dataset": ".../eval_dataset_nwp_ml.csv",
      "nwp_train_tensor_x": ".../train_nwp_x.npy"
    },
    "data_summary": {
      "dataset_mode": "nwp_aligned",
      "aligned_samples": 173
    }
  }
}
```

`dataset_mode=nwp_aligned` 表示已经使用 EC NWP 完成时空对齐；如果退回 `power_history_tabular`，需要检查 NWP 文件范围、经纬度和任务时间。

### 4.2 数据状态

`GET /api/v1/online-modeling/data/status?task_id={task_id}`

返回数据检查列表，业务后端重点看：

- `data_type=nwp`：NWP 根目录、文件数、首末报文时间。
- `data_type=power`：实发数据行数、缺失率、时间范围。
- `data_type=wind_cleaning`：风电清洗结果。
- `data_type=nwp_aligned_dataset`：NWP 样本数、张量通道、失败行数。
- `data_type=dataset`：最终训练/评估集摘要。

### 4.3 数据预览

`GET /api/v1/online-modeling/data/preview?task_id={task_id}&data_type=eval&limit=200`

`data_type` 可传：

- `train`：训练样本。
- `eval`：评估样本。
- 其他值当前回退到清洗后序列。

返回 `rows` 为表格记录，前端可直接画实发功率、风速、NWP 特征等预览曲线。

### 4.4 点位编辑记录

`POST /api/v1/online-modeling/data/edit`

请求：

```json
{
  "task_id": "backend_flow_demo",
  "point_edits": [
    {
      "time": "2025-11-03T00:00:00",
      "field": "power_mw",
      "value": null,
      "reason": "人工标记异常点"
    }
  ]
}
```

当前实现会把编辑记录保存到 `tasks/{task_id}/data/point_edits.json`，并登记 `data_type=point_edits` 的数据检查记录。第一阶段只做审计保存，不自动重放并重算样本。

### 4.5 触发建模

`POST /api/v1/online-modeling/train`

同步联调请求：

```json
{
  "task_id": "backend_flow_demo",
  "model_candidates": ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"],
  "train_mode": "local",
  "sync": true
}
```

异步请求：

```json
{
  "task_id": "backend_flow_demo",
  "model_candidates": ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"],
  "train_mode": "local",
  "sync": false
}
```

同步返回包含每个候选模型的训练状态、指标和产物路径。异步返回 `job_id/status/stage`，业务后端用训练状态接口轮询。

### 4.6 训练状态

`GET /api/v1/online-modeling/train/status?task_id={task_id}`

或：

`GET /api/v1/online-modeling/train/status?job_id={job_id}`

本地同步训练完成后，按 `task_id` 查询通常返回任务主表状态，例如 `TRAINED`。

### 4.7 模型列表

`GET /api/v1/online-modeling/model/list?station_type=wind&object_type=station`

返回当前可选模型方案。风电常用：

- `EC_LGB_WIND_V1`
- `EC_XGB_WIND_V1`
- `EC_SWIN3D_WIND_V1`
- `EC_LORA_WIND_V1.1`
- `PERSISTENCE_BASELINE`

光伏常用：

- `EC_LGB_PV_V1`
- `EC_XGB_PV_V1`
- `EC_SWIN3D_PV_V1`
- `EC_LORA_PV_V1.1`
- `PERSISTENCE_BASELINE`

### 4.8 评估

`POST /api/v1/online-modeling/evaluate`

```json
{
  "task_id": "backend_flow_demo",
  "sync": true
}
```

评估会在同一验证集上比较候选模型，并写入：

- `online_model_eval`：指标。
- `online_model_curve`：实发和预测曲线。
- `tasks/{task_id}/reports/eval_result.json`：评估汇总。

### 4.9 评估结果

`GET /api/v1/online-modeling/evaluate/result?task_id={task_id}`

业务后端/前端重点字段：

- `models`：候选模型列表、版本、指标。
- `curve.real`：验证时段实发功率序列。
- `curve.predictions`：各模型预测序列。
- `daily_accuracy`：日准确率。
- `avg_accuracy`：统计周期平均准确率。
- `selected_model`：算法侧按指标选出的最优模型。
- `quality_summary`：数据质量摘要。

### 4.10 发布模型

`POST /api/v1/online-modeling/publish`

自动发布评估最优模型：

```json
{
  "task_id": "backend_flow_demo"
}
```

指定模型发布：

```json
{
  "task_id": "backend_flow_demo",
  "selected_model_id": "backend_flow_demo_EC_LGB_WIND_V1_20260629150000"
}
```

返回 `model_id/version/artifact_path`。发布后，任务状态变为 `PUBLISHED`，任务主表记录 `published_model_id`。

### 4.11 推理

`POST /api/v1/online-modeling/infer`

按 `task_id` 使用已发布模型：

```json
{
  "task_id": "backend_flow_demo",
  "issue_time": "2025-11-03 12:00:00"
}
```

按 `model_id` 指定模型：

```json
{
  "model_id": "backend_flow_demo_PERSISTENCE_BASELINE_20260629150115",
  "issue_time": "2025-11-03 12:00:00"
}
```

返回：

```json
{
  "infer_id": "infer_xxxxxxxx",
  "task_id": "backend_flow_demo",
  "model_id": "...",
  "issue_time": "2025-11-03 12:00:00",
  "predictions": [
    {"valid_time": "2025-11-03 16:00:00", "p_pred_mw": 123.4}
  ]
}
```

当前通用推理接口对 tabular 模型直接加载模型文件；对 Swin3D/LoRA 模型，如果请求没有携带在线 NWP tensor，则使用评估预测曲线 fallback，权重和 adapter 仍保存在模型目录。生产级按 `issue_time` 实时读取 EC NetCDF 并构造 DL tensor 的能力属于下一阶段增强点。

### 4.12 推理状态

`GET /api/v1/online-modeling/infer/status?infer_id={infer_id}`

当前本地推理为同步执行，成功返回：

```json
{
  "infer_id": "infer_xxxxxxxx",
  "status": "SUCCESS",
  "note": "Local inference is synchronous in phase 1."
}
```

## 5. NWP 与时间约定

- NWP 测试路径：`/mnt/d/data/netcdf/ecmwf/jiangsu`，Windows 对应 `D:\data\netcdf\ecmwf\jiangsu`。
- 当前 EC HRES 训练优先使用 `12Z` 起报。
- `N1` 表示 `12Z` 起报后的下一个北京时间自然日 `00:00-23:45`。
- 接口时间支持 `"2025-11-02T00:00:00"` 或 `"2025-11-02 00:00:00"`。在 shell/CLI 中建议使用带 `T` 的 ISO 格式，避免空格被命令行错误拆分。
- 当前测试数据的 NWP 文件从 2025-11 开始，使用 2025-02 时间段会找不到对应 NWP 并退回历史功率表格样本。

## 6. 可重复验收脚本

工程内置了一个后端接口闭环脚本：

```bash
cd /mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj
source /home/lxce/miniconda3/etc/profile.d/conda.sh
conda activate ml_sc
PYTHONPATH=src python scripts/run_backend_api_flow.py \
  --home runtime_backend_api_flow_real1 \
  --task-id backend_flow_real1
```

该脚本按 docx 的接口流程调用 FastAPI app：

`health -> model/list -> ingest -> data/status -> data/preview -> data/edit -> train -> train/status -> evaluate -> evaluate/result -> publish -> infer -> infer/status`

本地已验证结果：

```json
{
  "task_id": "backend_flow_real1",
  "ingest_status": "CLEANED",
  "data_status": "CLEANED",
  "dataset_mode": "nwp_aligned",
  "aligned_samples": 173,
  "preview_rows": 3,
  "saved_point_edits": 1,
  "trained_models": 2,
  "train_status": "TRAINED",
  "eval_model_count": 2,
  "published_model_id": "backend_flow_real1_PERSISTENCE_BASELINE_20260629150115",
  "infer_points": 96,
  "infer_status": "SUCCESS"
}
```

脚本结果同时写入：

`runtime_backend_api_flow_real1/backend_api_flow_summary.json`

## 7. 业务后端落库建议

业务系统侧至少保存以下字段：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `task_id` | ingest 请求或返回 | 全链路关联 |
| `station_id/region_id` | ingest 请求 | 业务对象关联 |
| `station_type/object_type` | ingest 请求 | 风光/单站区域区分 |
| `train_start/train_end/eval_start/eval_end` | ingest 请求 | 复现实验范围 |
| `model_candidates` | ingest/train 请求 | 模型方案追溯 |
| `status` | 各状态接口 | 前端展示和轮询 |
| `selected_model_id` | evaluate/result | 发布确认 |
| `published_model_id` | publish | 推理默认模型 |
| `avg_accuracy/daily_accuracy` | evaluate/result | 前端评估展示 |
| `infer_id` | infer | 推理请求追踪 |

算法服务侧 SQLite 会保存运行明细，但业务系统仍建议保存上述主索引，方便跨系统查询和页面回显。

## 8. 常见失败与处理

| 问题 | 表现 | 处理 |
| --- | --- | --- |
| NWP 时间段不匹配 | `dataset_mode=power_history_tabular`，`nwp_error` 提示 missing issue | 换到有 NetCDF 的时间段，例如 2025-11 以后 |
| 场站坐标缺失 | NWP 对齐无法执行 | ingest 必须传 `station.longitude/latitude` |
| 装机容量缺失 | 准确率和归一化误差不稳定 | ingest 传 `station.capacity_mw` |
| 训练时间较长 | HTTP 长时间等待 | 生产改用 `sync=false` 并轮询 `train/status` |
| DL 推理不是实时 NWP | Swin3D/LoRA 使用 fallback 曲线 | 下一阶段接入按 issue_time 构造在线 NWP tensor |
| 重复 task_id | SQLite 主键冲突 | 后端保证 task_id 唯一，或每次联调使用新 task_id |

## 9. 当前阶段边界

- Airflow DAG 已提供，但本地测试默认走 FastAPI 内置 local runner。
- `data/edit` 已持久化编辑记录，但不自动重放并重算训练样本。
- tabular 模型可直接做通用推理；Swin3D/LoRA 的实时 NWP tensor 推理待下一阶段增强。
- 当前 SQLite 适合本机联调；现场部署可按业务环境替换为正式数据库，表结构已集中在 `src/zhejiangforecast_zj/db/schema.py`。
