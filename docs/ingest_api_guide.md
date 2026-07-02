# ingest 接口对接说明

本文档面向业务后端，说明在线建模 `ingest` 接口的请求格式、返回格式、字段含义和常见问题。

## 1. 接口用途

`ingest` 用于创建在线建模任务，并在 `run_etl=true` 时立即执行数据接入、清洗、NWP 对齐和训练/评估样本构造。

接口：

```http
POST /api/v1/online-modeling/ingest
Content-Type: application/json
```

统一返回外壳：

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {}
}
```

说明：

- HTTP 200 只表示接口调用成功。
- 业务是否成功看 `data.status`。
- `data.status=CLEANED` 表示 ETL 已完成，可以进入训练。
- `data.status=FAILED` 表示任务失败，查看 `data.error_message`。
- `run_etl=false` 时只创建任务，通常返回 `CREATED`。

## 2. 推荐请求格式

生产建议使用 `powerData` 内联数组，不强制传 `data_paths.power` 文件路径。

```json
{
  "station": {
    "longitude": 120.1364,
    "latitude": 30.6864,
    "capacity_mw": 500.0
  },
  "task_id": "task_test_solar_powerdata",
  "station_id": "1113300000021",
  "region_id": null,
  "object_type": "station",
  "station_type": "solar",
  "train_start": "2025-03-01 00:00:00",
  "train_end": "2025-06-30 23:45:00",
  "eval_start": "2025-09-01 00:00:00",
  "eval_end": "2025-09-30 23:45:00",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "feature_set": "ec_hres_solar_n1",
  "data_paths": {},
  "etl_options": {
    "sequence_steps": 9,
    "grid_size": 16,
    "horizon_codes": ["N1"]
  },
  "train_options": {
    "device": "cpu"
  },
  "run_etl": true,
  "powerData": [
    {
      "dataTime": "2025-03-01 00:00:00",
      "actualPower": 0.0,
      "theoryPower": null,
      "windSpeed": null,
      "actualIrradiance": 0.0
    }
  ]
}
```

完整风电/光伏测试 JSON：

- `experiments/task_test/wind/ingest.json`
- `experiments/task_test/solar/ingest.json`

## 3. 字段说明

### 3.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `task_id` | string | 建议必填 | 业务侧任务 ID。建议业务后端生成并保存。 |
| `station_id` | string | 单站建议必填 | 场站 ID。 |
| `region_id` | string/null | 否 | 区域 ID。当前主要按单站使用。 |
| `object_type` | string | 是 | 当前使用 `station`。 |
| `station_type` | string | 是 | `wind` 或 `solar`。 |
| `train_start` | string | 是 | 训练开始时间，北京时间。 |
| `train_end` | string | 是 | 训练结束时间，北京时间。 |
| `eval_start` | string | 是 | 评估开始时间，北京时间。 |
| `eval_end` | string | 是 | 评估结束时间，北京时间。 |
| `model_candidates` | array | 建议必填 | 候选模型列表。 |
| `feature_set` | string | 否 | 特征方案标识，用于记录。 |
| `station` | object | NWP 建模必填 | 场站经纬度、容量。 |
| `data_paths` | object | 可空 | 文件路径配置。生产用 `powerData` 时可传 `{}`。 |
| `etl_options` | object | 否 | ETL 参数。 |
| `train_options` | object | 否 | 训练参数。 |
| `run_etl` | boolean | 否 | 默认 `true`。 |
| `powerData` | array | 推荐必填 | 实发功率序列。 |

### 3.2 station

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `longitude` | number/string | NWP 必填 | 经度，支持十进制度。 |
| `latitude` | number/string | NWP 必填 | 纬度，支持十进制度。 |
| `capacity_mw` | number | 建议必填 | 装机容量，单位 MW。评估归一化误差和清洗会用到。 |
| `station_name` | string | 否 | 场站名称，仅记录。 |

### 3.3 powerData

每条记录代表一个时间点。

| 字段 | 类型 | 必填 | 内部映射 | 说明 |
| --- | --- | --- | --- | --- |
| `dataTime` | string | 是 | `time_bj` | 北京时间。 |
| `utcTime` | string | 否 | `time_utc` | 如果不传，系统按 `dataTime - 8h` 生成。 |
| `actualPower` | number | 是 | `power_mw` | 实发功率，单位 MW。建模标签 y 只使用这个字段。 |
| `theoryPower` | number/null | 否 | `theoretical_power` | 理论功率，清洗审计辅助字段。 |
| `windSpeed` | number/null | 风电建议 | `wind_speed_mean` | 风电场站平均风速。 |
| `directIrradiance` | number/null | 光伏可传 | `direct_irradiance` | 光伏辐照度。 |
| `actualIrradiance` | number/null | 光伏可传 | `direct_irradiance` | 当前可识别为辐照度。 |
| `irradiance` | number/null | 光伏可传 | `direct_irradiance` | 当前可识别为辐照度。 |

要求：

- 时间建议为 15 分钟间隔。
- `powerData` 必须覆盖完整训练窗口和评估窗口。
- `dataTime` 可倒序传入，系统会排序。
- 不建议重复 `dataTime`；如重复，系统会按时间去重并保留最后一条。
- 光伏夜间大功率、低辐照高功率等异常会被清洗规则剔除。

### 3.4 data_paths

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `power` | string/null | 否 | 实发数据文件路径。使用 `powerData` 时不需要。 |
| `power_path` | string/null | 否 | `power` 的别名。 |
| `station_info` | string/null | 否 | 场站信息文件路径。 |
| `nwp_root` | string/null | 否 | NWP 文件根目录。生产建议不传，走配置文件默认值。 |

NWP 路径优先级：

1. 请求中的 `data_paths.nwp_root`
2. `configs/default.yml` 中 `nwp.roots.{station_type}`
3. `configs/default.yml` 中 `nwp.default_root`

可通过健康检查确认服务实际读取到的配置：

```bash
curl http://127.0.0.1:8000/health
```

重点看：

- `data.config_path`
- `data.project_root`
- `data.nwp_roots`

### 3.5 etl_options

| 字段 | 类型 | 建议值 | 说明 |
| --- | --- | --- | --- |
| `sequence_steps` | int | `9` | DL NWP tensor 时间步。 |
| `grid_size` | int | `16` | 场站周边网格大小。 |
| `horizon_codes` | array | `["N1"]` | 当前短期建模主要用 N1。 |
| `enable_wind_cleaning` | boolean | 可选 | 是否启用风电清洗。 |
| `enable_solar_cleaning` | boolean | 可选 | 是否启用光伏辐照清洗。 |
| `max_nwp_samples` | int | 生产不建议传 | 调试用样本截断。正式建模不要传。 |

重要：`max_nwp_samples` 会在样本按时间排序后截取前 N 条。如果传 `192`，只会保留约 2 天 15 分钟样本，可能导致评估集为空。

### 3.6 train_options

`ingest` 阶段只记录训练选项，训练阶段会使用。

常见字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `device` | string | `cpu`、`cuda`、`xpu` 等。 |
| `dl_epochs` | int | DL 小样本训练轮数。 |
| `dl_batch_size` | int | DL batch size。 |

## 4. 返回格式

成功调用返回：

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {
    "task_id": "task_test_solar_powerdata",
    "object_type": "station",
    "station_type": "solar",
    "station_id": "1113300000021",
    "region_id": null,
    "status": "CLEANED",
    "train_start": "2025-03-01 00:00:00",
    "train_end": "2025-06-30 23:45:00",
    "eval_start": "2025-09-01 00:00:00",
    "eval_end": "2025-09-30 23:45:00",
    "feature_set": "ec_hres_solar_n1",
    "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
    "request_json": {
      "station": {
        "station_id": "1113300000021",
        "longitude": 120.1364,
        "latitude": 30.6864,
        "capacity_mw": 500.0
      },
      "artifacts": {
        "clean_series": ".../data/clean_series.csv",
        "train_dataset": ".../data/nwp_aligned/train_dataset_nwp_ml.csv",
        "eval_dataset": ".../data/nwp_aligned/eval_dataset_nwp_ml.csv",
        "feature_schema": ".../data/feature_schema.json",
        "nwp_train_tensor_x": ".../data/nwp_aligned/train_nwp_x.npy",
        "nwp_eval_tensor_x": ".../data/nwp_aligned/eval_nwp_x.npy"
      },
      "data_summary": {
        "check_result": "PASS",
        "dataset_mode": "nwp_aligned",
        "aligned_samples": 10000,
        "train_samples": 7000,
        "eval_samples": 3000,
        "feature_contract": "x=nwp_only,y=power_mw",
        "nwp": {
          "nwp_root": "/data/share/data/power_foreacst/re_prosses_ec1h/china",
          "file_count": 1000
        }
      },
      "powerData_total_size": 14592
    },
    "work_dir": ".../runtime/tasks/task_test_solar_powerdata",
    "config_path": ".../runtime/tasks/task_test_solar_powerdata/config/task_config.json",
    "error_message": null
  }
}
```

注意：

- 返回给业务后端的 `request_json` 不会回传完整 `powerData`，只返回 `powerData_total_size`。
- 完整原始 `powerData` 会保存到任务目录下的 `data/source_powerData.json`，用于审计。
- 训练/评估数据、NWP tensor、特征 schema 等仍按原目录结构输出。

## 5. 业务后端重点检查字段

### 5.1 任务状态

看：

```json
{
  "data": {
    "status": "CLEANED",
    "error_message": null
  }
}
```

含义：

- `CLEANED`：ETL 成功，可以调训练接口。
- `CREATED`：只创建任务，未执行 ETL；常见于 `run_etl=false` 或未提供实发数据。
- `FAILED`：ETL 失败，看 `error_message`。

### 5.2 数据模式

看：

```json
{
  "dataset_mode": "nwp_aligned"
}
```

含义：

- `nwp_aligned`：已使用 EC NWP 与实发功率完成对齐。
- EC 风/光模型要求必须是 `nwp_aligned`。
- 当前 EC XGB/LGB/Swin3D/LoRA 不允许静默退回历史功率特征。

### 5.3 特征契约

看：

```json
{
  "feature_contract": "x=nwp_only,y=power_mw"
}
```

含义：

- 训练标签 `y` 只使用 `actualPower -> power_mw`。
- 模型输入 `X` 只使用 NWP 提取后的 `lead_hours + nwp_*` 特征。
- 推理阶段不需要实发功率。

### 5.4 样本数

看：

```json
{
  "aligned_samples": 10000,
  "train_samples": 7000,
  "eval_samples": 3000
}
```

要求：

- `train_samples > 0`
- `eval_samples > 0`
- 如果 `eval_samples=0`，通常是 `powerData` 没覆盖评估窗口，或 `max_nwp_samples` 截断过小。

## 6. 常见问题

### 6.1 powerData 没覆盖训练/评估窗口

错误示例：

```json
{
  "train_start": "2025-08-11 00:00:00",
  "train_end": "2025-08-15 00:00:00",
  "eval_start": "2025-08-15 15:00:00",
  "eval_end": "2025-08-28 00:00:00",
  "powerData": [
    {"dataTime": "2025-08-11 00:00:00"},
    {"dataTime": "2025-08-21 00:45:00"}
  ]
}
```

这里 `powerData` 实际只到 `2025-08-21 00:45:00`，但 `eval_end` 到 `2025-08-28 00:00:00`，评估后半段没有实发数据。

### 6.2 max_nwp_samples 截断导致评估集为空

如果传：

```json
{
  "etl_options": {
    "max_nwp_samples": 192
  }
}
```

系统会在按时间排序后只保留前 192 个样本，大约 2 天数据。如果训练窗口在前面，评估窗口在后面，评估集可能直接变成 0 条。

正式建模建议不要传 `max_nwp_samples`。

### 6.3 NWP 文件范围不覆盖任务时间

如果 NWP 根目录没有对应起报文件，`nwp_aligned_dataset` 会记录失败原因。

检查：

```bash
GET /api/v1/online-modeling/data/status?task_id={task_id}
```

重点看：

- `data_type=nwp`
- `data_type=nwp_aligned_dataset`
- `data_type=dataset`

### 6.4 经纬度或容量缺失

EC NWP 模型必须有：

```json
{
  "station": {
    "longitude": 120.1364,
    "latitude": 30.6864,
    "capacity_mw": 500.0
  }
}
```

缺经纬度会导致无法提取场站周边 NWP。

### 6.5 光伏清洗删除较多样本

如果传了辐照字段，光伏清洗会检查：

- 夜间或低辐照高功率
- 高辐照零功率
- 功率超容量
- 辐照异常

如果业务传入的是模拟随机数，可能被清洗大量剔除。可在联调阶段临时传：

```json
{
  "etl_options": {
    "enable_solar_cleaning": false
  }
}
```

正式数据建议开启清洗。

## 7. curl 示例

```bash
curl -X POST http://127.0.0.1:8000/api/v1/online-modeling/ingest \
  -H "Content-Type: application/json" \
  -d @experiments/task_test/solar/ingest.json
```

然后查询数据状态：

```bash
curl "http://127.0.0.1:8000/api/v1/online-modeling/data/status?task_id=task_test_solar_powerdata"
```

成功后触发训练：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/online-modeling/train \
  -H "Content-Type: application/json" \
  -d @experiments/task_test/solar/train.json
```

