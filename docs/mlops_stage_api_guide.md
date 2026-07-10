# 浙江功率预测 MLOps 分阶段接口说明

本文档说明新增的资源化 MLOps 接口。旧接口 `/api/v1/online-modeling/ingest`、`train`、`evaluate`、`publish` 保持兼容。

统一返回格式：

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {}
}
```

## 1. 数据库配置

生产建议使用 PostgreSQL：

```bash
export ZJ_FORECAST_DB_URL='postgresql+psycopg://mlops:lxce@123A@127.0.0.1:5432/zj_mlops'
```

也可以写入 `configs/default.yml`：

```yaml
database:
  url: postgresql+psycopg://mlops:lxce@123A@127.0.0.1:5432/zj_mlops
```

建表方式：

```bash
alembic -c alembic.ini upgrade head
```

本地最小部署仍保留 SQLAlchemy `create_all` 自动建表。

## 2. 场站注册

`POST /api/v1/mlops/stations`

```json
{
  "station_id": "nmg_shuijinghu_solar_v1",
  "station_type": "solar",
  "station_name": "内蒙古水晶湖光伏",
  "longitude": 109.7,
  "latitude": 40.30528,
  "capacity_mw": 100.0,
  "metadata": {
    "owner": "backend"
  }
}
```

查询：

- `GET /api/v1/mlops/stations/{station_id}`
- `GET /api/v1/mlops/stations?station_type=solar&limit=200`

## 3. 数据资产注册

`POST /api/v1/mlops/data/assets`

```json
{
  "asset_id": "asset_sjh_raw_power_202503_202509",
  "station_id": "nmg_shuijinghu_solar_v1",
  "asset_type": "raw_power",
  "uri": "/data/share/data/power_forecast/test/sjh_power.csv",
  "format": "csv",
  "status": "READY"
}
```

查询：

- `GET /api/v1/mlops/data/assets/{asset_id}`
- `GET /api/v1/mlops/data/assets?station_id=nmg_shuijinghu_solar_v1&asset_type=raw_power`

## 4. 清洗

`POST /api/v1/mlops/cleaning/runs`

```json
{
  "task_id": "task_sjh_solar_stage_demo",
  "station_id": "nmg_shuijinghu_solar_v1",
  "station_type": "solar",
  "power_asset_id": "asset_sjh_raw_power_202503_202509",
  "etl_options": {
    "enable_solar_cleaning": true
  },
  "sync": true
}
```

返回重点：

```json
{
  "run_id": "run_xxx",
  "task_id": "task_sjh_solar_stage_demo",
  "status": "SUCCESS",
  "clean_asset": {
    "asset_type": "clean_power",
    "uri": ".../tasks/task_sjh_solar_stage_demo/data/clean_series.csv"
  },
  "summary": {
    "clean_rows": 1234,
    "removed_rows": 12
  }
}
```

查看清洗结果：

- `GET /api/v1/mlops/cleaning/runs/{run_id}`
- `GET /api/v1/mlops/cleaning/runs/{run_id}/summary`
- `GET /api/v1/mlops/cleaning/runs/{run_id}/preview?limit=200`

## 5. ETL

`POST /api/v1/mlops/etl/runs`

```json
{
  "task_id": "task_sjh_solar_stage_demo",
  "station_id": "nmg_shuijinghu_solar_v1",
  "station_type": "solar",
  "power_asset_id": "asset_sjh_raw_power_202503_202509",
  "train_start": "2025-03-01 00:00:00",
  "train_end": "2025-06-30 23:45:00",
  "eval_start": "2025-09-01 00:00:00",
  "eval_end": "2025-09-30 23:45:00",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "etl_options": {
    "horizon_codes": ["N1"],
    "sequence_steps": 5,
    "grid_size": 16
  },
  "sync": true
}
```

查看 ETL 样本：

- `GET /api/v1/mlops/etl/runs/{run_id}`
- `GET /api/v1/mlops/etl/runs/{run_id}/preview?data_type=train&limit=200`
- `GET /api/v1/mlops/etl/runs/{run_id}/preview?data_type=eval&limit=200`

`data_summary.dataset_mode=nwp_aligned` 表示 NWP 对齐成功。若任务失败，优先检查 NWP 根目录、站点经纬度、时间范围和报文完整性。

## 6. 训练

`POST /api/v1/mlops/training/runs`

```json
{
  "task_id": "task_sjh_solar_stage_demo",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "sync": true
}
```

异步：

```json
{
  "task_id": "task_sjh_solar_stage_demo",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "sync": false
}
```

状态：

- `GET /api/v1/mlops/training/runs/{run_id}`
- `GET /api/v1/mlops/training/runs?task_id=task_sjh_solar_stage_demo`

## 7. 评估

`POST /api/v1/mlops/evaluation/runs`

```json
{
  "task_id": "task_sjh_solar_stage_demo",
  "sync": true
}
```

状态：

- `GET /api/v1/mlops/evaluation/runs/{run_id}`
- `GET /api/v1/mlops/evaluation/runs?task_id=task_sjh_solar_stage_demo`

旧评估接口仍按业务要求返回每日精度：

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {
    "task_id": "task_sjh_solar_stage_demo",
    "daily_accuracy": []
  }
}
```

## 8. 发布

`POST /api/v1/mlops/models/publish`

```json
{
  "task_id": "task_sjh_solar_stage_demo",
  "selected_model_id": null
}
```

查询：

- `GET /api/v1/mlops/models/published?station_id=nmg_shuijinghu_solar_v1`
- `GET /api/v1/mlops/models/published/{published_model_id}`

## 9. 通用运行查询

- `GET /api/v1/mlops/pipeline/runs?task_id=task_sjh_solar_stage_demo`
- `GET /api/v1/mlops/pipeline/runs/{run_id}`

返回中包含：

- `params_json`：运行参数。
- `result_json`：阶段输出。
- `lineage`：输入资产和输出资产关系。
- `task`：关联旧任务主表摘要。
- `job`：异步训练/评估任务状态。

## 10. 本地测试

单元与接口测试：

```bash
cd /mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj
PYTHONPATH=src /home/lxce/miniconda3/envs/ml_sc/bin/python -m unittest discover -s tests -v
```

PG smoke：

```bash
cd /mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj
PYTHONPATH=src ZJ_FORECAST_DB_URL='postgresql+psycopg://mlops:lxce@123A@127.0.0.1:5432/zj_mlops' \
  /home/lxce/miniconda3/envs/ml_sc/bin/python scripts/pg_smoke.py
```
