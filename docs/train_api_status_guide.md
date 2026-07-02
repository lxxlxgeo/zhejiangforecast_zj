# Train 接口与状态查询说明

## 1. 是否异步

`POST /api/v1/online-modeling/train` 支持同步和异步两种模式。

同步：

```json
{
  "task_id": "task_sjh_solar_v4",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "train_mode": "local",
  "sync": true
}
```

`sync=true` 会在 HTTP 请求内直接执行训练，训练完成后才返回。返回里保留原来的 `task_id/models` 训练结果，并额外带：

```json
{
  "sync": true,
  "train_mode": "local",
  "runner": "local_inline",
  "task_status": "TRAINED",
  "status_url": "/api/v1/online-modeling/train/status?task_id=task_sjh_solar_v4"
}
```

异步：

```json
{
  "task_id": "task_sjh_solar_v4",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "train_mode": "local",
  "sync": false
}
```

`sync=false` 会提交到本地 `ThreadPoolExecutor` 后立即返回，业务后端应保存 `job_id` 并轮询 `train/status`。

返回重点字段：

```json
{
  "accepted": true,
  "sync": false,
  "train_mode": "local",
  "runner": "local_threadpool",
  "task_id": "task_sjh_solar_v4",
  "job_id": "job_xxxxxxxxxxxx",
  "status": "CREATED",
  "job_status": "CREATED",
  "task_status": "CLEANED",
  "progress": 0.0,
  "done": false,
  "success": false,
  "status_url": "/api/v1/online-modeling/train/status?job_id=job_xxxxxxxxxxxx",
  "task_status_url": "/api/v1/online-modeling/train/status?task_id=task_sjh_solar_v4"
}
```

## 2. 状态查询

按 `job_id` 查询：

```http
GET /api/v1/online-modeling/train/status?job_id=job_xxxxxxxxxxxx
```

按 `task_id` 查询最新训练 job：

```http
GET /api/v1/online-modeling/train/status?task_id=task_sjh_solar_v4
```

可选参数：

- `include_result=true|false`：是否返回 `train_result` 摘要，默认 `true`。
- `log_limit=20`：返回最近训练日志条数，范围 0-200。

返回重点字段：

```json
{
  "task_id": "task_sjh_solar_v4",
  "job_id": "job_xxxxxxxxxxxx",
  "status": "SUCCESS",
  "job_status": "SUCCESS",
  "task_status": "TRAINED",
  "stage": "train",
  "progress": 1.0,
  "done": true,
  "success": true,
  "error_message": null,
  "is_async": true,
  "runner": "local_threadpool",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1"],
  "model_count": 2,
  "trained_model_count": 2,
  "skipped_model_count": 0,
  "models": [
    {
      "model_id": "task_sjh_solar_v4_EC_XGB_PV_V1_20260702120000",
      "model_type": "xgboost",
      "status": "TRAINED",
      "artifact_path": ".../model.joblib",
      "metrics": {
        "mae": 1.23,
        "accuracy": 0.95
      }
    }
  ],
  "train_result_path": ".../reports/train_result.json",
  "latest_logs": [
    {
      "stage": "train",
      "log_level": "INFO",
      "message": "Training finished with 2 trained model(s)"
    }
  ]
}
```

## 3. 状态判断建议

业务后端建议优先使用：

- `done=false`：继续轮询。
- `done=true && success=true`：训练完成，可以进入评估。
- `done=true && success=false`：训练失败或取消，展示 `error_message` 和 `latest_logs`。

`status` 是面向轮询的主状态：如果存在异步 job，则优先等于 `job_status`；否则等于 `task_status`。

