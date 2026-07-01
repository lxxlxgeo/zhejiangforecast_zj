# 算法后端与算法工程重构说明

本文档记录本次重构后的工程边界、目录结构、外部算法迁移、数据库改造和验证结果。

## 1. 重构目标

本次重构解决三个问题：

1. 原工程通过 `zhejiangforecast_zj/src/zhejiangforecast_zj/adapters/external_projects.py` 动态引用 workspace 下的外部项目，部署时会依赖同级目录存在。
2. 算法后端接口、任务编排、数据库访问与算法 ETL/模型适配混在同一层级，不利于后续维护。
3. 数据库层是手写 SQLite SQL，后续切 PostgreSQL 成本较高；ETL 二进制缓存仍有 pickle 写出。

重构后保留原 FastAPI 接口和业务流程不变，重点调整内部边界。

## 2. 新工程边界

当前包内分为两层：

### 算法后端层

负责接收业务请求、任务状态、数据库、调度和结果返回：

```text
src/zhejiangforecast_zj/
  api/                FastAPI 路由
  schemas.py          API 请求/响应 schema
  services/           任务编排、训练触发、评估、发布、推理服务
  db/                 SQLAlchemy ORM、session、repository
  core/               配置、枚举、路径、时间语义
```

### 算法工程层

负责数据清洗、NWP ETL、模型适配和 vendored 算法代码：

```text
src/zhejiangforecast_zj/algorithm_engine/
  adapters/           服务层到算法实现的适配器
  etl/                后续 ETL 合同和数据集构造扩展点
  models/             后续模型推理/训练扩展点
  persistence/        joblib 等算法产物持久化工具
  vendor/             从原始算法工程迁入的源码
```

旧目录 `src/zhejiangforecast_zj/adapters/` 目前只保留兼容转发，不再维护外部 workspace 路径。

## 3. 外部算法迁移清单

已迁入本工程的源码：

| 原始工程 | 新位置 | 当前用途 |
| --- | --- | --- |
| `数据清洗_bygptpro/h3_wind_cleaning_project/src/wind_cleaning` | `algorithm_engine/vendor/wind_cleaning` | 风电清洗 pipeline |
| `baseline_mlops_by_gptpro/power_ml_baseline/src/power_ml_baseline` | `algorithm_engine/vendor/power_ml_baseline` | ML baseline 参考数据合同 |
| `nwp_temporal_downscaling_v2_project/.../src/nwp_temporal_downscaling` | `algorithm_engine/vendor/nwp_temporal_downscaling` | NWP 1h 到 15min 插值 |
| `短期模型修改部分/met_swin3d_nwp_power_v2/swin3d_nwp_power/swin3d_power` | `algorithm_engine/vendor/swin3d_power` | Swin3D 短期模型 |
| `lora_swin3d_power_project/.../src/lora_swin3d_power` | `algorithm_engine/vendor/lora_swin3d_power` | LoRA-Swin3D adapter 训练 |

新的路径管理文件：

`src/zhejiangforecast_zj/algorithm_engine/adapters/vendor_paths.py`

它只把 `algorithm_engine/vendor` 加入 `sys.path`，不再查找 workspace 下的外部项目目录。

## 4. ETL 与持久化调整

NWP ETL 主流程仍保持业务语义：

- ECMWF HRES NetCDF。
- 优先 12Z 起报。
- `N1` 为下一个北京时间自然日 `00:00-23:45`。
- 按场站经纬度裁剪网格。
- 对 1 小时 NWP 做 15 分钟插值。
- 输出 ML tabular CSV 和 DL tensor `.npy`。

二进制缓存已从 pickle 改为 joblib：

```text
tasks/{task_id}/data/nwp_aligned/ml_baseline_dataset.joblib
```

封装工具：

`src/zhejiangforecast_zj/algorithm_engine/persistence/joblib_store.py`

vendored `power_ml_baseline.data.dataset` 也已改为使用 `joblib.dump/load`。旧 `.pkl/.pickle` 后缀仅作为读取兼容，不再新写 pickle。

## 5. 数据库层调整

数据库层改为 SQLAlchemy ORM：

```text
src/zhejiangforecast_zj/db/models.py       ORM 表模型
src/zhejiangforecast_zj/db/session.py      engine/session/init_db
src/zhejiangforecast_zj/db/repository.py   Repository 方法封装
src/zhejiangforecast_zj/db/schema.py       兼容旧 init_db 导入
```

上层服务仍使用原来的 repository 方法，例如：

- `create_task`
- `get_task`
- `update_task`
- `add_data_check`
- `add_artifact`
- `replace_eval_rows`
- `add_eval_metric`
- `add_curve_rows`
- `list_curve`

因此 API、训练、评估、发布等业务代码改动较小。

## 6. SQLite 与 PostgreSQL 配置

默认仍使用 SQLite：

```bash
export ZJ_FORECAST_HOME=$PWD/runtime_api
```

显式 SQLite 文件：

```bash
export ZJ_FORECAST_DB=/data/online_modeling/zj_forecast.db
```

PostgreSQL 预留接口：

```bash
export ZJ_FORECAST_DB_URL='postgresql+psycopg://user:password@host:5432/dbname'
```

也可以把 URL 放在 `ZJ_FORECAST_DB`：

```bash
export ZJ_FORECAST_DB='postgresql+psycopg://user:password@host:5432/dbname'
```

依赖已补充：

- `sqlalchemy>=2.0`
- `psycopg[binary]>=3.1` 放在 `pg` optional extra

## 7. 参考原则

本次重构参考了以下官方或成熟实践：

- SQLAlchemy 2.0 ORM 使用 declarative mapping、engine、session 和 `Base.metadata.create_all()`，见 [SQLAlchemy ORM Quick Start](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)。
- NWP NetCDF 读取继续围绕 xarray 文件 I/O，见 [xarray Reading and writing files](https://docs.xarray.dev/en/stable/user-guide/io.html)。
- Python 对象持久化采用 joblib 的 `dump/load`，见 [joblib Persistence](https://joblib.readthedocs.io/en/stable/persistence.html)。
- 工程目录按“逻辑、标准化、便于协作”的数据科学工程结构思想拆分，参考 [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)。

## 8. 本次验证

编译检查：

```bash
PYTHONPATH=src python -m compileall -q src scripts
```

单测：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

结果：

```text
Ran 4 tests
OK
```

真实接口闭环：

```bash
PYTHONPATH=src python scripts/run_backend_api_flow.py \
  --home runtime_refactor_api_flow1 \
  --task-id refactor_api_flow1
```

结果摘要：

```json
{
  "dataset_mode": "nwp_aligned",
  "aligned_samples": 173,
  "trained_models": 2,
  "train_status": "TRAINED",
  "eval_model_count": 2,
  "published_model_id": "refactor_api_flow1_PERSISTENCE_BASELINE_20260629155357",
  "infer_points": 96,
  "infer_status": "SUCCESS"
}
```

ETL 产物确认：

```text
tasks/refactor_api_flow1/data/nwp_aligned/ml_baseline_dataset.joblib
```

## 9. 后续建议

1. PostgreSQL 真库联调：配置 `ZJ_FORECAST_DB_URL`，跑 `scripts/run_backend_api_flow.py`。
2. 数据库迁移工具：现场部署建议补 Alembic migration，而不是长期依赖 `Base.metadata.create_all()`。
3. DL 在线推理：当前 Swin3D/LoRA 通用推理仍有 fallback，下一步应按 `issue_time + station` 实时构造 NWP tensor。
4. 算法 vendor 精简：本次优先完整迁移源码，后续可以删除未使用的 demo/tests/example 文件。
