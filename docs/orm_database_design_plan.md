# ORM 与数据库设计方案

## 1. 目标

在不破坏现有 `ingest/train/evaluate/publish` 接口和 SQLite 可用性的前提下，引入更完整的 MLOps 数据模型：

- 支持 PostgreSQL 作为生产/联调数据库。
- 保留 SQLite 作为本地开发和最小部署数据库。
- 用 SQLAlchemy ORM 统一访问。
- 用 Alembic 管理迁移，避免生产库靠 `create_all` 静默变更。
- 为后续新增 `station/data/cleaning/etl/training/evaluation/publish` 分阶段接口提供稳定表结构。

## 2. 当前状态

当前工程已经使用 SQLAlchemy ORM：

- `OnlineModelTask`：任务主表。
- `OnlineModelDataCheck`：数据检查结果。
- `OnlineModelArtifact`：模型产物。
- `OnlineModelEval`：评估指标。
- `OnlineModelCurve`：预测曲线。
- `OnlineModelLog`：日志。
- `OnlineModelJob`：异步任务。

这套表能支撑现有一键式流程，但不够表达：

- 场站主数据。
- 原始数据、清洗数据、NWP 数据、dataset 的资产化管理。
- 清洗/ETL/训练/评估/发布各阶段 run 的输入输出血缘。
- 模型发布 alias、active 版本、回滚关系。

## 3. 数据库连接策略

配置优先级保持当前逻辑：

1. 环境变量 `ZJ_FORECAST_DB_URL`。
2. `configs/default.yml` 中 `database.url`。
3. SQLite 路径 `database.path`。

PostgreSQL 示例：

```yaml
database:
  url: postgresql+psycopg://mlops:lxce%40123A@127.0.0.1:5432/zj_mlops
  path:
```

注意：密码里的 `@` 在 URL 中应写成 `%40`。

本地 SQLite 示例：

```yaml
database:
  path: zj_forecast.db
  url:
```

## 4. 迁移策略

建议新增 Alembic：

```text
zhejiangforecast_zj/
  alembic.ini
  migrations/
    env.py
    script.py.mako
    versions/
```

执行方式：

```bash
export PYTHONPATH=src
export ZJ_FORECAST_DB_URL='postgresql+psycopg://mlops:lxce%40123A@127.0.0.1:5432/zj_mlops'
alembic upgrade head
```

开发期可以继续保留 `init_db(... create_all ...)`，但生产 PostgreSQL 建议只通过 Alembic 变更 schema。

## 5. 新增 ORM 表设计

### 5.1 station_registry

场站主数据表。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `station_id` | string pk | 场站 ID |
| `station_name` | string | 场站名 |
| `object_type` | string | `station` / `region` |
| `station_type` | string | `wind` / `solar` |
| `region_id` | string nullable | 区域 |
| `longitude` | float | 经度 |
| `latitude` | float | 纬度 |
| `capacity_mw` | float | 装机容量 |
| `timezone` | string | 默认 `Asia/Shanghai` |
| `status` | string | `ACTIVE` / `INACTIVE` |
| `metadata_json` | json/text | 扩展信息 |
| `created_time` | string | 创建时间 |
| `updated_time` | string | 更新时间 |

索引：

- `(station_type, region_id)`
- `status`

### 5.2 data_asset

统一数据资产表，表达原始实发、清洗后功率、NWP 根目录、NWP 对齐 dataset 等。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `asset_id` | string pk | 数据资产 ID |
| `station_id` | string index | 所属场站 |
| `asset_type` | string | `power_raw` / `power_clean` / `nwp_root` / `dataset` / `nwp_tensor` |
| `source_type` | string | `inline` / `path` / `generated` |
| `uri` | text | 文件路径、目录或逻辑 URI |
| `format` | string | `csv` / `json` / `netcdf` / `npy` / `joblib` |
| `schema_json` | json/text | 字段 schema |
| `summary_json` | json/text | 行数、时间范围、缺失率等 |
| `start_time` | string nullable | 数据开始时间 |
| `end_time` | string nullable | 数据结束时间 |
| `row_count` | integer nullable | 行数 |
| `checksum` | string nullable | 可选校验 |
| `status` | string | `REGISTERED` / `READY` / `FAILED` |
| `created_time` | string | 创建时间 |

索引：

- `(station_id, asset_type)`
- `(asset_type, status)`
- `(start_time, end_time)`

### 5.3 pipeline_run

统一记录 cleaning / etl / training / evaluation / publish 的执行。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `run_id` | string pk | run ID |
| `run_type` | string | `cleaning` / `etl` / `training` / `evaluation` / `publish` |
| `task_id` | string nullable index | 兼容现有 task |
| `station_id` | string index | 场站 |
| `job_id` | string nullable | 异步 job |
| `status` | string | `CREATED` / `RUNNING` / `SUCCESS` / `FAILED` |
| `input_assets_json` | json/text | 输入资产 ID 列表 |
| `output_assets_json` | json/text | 输出资产 ID 列表 |
| `params_json` | json/text | 运行参数 |
| `summary_json` | json/text | 运行摘要 |
| `metrics_json` | json/text | 指标 |
| `error_message` | text nullable | 错误 |
| `created_time` | string | 创建时间 |
| `updated_time` | string | 更新时间 |

索引：

- `(station_id, run_type)`
- `(task_id, run_type)`
- `(status, run_type)`

### 5.4 asset_lineage

记录资产血缘。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | integer pk | 自增 |
| `parent_asset_id` | string index | 输入资产 |
| `child_asset_id` | string index | 输出资产 |
| `run_id` | string index | 产生该血缘的 run |
| `relation_type` | string | `cleaned_from` / `aligned_from` / `trained_from` |
| `created_time` | string | 创建时间 |

### 5.5 model_registry / published_model

现有 `OnlineModelArtifact` 可继续作为第一阶段模型注册表。建议新增 `published_model` 表专门表达发布态。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `published_id` | string pk | 发布 ID |
| `station_id` | string index | 场站 |
| `model_id` | string index | 模型 ID |
| `alias` | string | 如 `default` / `short_term` |
| `version` | string | 版本 |
| `active` | bool | 是否当前在线 |
| `strategy` | string | `replace_active` / `manual` |
| `published_time` | string | 发布时间 |
| `metadata_json` | json/text | 扩展 |

约束：

- PostgreSQL 可用 partial unique index 保证同一 `station_id + alias` 只有一个 `active=true`。
- SQLite 阶段可由应用层事务保证。

## 6. Repository 分层建议

当前 `Repository` 已经承担所有读写。新增 MLOps 资源后建议拆分：

```text
db/
  models.py
  session.py
  repository.py              # 兼容现有任务接口
  repositories/
    station_repository.py
    asset_repository.py
    run_repository.py
    model_repository.py
```

第一阶段为了降低改动，也可以只扩展当前 `Repository`，但新增方法按资源分组命名：

- `upsert_station`
- `get_station`
- `create_data_asset`
- `list_data_assets`
- `create_pipeline_run`
- `update_pipeline_run`
- `add_asset_lineage`
- `publish_model_record`

## 7. 服务层建议

新增服务层，不把业务逻辑写进 FastAPI handler：

```text
services/
  station_registry.py
  data_assets.py
  mlops_runs.py
```

职责：

- API handler 只做参数校验和响应包装。
- service 负责组装现有 `create_or_ingest_task/run_data_pipeline/run_training/run_evaluation/publish_model`。
- repository 只做数据库读写。

## 8. 新旧接口兼容关系

现有接口保持不变：

```text
/api/v1/online-modeling/ingest
/api/v1/online-modeling/train
/api/v1/online-modeling/evaluate
/api/v1/online-modeling/publish
```

新增 MLOps 接口：

```text
/api/v1/mlops/stations
/api/v1/mlops/data/assets
/api/v1/mlops/cleaning/runs
/api/v1/mlops/etl/runs
/api/v1/mlops/training/runs
/api/v1/mlops/evaluation/runs
/api/v1/mlops/models/publish
```

第一阶段这些新接口可以复用现有 task 机制：

- `cleaning + etl` 最终仍落一个 `online_model_task`。
- `training/evaluation/publish` 仍调用现有服务函数。
- 同时额外写 `data_asset/pipeline_run/asset_lineage`。

## 9. 实施顺序

### Phase 1: DB 基础与迁移

1. 安装 PostgreSQL 驱动 `psycopg[binary]`。
2. 增加 Alembic。
3. 增加新 ORM 表。
4. 保持 SQLite 测试通过。
5. 增加 PostgreSQL 连接 smoke test。

### Phase 2: 资源接口骨架

1. `station` CRUD。
2. `data_asset` 注册和查询。
3. `pipeline_run` 状态查询。
4. 不改变现有 `ingest/evaluate`。

### Phase 3: MLOps 分阶段执行

1. `cleaning/runs` 调现有清洗逻辑。
2. `etl/runs` 调现有 NWP 对齐逻辑。
3. `training/runs` 调现有训练逻辑。
4. `evaluation/runs` 调现有评估逻辑。
5. `models/publish` 调现有发布逻辑。

### Phase 4: 发布治理

1. `published_model` active alias。
2. 模型回滚。
3. 自动发布策略。

## 10. PostgreSQL 验证命令

在 WSL 终端中执行：

```bash
sudo -u postgres psql
```

然后执行：

```sql
ALTER USER mlops WITH PASSWORD 'lxce@123A';
CREATE DATABASE zj_mlops OWNER mlops;
GRANT ALL PRIVILEGES ON DATABASE zj_mlops TO mlops;
```

如果数据库已存在：

```sql
ALTER DATABASE zj_mlops OWNER TO mlops;
GRANT ALL PRIVILEGES ON DATABASE zj_mlops TO mlops;
```

退出后验证：

```bash
PGPASSWORD='lxce@123A' psql -h 127.0.0.1 -U mlops -d zj_mlops -c 'select current_user, current_database();'
```

