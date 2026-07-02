# 浙江省调在线建模工程架构图集

本文档用 Mermaid 描述 `zhejiangforecast_zj` 当前工程的主要关系、流程和数据库结构。图中的核心约束是：风电/光伏 EC 模型的训练特征 `X` 只来自 NWP 提取后的特征，标签 `y` 只使用 `power_mw`；推理阶段不依赖实发数据或 label。

## 1. 总体架构关系图

```mermaid
flowchart TB
    backend["业务后端 / 调度系统"]
    api["FastAPI 接口层\nsrc/zhejiangforecast_zj/api/main.py"]
    schema["Pydantic Schema\nschemas.py"]
    services["服务编排层\nservices/*"]
    repo["Repository\nSQLAlchemy ORM"]
    db[("SQLite 当前默认\nPostgreSQL 可通过 database_url 切换")]
    runtime["运行产物目录\nruntime/tasks/{task_id}"]
    engine["算法工程层\nalgorithm_engine/*"]
    vendor["内置 vendor 算法\nwind_cleaning / solar_clean\nnwp_downscaling / swin3d / lora / ml_baseline"]
    nwp[("ECMWF HRES NetCDF\n/mnt/d/data/netcdf/ecmwf/...")]
    station[("场站实发/元数据 CSV/XLSX")]
    airflow["Airflow DAG 可选\nairflow/dags"]
    local_runner["LocalOrchestrator\nThreadPoolExecutor"]

    backend -->|"HTTP: ingest/train/evaluate/publish/infer"| api
    api --> schema
    api --> services
    services --> repo
    repo --> db
    services --> runtime
    services --> engine
    engine --> vendor
    engine --> nwp
    services --> station
    api -->|"sync=false"| local_runner
    airflow -.可替换本地编排.-> services
```

## 2. 模块依赖关系图

```mermaid
flowchart LR
    subgraph api_layer["接口层"]
        main_py["api/main.py"]
        schemas_py["schemas.py"]
    end

    subgraph service_layer["业务服务层"]
        tasks_py["tasks.py\n任务创建 + ETL"]
        training_py["training.py\n多候选模型训练"]
        evaluation_py["evaluation.py\n指标与最优模型"]
        publishing_py["publishing.py\n模型发布"]
        inference_py["inference.py\n在线推理"]
        orchestrator_py["orchestrator.py\n本地异步任务"]
    end

    subgraph core_layer["核心公共层"]
        config_py["config.py"]
        catalog_py["model_catalog.py"]
        time_py["time_semantics.py\nN1/N2/N3 语义"]
        paths_py["paths.py"]
    end

    subgraph db_layer["数据访问层"]
        models_py["db/models.py"]
        repository_py["db/repository.py"]
        session_py["db/session.py"]
    end

    subgraph algo_layer["算法工程层"]
        cleaning_py["adapters/cleaning.py"]
        nwp_py["adapters/nwp.py\nNWP 对齐 + NWP-only 特征"]
        dl_py["adapters/deep_learning.py"]
        store_py["persistence/joblib_store.py"]
        vendor_paths_py["adapters/vendor_paths.py"]
    end

    main_py --> schemas_py
    main_py --> tasks_py
    main_py --> training_py
    main_py --> evaluation_py
    main_py --> publishing_py
    main_py --> inference_py
    main_py --> orchestrator_py

    tasks_py --> cleaning_py
    tasks_py --> nwp_py
    tasks_py --> repository_py
    training_py --> dl_py
    training_py --> store_py
    evaluation_py --> repository_py
    publishing_py --> repository_py
    inference_py --> nwp_py
    inference_py --> repository_py

    service_layer --> core_layer
    repository_py --> models_py
    repository_py --> session_py
```

## 3. 任务生命周期状态图

```mermaid
stateDiagram-v2
    [*] --> CREATED: POST /ingest
    CREATED --> DATA_READY: 读取场站实发和元数据
    DATA_READY --> CLEANED: 清洗 + NWP 对齐 + 生成数据集
    CLEANED --> TRAINING: POST /train
    TRAINING --> TRAINED: 至少一个候选模型训练成功
    TRAINED --> EVALUATING: POST /evaluate
    EVALUATING --> EVALUATED: 写入指标和预测曲线
    EVALUATED --> PUBLISHED: POST /publish
    PUBLISHED --> INFERING: POST /infer
    INFERING --> PUBLISHED: 返回预测结果

    CREATED --> FAILED: 入参或数据读取失败
    DATA_READY --> FAILED: 清洗/ETL 失败
    CLEANED --> FAILED: EC 模型缺少 NWP 对齐数据
    TRAINING --> FAILED: 候选模型全部失败
    EVALUATING --> FAILED: 评估失败
```

## 4. 在线建模主时序图

```mermaid
sequenceDiagram
    autonumber
    participant B as 业务后端
    participant API as FastAPI
    participant T as Task Service
    participant E as Algorithm Engine
    participant R as Repository
    participant DB as SQLite/PostgreSQL
    participant FS as Runtime Artifacts
    participant TR as Training Service
    participant EV as Evaluation Service
    participant PB as Publishing Service

    B->>API: POST /ingest
    API->>T: create_or_ingest_task(payload)
    T->>R: create_task(status=CREATED)
    R->>DB: insert online_model_task
    T->>E: clean_wind_power / clean_solar_power
    T->>E: build_nwp_power_datasets
    E->>FS: clean_series.csv / train_dataset_nwp_ml.csv / eval_dataset_nwp_ml.csv / npy / joblib
    T->>R: add_data_check + update_task(CLEANED)
    R->>DB: insert checks + update task
    API-->>B: task_id + status=CLEANED

    B->>API: POST /train
    API->>TR: run_training(task_id)
    TR->>R: get_task + read artifacts
    TR->>FS: load feature_schema + train/eval dataset
    TR->>E: train XGB/LGB/Swin3D/LoRA
    TR->>FS: save model artifact + predictions
    TR->>R: add_artifact + update_task(TRAINED)
    API-->>B: train_result

    B->>API: POST /evaluate
    API->>EV: run_evaluation(task_id)
    EV->>FS: read predictions
    EV->>R: replace_eval_rows + add_eval_metric + add_curve_rows
    EV->>R: update_task(EVALUATED)
    API-->>B: selected_model + metrics

    B->>API: POST /publish
    API->>PB: publish_model(task_id)
    PB->>R: get selected artifact
    PB->>FS: copy model to published/{model_id}
    PB->>R: update_task(PUBLISHED, published_model_id)
    API-->>B: model_card
```

## 5. ETL 与特征契约流程图

```mermaid
flowchart TB
    raw_power[("场站实发数据\nbj_time / actual_power / utc_time / 辅助列")]
    station_meta[("场站元数据\n经纬度 / 装机容量 / 风机数")]
    nwp_files[("ECMWF HRES NetCDF\n12Z issue files")]

    read_power["read_power_timeseries\n统一 time_bj/time_utc/power_mw"]
    read_meta["read_station_metadata\n解析经纬度和容量"]
    clean{"station_type"}
    clean_wind["clean_wind_power\n容量边界/风电清洗"]
    clean_solar["clean_solar_power\n夜间/辐照/容量规则"]
    nwp_align["build_nwp_power_datasets\n按 N1 业务窗口匹配 12Z NWP"]
    downscale["TemporalDownscalePipeline\n或线性插值到 15min"]
    extract["站点周边网格提取\n统计特征 + NWP tensor"]
    contract{"特征契约校验"}
    ml_csv["ML CSV\nX=lead_hours+nwp_*\ny=power_mw"]
    dl_npy["DL NPY\nX=[N,C,S,H,W]\ny=power_mw/capacity"]
    fail["FAILED\nEC 模型禁止退回 history_power"]

    raw_power --> read_power
    station_meta --> read_meta
    read_power --> clean
    clean -->|wind| clean_wind
    clean -->|solar| clean_solar
    clean_wind --> nwp_align
    clean_solar --> nwp_align
    read_meta --> nwp_align
    nwp_files --> nwp_align
    nwp_align --> downscale
    downscale --> extract
    extract --> contract
    contract -->|"通过: no power/history/capacity in X"| ml_csv
    contract -->|"通过"| dl_npy
    contract -->|"缺 NWP 或含泄漏字段"| fail
```

## 6. 训练、评估与模型选择流程图

```mermaid
flowchart TB
    start["run_training(task_id)"]
    read["读取 task_config\nfeature_schema\ntrain/eval dataset"]
    candidates["遍历 model_candidates"]
    branch{"候选模型类型"}
    xgb["XGBRegressor\njoblib artifact"]
    lgb["LGBMRegressor\njoblib artifact"]
    swin["MetSwin3D\n读取 NWP tensor"]
    lora["LoRA-Swin3D\n保存 adapter"]
    persistence["PERSISTENCE_BASELINE\n显式历史基线"]
    predict_eval["对 eval 期生成 p_pred"]
    metrics["compute_metrics + daily_accuracy"]
    artifact["online_model_artifact\nmodel.joblib/model.pt/adapter"]
    eval_start["run_evaluation(task_id)"]
    compare["按 avg_accuracy / accuracy 选最优"]
    curve["online_model_curve\np_real/p_pred"]
    selected["selected_model"]

    start --> read --> candidates --> branch
    branch -->|"EC_XGB_*"| xgb
    branch -->|"EC_LGB_*"| lgb
    branch -->|"EC_SWIN3D_*"| swin
    branch -->|"EC_LORA_*"| lora
    branch -->|"PERSISTENCE_BASELINE"| persistence
    xgb --> predict_eval
    lgb --> predict_eval
    swin --> predict_eval
    lora --> predict_eval
    persistence --> predict_eval
    predict_eval --> metrics --> artifact
    artifact --> eval_start --> compare
    compare --> curve
    compare --> selected
```

## 7. 推理阶段时序图

```mermaid
sequenceDiagram
    autonumber
    participant B as 业务后端
    participant API as FastAPI /infer
    participant INF as Inference Service
    participant R as Repository
    participant NWP as NWP Adapter
    participant FS as Runtime/Published Artifacts
    participant M as Model Artifact

    B->>API: POST /infer(task_id, issue_time, model_id?)
    API->>INF: run_inference(...)
    INF->>R: get_task(task_id)
    INF->>R: get_artifact(model_id or published_model_id)
    INF->>FS: read model_meta.json / feature_names
    alt NWP-only feature set
        INF->>NWP: build_nwp_inference_features(issue_time, lon, lat, periods=96)
        NWP->>FS: read ECMWF NetCDF
        NWP-->>INF: DataFrame[lead_hours+nwp_*]
    else external feature data supplied
        B->>API: data / nwp_data
        API->>INF: DataFrame from request
    end
    INF->>M: load joblib/json model
    M-->>INF: predict p_pred
    INF-->>API: predictions[time, p_pred]
    API-->>B: infer result
```

推理阶段的关键点：

- 对 EC XGB/LGB 风光模型，`feature_names` 必须是 `lead_hours` 或 `nwp_*`。
- `build_nwp_inference_features` 只读 NWP、场站经纬度、起报时间，不读取 `power_mw`。
- 返回 96 个 15 分钟点时，业务后端不需要提供实发功率。

## 8. 数据库 ER 图

```mermaid
erDiagram
    ONLINE_MODEL_TASK ||--o{ ONLINE_MODEL_DATA_CHECK : has
    ONLINE_MODEL_TASK ||--o{ ONLINE_MODEL_ARTIFACT : trains
    ONLINE_MODEL_TASK ||--o{ ONLINE_MODEL_EVAL : evaluates
    ONLINE_MODEL_TASK ||--o{ ONLINE_MODEL_CURVE : stores
    ONLINE_MODEL_TASK ||--o{ ONLINE_MODEL_LOG : logs
    ONLINE_MODEL_TASK ||--o{ ONLINE_MODEL_JOB : runs
    ONLINE_MODEL_ARTIFACT ||--o{ ONLINE_MODEL_EVAL : measured_by
    ONLINE_MODEL_ARTIFACT ||--o{ ONLINE_MODEL_CURVE : predicts

    ONLINE_MODEL_TASK {
        string task_id PK
        string object_type
        string station_type
        string station_id
        string region_id
        string status
        string train_start
        string train_end
        string eval_start
        string eval_end
        string feature_set
        text model_candidates
        text request_json
        text config_path
        text work_dir
        string published_model_id
        text error_message
        string created_time
        string updated_time
    }

    ONLINE_MODEL_DATA_CHECK {
        int id PK
        string task_id FK
        string data_type
        float missing_rate
        string start_time
        string end_time
        string check_result
        text summary_json
        string created_time
    }

    ONLINE_MODEL_ARTIFACT {
        string model_id PK
        string task_id FK
        string model_type
        text base_id
        text adapter_id
        text artifact_path
        string version
        string status
        text metrics_json
        string created_time
    }

    ONLINE_MODEL_EVAL {
        int id PK
        string task_id FK
        string model_id FK
        string metric_name
        float metric_value
        string eval_date
        string created_time
    }

    ONLINE_MODEL_CURVE {
        int id PK
        string task_id FK
        string model_id FK
        string time
        float p_real
        float p_pred
        string created_time
    }

    ONLINE_MODEL_LOG {
        int id PK
        string task_id FK
        string stage
        string log_level
        text message
        string log_time
    }

    ONLINE_MODEL_JOB {
        string job_id PK
        string task_id FK
        string job_type
        string status
        string stage
        float progress
        text error_message
        string created_time
        string updated_time
    }
```

## 9. Runtime 产物关系图

```mermaid
flowchart TB
    task_dir["runtime/tasks/{task_id}"]
    config["config/task_config.json\n请求、场站、artifacts、data_summary"]
    data["data/"]
    cleaning["data/cleaning/\n清洗报告与中间产物"]
    clean_series["data/clean_series.csv\n清洗后的 label 序列"]
    nwp_aligned["data/nwp_aligned/"]
    train_csv["train_dataset_nwp_ml.csv\n训练表: time + power_mw + nwp features"]
    eval_csv["eval_dataset_nwp_ml.csv\n评估表: time + power_mw + nwp features"]
    schema["feature_schema.json\nfeature_contract=x=nwp_only,y=power_mw"]
    tensor["train/eval_nwp_x.npy\ntrain/eval_nwp_y.npy\nSwin3D/LoRA 输入"]
    models["models/{model_id}/"]
    model_file["model.joblib / model.pt / adapter"]
    meta["model_meta.json / metrics.json"]
    reports["reports/"]
    predictions["predictions_{model_id}.csv\n评估期 p_real/p_pred"]
    published["runtime/published/{model_id}/\n发布模型与 model_card.json"]

    task_dir --> config
    task_dir --> data
    data --> cleaning
    data --> clean_series
    data --> nwp_aligned
    nwp_aligned --> train_csv
    nwp_aligned --> eval_csv
    data --> schema
    nwp_aligned --> tensor
    task_dir --> models
    models --> model_file
    models --> meta
    task_dir --> reports
    reports --> predictions
    models --> published
```

## 10. Airflow / 本地编排关系图

```mermaid
flowchart LR
    user["业务后端"]
    api["FastAPI"]
    sync{"sync 参数"}
    local["LocalOrchestrator\nonline_model_job"]
    airflow["Airflow DAG\nonline_modeling_pipeline"]
    services["run_data_pipeline\nrun_training\nrun_evaluation\npublish_model"]
    db[("online_model_job\nonline_model_task")]

    user --> api --> sync
    sync -->|"sync=true"| services
    sync -->|"sync=false 当前默认本地"| local
    local --> services
    local --> db
    airflow -.可用于生产调度.-> services
    services --> db
```

## 11. 模型与数据契约关系图

```mermaid
flowchart TB
    contract["feature_schema.json"]
    target["target = power_mw"]
    nwp_x["EC 风/光模型 X\nlead_hours + nwp_*"]
    dl_x["DL 模型 X\nNWP tensor [N,C,S,H,W]"]
    forbid["禁止进入 EC X\npower_mw / history_power_* / capacity_mw / time metadata"]
    ml_models["EC_XGB_PV_V1 / EC_LGB_PV_V1\nEC_XGB_WIND_V1 / EC_LGB_WIND_V1"]
    dl_models["EC_SWIN3D_* / EC_LORA_*"]
    persistence["PERSISTENCE_BASELINE\n显式历史功率基线"]
    infer["推理\n只用 NWP + 经度纬度 + issue_time"]

    contract --> target
    contract --> nwp_x
    contract --> dl_x
    contract --> forbid
    nwp_x --> ml_models
    dl_x --> dl_models
    target --> ml_models
    target --> dl_models
    persistence -.独立 fallback，不代表 EC 模型特征.-> target
    ml_models --> infer
    dl_models --> infer
```

## 12. 业务后端最小调用顺序

```mermaid
sequenceDiagram
    autonumber
    participant B as 业务后端
    participant A as 算法服务

    B->>A: POST /api/v1/online-modeling/ingest
    A-->>B: task_id, status=CLEANED 或 FAILED
    B->>A: GET /api/v1/online-modeling/data/status?task_id=...
    A-->>B: data_checks
    B->>A: POST /api/v1/online-modeling/train
    A-->>B: models + train metrics
    B->>A: POST /api/v1/online-modeling/evaluate
    A-->>B: selected_model + eval metrics
    B->>A: POST /api/v1/online-modeling/publish
    A-->>B: model_id + model_card
    B->>A: POST /api/v1/online-modeling/infer
    A-->>B: 96 点预测曲线
```

