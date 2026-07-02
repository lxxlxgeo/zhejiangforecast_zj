# 浙江省调功率预测在线建模工程

本工程按 `task_md.md` 和《浙江省调在线建模_最小化实现路径技术文档.docx》的接口思路实现，目录名为 `zhejiangforecast_zj`。当前版本不是单纯冒烟程序，已经把数据清洗、NWP ETL、ML baseline、Swin3D、LoRA-Swin3D、评估、发布、推理和 FastAPI/SQLite 工程骨架接起来。

## 已实现范围

- FastAPI 接口：`/ingest`、`/data/status`、`/data/preview`、`/train`、`/evaluate`、`/publish`、`/infer`。
- SQLite 表：任务、数据检查、模型产物、评估结果、预测曲线、运行日志、作业记录。
- task 工作目录：`runtime/tasks/{task_id}/config|data|models|reports|logs`。
- 风电/光伏实发 ETL：读取 CSV/XLSX，自动识别北京时间、UTC 时间、实发功率、风速/辐照度等常见列。
- 风电清洗：优先调用 `数据清洗_bygptpro/h3_wind_cleaning_project` 的清洗流水线；不满足外部清洗条件时使用物理边界和容量约束 fallback。
- NWP ETL：读取 ECMWF HRES NetCDF，按场站经纬度裁剪 16x16 网格，按 12Z/N1 业务时序对齐实发功率，并调用 `nwp_temporal_downscaling_v2_project` 或线性插值生成 15 分钟序列。
- ML baseline：真实训练 LightGBM/XGBoost/ExtraTrees；缺少依赖时 fallback 到 Ridge/Persistence，模型以 joblib/json 形式登记。
- Swin3D：接入 `短期模型修改部分/met_swin3d_nwp_power_v2` 的 `MetSwin3DRegressor`，直接消费 ETL 生成的 `[N,C,S,H,W]` NWP 张量。
- LoRA-Swin3D：接入 `lora_swin3d_power_project`，训练并保存 LoRA adapter。
- 评估：多模型曲线、MAE/RMSE/Bias、容量归一化误差、日准确率、平均准确率和最优模型选择。
- 发布/推理：登记最优模型并支持通用推理；深度模型在没有在线 NWP tensor 输入时使用评估预测曲线作为 fallback，权重和 adapter 仍完整保存。
- Airflow：提供可选 `online_modeling_pipeline` DAG。

## N1/N2/N3 时间语义

代码显式记录并计算：

- `issue_time_utc`：ECMWF HRES 起报时次，当前训练约定优先使用 `12Z`。
- `valid_time_utc`：预测有效时间，统一 UTC。
- `lead_hours`：`valid_time_utc - issue_time_utc`。
- `horizon_code`：`N1/N2/N3`。

当前业务约定：`12Z` 起报的 `N1` 是下一个北京时间自然日 `00:00-23:45`；`N2`、`N3` 依次后推。核心代码在 `src/zhejiangforecast_zj/core/time_semantics.py`。

## WSL 运行

```bash
cd /mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj
source /home/lxce/miniconda3/etc/profile.d/conda.sh
conda activate ml_sc
export PYTHONPATH=src
export ZJ_FORECAST_HOME=$PWD/runtime
python -m zhejiangforecast_zj.cli init-db
```

启动 API：

```bash
uvicorn zhejiangforecast_zj.api.main:app --host 0.0.0.0 --port 8000
```

也可以使用脚本：

```bash
bash scripts/start_api_wsl.sh 8000
```

## 已验证命令

核心单测：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

真实 NWP + ML baseline 小样例：

```bash
PYTHONPATH=src python scripts/run_h3_smoke.py \
  --task-id task_h3_nwp_real3 \
  --home runtime_h3_nwp_real3 \
  --train-start 2025-11-02T00:00:00 \
  --train-end 2025-11-02T23:45:00 \
  --eval-start 2025-11-03T00:00:00 \
  --eval-end 2025-11-03T23:45:00 \
  --max-nwp-samples 192 \
  --candidates EC_LGB_WIND_V1 PERSISTENCE_BASELINE
```

本地验证结果：`dataset_mode=nwp_aligned`，`aligned_samples=173`，NWP 张量形状为 `[173,20,9,16,16]`。

Swin3D 小样例：

```bash
PYTHONPATH=src python scripts/run_h3_smoke.py \
  --task-id task_h3_swin_real2 \
  --home runtime_h3_swin_real2 \
  --train-start 2025-11-02T00:00:00 \
  --train-end 2025-11-02T11:45:00 \
  --eval-start 2025-11-02T12:00:00 \
  --eval-end 2025-11-02T23:45:00 \
  --max-nwp-samples 96 \
  --candidates EC_SWIN3D_WIND_V1
```

LoRA-Swin3D 小样例：

```bash
PYTHONPATH=src python scripts/run_h3_smoke.py \
  --task-id task_h3_lora_real2 \
  --home runtime_h3_lora_real2 \
  --train-start 2025-11-02T00:00:00 \
  --train-end 2025-11-02T05:45:00 \
  --eval-start 2025-11-02T06:00:00 \
  --eval-end 2025-11-02T11:45:00 \
  --max-nwp-samples 48 \
  --candidates EC_LORA_WIND_V1.1
```

## API 示例

业务后端完整对接说明见 `docs/backend_integration_guide.md`。`ingest` 接口单独说明见 `docs/ingest_api_guide.md`。`powerData` 新接口风电/光伏验收样例见 `docs/backend_powerdata_api_cases.md`。算法后端/算法工程拆分与 SQLAlchemy 重构说明见 `docs/refactor_algorithm_engine.md`。架构关系、时序图、ER 图和流程图见 `docs/architecture_mermaid_readme.md`。可重复执行的接口闭环测试脚本见 `scripts/run_backend_api_flow.py`。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/online-modeling/ingest \
  -H "Content-Type: application/json" \
  -d @configs/example_ingest_wind_h3.json
```

同步训练/评估：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/online-modeling/train \
  -H "Content-Type: application/json" \
  -d '{"task_id":"<task_id>","sync":true}'

curl -X POST http://127.0.0.1:8000/api/v1/online-modeling/evaluate \
  -H "Content-Type: application/json" \
  -d '{"task_id":"<task_id>","sync":true}'
```

## 说明

- 测试 NWP 路径默认使用 `/mnt/d/data/netcdf/ecmwf/jiangsu`，Windows 对应 `D:\data\netcdf\ecmwf\jiangsu`。
- H3 风电样例使用 `测试场站数据/js_yanchengH3_wind/station_power/station_power_original_2502-2601.csv`。
- 2025-02 的 H3 功率数据没有对应 NWP 文件；真实 NWP 联调请使用 2025-11 以后有 NetCDF 的时间段。
- WSL 的 `ml_sc` 当前没有 `pytest`，单测用标准库 `unittest` 执行即可。
