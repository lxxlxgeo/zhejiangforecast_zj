# 水镜湖光伏场站数据整理与清洗测试

## 1. 数据来源

新测试场站目录：

`E:\workspace\pwforecast\zj_mlops\测试场站数据\nmg_shuijinghu_solar`

原始文件：

- `station_info/水镜湖光伏电站收资表.xlsx`
- `station_power/station_power_original.csv`

收资表记录：

```json
{
  "station_id": "nmg_shuijinghu_solar",
  "station_name": "水镜湖光伏电站",
  "longitude": 109.7,
  "latitude": 40.30528,
  "capacity_mw": 100.0
}
```

## 2. 标准数据整理

整理脚本：

`scripts/prepare_nmg_shuijinghu_solar.py`

输出：

- `测试场站数据/nmg_shuijinghu_solar/station_power/nmg_shuijinghu_solar_real.csv`
- `测试场站数据/nmg_shuijinghu_solar/station_info/nmg_shuijinghu_station_info.txt`
- `测试场站数据/nmg_shuijinghu_solar/station_power/nmg_shuijinghu_prepare_summary.json`

标准 CSV 字段：

```text
bj_time,actual_power,direct_irradiance
```

说明：原始 `station_power_original.csv` 里的辐照度字段是 `radiation_total`，这里映射为 `direct_irradiance`，用于后续工程接口统一。

整理结果：

```json
{
  "rows_raw": 123475,
  "rows_output": 123475,
  "start_time": "2022-01-01 00:00:00",
  "end_time": "2025-11-17 00:00:00",
  "actual_power_missing_rate": 0.0037011540797732335,
  "direct_irradiance_missing_rate": 0.19899574812715123
}
```

## 3. 光伏清洗方案

新增模块：

`src/zhejiangforecast_zj/algorithm_engine/vendor/solar_clean`

当前实现不强制依赖 `pvlib`，先采用轻量规则，适合本地和 WSL 环境直接跑：

- 功率物理边界：严重负功率、超过容量上限。
- 小负功率修正：`>-1% capacity` 的负功率按夜间表计偏置处理，归零而不是删除。
- 辐照度边界：负辐照度、异常高辐照度。
- 夜间功率异常：低辐照度但高功率。
- 高辐照度零功率：疑似停机或数据异常。
- 功率-辐照度分箱 IQR 离群。
- 白天功率长时间卡死。
- 功率突变但辐照度变化很小。

后续增强方向可引入 `pvlib`：

- `pvlib.location.Location.get_solarposition` / `pvlib.solarposition.get_solarposition` 用于太阳高度角和昼夜判定。
- `pvlib.location.Location.get_clearsky` / clear-sky 模型用于 clear-sky index。
- `pvlib.irradiance.clearness_index` / `clearsky_index` 用于辐照度物理质控。

参考官方文档：

- [pvlib Solar Position](https://pvlib-python.readthedocs.io/en/stable/reference/solarposition.html)
- [pvlib Clear sky](https://pvlib-python.readthedocs.io/en/stable/reference/clearsky.html)

## 4. 清洗测试结果

直接清洗测试输出：

`zhejiangforecast_zj/runtime_solar_clean_probe/nmg_shuijinghu_v2/solar_clean/solar_clean_summary.json`

结果：

```json
{
  "rows_total": 123018,
  "clean_rows": 97404,
  "removed_rows": 25614,
  "clean_rate": 0.7917865678193435,
  "capacity_mw": 100.0,
  "flag_counts": {
    "flag_missing": 24115,
    "flag_power_negative": 3,
    "flag_power_over_capacity": 0,
    "flag_irradiance_negative": 0,
    "flag_irradiance_too_high": 0,
    "flag_night_power": 1173,
    "flag_low_irradiance_high_power": 1,
    "flag_high_irradiance_zero_power": 167,
    "flag_curve_outlier": 186,
    "flag_stuck_power": 0,
    "flag_power_spike": 9
  }
}
```

主要问题是辐照度缺失，约 2.4 万条。负功率中绝大多数是 -0.37 到 -0.47MW 的夜间小偏置，已按规则归零；严重负功率只剩 3 条被剔除。

## 5. 主工程闭环测试

测试任务：

`nmg_shuijinghu_solar_flow1`

运行目录：

`zhejiangforecast_zj/runtime_nmg_solar_flow1`

时间范围：

- train: `2024-06-01 00:00:00` 到 `2024-08-31 23:45:00`
- eval: `2024-09-01 00:00:00` 到 `2024-09-15 23:45:00`

本次没有接 NWP，因为当前可用 NWP 是江苏区域，而水镜湖在内蒙古，使用江苏 NWP 会造成误导。该次测试验证的是：

```text
solar CSV -> solar_clean -> tabular dataset -> LGB/Persistence -> evaluate -> publish -> infer
```

结果：

```json
{
  "task_status": "CLEANED",
  "final_status": "PUBLISHED",
  "rows_clean": 97404,
  "rows_train": 8628,
  "rows_eval": 1417,
  "dataset_mode": "power_history_tabular",
  "trained_models": 2,
  "selected_model": "nmg_shuijinghu_solar_flow1_EC_LGB_PV_V1_20260701204813",
  "infer_points": 96
}
```

评估摘要：

```json
{
  "model": "EC_LGB_PV_V1",
  "mae": 1.078638458117945,
  "rmse": 3.0553781398695206,
  "nmae_capacity": 0.01078638458117945,
  "nrmse_capacity": 0.030553781398695204,
  "accuracy": 0.9892136154188206,
  "avg_accuracy": 0.9892535321896383
}
```

## 6. 当前结论

水镜湖光伏数据可以作为当前光伏清洗和 solar tabular baseline 的测试场站。数据相比信利更完整，坐标和装机容量明确。

注意事项：

- 原始辐照度缺失率约 19.9%，清洗后可用样本仍足够。
- 当前清洗使用 `radiation_total -> direct_irradiance` 的统一字段映射。
- 后续做 NWP 对齐时，需要内蒙古区域 NWP；不要用当前江苏 NWP 测这个站。
