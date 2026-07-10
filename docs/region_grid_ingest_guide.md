# 浙江省/地市区域网格接入说明

本功能仍走旧接口：

- `POST /api/v1/online-modeling/ingest`
- `POST /api/v1/online-modeling/train`
- `POST /api/v1/online-modeling/evaluate`

新增的是区域字典和 ingest 入库前的自动规范化。

## 区域字典

区域数据来自：

`E:/工作文档/江苏应龙/场站侧/标准整改报告/各地市经纬度.xlsx`

代码位置：

`src/zhejiangforecast_zj/core/regions.py`

包含：

- `330000`：浙江省，全省范围由地市范围 union 得出。
- `330100`：杭州
- `330200`：宁波
- `330300`：温州
- `330400`：嘉兴
- `330500`：湖州
- `330600`：绍兴
- `330700`：金华
- `330800`：衢州
- `330900`：舟山
- `331000`：台州
- `331100`：丽水

## 网格规则

区域任务会把 bbox 转换成：

- 区域中心点：`station.longitude / station.latitude`
- 正方形网格：`etl_options.grid_size`
- 区域元信息：`etl_options.region_bounds`
- Swin 网格元信息：`etl_options.region_grid`

默认规则：

```yaml
region_grid:
  margin_deg: 0.5
  resolution_deg: 0.25
  swin_grid_multiple: 8
  min_grid_size: 16
```

计算逻辑：

1. 经度、纬度 bbox 各向外扩 `0.5deg`。
2. 取经纬跨度较大的那个方向，保证正方形 `H == W`。
3. 按 NWP 分辨率换算为格点数。
4. 向上取整到 `swin_grid_multiple`，保证满足 Swin patch/merge 的整除约束。

示例：

- 杭州：默认 `grid_size=16`
- 浙江全省：默认 `grid_size=24`

## 查看区域

`GET /api/v1/online-modeling/region/list`

返回区域字典。

`GET /api/v1/online-modeling/region/grid?region_id=330100`

返回某个区域的中心点、原始 bbox、外扩 bbox、grid_size。

## 省级任务示例

```json
{
  "task_id": "task_zhejiang_region_demo",
  "object_type": "region",
  "region_id": "330000",
  "station_type": "wind",
  "train_start": "2025-03-01 00:00:00",
  "train_end": "2025-06-30 23:45:00",
  "eval_start": "2025-09-01 00:00:00",
  "eval_end": "2025-09-30 23:45:00",
  "model_candidates": ["EC_XGB_WIND_V1", "EC_LGB_WIND_V1", "EC_SWIN3D_WIND_V1"],
  "data_paths": {
    "power": "/data/share/data/power_forecast/zhejiang_region_power.csv"
  },
  "etl_options": {
    "horizon_codes": ["N1"],
    "sequence_steps": 5
  }
}
```

ingest 内部会补成类似：

```json
{
  "region_id": "330000",
  "station": {
    "station_name": "浙江省",
    "longitude": 120.35,
    "latitude": 29.05
  },
  "etl_options": {
    "grid_size": 24,
    "region_grid": {
      "expanded_lon_min": 117.75,
      "expanded_lon_max": 122.95,
      "expanded_lat_min": 26.58,
      "expanded_lat_max": 31.52,
      "grid_size": 24,
      "grid_multiple": 8
    }
  }
}
```

## 地市任务示例

```json
{
  "task_id": "task_hangzhou_region_demo",
  "object_type": "region",
  "region_id": "330100",
  "station_type": "solar",
  "train_start": "2025-03-01 00:00:00",
  "train_end": "2025-06-30 23:45:00",
  "eval_start": "2025-09-01 00:00:00",
  "eval_end": "2025-09-30 23:45:00",
  "model_candidates": ["EC_XGB_PV_V1", "EC_LGB_PV_V1", "EC_SWIN3D_PV_V1"],
  "data_paths": {
    "power": "/data/share/data/power_forecast/hangzhou_solar_power.csv"
  },
  "etl_options": {
    "horizon_codes": ["N1"],
    "sequence_steps": 5
  }
}
```

杭州默认会生成：

```json
{
  "station": {
    "station_name": "浙江杭州",
    "longitude": 119.675,
    "latitude": 29.96
  },
  "etl_options": {
    "grid_size": 16
  }
}
```

## 备注

当前实现仍复用现有 NWP 裁剪函数：以中心点裁剪 `N x N` 格点。浙江 NWP 文件到位后即可直接跑真实 NWP 对齐；如果用江苏 NWP 文件测试，只能验证接口和裁剪流程是否通，不代表浙江区域气象覆盖有效。
