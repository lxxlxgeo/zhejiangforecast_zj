# PowerData 新接口风电/光伏验收样例

本报告验证 ingest 支持正式部署风格的 `powerData` 内联实发数组，并验证 `data_paths.nwp_root` 可省略，由 `configs/default.yml` 按 `station_type` 自动选择 NWP 根目录。

评估接口返回按业务后端要求包装为：`code/message/data`，目录产物仍保持原来的 `reports/eval_result.json`、模型 artifact、预测曲线等结构。

## wind

### Ingest 请求样例

```json
{
  "task_id": "api_contract_wind_powerdata",
  "station_id": "js_yancheng_h3",
  "region_id": null,
  "object_type": "station",
  "station_type": "wind",
  "train_start": "2025-11-02 00:00:00",
  "train_end": "2025-11-02 23:45:00",
  "eval_start": "2025-11-03 00:00:00",
  "eval_end": "2025-11-03 23:45:00",
  "model_candidates": [
    "EC_XGB_WIND_V1",
    "EC_LGB_WIND_V1"
  ],
  "feature_set": "ec_hres_wind_n1",
  "station": {
    "longitude": 120.60242679916666,
    "latitude": 34.31443469888889,
    "capacity_mw": 300.0
  },
  "data_paths": {
    "power": null
  },
  "etl_options": {
    "max_nwp_samples": 192,
    "sequence_steps": 9,
    "grid_size": 16,
    "horizon_codes": [
      "N1"
    ],
    "enable_wind_cleaning": false
  },
  "train_options": {
    "device": "cpu"
  },
  "run_etl": true,
  "powerData": [
    {
      "dataTime": "2025-11-02 00:00:00",
      "actualPower": 161.215,
      "theoryPower": 168.37,
      "windSpeed": 7.5992
    },
    {
      "dataTime": "2025-11-02 00:15:00",
      "actualPower": 121.606,
      "theoryPower": 136.47,
      "windSpeed": 7.052133333
    },
    {
      "dataTime": "2025-11-02 00:30:00",
      "actualPower": 76.426,
      "theoryPower": 86.37,
      "windSpeed": 5.997733333
    }
  ],
  "powerData_sample_size": 3,
  "powerData_total_size": 192
}
```

### Ingest 返回

```json
{
  "task_id": "api_contract_wind_powerdata",
  "object_type": "station",
  "station_type": "wind",
  "station_id": "js_yancheng_h3",
  "region_id": null,
  "status": "CLEANED",
  "train_start": "2025-11-02 00:00:00",
  "train_end": "2025-11-02 23:45:00",
  "eval_start": "2025-11-03 00:00:00",
  "eval_end": "2025-11-03 23:45:00",
  "feature_set": "ec_hres_wind_n1",
  "model_candidates": [
    "EC_XGB_WIND_V1",
    "EC_LGB_WIND_V1"
  ],
  "request_json": {
    "task_id": "api_contract_wind_powerdata",
    "station_id": "js_yancheng_h3",
    "object_type": "station",
    "station_type": "wind",
    "train_start": "2025-11-02 00:00:00",
    "train_end": "2025-11-02 23:45:00",
    "eval_start": "2025-11-03 00:00:00",
    "eval_end": "2025-11-03 23:45:00",
    "model_candidates": [
      "EC_XGB_WIND_V1",
      "EC_LGB_WIND_V1"
    ],
    "feature_set": "ec_hres_wind_n1",
    "station": {
      "station_id": "js_yancheng_h3",
      "station_name": null,
      "longitude": 120.60242679916666,
      "latitude": 34.31443469888889,
      "capacity_mw": 300.0
    },
    "data_paths": {},
    "powerData": [
      {
        "dataTime": "2025-11-02 00:00:00",
        "actualPower": 161.215,
        "theoryPower": 168.37,
        "windSpeed": 7.5992
      },
      {
        "dataTime": "2025-11-02 00:15:00",
        "actualPower": 121.606,
        "theoryPower": 136.47,
        "windSpeed": 7.052133333
      },
      {
        "dataTime": "2025-11-02 00:30:00",
        "actualPower": 76.426,
        "theoryPower": 86.37,
        "windSpeed": 5.997733333
      },
      {
        "dataTime": "2025-11-02 00:45:00",
        "actualPower": 96.872,
        "theoryPower": 106.17,
        "windSpeed": 6.267333333
      },
      {
        "dataTime": "2025-11-02 01:00:00",
        "actualPower": 137.81,
        "theoryPower": 158.91,
        "windSpeed": 7.392933333
      },
      {
        "dataTime": "2025-11-02 01:15:00",
        "actualPower": 247.772,
        "theoryPower": 254.16,
        "windSpeed": 9.1252
      },
      {
        "dataTime": "2025-11-02 01:30:00",
        "actualPower": 274.87,
        "theoryPower": 261.73,
        "windSpeed": 9.251333333
      },
      {
        "dataTime": "2025-11-02 01:45:00",
        "actualPower": 252.783,
        "theoryPower": 248.73,
        "windSpeed": 8.975733333
      },
      {
        "dataTime": "2025-11-02 02:00:00",
        "actualPower": 174.722,
        "theoryPower": 193.96,
        "windSpeed": 8.003466667
      },
      {
        "dataTime": "2025-11-02 02:15:00",
        "actualPower": 135.737,
        "theoryPower": 155.89,
        "windSpeed": 7.386266667
      },
      {
        "dataTime": "2025-11-02 02:30:00",
        "actualPower": 122.575,
        "theoryPower": 144.63,
        "windSpeed": 7.212
      },
      {
        "dataTime": "2025-11-02 02:45:00",
        "actualPower": 105.863,
        "theoryPower": 121.47,
        "windSpeed": 6.758933333
      },
      {
        "dataTime": "2025-11-02 03:00:00",
        "actualPower": 112.571,
        "theoryPower": 124.4,
        "windSpeed": 6.807466667
      },
      {
        "dataTime": "2025-11-02 03:15:00",
        "actualPower": 104.894,
        "theoryPower": 119.35,
        "windSpeed": 6.725066667
      },
      {
        "dataTime": "2025-11-02 03:30:00",
        "actualPower": 93.546,
        "theoryPower": 104.72,
        "windSpeed": 6.4332
      },
      {
        "dataTime": "2025-11-02 03:45:00",
        "actualPower": 120.299,
        "theoryPower": 132.99,
        "windSpeed": 7.018666667
      },
      {
        "dataTime": "2025-11-02 04:00:00",
        "actualPower": 154.119,
        "theoryPower": 176.45,
        "windSpeed": 7.700533333
      },
      {
        "dataTime": "2025-11-02 04:15:00",
        "actualPower": 160.415,
        "theoryPower": 175.67,
        "windSpeed": 7.708
      },
      {
        "dataTime": "2025-11-02 04:30:00",
        "actualPower": 148.201,
        "theoryPower": 168.52,
        "windSpeed": 7.591066667
      },
      {
        "dataTime": "2025-11-02 04:45:00",
        "actualPower": 163.901,
        "theoryPower": 180.48,
        "windSpeed": 7.761733333
      },
      {
        "dataTime": "2025-11-02 05:00:00",
        "actualPower": 164.392,
        "theoryPower": 175.96,
        "windSpeed": 7.709466667
      },
      {
        "dataTime": "2025-11-02 05:15:00",
        "actualPower": 180.934,
        "theoryPower": 192.07,
        "windSpeed": 7.9704
      },
      {
        "dataTime": "2025-11-02 05:30:00",
        "actualPower": 202.419,
        "theoryPower": 210.02,
        "windSpeed": 8.237066667
      },
      {
        "dataTime": "2025-11-02 05:45:00",
        "actualPower": 237.808,
        "theoryPower": 235.91,
        "windSpeed": 8.757066667
      },
      {
        "dataTime": "2025-11-02 06:00:00",
        "actualPower": 286.852,
        "theoryPower": 280.04,
        "windSpeed": 9.844
      },
      {
        "dataTime": "2025-11-02 06:15:00",
        "actualPower": 290.993,
        "theoryPower": 281.86,
        "windSpeed": 9.812266667
      },
      {
        "dataTime": "2025-11-02 06:30:00",
        "actualPower": 294.384,
        "theoryPower": 285.73,
        "windSpeed": 9.957866667
      },
      {
        "dataTime": "2025-11-02 06:45:00",
        "actualPower": 295.567,
        "theoryPower": 293.0,
        "windSpeed": 10.20733333
      },
      {
        "dataTime": "2025-11-02 07:00:00",
        "actualPower": 292.662,
        "theoryPower": 279.54,
        "windSpeed": 9.822933333
      },
      {
        "dataTime": "2025-11-02 07:15:00",
        "actualPower": 289.948,
        "theoryPower": 279.24,
        "windSpeed": 9.709066667
      },
      {
        "dataTime": "2025-11-02 07:30:00",
        "actualPower": 287.055,
        "theoryPower": 277.24,
        "windSpeed": 9.6876
      },
      {
        "dataTime": "2025-11-02 07:45:00",
        "actualPower": 283.328,
        "theoryPower": 273.28,
        "windSpeed": 9.4744
      },
      {
        "dataTime": "2025-11-02 08:00:00",
        "actualPower": 274.847,
        "theoryPower": 272.08,
        "windSpeed": 9.485066667
      },
      {
        "dataTime": "2025-11-02 08:15:00",
        "actualPower": 156.849,
        "theoryPower": 260.37,
        "windSpeed": 9.1424
      },
      {
        "dataTime": "2025-11-02 08:30:00",
        "actualPower": 261.479,
        "theoryPower": 278.18,
        "windSpeed": 9.627066667
      },
      {
        "dataTime": "2025-11-02 08:45:00",
        "actualPower": 212.501,
        "theoryPower": 277.89,
        "windSpeed": 9.537333333
      },
      {
        "dataTime": "2025-11-02 09:00:00",
        "actualPower": 283.903,
        "theoryPower": 275.66,
        "windSpeed": 9.627066667
      },
      {
        "dataTime": "2025-11-02 09:15:00",
        "actualPower": 76.302,
        "theoryPower": 266.8,
        "windSpeed": 9.323333333
      },
      {
        "dataTime": "2025-11-02 09:30:00",
        "actualPower": 60.474,
        "theoryPower": 251.36,
        "windSpeed": 9.015066667
      },
      {
        "dataTime": "2025-11-02 09:45:00",
        "actualPower": 82.455,
        "theoryPower": 255.47,
        "windSpeed": 9.039466667
      },
      {
        "dataTime": "2025-11-02 10:00:00",
        "actualPower": 87.682,
        "theoryPower": 231.31,
        "windSpeed": 8.6604
      },
      {
        "dataTime": "2025-11-02 10:15:00",
        "actualPower": 108.622,
        "theoryPower": 224.99,
        "windSpeed": 8.530133333
      },
      {
        "dataTime": "2025-11-02 10:30:00",
        "actualPower": 75.219,
        "theoryPower": 217.76,
        "windSpeed": 8.391866667
      },
      {
        "dataTime": "2025-11-02 10:45:00",
        "actualPower": 72.211,
        "theoryPower": 212.81,
        "windSpeed": 8.3012
      },
      {
        "dataTime": "2025-11-02 11:00:00",
        "actualPower": 93.448,
        "theoryPower": 210.52,
        "windSpeed": 8.272666667
      },
      {
        "dataTime": "2025-11-02 11:15:00",
        "actualPower": 91.548,
        "theoryPower": 209.16,
        "windSpeed": 8.254266667
      },
      {
        "dataTime": "2025-11-02 11:30:00",
        "actualPower": 76.656,
        "theoryPower": 164.38,
        "windSpeed": 7.554
      },
      {
        "dataTime": "2025-11-02 11:45:00",
        "actualPower": 67.748,
        "theoryPower": 149.15,
        "windSpeed": 7.282133333
      },
      {
        "dataTime": "2025-11-02 12:00:00",
        "actualPower": 61.263,
        "theoryPower": 130.17,
        "windSpeed": 6.970266667
      },
      {
        "dataTime": "2025-11-02 12:15:00",
        "actualPower": 61.253,
        "theoryPower": 120.71,
        "windSpeed": 6.7916
      },
      {
        "dataTime": "2025-11-02 12:30:00",
        "actualPower": 84.424,
        "theoryPower": 127.6,
        "windSpeed": 6.9236
      },
      {
        "dataTime": "2025-11-02 12:45:00",
        "actualPower": 76.281,
        "theoryPower": 128.47,
        "windSpeed": 6.943066667
      },
      {
        "dataTime": "2025-11-02 13:00:00",
        "actualPower": 61.05,
        "theoryPower": 126.1,
        "windSpeed": 6.895466667
      },
      {
        "dataTime": "2025-11-02 13:15:00",
        "actualPower": 71.143,
        "theoryPower": 127.81,
        "windSpeed": 6.9392
      },
      {
        "dataTime": "2025-11-02 13:30:00",
        "actualPower": 101.397,
        "theoryPower": 152.18,
        "windSpeed": 7.349733333
      },
      {
        "dataTime": "2025-11-02 13:45:00",
        "actualPower": 62.712,
        "theoryPower": 141.6,
        "windSpeed": 7.170933333
      },
      {
        "dataTime": "2025-11-02 14:00:00",
        "actualPower": 61.254,
        "theoryPower": 147.67,
        "windSpeed": 7.244933333
      },
      {
        "dataTime": "2025-11-02 14:15:00",
        "actualPower": 69.709,
        "theoryPower": 152.44,
        "windSpeed": 7.339333333
      },
      {
        "dataTime": "2025-11-02 14:30:00",
        "actualPower": 135.936,
        "theoryPower": 175.65,
        "windSpeed": 7.720133333
      },
      {
        "dataTime": "2025-11-02 14:45:00",
        "actualPower": 111.816,
        "theoryPower": 161.44,
        "windSpeed": 7.524533333
      },
      {
        "dataTime": "2025-11-02 15:00:00",
        "actualPower": 147.971,
        "theoryPower": 175.17,
        "windSpeed": 7.729733333
      },
      {
        "dataTime": "2025-11-02 15:15:00",
        "actualPower": 112.265,
        "theoryPower": 175.35,
        "windSpeed": 7.728266667
      },
      {
        "dataTime": "2025-11-02 15:30:00",
        "actualPower": 154.425,
        "theoryPower": 168.87,
        "windSpeed": 7.6344
      },
      {
        "dataTime": "2025-11-02 15:45:00",
        "actualPower": 158.685,
        "theoryPower": 184.08,
        "windSpeed": 7.8696
      },
      {
        "dataTime": "2025-11-02 16:00:00",
        "actualPower": 185.492,
        "theoryPower": 189.16,
        "windSpeed": 7.928133333
      },
      {
        "dataTime": "2025-11-02 16:15:00",
        "actualPower": 178.46,
        "theoryPower": 181.03,
        "windSpeed": 7.797466667
      },
      {
        "dataTime": "2025-11-02 16:30:00",
        "actualPower": 171.479,
        "theoryPower": 181.22,
        "windSpeed": 7.790933333
      },
      {
        "dataTime": "2025-11-02 16:45:00",
        "actualPower": 174.246,
        "theoryPower": 177.35,
        "windSpeed": 7.729466667
      },
      {
        "dataTime": "2025-11-02 17:00:00",
        "actualPower": 177.128,
        "theoryPower": 172.09,
        "windSpeed": 7.662266667
      },
      {
        "dataTime": "2025-11-02 17:15:00",
        "actualPower": 152.49,
        "theoryPower": 156.59,
        "windSpeed": 7.380666667
      },
      {
        "dataTime": "2025-11-02 17:30:00",
        "actualPower": 151.596,
        "theoryPower": 163.09,
        "windSpeed": 7.506133333
      },
      {
        "dataTime": "2025-11-02 17:45:00",
        "actualPower": 162.882,
        "theoryPower": 171.09,
        "windSpeed": 7.6428
      },
      {
        "dataTime": "2025-11-02 18:00:00",
        "actualPower": 186.602,
        "theoryPower": 193.6,
        "windSpeed": 7.9836
      },
      {
        "dataTime": "2025-11-02 18:15:00",
        "actualPower": 170.59,
        "theoryPower": 171.48,
        "windSpeed": 7.612266667
      },
      {
        "dataTime": "2025-11-02 18:30:00",
        "actualPower": 151.541,
        "theoryPower": 161.25,
        "windSpeed": 7.451333333
      },
      {
        "dataTime": "2025-11-02 18:45:00",
        "actualPower": 113.473,
        "theoryPower": 120.03,
        "windSpeed": 6.6788
      },
      {
        "dataTime": "2025-11-02 19:00:00",
        "actualPower": 101.484,
        "theoryPower": 114.26,
        "windSpeed": 6.589333333
      },
      {
        "dataTime": "2025-11-02 19:15:00",
        "actualPower": 69.26,
        "theoryPower": 80.3,
        "windSpeed": 5.752133333
      },
      {
        "dataTime": "2025-11-02 19:30:00",
        "actualPower": 83.133,
        "theoryPower": 96.83,
        "windSpeed": 6.1984
      },
      {
        "dataTime": "2025-11-02 19:45:00",
        "actualPower": 65.889,
        "theoryPower": 75.82,
        "windSpeed": 5.652666667
      },
      {
        "dataTime": "2025-11-02 20:00:00",
        "actualPower": 74.133,
        "theoryPower": 79.6,
        "windSpeed": 5.775333333
      },
      {
        "dataTime": "2025-11-02 20:15:00",
        "actualPower": 66.287,
        "theoryPower": 75.12,
        "windSpeed": 5.729066667
      },
      {
        "dataTime": "2025-11-02 20:30:00",
        "actualPower": 66.091,
        "theoryPower": 77.74,
        "windSpeed": 5.745733333
      },
      {
        "dataTime": "2025-11-02 20:45:00",
        "actualPower": 57.267,
        "theoryPower": 69.91,
        "windSpeed": 5.494266667
      },
      {
        "dataTime": "2025-11-02 21:00:00",
        "actualPower": 68.472,
        "theoryPower": 79.97,
        "windSpeed": 5.785066667
      },
      {
        "dataTime": "2025-11-02 21:15:00",
        "actualPower": 79.311,
        "theoryPower": 95.37,
        "windSpeed": 6.192533333
      },
      {
        "dataTime": "2025-11-02 21:30:00",
        "actualPower": 79.722,
        "theoryPower": 90.21,
        "windSpeed": 6.05
      },
      {
        "dataTime": "2025-11-02 21:45:00",
        "actualPower": 98.009,
        "theoryPower": 107.35,
        "windSpeed": 6.444533333
      },
      {
        "dataTime": "2025-11-02 22:00:00",
        "actualPower": 110.886,
        "theoryPower": 130.68,
        "windSpeed": 6.891066667
      },
      {
        "dataTime": "2025-11-02 22:15:00",
        "actualPower": 96.963,
        "theoryPower": 109.84,
        "windSpeed": 6.510533333
      },
      {
        "dataTime": "2025-11-02 22:30:00",
        "actualPower": 105.521,
        "theoryPower": 118.21,
        "windSpeed": 6.643466667
      },
      {
        "dataTime": "2025-11-02 22:45:00",
        "actualPower": 91.172,
        "theoryPower": 103.68,
        "windSpeed": 6.400933333
      },
      {
        "dataTime": "2025-11-02 23:00:00",
        "actualPower": 87.133,
        "theoryPower": 99.82,
        "windSpeed": 6.296933333
      },
      {
        "dataTime": "2025-11-02 23:15:00",
        "actualPower": 90.665,
        "theoryPower": 98.43,
        "windSpeed": 6.260133333
      },
      {
        "dataTime": "2025-11-02 23:30:00",
        "actualPower": 75.161,
        "theoryPower": 88.74,
        "windSpeed": 5.986933333
      },
      {
        "dataTime": "2025-11-02 23:45:00",
        "actualPower": 98.925,
        "theoryPower": 109.99,
        "windSpeed": 6.481866667
      },
      {
        "dataTime": "2025-11-03 00:00:00",
        "actualPower": 98.856,
        "theoryPower": 114.79,
        "windSpeed": 6.5732
      },
      {
        "dataTime": "2025-11-03 00:15:00",
        "actualPower": 100.521,
        "theoryPower": 110.44,
        "windSpeed": 6.526533333
      },
      {
        "dataTime": "2025-11-03 00:30:00",
        "actualPower": 99.634,
        "theoryPower": 108.67,
        "windSpeed": 6.432533333
      },
      {
        "dataTime": "2025-11-03 00:45:00",
        "actualPower": 108.965,
        "theoryPower": 122.57,
        "windSpeed": 6.774666667
      },
      {
        "dataTime": "2025-11-03 01:00:00",
        "actualPower": 95.008,
        "theoryPower": 103.43,
        "windSpeed": 6.377466667
      },
      {
        "dataTime": "2025-11-03 01:15:00",
        "actualPower": 89.087,
        "theoryPower": 97.33,
        "windSpeed": 6.2256
      },
      {
        "dataTime": "2025-11-03 01:30:00",
        "actualPower": 86.87,
        "theoryPower": 91.81,
        "windSpeed": 6.162933333
      },
      {
        "dataTime": "2025-11-03 01:45:00",
        "actualPower": 91.692,
        "theoryPower": 101.41,
        "windSpeed": 6.318133333
      },
      {
        "dataTime": "2025-11-03 02:00:00",
        "actualPower": 77.762,
        "theoryPower": 90.99,
        "windSpeed": 6.084266667
      },
      {
        "dataTime": "2025-11-03 02:15:00",
        "actualPower": 67.286,
        "theoryPower": 74.68,
        "windSpeed": 5.697066667
      },
      {
        "dataTime": "2025-11-03 02:30:00",
        "actualPower": 55.449,
        "theoryPower": 62.01,
        "windSpeed": 5.396933333
      },
      {
        "dataTime": "2025-11-03 02:45:00",
        "actualPower": 66.606,
        "theoryPower": 71.93,
        "windSpeed": 5.625333333
      },
      {
        "dataTime": "2025-11-03 03:00:00",
        "actualPower": 52.271,
        "theoryPower": 59.03,
        "windSpeed": 5.272133333
      },
      {
        "dataTime": "2025-11-03 03:15:00",
        "actualPower": 52.751,
        "theoryPower": 62.85,
        "windSpeed": 5.3636
      },
      {
        "dataTime": "2025-11-03 03:30:00",
        "actualPower": 42.309,
        "theoryPower": 46.09,
        "windSpeed": 4.850666667
      },
      {
        "dataTime": "2025-11-03 03:45:00",
        "actualPower": 35.971,
        "theoryPower": 41.53,
        "windSpeed": 4.673333333
      },
      {
        "dataTime": "2025-11-03 04:00:00",
        "actualPower": 47.825,
        "theoryPower": 54.93,
        "windSpeed": 5.148533333
      },
      {
        "dataTime": "2025-11-03 04:15:00",
        "actualPower": 40.936,
        "theoryPower": 48.35,
        "windSpeed": 4.954133333
      },
      {
        "dataTime": "2025-11-03 04:30:00",
        "actualPower": 37.341,
        "theoryPower": 46.64,
        "windSpeed": 4.8124
      },
      {
        "dataTime": "2025-11-03 04:45:00",
        "actualPower": 39.851,
        "theoryPower": 48.57,
        "windSpeed": 4.883866667
      },
      {
        "dataTime": "2025-11-03 05:00:00",
        "actualPower": 40.655,
        "theoryPower": 46.88,
        "windSpeed": 4.8624
      },
      {
        "dataTime": "2025-11-03 05:15:00",
        "actualPower": 35.655,
        "theoryPower": 42.94,
        "windSpeed": 4.684266667
      },
      {
        "dataTime": "2025-11-03 05:30:00",
        "actualPower": 26.954,
        "theoryPower": 32.08,
        "windSpeed": 4.245733333
      },
      {
        "dataTime": "2025-11-03 05:45:00",
        "actualPower": 15.324,
        "theoryPower": 22.13,
        "windSpeed": 3.765333333
      },
      {
        "dataTime": "2025-11-03 06:00:00",
        "actualPower": 7.079,
        "theoryPower": 15.94,
        "windSpeed": 3.406133333
      },
      {
        "dataTime": "2025-11-03 06:15:00",
        "actualPower": 18.526,
        "theoryPower": 24.1,
        "windSpeed": 3.916533333
      },
      {
        "dataTime": "2025-11-03 06:30:00",
        "actualPower": 34.12,
        "theoryPower": 40.71,
        "windSpeed": 4.661333333
      },
      {
        "dataTime": "2025-11-03 06:45:00",
        "actualPower": 40.298,
        "theoryPower": 48.33,
        "windSpeed": 4.9372
      },
      {
        "dataTime": "2025-11-03 07:00:00",
        "actualPower": 37.24,
        "theoryPower": 44.54,
        "windSpeed": 4.814
      },
      {
        "dataTime": "2025-11-03 07:15:00",
        "actualPower": 33.16,
        "theoryPower": 41.39,
        "windSpeed": 4.661066667
      },
      {
        "dataTime": "2025-11-03 07:30:00",
        "actualPower": 26.673,
        "theoryPower": 33.98,
        "windSpeed": 4.3532
      },
      {
        "dataTime": "2025-11-03 07:45:00",
        "actualPower": 16.004,
        "theoryPower": 22.15,
        "windSpeed": 3.574133333
      },
      {
        "dataTime": "2025-11-03 08:00:00",
        "actualPower": 18.419,
        "theoryPower": 24.93,
        "windSpeed": 3.9116
      },
      {
        "dataTime": "2025-11-03 08:15:00",
        "actualPower": 19.521,
        "theoryPower": 26.65,
        "windSpeed": 4.0116
      },
      {
        "dataTime": "2025-11-03 08:30:00",
        "actualPower": 28.581,
        "theoryPower": 36.19,
        "windSpeed": 4.499866667
      },
      {
        "dataTime": "2025-11-03 08:45:00",
        "actualPower": 28.164,
        "theoryPower": 36.24,
        "windSpeed": 4.467466667
      },
      {
        "dataTime": "2025-11-03 09:00:00",
        "actualPower": 27.51,
        "theoryPower": 30.62,
        "windSpeed": 4.260933333
      },
      {
        "dataTime": "2025-11-03 09:15:00",
        "actualPower": 34.322,
        "theoryPower": 41.0,
        "windSpeed": 4.6944
      },
      {
        "dataTime": "2025-11-03 09:30:00",
        "actualPower": 11.529,
        "theoryPower": 18.1,
        "windSpeed": 3.631866667
      },
      {
        "dataTime": "2025-11-03 09:45:00",
        "actualPower": 14.56,
        "theoryPower": 20.62,
        "windSpeed": 3.645333333
      },
      {
        "dataTime": "2025-11-03 10:00:00",
        "actualPower": 8.149,
        "theoryPower": 15.57,
        "windSpeed": 3.4388
      },
      {
        "dataTime": "2025-11-03 10:15:00",
        "actualPower": 1.321,
        "theoryPower": 7.44,
        "windSpeed": 2.6672
      },
      {
        "dataTime": "2025-11-03 10:30:00",
        "actualPower": -2.279,
        "theoryPower": 3.36,
        "windSpeed": 2.2288
      },
      {
        "dataTime": "2025-11-03 10:45:00",
        "actualPower": -2.512,
        "theoryPower": 2.59,
        "windSpeed": 2.022533333
      },
      {
        "dataTime": "2025-11-03 11:00:00",
        "actualPower": -2.75,
        "theoryPower": 2.12,
        "windSpeed": 1.992933333
      },
      {
        "dataTime": "2025-11-03 11:15:00",
        "actualPower": -2.911,
        "theoryPower": 1.49,
        "windSpeed": 2.035866667
      },
      {
        "dataTime": "2025-11-03 11:30:00",
        "actualPower": -2.733,
        "theoryPower": 0.7,
        "windSpeed": 1.765333333
      },
      {
        "dataTime": "2025-11-03 11:45:00",
        "actualPower": -2.762,
        "theoryPower": 1.31,
        "windSpeed": 1.8856
      },
      {
        "dataTime": "2025-11-03 12:00:00",
        "actualPower": -2.662,
        "theoryPower": 1.64,
        "windSpeed": 1.851066667
      },
      {
        "dataTime": "2025-11-03 12:15:00",
        "actualPower": -2.663,
        "theoryPower": 0.76,
        "windSpeed": 1.710933333
      },
      {
        "dataTime": "2025-11-03 12:30:00",
        "actualPower": -2.662,
        "theoryPower": 0.28,
        "windSpeed": 1.785066667
      },
      {
        "dataTime": "2025-11-03 12:45:00",
        "actualPower": -2.612,
        "theoryPower": 0.69,
        "windSpeed": 1.613466667
      },
      {
        "dataTime": "2025-11-03 13:00:00",
        "actualPower": -2.694,
        "theoryPower": 0.38,
        "windSpeed": 1.426933333
      },
      {
        "dataTime": "2025-11-03 13:15:00",
        "actualPower": -2.625,
        "theoryPower": 0.63,
        "windSpeed": 1.564133333
      },
      {
        "dataTime": "2025-11-03 13:30:00",
        "actualPower": -2.588,
        "theoryPower": 0.58,
        "windSpeed": 1.375466667
      },
      {
        "dataTime": "2025-11-03 13:45:00",
        "actualPower": -2.564,
        "theoryPower": -0.04,
        "windSpeed": 1.152533333
      },
      {
        "dataTime": "2025-11-03 14:00:00",
        "actualPower": -2.476,
        "theoryPower": 0.04,
        "windSpeed": 1.068
      },
      {
        "dataTime": "2025-11-03 14:15:00",
        "actualPower": -2.452,
        "theoryPower": 0.05,
        "windSpeed": 0.995333333
      },
      {
        "dataTime": "2025-11-03 14:30:00",
        "actualPower": -2.546,
        "theoryPower": -0.01,
        "windSpeed": 0.924933333
      },
      {
        "dataTime": "2025-11-03 14:45:00",
        "actualPower": -2.545,
        "theoryPower": 0.15,
        "windSpeed": 0.910933333
      },
      {
        "dataTime": "2025-11-03 15:00:00",
        "actualPower": -2.431,
        "theoryPower": 0.05,
        "windSpeed": 0.9992
      },
      {
        "dataTime": "2025-11-03 15:15:00",
        "actualPower": -2.382,
        "theoryPower": 0.27,
        "windSpeed": 1.3868
      },
      {
        "dataTime": "2025-11-03 15:30:00",
        "actualPower": -2.38,
        "theoryPower": 0.17,
        "windSpeed": 1.385066667
      },
      {
        "dataTime": "2025-11-03 15:45:00",
        "actualPower": -2.403,
        "theoryPower": 0.09,
        "windSpeed": 1.273733333
      },
      {
        "dataTime": "2025-11-03 16:00:00",
        "actualPower": -2.396,
        "theoryPower": 0.63,
        "windSpeed": 1.420266667
      },
      {
        "dataTime": "2025-11-03 16:15:00",
        "actualPower": -2.381,
        "theoryPower": 0.26,
        "windSpeed": 1.477066667
      },
      {
        "dataTime": "2025-11-03 16:30:00",
        "actualPower": -2.396,
        "theoryPower": 0.43,
        "windSpeed": 1.365333333
      },
      {
        "dataTime": "2025-11-03 16:45:00",
        "actualPower": -2.364,
        "theoryPower": 0.03,
        "windSpeed": 1.2908
      },
      {
        "dataTime": "2025-11-03 17:00:00",
        "actualPower": -2.325,
        "theoryPower": 0.22,
        "windSpeed": 1.2456
      },
      {
        "dataTime": "2025-11-03 17:15:00",
        "actualPower": -2.433,
        "theoryPower": 0.26,
        "windSpeed": 1.4824
      },
      {
        "dataTime": "2025-11-03 17:30:00",
        "actualPower": -2.359,
        "theoryPower": 0.24,
        "windSpeed": 1.6612
      },
      {
        "dataTime": "2025-11-03 17:45:00",
        "actualPower": -2.435,
        "theoryPower": 1.09,
        "windSpeed": 1.604133333
      },
      {
        "dataTime": "2025-11-03 18:00:00",
        "actualPower": -2.34,
        "theoryPower": 0.54,
        "windSpeed": 1.5956
      },
      {
        "dataTime": "2025-11-03 18:15:00",
        "actualPower": -2.396,
        "theoryPower": 0.77,
        "windSpeed": 1.646666667
      },
      {
        "dataTime": "2025-11-03 18:30:00",
        "actualPower": -2.331,
        "theoryPower": 2.34,
        "windSpeed": 1.9428
      },
      {
        "dataTime": "2025-11-03 18:45:00",
        "actualPower": -2.444,
        "theoryPower": 1.4,
        "windSpeed": 1.937866667
      },
      {
        "dataTime": "2025-11-03 19:00:00",
        "actualPower": -2.51,
        "theoryPower": 1.66,
        "windSpeed": 1.994
      },
      {
        "dataTime": "2025-11-03 19:15:00",
        "actualPower": -2.435,
        "theoryPower": 0.91,
        "windSpeed": 1.909466667
      },
      {
        "dataTime": "2025-11-03 19:30:00",
        "actualPower": -2.272,
        "theoryPower": 1.26,
        "windSpeed": 1.804533333
      },
      {
        "dataTime": "2025-11-03 19:45:00",
        "actualPower": -1.975,
        "theoryPower": 3.25,
        "windSpeed": 2.1612
      },
      {
        "dataTime": "2025-11-03 20:00:00",
        "actualPower": -2.039,
        "theoryPower": 4.0,
        "windSpeed": 2.396666667
      },
      {
        "dataTime": "2025-11-03 20:15:00",
        "actualPower": -0.19,
        "theoryPower": 5.37,
        "windSpeed": 2.61
      },
      {
        "dataTime": "2025-11-03 20:30:00",
        "actualPower": -1.902,
        "theoryPower": 2.48,
        "windSpeed": 2.175866667
      },
      {
        "dataTime": "2025-11-03 20:45:00",
        "actualPower": -1.362,
        "theoryPower": 3.52,
        "windSpeed": 2.339466667
      },
      {
        "dataTime": "2025-11-03 21:00:00",
        "actualPower": -2.261,
        "theoryPower": 3.7,
        "windSpeed": 2.403733333
      },
      {
        "dataTime": "2025-11-03 21:15:00",
        "actualPower": 2.29,
        "theoryPower": 9.76,
        "windSpeed": 2.9484
      },
      {
        "dataTime": "2025-11-03 21:30:00",
        "actualPower": 7.093,
        "theoryPower": 13.2,
        "windSpeed": 3.257466667
      },
      {
        "dataTime": "2025-11-03 21:45:00",
        "actualPower": 1.203,
        "theoryPower": 5.47,
        "windSpeed": 2.515333333
      },
      {
        "dataTime": "2025-11-03 22:00:00",
        "actualPower": 1.153,
        "theoryPower": 6.61,
        "windSpeed": 2.6264
      },
      {
        "dataTime": "2025-11-03 22:15:00",
        "actualPower": -1.871,
        "theoryPower": 3.68,
        "windSpeed": 2.416266667
      },
      {
        "dataTime": "2025-11-03 22:30:00",
        "actualPower": 0.85,
        "theoryPower": 6.5,
        "windSpeed": 2.697333333
      },
      {
        "dataTime": "2025-11-03 22:45:00",
        "actualPower": -2.162,
        "theoryPower": 1.59,
        "windSpeed": 2.032933333
      },
      {
        "dataTime": "2025-11-03 23:00:00",
        "actualPower": -2.226,
        "theoryPower": 3.62,
        "windSpeed": 2.406266667
      },
      {
        "dataTime": "2025-11-03 23:15:00",
        "actualPower": -2.242,
        "theoryPower": 2.14,
        "windSpeed": 2.198933333
      },
      {
        "dataTime": "2025-11-03 23:30:00",
        "actualPower": -2.21,
        "theoryPower": 1.67,
        "windSpeed": 2.006133333
      },
      {
        "dataTime": "2025-11-03 23:45:00",
        "actualPower": -2.499,
        "theoryPower": 1.05,
        "windSpeed": 1.958533333
      }
    ],
    "etl_options": {
      "max_nwp_samples": 192,
      "sequence_steps": 9,
      "grid_size": 16,
      "horizon_codes": [
        "N1"
      ],
      "enable_wind_cleaning": false
    },
    "train_options": {
      "device": "cpu"
    },
    "capacity_mw": 300.0,
    "artifacts": {
      "clean_series": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/clean_series.csv",
      "train_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/train_dataset_nwp_ml.csv",
      "eval_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/eval_dataset_nwp_ml.csv",
      "feature_schema": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/feature_schema.json",
      "summary": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/data_check_summary.json",
      "nwp_nwp_aligned_table": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/nwp_aligned_table.csv",
      "nwp_train_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/train_dataset_nwp_ml.csv",
      "nwp_eval_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/eval_dataset_nwp_ml.csv",
      "nwp_train_tensor_x": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/train_nwp_x.npy",
      "nwp_train_tensor_y": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/train_nwp_y.npy",
      "nwp_eval_tensor_x": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/eval_nwp_x.npy",
      "nwp_eval_tensor_y": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/eval_nwp_y.npy",
      "nwp_tensor_meta": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/nwp_tensor_meta.json",
      "nwp_ml_baseline_joblib": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/data/nwp_aligned/ml_baseline_dataset.joblib"
    },
    "data_summary": {
      "check_result": "PASS",
      "aligned_samples": 192,
      "train_samples": 96,
      "eval_samples": 96,
      "missing_issue_count": 0,
      "missing_issues_sample": [],
      "failed_rows": 0,
      "channels": [
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_dir_10m_sin",
        "wind_dir_10m_cos",
        "wind_dir_100m_sin",
        "wind_dir_100m_cos",
        "t2m",
        "sp",
        "pressure_wind_speed_900",
        "pressure_wind_speed_850",
        "pressure_wind_speed_800",
        "pressure_wind_speed_700",
        "pressure_wind_dir_900_sin",
        "pressure_wind_dir_900_cos",
        "pressure_wind_dir_850_sin",
        "pressure_wind_dir_850_cos",
        "pressure_wind_dir_800_sin",
        "pressure_wind_dir_800_cos",
        "pressure_wind_dir_700_sin",
        "pressure_wind_dir_700_cos"
      ],
      "feature_count": 101,
      "feature_contract": "x=nwp_only,y=power_mw",
      "features": [
        "lead_hours",
        "nwp_wind_speed_10m_mean",
        "nwp_wind_speed_10m_std",
        "nwp_wind_speed_10m_min",
        "nwp_wind_speed_10m_max",
        "nwp_wind_speed_10m_center_t0",
        "nwp_wind_speed_100m_mean",
        "nwp_wind_speed_100m_std",
        "nwp_wind_speed_100m_min",
        "nwp_wind_speed_100m_max",
        "nwp_wind_speed_100m_center_t0",
        "nwp_wind_dir_10m_sin_mean",
        "nwp_wind_dir_10m_sin_std",
        "nwp_wind_dir_10m_sin_min",
        "nwp_wind_dir_10m_sin_max",
        "nwp_wind_dir_10m_sin_center_t0",
        "nwp_wind_dir_10m_cos_mean",
        "nwp_wind_dir_10m_cos_std",
        "nwp_wind_dir_10m_cos_min",
        "nwp_wind_dir_10m_cos_max",
        "nwp_wind_dir_10m_cos_center_t0",
        "nwp_wind_dir_100m_sin_mean",
        "nwp_wind_dir_100m_sin_std",
        "nwp_wind_dir_100m_sin_min",
        "nwp_wind_dir_100m_sin_max",
        "nwp_wind_dir_100m_sin_center_t0",
        "nwp_wind_dir_100m_cos_mean",
        "nwp_wind_dir_100m_cos_std",
        "nwp_wind_dir_100m_cos_min",
        "nwp_wind_dir_100m_cos_max",
        "nwp_wind_dir_100m_cos_center_t0",
        "nwp_t2m_mean",
        "nwp_t2m_std",
        "nwp_t2m_min",
        "nwp_t2m_max",
        "nwp_t2m_center_t0",
        "nwp_sp_mean",
        "nwp_sp_std",
        "nwp_sp_min",
        "nwp_sp_max",
        "nwp_sp_center_t0",
        "nwp_pressure_wind_speed_900_mean",
        "nwp_pressure_wind_speed_900_std",
        "nwp_pressure_wind_speed_900_min",
        "nwp_pressure_wind_speed_900_max",
        "nwp_pressure_wind_speed_900_center_t0",
        "nwp_pressure_wind_speed_850_mean",
        "nwp_pressure_wind_speed_850_std",
        "nwp_pressure_wind_speed_850_min",
        "nwp_pressure_wind_speed_850_max",
        "nwp_pressure_wind_speed_850_center_t0",
        "nwp_pressure_wind_speed_800_mean",
        "nwp_pressure_wind_speed_800_std",
        "nwp_pressure_wind_speed_800_min",
        "nwp_pressure_wind_speed_800_max",
        "nwp_pressure_wind_speed_800_center_t0",
        "nwp_pressure_wind_speed_700_mean",
        "nwp_pressure_wind_speed_700_std",
        "nwp_pressure_wind_speed_700_min",
        "nwp_pressure_wind_speed_700_max",
        "nwp_pressure_wind_speed_700_center_t0",
        "nwp_pressure_wind_dir_900_sin_mean",
        "nwp_pressure_wind_dir_900_sin_std",
        "nwp_pressure_wind_dir_900_sin_min",
        "nwp_pressure_wind_dir_900_sin_max",
        "nwp_pressure_wind_dir_900_sin_center_t0",
        "nwp_pressure_wind_dir_900_cos_mean",
        "nwp_pressure_wind_dir_900_cos_std",
        "nwp_pressure_wind_dir_900_cos_min",
        "nwp_pressure_wind_dir_900_cos_max",
        "nwp_pressure_wind_dir_900_cos_center_t0",
        "nwp_pressure_wind_dir_850_sin_mean",
        "nwp_pressure_wind_dir_850_sin_std",
        "nwp_pressure_wind_dir_850_sin_min",
        "nwp_pressure_wind_dir_850_sin_max",
        "nwp_pressure_wind_dir_850_sin_center_t0",
        "nwp_pressure_wind_dir_850_cos_mean",
        "nwp_pressure_wind_dir_850_cos_std",
        "nwp_pressure_wind_dir_850_cos_min",
        "nwp_pressure_wind_dir_850_cos_max",
        "nwp_pressure_wind_dir_850_cos_center_t0",
        "nwp_pressure_wind_dir_800_sin_mean",
        "nwp_pressure_wind_dir_800_sin_std",
        "nwp_pressure_wind_dir_800_sin_min",
        "nwp_pressure_wind_dir_800_sin_max",
        "nwp_pressure_wind_dir_800_sin_center_t0",
        "nwp_pressure_wind_dir_800_cos_mean",
        "nwp_pressure_wind_dir_800_cos_std",
        "nwp_pressure_wind_dir_800_cos_min",
        "nwp_pressure_wind_dir_800_cos_max",
        "nwp_pressure_wind_dir_800_cos_center_t0",
        "nwp_pressure_wind_dir_700_sin_mean",
        "nwp_pressure_wind_dir_700_sin_std",
        "nwp_pressure_wind_dir_700_sin_min",
        "nwp_pressure_wind_dir_700_sin_max",
        "nwp_pressure_wind_dir_700_sin_center_t0",
        "nwp_pressure_wind_dir_700_cos_mean",
        "nwp_pressure_wind_dir_700_cos_std",
        "nwp_pressure_wind_dir_700_cos_min",
        "nwp_pressure_wind_dir_700_cos_max",
        "nwp_pressure_wind_dir_700_cos_center_t0"
      ],
      "start_time": "2025-11-02 00:00:00",
      "end_time": "2025-11-03 23:45:00",
      "dataset_mode": "nwp_aligned",
      "nwp_error": null,
      "station": {
        "station_id": "js_yancheng_h3",
        "station_name": null,
        "longitude": 120.60242679916666,
        "latitude": 34.31443469888889,
        "capacity_mw": 300.0
      },
      "nwp": {
        "nwp_root": "/mnt/d/data/netcdf/ecmwf/jiangsu",
        "file_count": 163,
        "first_issue": "2025110100",
        "last_issue": "2026012212",
        "sample_files": [
          "jiangsu_2025110100.nc",
          "jiangsu_2025110112.nc",
          "jiangsu_2025110200.nc",
          "jiangsu_2025110212.nc",
          "jiangsu_2025110300.nc"
        ],
        "issues": []
      },
      "capacity_mw": 300.0
    }
  },
  "config_path": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata/config/task_config.json",
  "work_dir": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_wind_powerdata",
  "published_model_id": null,
  "error_message": null,
  "created_time": "2026-07-02T06:53:49.973878+00:00",
  "updated_time": "2026-07-02T06:53:53.844310+00:00"
}
```

### Train 返回摘要

```json
{
  "task_id": "api_contract_wind_powerdata",
  "trained_models": [
    {
      "candidate": "EC_XGB_WIND_V1",
      "status": "TRAINED",
      "model_id": "api_contract_wind_powerdata_EC_XGB_WIND_V1_20260702145353"
    },
    {
      "candidate": "EC_LGB_WIND_V1",
      "status": "TRAINED",
      "model_id": "api_contract_wind_powerdata_EC_LGB_WIND_V1_20260702145437"
    }
  ]
}
```

### Evaluate 返回

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {
    "task_id": "api_contract_wind_powerdata",
    "daily_accuracy": [
      {
        "date": "2025-11-03 00:00:00",
        "mae": 88.393799825553,
        "accuracy": 0.70535400058149,
        "n": 96.0
      }
    ]
  }
}
```

### Evaluate Result 返回

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {
    "task_id": "api_contract_wind_powerdata",
    "daily_accuracy": [
      {
        "date": "2025-11-03 00:00:00",
        "mae": 88.393799825553,
        "accuracy": 0.70535400058149,
        "n": 96.0
      }
    ]
  }
}
```

### 数据契约检查

```json
{
  "feature_contract": "x=nwp_only,y=power_mw",
  "target": "power_mw",
  "feature_count": 101,
  "pure_nwp_features": true,
  "non_nwp_features": [],
  "nwp_root_from_config": "/mnt/d/data/netcdf/ecmwf/jiangsu",
  "dataset_mode": "nwp_aligned"
}
```

## solar

### Ingest 请求样例

```json
{
  "task_id": "api_contract_solar_powerdata",
  "station_id": "nmg_shuijinghu_solar",
  "region_id": null,
  "object_type": "station",
  "station_type": "solar",
  "train_start": "2025-09-02 00:00:00",
  "train_end": "2025-09-02 23:45:00",
  "eval_start": "2025-09-03 00:00:00",
  "eval_end": "2025-09-03 23:45:00",
  "model_candidates": [
    "EC_XGB_PV_V1",
    "EC_LGB_PV_V1"
  ],
  "feature_set": "ec_hres_solar_n1",
  "station": {
    "longitude": 109.7,
    "latitude": 40.30528,
    "capacity_mw": 100.0
  },
  "data_paths": {
    "power": null
  },
  "etl_options": {
    "max_nwp_samples": 192,
    "sequence_steps": 9,
    "grid_size": 16,
    "horizon_codes": [
      "N1"
    ],
    "enable_solar_cleaning": false
  },
  "train_options": {
    "device": "cpu"
  },
  "run_etl": true,
  "powerData": [
    {
      "dataTime": "2025-09-02 00:00:00",
      "actualPower": 0.0,
      "directIrradiance": null
    },
    {
      "dataTime": "2025-09-02 00:15:00",
      "actualPower": 0.0,
      "directIrradiance": null
    },
    {
      "dataTime": "2025-09-02 00:30:00",
      "actualPower": 0.0,
      "directIrradiance": null
    }
  ],
  "powerData_sample_size": 3,
  "powerData_total_size": 192
}
```

### Ingest 返回

```json
{
  "task_id": "api_contract_solar_powerdata",
  "object_type": "station",
  "station_type": "solar",
  "station_id": "nmg_shuijinghu_solar",
  "region_id": null,
  "status": "CLEANED",
  "train_start": "2025-09-02 00:00:00",
  "train_end": "2025-09-02 23:45:00",
  "eval_start": "2025-09-03 00:00:00",
  "eval_end": "2025-09-03 23:45:00",
  "feature_set": "ec_hres_solar_n1",
  "model_candidates": [
    "EC_XGB_PV_V1",
    "EC_LGB_PV_V1"
  ],
  "request_json": {
    "task_id": "api_contract_solar_powerdata",
    "station_id": "nmg_shuijinghu_solar",
    "object_type": "station",
    "station_type": "solar",
    "train_start": "2025-09-02 00:00:00",
    "train_end": "2025-09-02 23:45:00",
    "eval_start": "2025-09-03 00:00:00",
    "eval_end": "2025-09-03 23:45:00",
    "model_candidates": [
      "EC_XGB_PV_V1",
      "EC_LGB_PV_V1"
    ],
    "feature_set": "ec_hres_solar_n1",
    "station": {
      "station_id": "nmg_shuijinghu_solar",
      "station_name": null,
      "longitude": 109.7,
      "latitude": 40.30528,
      "capacity_mw": 100.0
    },
    "data_paths": {},
    "powerData": [
      {
        "dataTime": "2025-09-02 00:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 00:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 00:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 00:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 01:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 01:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 01:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 01:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 02:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 02:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 02:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 02:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 03:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 03:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 03:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 03:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 04:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 04:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 04:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 04:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 05:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 05:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 05:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 05:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 06:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 06:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 06:30:00",
        "actualPower": 0.47
      },
      {
        "dataTime": "2025-09-02 06:45:00",
        "actualPower": 2.28
      },
      {
        "dataTime": "2025-09-02 07:00:00",
        "actualPower": 3.52
      },
      {
        "dataTime": "2025-09-02 07:15:00",
        "actualPower": 4.62
      },
      {
        "dataTime": "2025-09-02 07:30:00",
        "actualPower": 8.24
      },
      {
        "dataTime": "2025-09-02 07:45:00",
        "actualPower": 16.52
      },
      {
        "dataTime": "2025-09-02 08:00:00",
        "actualPower": 12.53
      },
      {
        "dataTime": "2025-09-02 08:15:00",
        "actualPower": 24.92
      },
      {
        "dataTime": "2025-09-02 08:30:00",
        "actualPower": 22.68
      },
      {
        "dataTime": "2025-09-02 08:45:00",
        "actualPower": 45.32
      },
      {
        "dataTime": "2025-09-02 09:00:00",
        "actualPower": 39.76
      },
      {
        "dataTime": "2025-09-02 09:15:00",
        "actualPower": 51.05
      },
      {
        "dataTime": "2025-09-02 09:30:00",
        "actualPower": 57.15
      },
      {
        "dataTime": "2025-09-02 09:45:00",
        "actualPower": 54.5
      },
      {
        "dataTime": "2025-09-02 10:00:00",
        "actualPower": 61.67
      },
      {
        "dataTime": "2025-09-02 10:15:00",
        "actualPower": 62.21
      },
      {
        "dataTime": "2025-09-02 10:30:00",
        "actualPower": 34.0
      },
      {
        "dataTime": "2025-09-02 10:45:00",
        "actualPower": 26.46
      },
      {
        "dataTime": "2025-09-02 11:00:00",
        "actualPower": 30.42
      },
      {
        "dataTime": "2025-09-02 11:15:00",
        "actualPower": 45.19
      },
      {
        "dataTime": "2025-09-02 11:30:00",
        "actualPower": 47.0
      },
      {
        "dataTime": "2025-09-02 11:45:00",
        "actualPower": 52.86
      },
      {
        "dataTime": "2025-09-02 12:00:00",
        "actualPower": 54.1
      },
      {
        "dataTime": "2025-09-02 12:15:00",
        "actualPower": 53.03
      },
      {
        "dataTime": "2025-09-02 12:30:00",
        "actualPower": 56.55
      },
      {
        "dataTime": "2025-09-02 12:45:00",
        "actualPower": 46.03
      },
      {
        "dataTime": "2025-09-02 13:00:00",
        "actualPower": 21.94
      },
      {
        "dataTime": "2025-09-02 13:15:00",
        "actualPower": 30.65
      },
      {
        "dataTime": "2025-09-02 13:30:00",
        "actualPower": 21.57
      },
      {
        "dataTime": "2025-09-02 13:45:00",
        "actualPower": 16.38
      },
      {
        "dataTime": "2025-09-02 14:00:00",
        "actualPower": 11.46
      },
      {
        "dataTime": "2025-09-02 14:15:00",
        "actualPower": 7.57
      },
      {
        "dataTime": "2025-09-02 14:30:00",
        "actualPower": 7.5
      },
      {
        "dataTime": "2025-09-02 14:45:00",
        "actualPower": 9.31
      },
      {
        "dataTime": "2025-09-02 15:00:00",
        "actualPower": 11.93
      },
      {
        "dataTime": "2025-09-02 15:15:00",
        "actualPower": 13.47
      },
      {
        "dataTime": "2025-09-02 15:30:00",
        "actualPower": 9.08
      },
      {
        "dataTime": "2025-09-02 15:45:00",
        "actualPower": 11.39
      },
      {
        "dataTime": "2025-09-02 16:00:00",
        "actualPower": 13.84
      },
      {
        "dataTime": "2025-09-02 16:15:00",
        "actualPower": 16.85
      },
      {
        "dataTime": "2025-09-02 16:30:00",
        "actualPower": 22.31
      },
      {
        "dataTime": "2025-09-02 16:45:00",
        "actualPower": 25.49
      },
      {
        "dataTime": "2025-09-02 17:00:00",
        "actualPower": 30.05
      },
      {
        "dataTime": "2025-09-02 17:15:00",
        "actualPower": 28.31
      },
      {
        "dataTime": "2025-09-02 17:30:00",
        "actualPower": 23.95
      },
      {
        "dataTime": "2025-09-02 17:45:00",
        "actualPower": 20.94
      },
      {
        "dataTime": "2025-09-02 18:00:00",
        "actualPower": 15.85
      },
      {
        "dataTime": "2025-09-02 18:15:00",
        "actualPower": 11.22
      },
      {
        "dataTime": "2025-09-02 18:30:00",
        "actualPower": 4.49
      },
      {
        "dataTime": "2025-09-02 18:45:00",
        "actualPower": 3.62
      },
      {
        "dataTime": "2025-09-02 19:00:00",
        "actualPower": 1.14
      },
      {
        "dataTime": "2025-09-02 19:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 19:30:00",
        "actualPower": -0.44
      },
      {
        "dataTime": "2025-09-02 19:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 20:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 20:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 20:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 20:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 21:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 21:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 21:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 21:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 22:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 22:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 22:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 22:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 23:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 23:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 23:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-02 23:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 00:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 00:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 00:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 00:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 01:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 01:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 01:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 01:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 02:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 02:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 02:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 02:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 03:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 03:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 03:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 03:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 04:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 04:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 04:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 04:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 05:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 05:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 05:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 05:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 06:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 06:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 06:30:00",
        "actualPower": 2.18
      },
      {
        "dataTime": "2025-09-03 06:45:00",
        "actualPower": 4.56
      },
      {
        "dataTime": "2025-09-03 07:00:00",
        "actualPower": 7.14
      },
      {
        "dataTime": "2025-09-03 07:15:00",
        "actualPower": 7.2
      },
      {
        "dataTime": "2025-09-03 07:30:00",
        "actualPower": 7.34
      },
      {
        "dataTime": "2025-09-03 07:45:00",
        "actualPower": 7.17
      },
      {
        "dataTime": "2025-09-03 08:00:00",
        "actualPower": 7.17
      },
      {
        "dataTime": "2025-09-03 08:15:00",
        "actualPower": 7.91
      },
      {
        "dataTime": "2025-09-03 08:30:00",
        "actualPower": 8.58
      },
      {
        "dataTime": "2025-09-03 08:45:00",
        "actualPower": 9.58
      },
      {
        "dataTime": "2025-09-03 09:00:00",
        "actualPower": 10.28
      },
      {
        "dataTime": "2025-09-03 09:15:00",
        "actualPower": 10.72
      },
      {
        "dataTime": "2025-09-03 09:30:00",
        "actualPower": 6.33
      },
      {
        "dataTime": "2025-09-03 09:45:00",
        "actualPower": 6.7
      },
      {
        "dataTime": "2025-09-03 10:00:00",
        "actualPower": 7.04
      },
      {
        "dataTime": "2025-09-03 10:15:00",
        "actualPower": 7.37
      },
      {
        "dataTime": "2025-09-03 10:30:00",
        "actualPower": 8.58
      },
      {
        "dataTime": "2025-09-03 10:45:00",
        "actualPower": 8.27
      },
      {
        "dataTime": "2025-09-03 11:00:00",
        "actualPower": 11.52
      },
      {
        "dataTime": "2025-09-03 11:15:00",
        "actualPower": 12.5
      },
      {
        "dataTime": "2025-09-03 11:30:00",
        "actualPower": 19.4
      },
      {
        "dataTime": "2025-09-03 11:45:00",
        "actualPower": 29.11
      },
      {
        "dataTime": "2025-09-03 12:00:00",
        "actualPower": 27.03
      },
      {
        "dataTime": "2025-09-03 12:15:00",
        "actualPower": 40.7
      },
      {
        "dataTime": "2025-09-03 12:30:00",
        "actualPower": 29.78
      },
      {
        "dataTime": "2025-09-03 12:45:00",
        "actualPower": 33.5
      },
      {
        "dataTime": "2025-09-03 13:00:00",
        "actualPower": 52.83
      },
      {
        "dataTime": "2025-09-03 13:15:00",
        "actualPower": 42.34
      },
      {
        "dataTime": "2025-09-03 13:30:00",
        "actualPower": 48.98
      },
      {
        "dataTime": "2025-09-03 13:45:00",
        "actualPower": 63.01
      },
      {
        "dataTime": "2025-09-03 14:00:00",
        "actualPower": 23.58
      },
      {
        "dataTime": "2025-09-03 14:15:00",
        "actualPower": 27.23
      },
      {
        "dataTime": "2025-09-03 14:30:00",
        "actualPower": 23.95
      },
      {
        "dataTime": "2025-09-03 14:45:00",
        "actualPower": 26.43
      },
      {
        "dataTime": "2025-09-03 15:00:00",
        "actualPower": 12.96
      },
      {
        "dataTime": "2025-09-03 15:15:00",
        "actualPower": 13.6
      },
      {
        "dataTime": "2025-09-03 15:30:00",
        "actualPower": 13.1
      },
      {
        "dataTime": "2025-09-03 15:45:00",
        "actualPower": 14.87
      },
      {
        "dataTime": "2025-09-03 16:00:00",
        "actualPower": 11.72
      },
      {
        "dataTime": "2025-09-03 16:15:00",
        "actualPower": 9.35
      },
      {
        "dataTime": "2025-09-03 16:30:00",
        "actualPower": 5.46
      },
      {
        "dataTime": "2025-09-03 16:45:00",
        "actualPower": 5.23
      },
      {
        "dataTime": "2025-09-03 17:00:00",
        "actualPower": 31.15
      },
      {
        "dataTime": "2025-09-03 17:15:00",
        "actualPower": 29.68
      },
      {
        "dataTime": "2025-09-03 17:30:00",
        "actualPower": 25.33
      },
      {
        "dataTime": "2025-09-03 17:45:00",
        "actualPower": 18.53
      },
      {
        "dataTime": "2025-09-03 18:00:00",
        "actualPower": 13.27
      },
      {
        "dataTime": "2025-09-03 18:15:00",
        "actualPower": 5.7
      },
      {
        "dataTime": "2025-09-03 18:30:00",
        "actualPower": 4.19
      },
      {
        "dataTime": "2025-09-03 18:45:00",
        "actualPower": 2.01
      },
      {
        "dataTime": "2025-09-03 19:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 19:15:00",
        "actualPower": -0.4
      },
      {
        "dataTime": "2025-09-03 19:30:00",
        "actualPower": -0.37
      },
      {
        "dataTime": "2025-09-03 19:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 20:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 20:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 20:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 20:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 21:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 21:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 21:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 21:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 22:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 22:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 22:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 22:45:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 23:00:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 23:15:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 23:30:00",
        "actualPower": 0.0
      },
      {
        "dataTime": "2025-09-03 23:45:00",
        "actualPower": 0.0
      }
    ],
    "etl_options": {
      "max_nwp_samples": 192,
      "sequence_steps": 9,
      "grid_size": 16,
      "horizon_codes": [
        "N1"
      ],
      "enable_solar_cleaning": false
    },
    "train_options": {
      "device": "cpu"
    },
    "capacity_mw": 100.0,
    "artifacts": {
      "clean_series": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/clean_series.csv",
      "train_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/train_dataset_nwp_ml.csv",
      "eval_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/eval_dataset_nwp_ml.csv",
      "feature_schema": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/feature_schema.json",
      "summary": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/data_check_summary.json",
      "nwp_nwp_aligned_table": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/nwp_aligned_table.csv",
      "nwp_train_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/train_dataset_nwp_ml.csv",
      "nwp_eval_dataset": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/eval_dataset_nwp_ml.csv",
      "nwp_train_tensor_x": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/train_nwp_x.npy",
      "nwp_train_tensor_y": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/train_nwp_y.npy",
      "nwp_eval_tensor_x": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/eval_nwp_x.npy",
      "nwp_eval_tensor_y": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/eval_nwp_y.npy",
      "nwp_tensor_meta": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/nwp_tensor_meta.json",
      "nwp_ml_baseline_joblib": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/data/nwp_aligned/ml_baseline_dataset.joblib"
    },
    "data_summary": {
      "check_result": "PASS",
      "aligned_samples": 192,
      "train_samples": 96,
      "eval_samples": 96,
      "missing_issue_count": 0,
      "missing_issues_sample": [],
      "failed_rows": 0,
      "channels": [
        "ssrd",
        "fdir",
        "cdir",
        "t2m",
        "tcc",
        "lcc",
        "mcc",
        "hcc",
        "sp"
      ],
      "feature_count": 46,
      "feature_contract": "x=nwp_only,y=power_mw",
      "features": [
        "lead_hours",
        "nwp_ssrd_mean",
        "nwp_ssrd_std",
        "nwp_ssrd_min",
        "nwp_ssrd_max",
        "nwp_ssrd_center_t0",
        "nwp_fdir_mean",
        "nwp_fdir_std",
        "nwp_fdir_min",
        "nwp_fdir_max",
        "nwp_fdir_center_t0",
        "nwp_cdir_mean",
        "nwp_cdir_std",
        "nwp_cdir_min",
        "nwp_cdir_max",
        "nwp_cdir_center_t0",
        "nwp_t2m_mean",
        "nwp_t2m_std",
        "nwp_t2m_min",
        "nwp_t2m_max",
        "nwp_t2m_center_t0",
        "nwp_tcc_mean",
        "nwp_tcc_std",
        "nwp_tcc_min",
        "nwp_tcc_max",
        "nwp_tcc_center_t0",
        "nwp_lcc_mean",
        "nwp_lcc_std",
        "nwp_lcc_min",
        "nwp_lcc_max",
        "nwp_lcc_center_t0",
        "nwp_mcc_mean",
        "nwp_mcc_std",
        "nwp_mcc_min",
        "nwp_mcc_max",
        "nwp_mcc_center_t0",
        "nwp_hcc_mean",
        "nwp_hcc_std",
        "nwp_hcc_min",
        "nwp_hcc_max",
        "nwp_hcc_center_t0",
        "nwp_sp_mean",
        "nwp_sp_std",
        "nwp_sp_min",
        "nwp_sp_max",
        "nwp_sp_center_t0"
      ],
      "start_time": "2025-09-02 00:00:00",
      "end_time": "2025-09-03 23:45:00",
      "dataset_mode": "nwp_aligned",
      "nwp_error": null,
      "station": {
        "station_id": "nmg_shuijinghu_solar",
        "station_name": null,
        "longitude": 109.7,
        "latitude": 40.30528,
        "capacity_mw": 100.0
      },
      "nwp": {
        "nwp_root": "/mnt/d/data/netcdf/ecmwf/neimeng",
        "file_count": 204,
        "first_issue": "2025090112",
        "last_issue": "2026033112",
        "sample_files": [
          "2025090112.nc",
          "2025090212.nc",
          "2025090312.nc",
          "2025090412.nc",
          "2025090512.nc"
        ],
        "issues": []
      },
      "capacity_mw": 100.0
    }
  },
  "config_path": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata/config/task_config.json",
  "work_dir": "/mnt/e/workspace/pwforecast/zj_mlops/zhejiangforecast_zj/runtime_backend_powerdata_api_cases_v2/tasks/api_contract_solar_powerdata",
  "published_model_id": null,
  "error_message": null,
  "created_time": "2026-07-02T06:54:50.450748+00:00",
  "updated_time": "2026-07-02T06:54:52.640073+00:00"
}
```

### Train 返回摘要

```json
{
  "task_id": "api_contract_solar_powerdata",
  "trained_models": [
    {
      "candidate": "EC_XGB_PV_V1",
      "status": "TRAINED",
      "model_id": "api_contract_solar_powerdata_EC_XGB_PV_V1_20260702145452"
    },
    {
      "candidate": "EC_LGB_PV_V1",
      "status": "TRAINED",
      "model_id": "api_contract_solar_powerdata_EC_LGB_PV_V1_20260702145523"
    }
  ]
}
```

### Evaluate 返回

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {
    "task_id": "api_contract_solar_powerdata",
    "daily_accuracy": [
      {
        "date": "2025-09-03 00:00:00",
        "mae": 8.934934127853873,
        "accuracy": 0.9106506587214612,
        "n": 96.0
      }
    ]
  }
}
```

### Evaluate Result 返回

```json
{
  "code": 200,
  "message": "请求成功！",
  "data": {
    "task_id": "api_contract_solar_powerdata",
    "daily_accuracy": [
      {
        "date": "2025-09-03 00:00:00",
        "mae": 8.934934127853873,
        "accuracy": 0.9106506587214612,
        "n": 96.0
      }
    ]
  }
}
```

### 数据契约检查

```json
{
  "feature_contract": "x=nwp_only,y=power_mw",
  "target": "power_mw",
  "feature_count": 46,
  "pure_nwp_features": true,
  "non_nwp_features": [],
  "nwp_root_from_config": "/mnt/d/data/netcdf/ecmwf/neimeng",
  "dataset_mode": "nwp_aligned"
}
```
