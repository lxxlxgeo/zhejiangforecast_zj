# NWP 读取性能优化说明

## 目标

在线建模的 NWP ETL 尽量减少 NetCDF I/O、插值计算和重复 xarray 索引开销，同时保证不同 NWP 文件变量不一致时不会破坏整体任务。

## 当前读取顺序

`build_nwp_power_datasets` 现在按 NWP 起报文件分组处理：

1. 根据 label 的 `time_bj` 和 `horizon_code=N1` 计算 `issue_time_utc`。
2. 按 `issue_time_utc` 找对应 NetCDF 文件。
3. 同一个 NWP 文件内的 96 个或更多 label 样本合并为一个 group。
4. 每个 group 只打开一次 NetCDF。
5. 先裁剪变量和站点周边网格：`ds_sub = vars + station grid`。
6. 再裁剪该 group 实际需要的 `valid_time` 窗口。
7. 最后只对这个小窗口做 15 分钟插值。
8. 在小窗口上逐样本生成 `[C,S,H,W]` tensor 和 ML 统计特征。

这个顺序比“整文件插值后再逐行取样”更省时、省内存。

## 并行策略

NWP group 级别使用 joblib 并行：

```yaml
nwp:
  workers: 4
  parallel_backend: loky
```

也可以在单个 ingest 请求中覆盖：

```json
{
  "etl_options": {
    "nwp_workers": 6,
    "nwp_parallel_backend": "loky"
  }
}
```

推荐生产默认 `4`，机器 I/O 和内存充足时可试 `6` 或 `8`。

## 为什么默认不用线程

NetCDF/HDF5 读取和 xarray 后端在多线程下容易遇到底层 I/O 锁或线程安全问题。默认使用 joblib `loky`，即进程并行：

- 每个进程独立打开自己的 NetCDF 文件。
- 不共享 xarray Dataset 句柄。
- 避免多个线程同时读同一个 HDF5/NetCDF 句柄。

`threading` 后端只建议在确认本机 NetCDF 后端线程安全、且进程启动开销明显大于 I/O 开销时使用。

## 诊断字段

`data/status` 的 `nwp_aligned_dataset` 会包含：

- `nwp_group_count`：参与处理的 NWP 文件组数。
- `nwp_workers`：本次使用的 worker 数。
- `nwp_parallel_backend`：joblib backend。
- `nwp_parallel_fallback_reason`：并行失败后回退串行的原因。
- `nwp_prefilter_order`：当前裁剪/插值顺序。
- `channel_mismatch_count`：不同样本 NWP 通道不一致次数。
- `shape_mismatch_count`：tensor 形状不一致次数。
- `failure_samples`：前若干个失败样本。

