# 浙江省调在线建模前端

Vue 3 + Vite + TypeScript 的在线建模工作台。前端按现有 FastAPI 契约调用，不修改 `ingest`、`train`、`evaluate` 的请求和返回形式。

## 启动

先启动后端：

```powershell
cd E:\workspace\pwforecast\zj_mlops\zhejiangforecast_zj
$env:PYTHONPATH="src"
$env:ZJ_FORECAST_HOME="$PWD\runtime"
uvicorn zhejiangforecast_zj.api.main:app --host 127.0.0.1 --port 8000
```

再启动前端：

```powershell
cd E:\workspace\pwforecast\zj_mlops\zhejiangforecast_zj\frontend
npm install
npm run dev
```

访问：

```text
http://127.0.0.1:5174/
```

如果 `5174` 被占用，Vite 会自动切到下一个可用端口，例如 `5175`。

## 说明

- Vite 开发代理将 `/api`、`/health`、`/docs` 转发到 `http://127.0.0.1:8000`。
- `evaluate` 当前冻结返回只包含 `task_id` 和 `daily_accuracy`，前端按该结构展示评估结果。
- 模型指标和训练日志来自 `GET /api/v1/online-modeling/train/status`。
