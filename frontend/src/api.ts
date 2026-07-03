import type {
  ApiEnvelope,
  DataEditPayload,
  DataPreviewPayload,
  DataStatusPayload,
  EvaluatePayload,
  HealthPayload,
  InferPayload,
  IngestRequest,
  IngestPayload,
  ModelSpec,
  PointEdit,
  PublishPayload,
  TrainRequest,
  TrainStatusPayload
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const payload = (await response.json()) as ApiEnvelope<T> | T;
  if (typeof payload === "object" && payload !== null && "data" in payload) {
    return (payload as ApiEnvelope<T>).data;
  }
  return payload as T;
}

export function health() {
  return request<HealthPayload>("/health");
}

export function listModels(stationType: string, objectType = "station") {
  const params = new URLSearchParams({ station_type: stationType, object_type: objectType });
  return request<{ models: ModelSpec[] }>(`/api/v1/online-modeling/model/list?${params.toString()}`);
}

export function ingest(payload: IngestRequest) {
  return request<IngestPayload>("/api/v1/online-modeling/ingest", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function dataStatus(taskId: string) {
  return request<DataStatusPayload>(`/api/v1/online-modeling/data/status?task_id=${encodeURIComponent(taskId)}`);
}

export function dataPreview(taskId: string, dataType: "train" | "eval" = "eval", limit = 300) {
  const params = new URLSearchParams({ task_id: taskId, data_type: dataType, limit: String(limit) });
  return request<DataPreviewPayload>(`/api/v1/online-modeling/data/preview?${params.toString()}`);
}

export function savePointEdits(taskId: string, pointEdits: PointEdit[]) {
  return request<DataEditPayload>("/api/v1/online-modeling/data/edit", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId, point_edits: pointEdits })
  });
}

export function train(payload: TrainRequest) {
  return request<TrainStatusPayload>("/api/v1/online-modeling/train", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function trainStatus(input: { taskId?: string; jobId?: string; includeResult?: boolean; logLimit?: number }) {
  const params = new URLSearchParams();
  if (input.taskId) params.set("task_id", input.taskId);
  if (input.jobId) params.set("job_id", input.jobId);
  params.set("include_result", String(input.includeResult ?? true));
  params.set("log_limit", String(input.logLimit ?? 20));
  return request<TrainStatusPayload>(`/api/v1/online-modeling/train/status?${params.toString()}`);
}

export function cancelTrain(jobId: string) {
  return request<Record<string, unknown>>(`/api/v1/online-modeling/train/cancel?job_id=${encodeURIComponent(jobId)}`, {
    method: "POST"
  });
}

export function evaluate(taskId: string, sync = true) {
  return request<EvaluatePayload>("/api/v1/online-modeling/evaluate", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId, sync })
  });
}

export function evaluateResult(taskId: string) {
  return request<EvaluatePayload>(`/api/v1/online-modeling/evaluate/result?task_id=${encodeURIComponent(taskId)}`);
}

export function publish(taskId: string, selectedModelId?: string) {
  return request<PublishPayload>("/api/v1/online-modeling/publish", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId, selected_model_id: selectedModelId || undefined })
  });
}

export function infer(payload: { taskId?: string; modelId?: string; issueTime?: string; data?: Record<string, unknown>[] }) {
  return request<InferPayload>("/api/v1/online-modeling/infer", {
    method: "POST",
    body: JSON.stringify({
      task_id: payload.taskId || undefined,
      model_id: payload.modelId || undefined,
      issue_time: payload.issueTime || undefined,
      data: payload.data
    })
  });
}
