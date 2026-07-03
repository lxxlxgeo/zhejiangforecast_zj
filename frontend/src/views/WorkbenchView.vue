<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { BarChart, LineChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  CloudSun,
  Cpu,
  Database,
  Gauge,
  GitBranch,
  LayoutDashboard,
  PackageCheck,
  Play,
  Plus,
  RefreshCw,
  Rocket,
  Save,
  Search,
  Send,
  Settings,
  Table2,
  Zap
} from "@lucide/vue";
import {
  cancelTrain,
  dataPreview,
  dataStatus,
  evaluate,
  evaluateResult,
  health,
  infer,
  ingest,
  listModels,
  publish,
  savePointEdits,
  train,
  trainStatus
} from "../api";
import { useTaskStore } from "../stores/taskStore";
import type {
  DataCheck,
  DataPreviewPayload,
  DataStatusPayload,
  EvaluatePayload,
  HealthPayload,
  InferPayload,
  IngestPayload,
  IngestRequest,
  ModelSpec,
  PointEdit,
  StationType,
  TrainModelArtifact,
  TrainStatusPayload
} from "../types";
import {
  compactJson,
  detectPowerField,
  detectTimeField,
  formatDateTime,
  formatNumber,
  formatPercent,
  parseJsonObject,
  readError,
  toNumber
} from "../utils";

use([CanvasRenderer, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent]);

type FormState = {
  task_id: string;
  object_type: "station" | "region";
  station_type: StationType;
  station_id: string;
  region_id: string;
  station_name: string;
  longitude: string;
  latitude: string;
  capacity_mw: number | undefined;
  train_start: string;
  train_end: string;
  eval_start: string;
  eval_end: string;
  feature_set: string;
  power_path: string;
  nwp_root: string;
  run_etl: boolean;
};

type RunRow = {
  task_id: string;
  target: string;
  station_type: string;
  status: string;
  train_samples: string;
  eval_samples: string;
  models: string;
  accuracy: string;
  updated_time: string;
  progress: number;
};

type NavKey = "overview" | "runs" | "data" | "train" | "evaluate" | "registry" | "inference" | "settings";

const store = useTaskStore();
const form = reactive<FormState>({
  task_id: "frontend_demo_wind",
  object_type: "station",
  station_type: "wind",
  station_id: "js_yancheng_h3",
  region_id: "",
  station_name: "江苏盐城 H3 测试风电场",
  longitude: "120°36'08.736477\"",
  latitude: "034°18'51.964916\"",
  capacity_mw: 300,
  train_start: "2025-11-02 00:00:00",
  train_end: "2025-11-02 23:45:00",
  eval_start: "2025-11-03 00:00:00",
  eval_end: "2025-11-03 23:45:00",
  feature_set: "ec_hres_wind_n1",
  power_path: "../测试场站数据/js_yanchengH3_wind/station_power/station_power_original_2502-2601.csv",
  nwp_root: "/mnt/d/data/netcdf/ecmwf/jiangsu",
  run_etl: true
});

const etlOptionsJson = ref(
  compactJson({
    max_nwp_samples: 192,
    sequence_steps: 9,
    grid_size: 16,
    horizon_codes: ["N1"]
  })
);
const trainOptionsJson = ref(compactJson({ device: "cpu", dl_epochs: 1, dl_batch_size: 4 }));
const powerDataJson = ref("[]");
const modelCandidates = ref<string[]>(["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"]);
const activeTaskInput = ref("");
const selectedDataType = ref<"train" | "eval">("eval");
const syncTrain = ref(false);
const syncEvaluate = ref(true);
const selectedModelId = ref("");
const issueTime = ref("2025-11-03 12:00:00");
const pointEdit = reactive<PointEdit>({
  time: "2025-11-03T00:00:00",
  field: "power_mw",
  value: null,
  reason: "人工标记异常点"
});

const loading = reactive({
  health: false,
  models: false,
  ingest: false,
  status: false,
  preview: false,
  edit: false,
  train: false,
  trainStatus: false,
  evaluate: false,
  publish: false,
  infer: false
});

const createDrawerVisible = ref(false);
const createStep = ref(0);
const activeSection = ref<NavKey>("overview");
const activeDetailTab = ref("data");
const runSearch = ref("");
const healthPayload = ref<HealthPayload | null>(null);
const modelOptions = ref<ModelSpec[]>([]);
const ingestResult = ref<IngestPayload | null>(null);
const statusResult = ref<DataStatusPayload | null>(null);
const previewResult = ref<DataPreviewPayload | null>(null);
const trainResult = ref<TrainStatusPayload | null>(null);
const evaluatePayload = ref<EvaluatePayload | null>(null);
const publishPayload = ref<Record<string, unknown> | null>(null);
const inferPayload = ref<InferPayload | null>(null);
let pollTimer: number | undefined;

const statusMeta: Record<string, { text: string; type: "primary" | "success" | "warning" | "danger" | "info" }> = {
  CREATED: { text: "已创建", type: "info" },
  DATA_READY: { text: "数据就绪", type: "primary" },
  CLEANED: { text: "已清洗", type: "success" },
  TRAINING: { text: "训练中", type: "warning" },
  TRAINED: { text: "已训练", type: "success" },
  EVALUATED: { text: "已评估", type: "success" },
  PUBLISHED: { text: "已发布", type: "success" },
  FAILED: { text: "失败", type: "danger" },
  CANCELED: { text: "已取消", type: "info" },
  SUCCESS: { text: "完成", type: "success" },
  RUNNING: { text: "运行中", type: "warning" },
  UNKNOWN: { text: "未加载", type: "info" }
};

const sidebarItems: Array<{ key: NavKey; label: string; icon: unknown }> = [
  { key: "overview", label: "Overview", icon: LayoutDashboard },
  { key: "runs", label: "Runs", icon: Table2 },
  { key: "data", label: "Data", icon: Database },
  { key: "train", label: "Training", icon: Cpu },
  { key: "evaluate", label: "Evaluation", icon: BarChart3 },
  { key: "registry", label: "Registry", icon: PackageCheck },
  { key: "inference", label: "Inference", icon: Zap },
  { key: "settings", label: "Settings", icon: Settings }
];

const activeTaskId = computed(() => store.activeTaskId || ingestResult.value?.task_id || form.task_id);
const currentStatus = computed(() => trainResult.value?.task_status || statusResult.value?.status || ingestResult.value?.status);
const latestChecks = computed(() => statusResult.value?.data_checks ?? []);
const datasetCheck = computed(() => latestChecks.value.find((item) => item.data_type === "dataset"));
const datasetSummary = computed(() => datasetCheck.value?.summary_json ?? ingestResult.value?.request_json?.data_summary ?? {});
const trainedModels = computed(() => trainResult.value?.models ?? []);
const selectedModelOptions = computed(() => trainedModels.value.filter((item) => item.status === "TRAINED"));
const previewRows = computed(() => previewResult.value?.rows ?? []);
const previewColumns = computed(() => buildTableColumns(previewRows.value));
const evaluationDays = computed(() => evaluatePayload.value?.daily_accuracy ?? []);
const progressPercent = computed(() => Math.round((trainResult.value?.progress ?? 0) * 100));
const serviceLabel = computed(() => (healthPayload.value ? `API ${healthPayload.value.status}` : "API 未连接"));
const targetLabel = computed(() => form.station_name || form.station_id || form.region_id || "-");
const trainWindowText = computed(() => `${form.train_start || "-"} ~ ${form.train_end || "-"}`);
const evalWindowText = computed(() => `${form.eval_start || "-"} ~ ${form.eval_end || "-"}`);

const dataCheckColumns = [
  { prop: "data_type", label: "类型", minWidth: 130 },
  { prop: "check_result", label: "结果", width: 90 },
  { prop: "missing_rate", label: "缺失率", width: 90 },
  { prop: "start_time", label: "开始", minWidth: 150 },
  { prop: "end_time", label: "结束", minWidth: 150 }
];

const payloadPreview = computed(() => {
  try {
    return compactJson(buildIngestPayload());
  } catch (error) {
    return readError(error);
  }
});

const averageAccuracy = computed(() => {
  const values = evaluationDays.value.map((row) => row.accuracy).filter((value): value is number => typeof value === "number");
  if (!values.length) return "-";
  return formatPercent(values.reduce((sum, value) => sum + value, 0) / values.length);
});

const metricStrip = computed(() => [
  { label: "Run 状态", value: statusTag(currentStatus.value).text, note: activeTaskId.value, tone: "blue" },
  { label: "数据模式", value: String(datasetSummary.value.dataset_mode ?? "-"), note: "dataset", tone: "green" },
  { label: "对齐样本", value: formatNumber(datasetSummary.value.aligned_samples, 0), note: "aligned", tone: "blue" },
  { label: "训练样本", value: formatNumber(datasetSummary.value.train_samples, 0), note: "train", tone: "blue" },
  { label: "评估均值", value: averageAccuracy.value, note: "daily accuracy", tone: "green" },
  { label: "模型产物", value: `${trainResult.value?.trained_model_count ?? 0}/${trainResult.value?.model_count ?? 0}`, note: "trained/all", tone: "purple" }
]);

const allRunRows = computed<RunRow[]>(() => {
  const taskIds = Array.from(new Set([activeTaskId.value, activeTaskInput.value, form.task_id, ...store.recentTasks].filter(Boolean)));
  return taskIds.map((taskId) => {
    const isActive = taskId === activeTaskId.value;
    return {
      task_id: taskId,
      target: isActive ? targetLabel.value : "-",
      station_type: isActive ? form.station_type : "-",
      status: isActive ? String(currentStatus.value ?? "CREATED") : "UNKNOWN",
      train_samples: isActive ? formatNumber(datasetSummary.value.train_samples, 0) : "-",
      eval_samples: isActive ? formatNumber(datasetSummary.value.eval_samples, 0) : "-",
      models: isActive ? `${trainResult.value?.trained_model_count ?? 0}/${trainResult.value?.model_count ?? 0}` : "-",
      accuracy: isActive ? averageAccuracy.value : "-",
      updated_time: isActive ? formatDateTime(trainResult.value?.updated_time ?? ingestResult.value?.updated_time) : "-",
      progress: isActive ? progressPercent.value : 0
    };
  });
});

const runRows = computed(() => {
  const query = runSearch.value.trim().toLowerCase();
  if (!query) return allRunRows.value;
  return allRunRows.value.filter((row) => [row.task_id, row.target, row.status].some((value) => value.toLowerCase().includes(query)));
});

const selectedRun = computed(() => allRunRows.value.find((row) => row.task_id === activeTaskId.value) ?? allRunRows.value[0]);

const lifecycleCards = computed(() => {
  const status = String(currentStatus.value ?? "CREATED");
  const isFailed = status === "FAILED";
  return [
    {
      title: "Ingest",
      desc: datasetSummary.value.dataset_mode ? `${datasetSummary.value.dataset_mode} / ${formatNumber(datasetSummary.value.aligned_samples, 0)} rows` : "等待数据接入",
      state: isFailed ? "error" : status ? "done" : "wait"
    },
    {
      title: "Train",
      desc: trainResult.value?.job_status || `${trainResult.value?.trained_model_count ?? 0} trained`,
      state: isFailed ? "error" : ["TRAINING"].includes(status) ? "process" : trainedModels.value.length ? "done" : "wait"
    },
    {
      title: "Evaluate",
      desc: evaluationDays.value.length ? `${evaluationDays.value.length} days / ${averageAccuracy.value}` : "等待评估结果",
      state: isFailed ? "error" : evaluationDays.value.length ? "done" : "wait"
    },
    {
      title: "Deploy",
      desc: selectedModelId.value || String(publishPayload.value?.model_id ?? "未发布"),
      state: isFailed ? "error" : publishPayload.value || status === "PUBLISHED" ? "done" : "wait"
    }
  ];
});

const previewChartOption = computed(() => {
  const rows = previewRows.value;
  const first = rows[0] ?? {};
  const timeField = detectTimeField(first);
  const powerField = detectPowerField(first);
  const windField = "wind_speed_mean" in first ? "wind_speed_mean" : "windSpeed" in first ? "windSpeed" : "";
  return {
    color: ["#2563eb", "#10b981"],
    tooltip: { trigger: "axis" },
    legend: { top: 0, right: 0 },
    grid: { left: 54, right: 18, top: 40, bottom: 42 },
    xAxis: { type: "category", data: rows.map((row) => String(row[timeField] ?? "")), axisLabel: { hideOverlap: true } },
    yAxis: [{ type: "value", name: powerField || "value" }],
    series: [
      {
        name: powerField || "功率",
        type: "line",
        showSymbol: false,
        smooth: true,
        data: rows.map((row) => toNumber(row[powerField]))
      },
      ...(windField
        ? [
            {
              name: windField,
              type: "line",
              showSymbol: false,
              smooth: true,
              data: rows.map((row) => toNumber(row[windField]))
            }
          ]
        : [])
    ]
  };
});

const modelMetricChartOption = computed(() => {
  const rows = trainedModels.value;
  return {
    color: ["#2563eb", "#10b981", "#f59e0b"],
    tooltip: { trigger: "axis" },
    legend: { top: 0, right: 0 },
    grid: { left: 54, right: 18, top: 42, bottom: 72 },
    xAxis: { type: "category", data: rows.map((item) => shortModelName(item.model_id)), axisLabel: { rotate: 24 } },
    yAxis: { type: "value" },
    series: [
      { name: "accuracy", type: "bar", data: rows.map((item) => metricValue(item, "accuracy")) },
      { name: "avg_accuracy", type: "bar", data: rows.map((item) => metricValue(item, "avg_accuracy")) },
      { name: "mae", type: "bar", data: rows.map((item) => metricValue(item, "mae")) }
    ]
  };
});

const dailyAccuracyChartOption = computed(() => ({
  color: ["#2563eb"],
  tooltip: { trigger: "axis" },
  grid: { left: 54, right: 18, top: 28, bottom: 42 },
  xAxis: { type: "category", data: evaluationDays.value.map((row) => row.date) },
  yAxis: { type: "value", min: 0, max: 1, name: "accuracy" },
  series: [
    {
      name: "日准确率",
      type: "bar",
      barMaxWidth: 34,
      data: evaluationDays.value.map((row) => row.accuracy)
    }
  ]
}));

const inferChartOption = computed(() => ({
  color: ["#2563eb"],
  tooltip: { trigger: "axis" },
  grid: { left: 54, right: 18, top: 28, bottom: 42 },
  xAxis: {
    type: "category",
    data: (inferPayload.value?.predictions ?? []).map((row) => String(row.valid_time ?? row.time ?? ""))
  },
  yAxis: { type: "value", name: "MW" },
  series: [
    {
      name: "预测功率",
      type: "line",
      showSymbol: false,
      smooth: true,
      data: (inferPayload.value?.predictions ?? []).map((row) => toNumber(row.p_pred_mw ?? row.p_pred))
    }
  ]
}));

onMounted(async () => {
  await Promise.allSettled([refreshModels(true), checkHealth(true)]);
  if (store.activeTaskId) {
    activeTaskInput.value = store.activeTaskId;
    await loadTaskBundle(store.activeTaskId);
  }
});

watch(
  () => form.station_type,
  async (stationType) => {
    applyStationDefaults(stationType);
    await refreshModels(true);
  }
);

watch(
  () => form.object_type,
  async () => {
    await refreshModels(true);
  }
);

watch(
  () => trainResult.value?.done,
  (done) => {
    if (done && pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = undefined;
    }
  }
);

function buildIngestPayload(): IngestRequest {
  const powerData = parseJsonObject<unknown>(powerDataJson.value, []);
  if (!Array.isArray(powerData)) {
    throw new Error("powerData 必须是数组。");
  }
  const etlOptions = parseJsonObject<Record<string, unknown>>(etlOptionsJson.value, {});
  const trainOptions = parseJsonObject<Record<string, unknown>>(trainOptionsJson.value, {});
  return {
    task_id: form.task_id || undefined,
    station_id: form.station_id || undefined,
    region_id: form.region_id || undefined,
    object_type: form.object_type,
    station_type: form.station_type,
    train_start: form.train_start || undefined,
    train_end: form.train_end || undefined,
    eval_start: form.eval_start || undefined,
    eval_end: form.eval_end || undefined,
    model_candidates: modelCandidates.value,
    feature_set: form.feature_set || undefined,
    station: {
      station_name: form.station_name || undefined,
      longitude: form.longitude || undefined,
      latitude: form.latitude || undefined,
      capacity_mw: form.capacity_mw
    },
    data_paths: {
      power: form.power_path || undefined,
      nwp_root: form.nwp_root || undefined
    },
    powerData: powerData.length ? powerData : undefined,
    etl_options: etlOptions,
    train_options: trainOptions,
    run_etl: form.run_etl
  };
}

async function checkHealth(silent = false) {
  loading.health = true;
  try {
    healthPayload.value = await health();
  } catch (error) {
    if (!silent) ElMessage.error(readError(error));
  } finally {
    loading.health = false;
  }
}

async function refreshModels(silent = false) {
  loading.models = true;
  try {
    const response = await listModels(form.station_type, form.object_type);
    modelOptions.value = response.models;
    if (!modelCandidates.value.length) {
      modelCandidates.value = response.models.slice(0, 2).map((item) => item.model_name);
    }
  } catch (error) {
    if (!silent) ElMessage.error(readError(error));
  } finally {
    loading.models = false;
  }
}

async function submitIngest() {
  loading.ingest = true;
  try {
    const result = await ingest(buildIngestPayload());
    ingestResult.value = result;
    store.setActiveTask(result.task_id);
    activeTaskInput.value = result.task_id;
    ElMessage.success(`任务 ${result.task_id} 已创建`);
    await loadTaskBundle(result.task_id);
    return true;
  } catch (error) {
    ElMessage.error(readError(error));
    return false;
  } finally {
    loading.ingest = false;
  }
}

async function loadTaskBundle(taskId = activeTaskInput.value || activeTaskId.value) {
  if (!taskId) return;
  store.setActiveTask(taskId);
  activeTaskInput.value = taskId;
  await Promise.allSettled([loadStatus(taskId), loadPreview(taskId), refreshTrainStatus(taskId), loadEvaluateResult(taskId)]);
}

async function loadStatus(taskId = activeTaskId.value) {
  if (!taskId) return;
  loading.status = true;
  try {
    statusResult.value = await dataStatus(taskId);
  } catch (error) {
    ElMessage.warning(readError(error));
  } finally {
    loading.status = false;
  }
}

async function loadPreview(taskId = activeTaskId.value) {
  if (!taskId) return;
  loading.preview = true;
  try {
    previewResult.value = await dataPreview(taskId, selectedDataType.value, 400);
  } catch (error) {
    ElMessage.warning(readError(error));
  } finally {
    loading.preview = false;
  }
}

async function submitPointEdit() {
  if (!activeTaskId.value) return;
  loading.edit = true;
  try {
    await savePointEdits(activeTaskId.value, [{ ...pointEdit }]);
    ElMessage.success("点位记录已保存");
    await loadStatus(activeTaskId.value);
  } catch (error) {
    ElMessage.error(readError(error));
  } finally {
    loading.edit = false;
  }
}

async function triggerTrain() {
  if (!activeTaskId.value) return;
  loading.train = true;
  try {
    const result = await train({
      task_id: activeTaskId.value,
      model_candidates: modelCandidates.value,
      train_mode: "local",
      sync: syncTrain.value
    });
    trainResult.value = result;
    store.setActiveJob(result.job_id);
    selectedModelId.value = firstTrainedModelId(result.models);
    activeDetailTab.value = "train";
    ElMessage.success(syncTrain.value ? "训练完成" : "训练已提交");
    if (!result.done) startTrainPolling(result.job_id ?? undefined);
  } catch (error) {
    ElMessage.error(readError(error));
  } finally {
    loading.train = false;
  }
}

async function refreshTrainStatus(taskId = activeTaskId.value) {
  if (!taskId && !store.activeJobId) return;
  loading.trainStatus = true;
  try {
    trainResult.value = await trainStatus({
      taskId: store.activeJobId ? undefined : taskId,
      jobId: store.activeJobId || undefined,
      includeResult: true,
      logLimit: 20
    });
    selectedModelId.value ||= firstTrainedModelId(trainResult.value.models);
  } catch (error) {
    if (taskId) {
      ElMessage.warning(readError(error));
    }
  } finally {
    loading.trainStatus = false;
  }
}

function startTrainPolling(jobId?: string) {
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = window.setInterval(() => {
    void refreshTrainStatus(jobId ? "" : activeTaskId.value);
  }, 3000);
}

async function stopTrain() {
  const jobId = trainResult.value?.job_id || store.activeJobId;
  if (!jobId) return;
  try {
    await cancelTrain(jobId);
    ElMessage.success("取消请求已记录");
    await refreshTrainStatus();
  } catch (error) {
    ElMessage.error(readError(error));
  }
}

async function triggerEvaluate() {
  if (!activeTaskId.value) return;
  loading.evaluate = true;
  try {
    evaluatePayload.value = await evaluate(activeTaskId.value, syncEvaluate.value);
    activeDetailTab.value = "evaluate";
    ElMessage.success("评估完成");
    await loadTaskBundle(activeTaskId.value);
  } catch (error) {
    ElMessage.error(readError(error));
  } finally {
    loading.evaluate = false;
  }
}

async function loadEvaluateResult(taskId = activeTaskId.value) {
  if (!taskId) return;
  try {
    evaluatePayload.value = await evaluateResult(taskId);
  } catch {
    evaluatePayload.value = null;
  }
}

async function triggerPublish() {
  if (!activeTaskId.value) return;
  loading.publish = true;
  try {
    const result = await publish(activeTaskId.value, selectedModelId.value || undefined);
    publishPayload.value = result;
    if (typeof result.model_id === "string") {
      selectedModelId.value = result.model_id;
    }
    activeDetailTab.value = "deploy";
    ElMessage.success("模型已发布");
    await loadStatus(activeTaskId.value);
  } catch (error) {
    ElMessage.error(readError(error));
  } finally {
    loading.publish = false;
  }
}

async function triggerInfer() {
  loading.infer = true;
  try {
    inferPayload.value = await infer({
      taskId: activeTaskId.value,
      modelId: selectedModelId.value,
      issueTime: issueTime.value
    });
    activeDetailTab.value = "deploy";
    ElMessage.success("推理完成");
  } catch (error) {
    ElMessage.error(readError(error));
  } finally {
    loading.infer = false;
  }
}

function applyStationDefaults(stationType: StationType) {
  if (stationType === "solar") {
    form.task_id = "frontend_demo_solar";
    form.station_id = "nmg_shuijinghu_solar";
    form.station_name = "内蒙古水晶湖光伏测试场站";
    form.longitude = "109.7";
    form.latitude = "40.30528";
    form.capacity_mw = 100;
    form.train_start = "2025-09-02 00:00:00";
    form.train_end = "2025-09-02 23:45:00";
    form.eval_start = "2025-09-03 00:00:00";
    form.eval_end = "2025-09-03 23:45:00";
    form.feature_set = "ec_hres_solar_n1";
    form.power_path = "";
    form.nwp_root = "/mnt/d/data/netcdf/ecmwf/neimeng";
    modelCandidates.value = ["EC_XGB_PV_V1", "EC_LGB_PV_V1"];
    etlOptionsJson.value = compactJson({
      max_nwp_samples: 192,
      sequence_steps: 9,
      grid_size: 16,
      horizon_codes: ["N1"],
      enable_solar_cleaning: false
    });
  } else {
    form.task_id = "frontend_demo_wind";
    form.station_id = "js_yancheng_h3";
    form.station_name = "江苏盐城 H3 测试风电场";
    form.longitude = "120°36'08.736477\"";
    form.latitude = "034°18'51.964916\"";
    form.capacity_mw = 300;
    form.train_start = "2025-11-02 00:00:00";
    form.train_end = "2025-11-02 23:45:00";
    form.eval_start = "2025-11-03 00:00:00";
    form.eval_end = "2025-11-03 23:45:00";
    form.feature_set = "ec_hres_wind_n1";
    form.power_path = "../测试场站数据/js_yanchengH3_wind/station_power/station_power_original_2502-2601.csv";
    form.nwp_root = "/mnt/d/data/netcdf/ecmwf/jiangsu";
    modelCandidates.value = ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"];
    etlOptionsJson.value = compactJson({
      max_nwp_samples: 192,
      sequence_steps: 9,
      grid_size: 16,
      horizon_codes: ["N1"]
    });
  }
}

function buildTableColumns(rows: Record<string, unknown>[]) {
  const columns = Object.keys(rows[0] ?? {}).slice(0, 18);
  return columns.map((key) => ({ prop: key, label: key, minWidth: key.includes("time") ? 170 : 120 }));
}

function statusTag(status?: string | null) {
  return statusMeta[String(status ?? "")] ?? { text: status ?? "-", type: "info" as const };
}

function checkSummaryText(row: DataCheck) {
  const summary = row.summary_json ?? {};
  if (row.data_type === "dataset") {
    return `${summary.dataset_mode ?? "-"} / train=${summary.train_samples ?? "-"} / eval=${summary.eval_samples ?? "-"}`;
  }
  if (row.data_type === "nwp") {
    const nwp = summary.nwp as Record<string, unknown> | undefined;
    return `${summary.file_count ?? nwp?.file_count ?? "-"} files`;
  }
  return String(summary.check_result ?? row.check_result ?? "-");
}

function metricValue(model: TrainModelArtifact, name: string) {
  const value = model.metrics?.[name];
  return typeof value === "number" ? value : null;
}

function shortModelName(modelId?: string) {
  if (!modelId) return "-";
  const parts = modelId.split("_");
  return parts.length > 4 ? parts.slice(-4, -1).join("_") : modelId.slice(0, 18);
}

function firstTrainedModelId(models?: TrainModelArtifact[]) {
  return models?.find((item) => item.status === "TRAINED")?.model_id ?? "";
}

function openDocs() {
  window.open("/docs", "_blank");
}

function openCreateDrawer() {
  createStep.value = 0;
  createDrawerVisible.value = true;
}

function nextCreateStep() {
  createStep.value = Math.min(createStep.value + 1, 2);
}

function previousCreateStep() {
  createStep.value = Math.max(createStep.value - 1, 0);
}

async function submitIngestFromDrawer() {
  const ok = await submitIngest();
  if (ok) {
    createDrawerVisible.value = false;
    createStep.value = 0;
  }
}

function selectRun(row: RunRow) {
  void loadTaskBundle(row.task_id);
}

function handleNavSelect(key: NavKey) {
  activeSection.value = key;
  if (key === "data") activeDetailTab.value = "data";
  if (key === "train") activeDetailTab.value = "train";
  if (key === "evaluate") activeDetailTab.value = "evaluate";
  if (key === "registry" || key === "inference") activeDetailTab.value = "deploy";
  if (key === "settings") activeDetailTab.value = "settings";
}
</script>

<template>
  <div class="mlops-shell">
    <aside class="nav-rail">
      <div class="brand-lockup">
        <div class="brand-mark">
          <CloudSun :size="22" />
        </div>
        <div>
          <strong>浙江预测 MLOps</strong>
          <span>Online Modeling</span>
        </div>
      </div>

      <nav class="module-nav">
        <button
          v-for="item in sidebarItems"
          :key="item.key"
          type="button"
          :class="{ active: activeSection === item.key }"
          @click="handleNavSelect(item.key)"
        >
          <component :is="item.icon" :size="17" />
          <span>{{ item.label }}</span>
        </button>
      </nav>

      <div class="rail-footer">
        <span class="env-label">Backend</span>
        <strong>{{ serviceLabel }}</strong>
        <el-button size="small" :icon="RefreshCw" :loading="loading.health" @click="checkHealth">检查</el-button>
      </div>
    </aside>

    <section class="platform">
      <header class="command-bar">
        <div class="page-title">
          <span class="eyebrow">Experiment workspace</span>
          <h1>在线建模运行中心</h1>
          <p>围绕 run 管理数据接入、训练、评估、发布和推理。</p>
        </div>
        <div class="command-actions">
          <el-input v-model="activeTaskInput" class="task-search" placeholder="输入 task_id 加载运行">
            <template #prefix><Search :size="16" /></template>
          </el-input>
          <el-button :icon="RefreshCw" :loading="loading.status || loading.preview" @click="loadTaskBundle(activeTaskInput || activeTaskId)">
            加载
          </el-button>
          <el-button :icon="Database" @click="openDocs">API</el-button>
          <el-button type="primary" :icon="Plus" @click="openCreateDrawer">New run</el-button>
        </div>
      </header>

      <main class="workspace-grid">
        <section class="metric-strip" aria-label="运行指标摘要">
          <div v-for="metric in metricStrip" :key="metric.label" class="metric-tile" :data-tone="metric.tone">
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.note }}</small>
          </div>
        </section>

        <section class="runs-panel">
          <div class="panel-heading">
            <div>
              <h2>Runs</h2>
              <p>最近任务、当前运行和操作入口。</p>
            </div>
            <div class="panel-tools">
              <el-input v-model="runSearch" clearable placeholder="过滤 run / 状态 / 场站">
                <template #prefix><Search :size="15" /></template>
              </el-input>
              <el-button :icon="RefreshCw" :loading="loading.status || loading.trainStatus" @click="loadTaskBundle(activeTaskId)" />
            </div>
          </div>

          <el-table
            :data="runRows"
            class="runs-table"
            height="360"
            size="small"
            highlight-current-row
            :default-sort="{ prop: 'updated_time', order: 'descending' }"
            @row-click="selectRun"
          >
            <el-table-column prop="task_id" label="Run" min-width="220" show-overflow-tooltip>
              <template #default="{ row }">
                <div class="run-name">
                  <GitBranch :size="15" />
                  <span class="mono">{{ row.task_id }}</span>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="target" label="对象" min-width="170" show-overflow-tooltip />
            <el-table-column prop="station_type" label="类型" width="82" />
            <el-table-column prop="status" label="状态" width="110">
              <template #default="{ row }">
                <el-tag size="small" :type="statusTag(row.status).type">{{ statusTag(row.status).text }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="train_samples" label="Train" width="90" align="right" />
            <el-table-column prop="eval_samples" label="Eval" width="90" align="right" />
            <el-table-column prop="models" label="Models" width="90" align="center" />
            <el-table-column prop="accuracy" label="Accuracy" width="118" />
            <el-table-column prop="progress" label="进度" width="120">
              <template #default="{ row }">
                <el-progress :percentage="row.progress" :show-text="false" :stroke-width="6" />
              </template>
            </el-table-column>
            <el-table-column label="动作" width="260" fixed="right">
              <template #default="{ row }">
                <div class="table-actions">
                  <el-button size="small" :icon="RefreshCw" @click.stop="loadTaskBundle(row.task_id)">加载</el-button>
                  <el-button size="small" :icon="Play" :loading="loading.train" @click.stop="triggerTrain">训练</el-button>
                  <el-button size="small" :icon="CheckCircle2" :loading="loading.evaluate" @click.stop="triggerEvaluate">评估</el-button>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </section>

        <aside class="run-inspector">
          <div class="inspector-head">
            <div>
              <span class="eyebrow">Selected run</span>
              <h2 class="mono">{{ selectedRun?.task_id || "-" }}</h2>
            </div>
            <el-tag :type="statusTag(currentStatus).type">{{ statusTag(currentStatus).text }}</el-tag>
          </div>

          <div class="run-meta">
            <div>
              <span>对象</span>
              <strong>{{ targetLabel }}</strong>
            </div>
            <div>
              <span>训练窗口</span>
              <strong>{{ trainWindowText }}</strong>
            </div>
            <div>
              <span>评估窗口</span>
              <strong>{{ evalWindowText }}</strong>
            </div>
          </div>

          <div class="lifecycle">
            <div v-for="item in lifecycleCards" :key="item.title" class="lifecycle-item" :data-state="item.state">
              <span class="dot" />
              <div>
                <strong>{{ item.title }}</strong>
                <small>{{ item.desc }}</small>
              </div>
            </div>
          </div>

          <div class="quick-actions">
            <el-switch v-model="syncTrain" active-text="同步训练" inactive-text="异步训练" />
            <el-button type="primary" :icon="Play" :loading="loading.train" @click="triggerTrain">启动训练</el-button>
            <el-button :icon="RefreshCw" :loading="loading.trainStatus" @click="refreshTrainStatus(activeTaskId)">刷新训练</el-button>
            <el-button :disabled="!trainResult?.job_id" @click="stopTrain">取消</el-button>
          </div>
        </aside>

        <section class="insight-panel data-preview-panel">
          <div class="panel-heading compact">
            <div>
              <h2>Data preview</h2>
              <p>按 train/eval 查看已对齐样本曲线。</p>
            </div>
            <el-radio-group v-model="selectedDataType" size="small" @change="loadPreview(activeTaskId)">
              <el-radio-button label="train">train</el-radio-button>
              <el-radio-button label="eval">eval</el-radio-button>
            </el-radio-group>
          </div>
          <VChart class="chart" :option="previewChartOption" autoresize />
        </section>

        <section class="insight-panel model-panel">
          <div class="panel-heading compact">
            <div>
              <h2>Model comparison</h2>
              <p>训练产物的核心指标对比。</p>
            </div>
            <el-button size="small" :icon="RefreshCw" :loading="loading.trainStatus" @click="refreshTrainStatus(activeTaskId)" />
          </div>
          <VChart class="chart" :option="modelMetricChartOption" autoresize />
        </section>

        <section class="detail-panel">
          <el-tabs v-model="activeDetailTab" class="detail-tabs">
            <el-tab-pane name="data">
              <template #label><Activity :size="15" /> Data</template>
              <div class="detail-grid">
                <div class="detail-block wide">
                  <div class="block-title">
                    <span>数据检查</span>
                    <el-button size="small" :icon="RefreshCw" :loading="loading.status" @click="loadStatus(activeTaskId)" />
                  </div>
                  <el-table :data="latestChecks" size="small" height="260">
                    <el-table-column v-for="col in dataCheckColumns" :key="col.prop" v-bind="col">
                      <template #default="{ row }">
                        <span v-if="col.prop === 'missing_rate'">{{ row.missing_rate == null ? "-" : formatPercent(row.missing_rate) }}</span>
                        <el-tag v-else-if="col.prop === 'check_result'" :type="row.check_result === 'PASS' ? 'success' : 'warning'" size="small">
                          {{ row.check_result }}
                        </el-tag>
                        <span v-else>{{ row[col.prop] || "-" }}</span>
                      </template>
                    </el-table-column>
                    <el-table-column label="摘要" min-width="220">
                      <template #default="{ row }">{{ checkSummaryText(row) }}</template>
                    </el-table-column>
                  </el-table>
                </div>

                <div class="detail-block">
                  <div class="block-title"><span>点位审计</span></div>
                  <el-form label-position="top" class="compact-form">
                    <el-form-item label="时间">
                      <el-input v-model="pointEdit.time" />
                    </el-form-item>
                    <el-form-item label="字段">
                      <el-input v-model="pointEdit.field" />
                    </el-form-item>
                    <el-form-item label="值">
                      <el-input-number v-model="pointEdit.value" class="full-width" />
                    </el-form-item>
                    <el-form-item label="原因">
                      <el-input v-model="pointEdit.reason" />
                    </el-form-item>
                    <el-button type="primary" :icon="Save" :loading="loading.edit" @click="submitPointEdit">保存记录</el-button>
                  </el-form>
                </div>

                <div class="detail-block wide">
                  <div class="block-title"><span>样本表</span></div>
                  <el-table :data="previewRows" size="small" height="280">
                    <el-table-column v-for="col in previewColumns" :key="col.prop" v-bind="col">
                      <template #default="{ row }">{{ typeof row[col.prop] === "number" ? formatNumber(row[col.prop]) : row[col.prop] }}</template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane name="train">
              <template #label><Gauge :size="15" /> Training</template>
              <div class="train-toolbar">
                <el-switch v-model="syncTrain" active-text="同步" inactive-text="异步" />
                <el-button type="primary" :icon="Play" :loading="loading.train" @click="triggerTrain">启动训练</el-button>
                <el-button :icon="RefreshCw" :loading="loading.trainStatus" @click="refreshTrainStatus(activeTaskId)">刷新状态</el-button>
                <el-button :disabled="!trainResult?.job_id" @click="stopTrain">取消</el-button>
              </div>
              <el-progress :percentage="progressPercent" :status="trainResult?.success ? 'success' : undefined" />
              <div class="detail-grid train-detail-grid">
                <div class="detail-block wide">
                  <div class="block-title"><span>模型产物</span></div>
                  <el-table :data="trainedModels" size="small" height="320">
                    <el-table-column label="模型" min-width="280">
                      <template #default="{ row }"><span class="mono">{{ row.model_id }}</span></template>
                    </el-table-column>
                    <el-table-column prop="model_type" label="类型" width="120" />
                    <el-table-column label="状态" width="100">
                      <template #default="{ row }">
                        <el-tag :type="row.status === 'TRAINED' ? 'success' : 'warning'" size="small">{{ row.status }}</el-tag>
                      </template>
                    </el-table-column>
                    <el-table-column label="accuracy" width="120">
                      <template #default="{ row }">{{ formatNumber(row.metrics?.accuracy) }}</template>
                    </el-table-column>
                    <el-table-column label="MAE" width="100">
                      <template #default="{ row }">{{ formatNumber(row.metrics?.mae) }}</template>
                    </el-table-column>
                    <el-table-column prop="artifact_path" label="产物路径" min-width="260" show-overflow-tooltip />
                  </el-table>
                </div>
                <div class="detail-block">
                  <div class="block-title"><span>训练日志</span></div>
                  <div class="log-list">
                    <div v-for="log in trainResult?.latest_logs ?? []" :key="`${log.log_time}-${log.message}`">
                      <span>{{ formatDateTime(log.log_time) }}</span>
                      <strong>{{ log.log_level }}</strong>
                      <p>{{ log.message }}</p>
                    </div>
                  </div>
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane name="evaluate">
              <template #label><BarChart3 :size="15" /> Evaluation</template>
              <div class="train-toolbar">
                <el-switch v-model="syncEvaluate" active-text="同步" inactive-text="异步" />
                <el-button type="primary" :icon="CheckCircle2" :loading="loading.evaluate" @click="triggerEvaluate">执行评估</el-button>
                <el-button :icon="RefreshCw" @click="loadEvaluateResult(activeTaskId)">获取结果</el-button>
              </div>
              <div class="detail-grid">
                <div class="detail-block">
                  <div class="block-title"><span>日准确率</span></div>
                  <VChart class="small-chart" :option="dailyAccuracyChartOption" autoresize />
                </div>
                <div class="detail-block wide">
                  <div class="block-title"><span>评估明细</span></div>
                  <el-table :data="evaluationDays" size="small" height="300">
                    <el-table-column prop="date" label="日期" min-width="170" />
                    <el-table-column label="准确率" width="120">
                      <template #default="{ row }">{{ formatPercent(row.accuracy) }}</template>
                    </el-table-column>
                    <el-table-column label="MAE" width="110">
                      <template #default="{ row }">{{ formatNumber(row.mae) }}</template>
                    </el-table-column>
                    <el-table-column label="样本" width="90">
                      <template #default="{ row }">{{ formatNumber(row.n, 0) }}</template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane name="deploy">
              <template #label><Rocket :size="15" /> Deploy</template>
              <div class="deploy-toolbar">
                <el-select v-model="selectedModelId" filterable clearable placeholder="默认发布最优模型">
                  <el-option v-for="model in selectedModelOptions" :key="model.model_id" :label="model.model_id" :value="model.model_id" />
                </el-select>
                <el-button type="primary" :icon="Rocket" :loading="loading.publish" @click="triggerPublish">发布</el-button>
                <el-date-picker v-model="issueTime" type="datetime" value-format="YYYY-MM-DD HH:mm:ss" />
                <el-button type="primary" plain :icon="Zap" :loading="loading.infer" @click="triggerInfer">推理</el-button>
              </div>
              <div class="detail-grid">
                <div class="detail-block">
                  <div class="block-title"><span>推理曲线</span></div>
                  <VChart class="small-chart" :option="inferChartOption" autoresize />
                </div>
                <div class="detail-block">
                  <div class="block-title"><span>发布结果</span></div>
                  <pre class="json-preview result-json">{{ compactJson(publishPayload ?? {}) }}</pre>
                </div>
                <div class="detail-block wide">
                  <div class="block-title"><span>推理结果</span></div>
                  <el-table :data="inferPayload?.predictions ?? []" size="small" height="280">
                    <el-table-column label="时间" min-width="180">
                      <template #default="{ row }">{{ row.valid_time ?? row.time }}</template>
                    </el-table-column>
                    <el-table-column label="预测 MW" min-width="120">
                      <template #default="{ row }">{{ formatNumber(row.p_pred_mw ?? row.p_pred) }}</template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>
            </el-tab-pane>

            <el-tab-pane name="settings">
              <template #label><Settings :size="15" /> Settings</template>
              <div class="detail-grid">
                <div class="detail-block">
                  <div class="block-title"><span>API 状态</span></div>
                  <dl class="settings-list">
                    <dt>Status</dt>
                    <dd>{{ healthPayload?.status ?? "-" }}</dd>
                    <dt>DB</dt>
                    <dd>{{ healthPayload?.db_path ?? "-" }}</dd>
                    <dt>NWP root</dt>
                    <dd>{{ healthPayload?.nwp_root ?? "-" }}</dd>
                  </dl>
                </div>
                <div class="detail-block wide">
                  <div class="block-title"><span>当前 ingest 请求预览</span></div>
                  <pre class="json-preview">{{ payloadPreview }}</pre>
                </div>
              </div>
            </el-tab-pane>
          </el-tabs>
        </section>
      </main>
    </section>

    <el-drawer v-model="createDrawerVisible" title="New online modeling run" size="min(720px, 94vw)" class="create-drawer">
      <div class="drawer-body">
        <el-steps :active="createStep" finish-status="success" process-status="process" simple>
          <el-step title="Scope" />
          <el-step title="Data" />
          <el-step title="Models" />
        </el-steps>

        <el-form v-if="createStep === 0" label-position="top" class="drawer-form">
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item label="对象类型">
                <el-segmented
                  v-model="form.object_type"
                  :options="[
                    { label: '单站', value: 'station' },
                    { label: '区域', value: 'region' }
                  ]"
                  block
                />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="能源类型">
                <el-segmented
                  v-model="form.station_type"
                  :options="[
                    { label: '风电', value: 'wind' },
                    { label: '光伏', value: 'solar' }
                  ]"
                  block
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-form-item label="任务 ID">
            <el-input v-model="form.task_id" />
          </el-form-item>
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item label="场站 ID">
                <el-input v-model="form.station_id" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="区域 ID">
                <el-input v-model="form.region_id" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-form-item label="场站名称">
            <el-input v-model="form.station_name" />
          </el-form-item>
          <el-row :gutter="12">
            <el-col :span="8">
              <el-form-item label="经度">
                <el-input v-model="form.longitude" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="纬度">
                <el-input v-model="form.latitude" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="容量 MW">
                <el-input-number v-model="form.capacity_mw" :min="0" class="full-width" />
              </el-form-item>
            </el-col>
          </el-row>
        </el-form>

        <el-form v-else-if="createStep === 1" label-position="top" class="drawer-form">
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item label="训练开始">
                <el-date-picker v-model="form.train_start" type="datetime" value-format="YYYY-MM-DD HH:mm:ss" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="训练结束">
                <el-date-picker v-model="form.train_end" type="datetime" value-format="YYYY-MM-DD HH:mm:ss" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item label="评估开始">
                <el-date-picker v-model="form.eval_start" type="datetime" value-format="YYYY-MM-DD HH:mm:ss" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="评估结束">
                <el-date-picker v-model="form.eval_end" type="datetime" value-format="YYYY-MM-DD HH:mm:ss" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-form-item label="实发文件路径">
            <el-input v-model="form.power_path" />
          </el-form-item>
          <el-form-item label="NWP 根目录">
            <el-input v-model="form.nwp_root" />
          </el-form-item>
          <el-form-item label="powerData">
            <el-input v-model="powerDataJson" type="textarea" :rows="5" resize="vertical" />
          </el-form-item>
          <el-form-item label="run_etl">
            <el-switch v-model="form.run_etl" />
          </el-form-item>
        </el-form>

        <el-form v-else label-position="top" class="drawer-form">
          <el-form-item label="模型候选">
            <el-select v-model="modelCandidates" multiple filterable :loading="loading.models" class="full-width">
              <el-option v-for="model in modelOptions" :key="model.model_name" :label="model.model_name" :value="model.model_name" />
            </el-select>
          </el-form-item>
          <el-form-item label="特征集">
            <el-input v-model="form.feature_set" />
          </el-form-item>
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item label="etl_options">
                <el-input v-model="etlOptionsJson" type="textarea" :rows="9" resize="vertical" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="train_options">
                <el-input v-model="trainOptionsJson" type="textarea" :rows="9" resize="vertical" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-collapse>
            <el-collapse-item name="payload">
              <template #title><ClipboardList :size="16" /> 请求预览</template>
              <pre class="json-preview">{{ payloadPreview }}</pre>
            </el-collapse-item>
          </el-collapse>
        </el-form>
      </div>

      <template #footer>
        <div class="drawer-footer">
          <el-button @click="createDrawerVisible = false">取消</el-button>
          <el-button :disabled="createStep === 0" @click="previousCreateStep">上一步</el-button>
          <el-button v-if="createStep < 2" type="primary" @click="nextCreateStep">下一步</el-button>
          <el-button v-else type="primary" :icon="Send" :loading="loading.ingest" @click="submitIngestFromDrawer">提交 ingest</el-button>
        </div>
      </template>
    </el-drawer>
  </div>
</template>
