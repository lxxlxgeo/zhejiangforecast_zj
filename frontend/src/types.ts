export type ObjectType = "station" | "region";
export type StationType = "wind" | "solar";
export type TrainMode = "local" | "airflow";

export interface ApiEnvelope<T> {
  code: number;
  message: string;
  data: T;
}

export interface HealthPayload {
  status: string;
  config_path?: string | null;
  project_root?: string;
  db_path?: string;
  database_url?: string;
  nwp_root?: string | null;
  nwp_roots?: Record<string, string>;
  nwp_workers?: number;
  nwp_parallel_backend?: string;
}

export interface ModelSpec {
  model_name: string;
  station_type: StationType | string;
  model_family: string;
  feature_set: string;
  required_features: string[];
  description: string;
}

export interface StationPayload {
  station_name?: string;
  longitude?: number | string;
  latitude?: number | string;
  capacity_mw?: number;
}

export interface PowerDataPoint {
  dataTime?: string;
  utcTime?: string;
  actualPower?: number;
  theoryPower?: number | null;
  windSpeed?: number | null;
  directIrradiance?: number | null;
  actualIrradiance?: number | null;
  irradiance?: number | null;
}

export interface IngestRequest {
  task_id?: string;
  station_id?: string;
  region_id?: string;
  object_type: ObjectType;
  station_type: StationType;
  train_start?: string;
  train_end?: string;
  eval_start?: string;
  eval_end?: string;
  model_candidates?: string[];
  feature_set?: string;
  station?: StationPayload;
  data_paths?: {
    power?: string | null;
    power_path?: string | null;
    station_info?: string | null;
    nwp_root?: string | null;
  };
  powerData?: PowerDataPoint[];
  etl_options?: Record<string, unknown>;
  train_options?: Record<string, unknown>;
  run_etl?: boolean;
}

export interface IngestPayload {
  task_id: string;
  object_type: ObjectType;
  station_type: StationType;
  station_id?: string | null;
  region_id?: string | null;
  status: TaskStatus;
  train_start?: string | null;
  train_end?: string | null;
  eval_start?: string | null;
  eval_end?: string | null;
  feature_set?: string | null;
  model_candidates?: string[];
  request_json?: {
    station?: StationPayload;
    artifacts?: Record<string, string>;
    data_summary?: DataSummary;
    powerData_total_size?: number | null;
  };
  work_dir?: string;
  config_path?: string;
  error_message?: string | null;
  published_model_id?: string | null;
  created_time?: string;
  updated_time?: string;
}

export type TaskStatus =
  | "CREATED"
  | "DATA_READY"
  | "CLEANED"
  | "TRAINING"
  | "TRAINED"
  | "EVALUATED"
  | "PUBLISHED"
  | "FAILED"
  | "CANCELED";

export interface DataSummary {
  check_result?: string;
  dataset_mode?: string;
  aligned_samples?: number;
  train_samples?: number;
  eval_samples?: number;
  missing_issue_count?: number;
  failed_rows?: number;
  feature_count?: number;
  feature_contract?: string;
  nwp_error?: string | null;
  capacity_mw?: number;
  start_time?: string;
  end_time?: string;
  nwp?: {
    nwp_root?: string;
    file_count?: number;
    first_issue?: string;
    last_issue?: string;
  };
  [key: string]: unknown;
}

export interface DataCheck {
  id: number;
  task_id: string;
  data_type: string;
  missing_rate?: number | null;
  start_time?: string | null;
  end_time?: string | null;
  check_result: string;
  summary_json?: DataSummary;
  created_time: string;
}

export interface DataStatusPayload {
  task_id: string;
  status: TaskStatus;
  data_checks: DataCheck[];
}

export interface DataPreviewPayload {
  task_id: string;
  data_type: string;
  rows: Record<string, unknown>[];
}

export interface PointEdit {
  time: string;
  field: string;
  value?: number | null;
  reason?: string;
}

export interface DataEditPayload {
  task_id: string;
  saved: number;
  total_edits: number;
  edit_path: string;
  note?: string;
}

export interface TrainRequest {
  task_id: string;
  model_name?: string;
  model_candidates?: string[];
  train_mode: TrainMode;
  sync: boolean;
}

export interface TrainModelArtifact {
  model_id: string;
  model_type?: string;
  status: "TRAINED" | "SKIPPED" | string;
  artifact_path?: string;
  version?: string;
  metrics?: Record<string, number>;
  created_time?: string;
}

export interface TrainStatusPayload {
  task_id: string;
  job_id?: string | null;
  status: string;
  job_status?: string | null;
  task_status?: TaskStatus | string | null;
  stage?: string;
  progress?: number;
  done?: boolean;
  success?: boolean;
  error_message?: string | null;
  is_async?: boolean;
  runner?: string;
  model_candidates?: string[];
  model_count?: number;
  trained_model_count?: number;
  skipped_model_count?: number;
  models?: TrainModelArtifact[];
  train_result?: {
    task_id?: string;
    models?: Array<{
      candidate?: string;
      model_id?: string;
      status?: string;
      metrics?: Record<string, number>;
      prediction_path?: string;
      reason?: string;
    }>;
  } | null;
  latest_logs?: Array<{
    id?: number;
    stage?: string;
    log_level?: string;
    message?: string;
    log_time?: string;
  }>;
  status_url?: string;
  task_status_url?: string;
  work_dir?: string;
  created_time?: string;
  updated_time?: string;
}

export interface EvaluatePayload {
  task_id: string;
  daily_accuracy: Array<{
    date: string;
    mae?: number;
    accuracy?: number;
    n?: number;
  }>;
}

export interface PublishPayload {
  task_id?: string;
  model_id?: string;
  version?: string;
  artifact_path?: string;
  published_dir?: string;
  model_card?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface InferPayload {
  infer_id: string;
  task_id?: string | null;
  model_id?: string | null;
  issue_time?: string | null;
  predictions: Array<{
    valid_time?: string;
    time?: string;
    p_pred_mw?: number;
    p_pred?: number;
    [key: string]: unknown;
  }>;
}
