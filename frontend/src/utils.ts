import dayjs from "dayjs";

export function formatDateTime(value?: string | null) {
  return value ? dayjs(value).format("YYYY-MM-DD HH:mm:ss") : "-";
}

export function formatNumber(value: unknown, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: Number.isInteger(number) ? 0 : Math.min(digits, 2)
  });
}

export function formatPercent(value: unknown, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${(number <= 1 ? number * 100 : number).toFixed(digits)}%`;
}

export function toNumber(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : Number.NaN;
}

export function readError(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

export function compactJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

export function parseJsonObject<T>(value: string, fallback: T): T {
  const trimmed = value.trim();
  if (!trimmed) return fallback;
  return JSON.parse(trimmed) as T;
}

export function detectTimeField(row: Record<string, unknown>) {
  return ["time", "time_bj", "dataTime", "valid_time"].find((key) => key in row) ?? "";
}

export function detectPowerField(row: Record<string, unknown>) {
  return ["power_mw", "actualPower", "p_real", "p_pred_mw", "p_pred"].find((key) => key in row) ?? "";
}
