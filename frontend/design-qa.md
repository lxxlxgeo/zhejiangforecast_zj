# Design QA

final result: passed

Reference pattern: run-centric MLOps consoles such as MLflow Tracking, Weights & Biases runs, ClearML WebApp, Kubeflow Pipelines runs, and SageMaker Experiments. The redesign moves task creation into a drawer wizard and keeps the first screen focused on navigation, run tables, selected-run status, metrics, charts, and operational actions.

Checked:

- Desktop viewport `1440x980`: sidebar navigation, command bar, run table, selected-run inspector, metric tiles, and chart panels render without page-level horizontal overflow.
- Drawer workflow: `New run` opens a three-step `Scope / Data / Models` wizard; the final step exposes request preview and `提交 ingest` without flattening the form on the homepage.
- Mobile viewport `390x900`: layout stacks into a single column, top module navigation remains horizontally scrollable with hidden scrollbar, and no page-level horizontal overflow is present.
- Frozen backend contracts: `ingest`, `train`, and `evaluate` are only consumed by the frontend; no backend request or response shape was changed.
- Build gate: `vue-tsc --noEmit` and Vite production build pass. The system `npm` entry is broken on this machine, so validation used the local project binaries with the Codex bundled Node runtime.

Notes:

- If the backend is not running, initial API/model discovery is silent; manual refresh buttons surface errors.
- Bundle-size warning is expected from Element Plus and ECharts and can be optimized later with route/chunk splitting.
