/**
 * Typed client for the AgentDS backend (FastAPI, `agentds/backend`).
 *
 * One exported function per endpoint, sharing a base URL + fetch helper.
 * Later phases extend this file with the per-module trigger/report
 * endpoints (`/clean`, `/visualize`, `/recommend`, `/train`, `/explain`,
 * `/report`, ...) — keep that same shape when adding to it.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** Real pipeline stage kinds the backend's `/artifacts` endpoint can report
 * on today. The backend's own `UPSTREAM_KINDS` list also carries an 8th
 * entry, `"whatif"`, reserved for a future Module 8 that is NOT built yet —
 * it must never surface in the UI. Everything in this file that touches
 * artifact kinds works from this filtered list, not the raw response. */
export const PIPELINE_KINDS = [
  "understanding",
  "cleaning",
  "visualization",
  "recommendation",
  "training",
  "explainability",
] as const;

export type PipelineKind = (typeof PIPELINE_KINDS)[number];

export interface DatasetListEntry {
  dataset_id: string;
  filename: string;
  uploaded_at: string | null;
}

export interface UploadResponse {
  dataset_id: string;
  filename: string;
}

export interface ArtifactsStatusRaw {
  present: string[];
  missing: string[];
}

export interface ArtifactsStatus {
  present: PipelineKind[];
  missing: PipelineKind[];
}

/** Thrown for any non-2xx response. `detail` carries the backend's own
 * error message (FastAPI's `{"detail": "..."}` body) so callers can show it
 * verbatim instead of inventing a nicer one. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function extractDetail(res: Response): Promise<string> {
  try {
    const body = await res.clone().json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    // body wasn't JSON — fall through to statusText.
  }
  return res.statusText || `Request failed with status ${res.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as T;
}

/** `GET /datasets` — every stored dataset, newest first (already sorted
 * server-side). */
export async function listDatasets(): Promise<DatasetListEntry[]> {
  return request<DatasetListEntry[]>("/datasets");
}

/** `POST /datasets/upload` — multipart, field name `file`, `.csv` only. */
export async function uploadDataset(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<UploadResponse>("/datasets/upload", {
    method: "POST",
    body: formData,
  });
}

/** `GET /datasets/{id}/artifacts` — which of the 7 real pipeline sidecars
 * exist for a dataset. Filters the backend's reserved `"whatif"` entry out
 * of both `present` and `missing` before returning. */
export async function getArtifactsStatus(
  datasetId: string,
): Promise<ArtifactsStatus> {
  const raw = await request<ArtifactsStatusRaw>(
    `/datasets/${encodeURIComponent(datasetId)}/artifacts`,
  );
  const isPipelineKind = (k: string): k is PipelineKind =>
    (PIPELINE_KINDS as readonly string[]).includes(k);
  return {
    present: raw.present.filter(isPipelineKind),
    missing: raw.missing.filter(isPipelineKind),
  };
}

// ---------------------------------------------------------------------------
// Module 1 — Data Understanding
// ---------------------------------------------------------------------------

export interface ColumnOverview {
  name: string;
  dtype: string;
  missing_count: number;
  missing_pct: number;
  n_unique: number;
}

export interface DuplicatesReport {
  n_duplicate_rows: number;
  percent: number;
  example_rows: Record<string, unknown>[];
}

export interface CorrelationPair {
  column_a: string;
  column_b: string;
  correlation: number;
}

export type ProblemType = "classification" | "regression" | "unclear";

export interface DataUnderstandingReport {
  dataset_id: string;
  n_rows: number;
  n_columns: number;
  columns: ColumnOverview[];
  duplicates: DuplicatesReport | null;
  correlations: CorrelationPair[] | null;
  target_candidate: string | null;
  problem_type: ProblemType;
  reasoning: string;
  narrative: string;
  key_findings: string[];
}

/** `POST /datasets/{id}/quick-stats` — deterministic, LLM-free, no API key
 * required. Idempotent and cheap, so it's fine to call on every page load. */
export async function getQuickStats(
  datasetId: string,
): Promise<DataUnderstandingReport> {
  return request<DataUnderstandingReport>(
    `/datasets/${encodeURIComponent(datasetId)}/quick-stats`,
    { method: "POST" },
  );
}

// ---------------------------------------------------------------------------
// Module 2 — Cleaning
// ---------------------------------------------------------------------------

export interface CleaningStep {
  order: number;
  tool: string;
  reason: string;
  params: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface CleaningReport {
  dataset_id: string;
  cleaned_dataset_id: string | null;
  target_candidate: string | null;
  problem_type: ProblemType;
  initial_shape: [number, number];
  final_shape: [number, number];
  initial_missing_cells: number;
  final_missing_cells: number;
  steps: CleaningStep[];
  summary: string[];
  remaining_issues: string[];
}

/** `POST /datasets/{id}/clean` — agentic, requires `ANTHROPIC_API_KEY`
 * server-side (503 if unset). Not idempotent/cheap: creates a new derived
 * dataset each run, so callers must gate this behind an explicit action,
 * never auto-trigger it on page load. */
export async function runClean(datasetId: string): Promise<CleaningReport> {
  return request<CleaningReport>(
    `/datasets/${encodeURIComponent(datasetId)}/clean`,
    { method: "POST" },
  );
}

// ---------------------------------------------------------------------------
// Module 3 — Visualization
// ---------------------------------------------------------------------------

export interface ChartSpec {
  chart_type: string;
  title: string;
  columns: string[];
  plotly: Record<string, unknown>;
  insight: string;
}

export interface VisualizationReport {
  dataset_id: string;
  source: "original" | "cleaned";
  target_column: string | null;
  problem_type: ProblemType;
  n_charts: number;
  charts: ChartSpec[];
  skipped: string[];
  narrative: string;
}

/** `POST /datasets/{id}/visualize` — requires `ANTHROPIC_API_KEY` server-side
 * (503 if unset). Not idempotent/cheap (LLM narration call) — gate behind an
 * explicit action. */
export async function runVisualize(
  datasetId: string,
): Promise<VisualizationReport> {
  return request<VisualizationReport>(
    `/datasets/${encodeURIComponent(datasetId)}/visualize`,
    { method: "POST" },
  );
}

/** `GET /datasets/{id}/visualize` — a previously-computed result, or `null`
 * when the stage has never been run (backend 404s). Lets the page show a
 * cached result on load instead of forcing a fresh recompute. */
export async function getVisualization(
  datasetId: string,
): Promise<VisualizationReport | null> {
  const res = await fetch(
    `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/visualize`,
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as VisualizationReport;
}

// ---------------------------------------------------------------------------
// Module 4 — Model Recommendation
// ---------------------------------------------------------------------------

export interface ModelCandidate {
  name: string;
  library: "sklearn" | "xgboost" | "lightgbm";
  rationale: string;
  hyperparameters: Record<string, unknown>;
}

export interface RecommendationReport {
  dataset_id: string | null;
  source_dataset_id: string | null;
  used_cleaned_dataset: boolean;
  problem_type: "classification" | "regression";
  modeling_profile: Record<string, unknown>;
  candidates: ModelCandidate[];
  preprocessing_recommendations: string[];
  primary_metric: string;
  reasoning: string;
}

/** `POST /datasets/{id}/recommend` — requires `ANTHROPIC_API_KEY` server-side
 * (503 if unset). Not idempotent/cheap (one LLM reasoning call) — gate
 * behind an explicit action. */
export async function runRecommend(
  datasetId: string,
): Promise<RecommendationReport> {
  return request<RecommendationReport>(
    `/datasets/${encodeURIComponent(datasetId)}/recommend`,
    { method: "POST" },
  );
}

/** `GET /datasets/{id}/recommend` — a previously-computed result, or `null`
 * when the stage has never been run (backend 404s). Lets the page show a
 * cached result on load instead of forcing a fresh recompute. */
export async function getRecommendation(
  datasetId: string,
): Promise<RecommendationReport | null> {
  const res = await fetch(
    `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/recommend`,
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as RecommendationReport;
}

/** `GET /datasets/{id}/report` — no dedicated status field exists for the
 * Report stage on `/artifacts` (it isn't one of the upstream module kinds),
 * so presence is determined the same way the report page itself will:
 * a 404 means "not built yet", a 200 means it's cached and ready to view. */
export async function getReportBuilt(datasetId: string): Promise<boolean> {
  const res = await fetch(
    `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/report`,
  );
  if (res.status === 404) return false;
  if (!res.ok) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Module 5 — Training & Evaluation
// ---------------------------------------------------------------------------

export interface ColumnRoles {
  numeric: string[];
  categorical: string[];
  dropped_high_cardinality: string[];
  dropped_datetime: string[];
}

export interface MetricStat {
  mean: number;
  std: number;
}

export interface CandidateResult {
  name: string;
  estimator_class: string;
  params: Record<string, unknown>;
  cv_metrics: Record<string, MetricStat>;
  test_metrics: Record<string, number>;
  fit_seconds: number;
  model_path: string | null;
  error: string | null;
}

export interface LeaderboardRow {
  rank: number;
  name: string;
  primary_metric: string;
  cv_mean: number | null;
  cv_std: number | null;
  test_score: number | null;
  failed: boolean;
}

export interface TrainingReport {
  dataset_id: string;
  problem_type: "classification" | "regression";
  target: string;
  n_rows: number;
  feature_columns: string[];
  column_roles: ColumnRoles;
  train_rows: number;
  test_rows: number;
  test_size: number;
  random_state: number;
  cv_splits: number;
  stratified: boolean;
  primary_metric: string;
  candidates: CandidateResult[];
  best_model: string | null;
  best_score: number | null;
  leaderboard: LeaderboardRow[];
  narrative: string;
  warnings: string[];
}

/** `POST /datasets/{id}/train` — deterministic AutoML sweep over the fixed
 * catalog. Unlike every other agentic module, this one does NOT require
 * `ANTHROPIC_API_KEY`: with no key, `narrative` just comes back as `""`
 * rather than a 503. Not idempotent/cheap (refits every candidate each
 * run) — gate behind an explicit action. */
export async function runTrain(datasetId: string): Promise<TrainingReport> {
  return request<TrainingReport>(
    `/datasets/${encodeURIComponent(datasetId)}/train`,
    { method: "POST" },
  );
}

/** `GET /datasets/{id}/train` — a previously-computed result, or `null` when
 * the stage has never been run (backend 404s). Lets the page show a cached
 * result on load instead of forcing a fresh recompute. */
export async function getTraining(
  datasetId: string,
): Promise<TrainingReport | null> {
  const res = await fetch(
    `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/train`,
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as TrainingReport;
}

// ---------------------------------------------------------------------------
// Module 6 — Explainability
// ---------------------------------------------------------------------------

export interface FeatureImportance {
  feature: string;
  mean_abs_shap: number;
}

export interface ShapReason {
  feature: string;
  shap_value: number;
  feature_value: number;
}

export interface SamplePrediction {
  row_index: number;
  predicted_value: number | string;
  top_reasons: ShapReason[];
}

export interface ExplainabilityReport {
  dataset_id: string;
  model_name: string;
  problem_type: "classification" | "regression";
  target: string;
  explainer_type: "tree" | "linear" | "kernel";
  n_rows_explained: number;
  feature_importance: FeatureImportance[];
  sample_explanations: SamplePrediction[];
  narrative: string;
  warnings: string[];
}

/** `POST /datasets/{id}/explain` — requires `ANTHROPIC_API_KEY` server-side
 * (503 if unset). Also requires a training report to already exist (422 if
 * not, or if the recorded `best_model` failed to fit) — surface that
 * message rather than inventing one. Not idempotent/cheap — gate behind an
 * explicit action. */
export async function runExplain(
  datasetId: string,
): Promise<ExplainabilityReport> {
  return request<ExplainabilityReport>(
    `/datasets/${encodeURIComponent(datasetId)}/explain`,
    { method: "POST" },
  );
}

/** `GET /datasets/{id}/explain` — a previously-computed result, or `null`
 * when the stage has never been run (backend 404s). Lets the page show a
 * cached result on load instead of forcing a fresh recompute. */
export async function getExplainability(
  datasetId: string,
): Promise<ExplainabilityReport | null> {
  const res = await fetch(
    `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/explain`,
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as ExplainabilityReport;
}

// ---------------------------------------------------------------------------
// Module 7 — Final Report
// ---------------------------------------------------------------------------

export interface ReportSection {
  key: string;
  title: string;
  body_markdown: string;
  source_module: string;
  status: "ok" | "not_run";
}

export interface FinalReport {
  dataset_id: string;
  generated_at: string;
  source_dataset_id: string;
  modules_present: string[];
  modules_missing: string[];
  sections: ReportSection[];
  executive_summary: string;
  markdown: string;
}

/** `POST /datasets/{id}/report` — assembles + caches the final report.
 * Requires `ANTHROPIC_API_KEY` server-side (503 if unset; the deterministic
 * section bodies are ready either way, but the prose layer needs the key).
 * Not idempotent/cheap — gate behind an explicit action. */
export async function runReport(datasetId: string): Promise<FinalReport> {
  return request<FinalReport>(
    `/datasets/${encodeURIComponent(datasetId)}/report`,
    { method: "POST" },
  );
}

/** `GET /datasets/{id}/report` — view a previously-built report. No key
 * required. Returns `null` (not a thrown error) when none has been built
 * yet (backend 404s with a "POST to build one" message) so callers can
 * render a "build the report" prompt instead of an error state. */
export async function getReport(
  datasetId: string,
): Promise<FinalReport | null> {
  const res = await fetch(
    `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/report`,
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as FinalReport;
}

/** Direct backend URL for the PDF export of a built report — plain `<a
 * href>` target, the backend sets `Content-Disposition` so no client-side
 * fetch+blob is needed. 404s if no report has been built yet. */
export function reportPdfUrl(datasetId: string): string {
  return `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/report/pdf`;
}

/** Direct backend URL for the DOCX export of a built report. See
 * `reportPdfUrl`. */
export function reportDocxUrl(datasetId: string): string {
  return `${API_BASE_URL}/datasets/${encodeURIComponent(datasetId)}/report/docx`;
}
