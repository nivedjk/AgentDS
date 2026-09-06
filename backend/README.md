# AgentDS — Backend

FastAPI backend for AgentDS, a multi-agent AI data scientist platform.

This is an early skeleton: dataset upload, Module 1 (Data Understanding),
Module 2 (Cleaning), Module 3 (Visualization), Module 4 (Model
Recommendation), Module 5 (Training & AutoML), Module 6 (Explainability),
and Module 7 (Report). The What-If Agent is not built yet.

## Module 1: Data Understanding Agent

Investigates an uploaded dataset and returns a `DataUnderstandingReport`
(column overview, duplicates, correlations, a target/problem-type guess, and
a plain-English narrative). Two ways to run it, both defined in
`app/agents/data_understanding.py`:

- **`POST /datasets/{dataset_id}/analyze`** — agentic. An LLM drives an
  investigative tool-use loop: it decides which tools to call, in what
  order, and how many times (capped at 8 investigative calls) before
  submitting its conclusions via a terminal `submit_report` tool call.
  **Provider-gated, not key-gated** — see [LLM provider](#llm-provider-all-modules)
  below. Defaults to a local Ollama model; `503` if that provider isn't
  currently usable.
- **`POST /datasets/{dataset_id}/quick-stats`** — deterministic, LLM-free
  heuristic pass over the same building blocks (rule-based target/
  problem-type guess, no narrative reasoning). **No API key required** —
  a fast fallback for offline use or if live API access has issues during a
  demo.

The agent's tools, all pure pandas functions reused as both the deterministic
report's building blocks and the agent's tool implementations:

| Tool | Purpose |
| --- | --- |
| `inspect_column(column_name)` | Full stats for one column — numeric: count/mean/median/std/min/max/quartiles/skew; categorical: cardinality + top 10 value counts. Missing count/% either way. |
| `sample_rows(n=5)` | `n` random rows as a list of dicts. |
| `check_duplicates()` | Count/percent of fully duplicated rows, plus up to 3 example rows. |
| `get_correlations(threshold=0.85)` | Numeric column pairs with `\|correlation\| >= threshold`, sorted descending. |
| `submit_report(...)` | Terminal tool — ends the investigation with the agent's conclusions (target candidate, problem type, reasoning, narrative, key findings), schema-validated via Pydantic. |

Factual fields in the final report (columns, duplicates, correlations) are
always taken from the actual tool results the agent retrieved during the
loop — never parsed from Claude's free text.

## Module 2: Cleaning Agent

Cleans an uploaded dataset with a Claude tool-use loop, seeded with the
Module 1 report as starting context. Defined in `app/agents/cleaning.py`.

- **`POST /datasets/{dataset_id}/clean`** — agentic. Loads the cached
  Module 1 `DataUnderstandingReport` (runs `/analyze` internally first if
  the dataset was never analyzed), then an LLM drives a loop of mutating
  cleaning actions before submitting conclusions via a terminal
  `submit_cleaning_report` call. Caps: 25 mutating actions
  (`inspect_column` is free), 40 total tool calls as a backstop.
  **Provider-gated, not key-gated** — same as `/analyze`, see
  [LLM provider](#llm-provider-all-modules) below.

The original dataset is never modified: cleaning writes a new CSV under a
fresh `dataset_id` (returned as `cleaned_dataset_id`). The full
`CleaningReport` — the ordered step log (each with the `reason` the agent
gave), initial/final shape, missing-cell counts, summary, and remaining
issues — is cached next to the dataset as a JSON sidecar.

| Tool | Purpose |
| --- | --- |
| `inspect_column(column_name)` | Full stats for one column of the current working frame (reused from Module 1). Read-only; does not count against the action budget. |
| `impute_column(column, strategy, reason, constant_value=None)` | Fill missing values (`mean`/`median`/`mode`/`constant`) or `drop_rows`. Returns before/after missing counts. |
| `drop_column(column, reason)` | Remove a column. The target column is protected. |
| `remove_duplicates(reason)` | Drop fully duplicated rows (keep first). |
| `flag_outliers(column, method, action, reason)` | `iqr`/`zscore` detection; `flag` adds `<column>_outlier`, `cap` winsorizes, `remove` drops rows. |
| `encode_column(column, method, reason)` | `one_hot` (new int columns, original dropped) or `label` (in-place integer mapping). |
| `scale_column(column, method, reason)` | `standard` / `minmax` / `robust`. Returns before/after mean/std or min/max. |
| `submit_cleaning_report(summary, remaining_issues)` | Terminal tool — ends the session. `summary` is a list of what-was-done-and-why entries. |

Action tools that would remove more than 50% of the current rows
(`drop_rows` imputation, outlier `remove`, `remove_duplicates`) are
refused and the reason is fed back to the agent. Column references are
validated against the *current* frame, so a step that names a column an
earlier step dropped comes back as an error the agent can recover from.

## Module 3: Visualization Agent

Turns a dataset into a set of Plotly-ready chart specifications. Defined in
`app/agents/visualization.py`.

- **`POST /datasets/{dataset_id}/visualize`** — deterministic pandas/numpy
  builders compute the actual chart data (histogram bin counts, category
  counts, correlation values, box five-number summaries, resampled time
  series) and embed it in literal Plotly figure dicts. A single LLM
  narration call then selects and orders the most insightful charts, writes
  a one-line `insight` per chart, and an overall `narrative` — it never
  produces or edits chart data. Operates on the **cleaned** dataset when
  Module 2 produced one, else the original; regenerates the Module 1 report
  first if it was never cached. **Provider-gated** — see
  [LLM provider](#llm-provider-all-modules); `503` if the provider isn't
  currently usable.

Coverage: numeric (histogram + box), categorical (value-count bar,
low-cardinality only), temporal (line chart when a parseable datetime
column exists), correlation (numeric corr-matrix heatmap), and
target-relationship charts (numeric-vs-target box, categorical-vs-target
grouped bars) using the target from the Module 1 report. Columns that
cannot be charted (too high cardinality, all-null, unparseable dates,
zero variance) are skipped and recorded with a reason in the report's
`skipped` list. The full `VisualizationReport` (`charts`, `skipped`,
`narrative`, `source`, `target_column`, `problem_type`) is cached next to
the dataset as `{id}.visualization.report.json`.

## Module 4: Model Recommendation Agent

Deterministically profiles the dataset, then makes ONE LLM reasoning
call to recommend candidate models. Defined in
`app/agents/recommendation.py`. No training happens here and no metric
numbers are produced.

- **`POST /datasets/{dataset_id}/recommend`** — profiles the **cleaned**
  dataset when Module 2 produced one (else the original), reads the cached
  Module 1 report (runs `/analyze` internally first if absent), builds a
  `modeling_profile` (row/feature counts, numeric vs categorical split,
  target type, class count + balance, missing %, high-cardinality
  categorical count, size bucket, datetime presence), then the LLM picks
  3–4 models from a **fixed catalog**, writes a rationale for each, names
  the preprocessing needed, and picks a primary metric. **Provider-gated**
  — see [LLM provider](#llm-provider-all-modules); `503` if the provider
  isn't currently usable; `404` for an unknown dataset; `422` if the
  problem type is `unclear` or the target column is absent.

The LLM only chooses catalog names and writes prose. `library` and the
**fixed default hyperparameters** (`random_state=42` where applicable) are
copied from the catalog verbatim — they are the contract Module 5 consumes.
Candidate names outside the catalog are dropped and back-filled
deterministically; an out-of-range metric falls back to `f1_macro`
(classification) / `r2` (regression). The full `RecommendationReport` is
cached as the `recommendation` JSON sidecar.

| Catalog (classification) | Catalog (regression) | Library |
| --- | --- | --- |
| `logistic_regression` | `ridge` | sklearn |
| `decision_tree_classifier` | `decision_tree_regressor` | sklearn |
| `random_forest_classifier` | `random_forest_regressor` | sklearn |
| `hist_gradient_boosting_classifier` | `hist_gradient_boosting_regressor` | sklearn |
| `xgboost_classifier` | `xgboost_regressor` | xgboost |
| `lightgbm_classifier` | `lightgbm_regressor` | lightgbm |

Allowed primary metrics: `accuracy`, `precision_macro`, `recall_macro`,
`f1_macro`, `roc_auc` (classification); `r2`, `mae`, `rmse` (regression).

## Module 5: Training & AutoML Agent

Fully deterministic training — **no LLM required in the training path**,
just one optional, best-effort narration call at the end. Defined in
`app/agents/training.py`.

- **`POST /datasets/{dataset_id}/train`** — fits every shortlisted catalog
  model, cross-validates and scores each on a held-out split, ranks them by a
  primary metric, and persists both a `TrainingReport` sidecar and every
  fitted pipeline. Operates on the **cleaned** dataset when Module 2 produced
  one, else the original; runs `/quick-stats` internally if the dataset was
  never analyzed. **Not provider-gated** — returns `200` whether or not
  `AGENTDS_LLM_PROVIDER`'s provider is available (unavailable, or failing
  mid-call, only means an empty `narrative`, never a failed request);
  `404` for an unknown dataset; `422` if the problem type is unclear or every
  candidate fails to fit.

Preprocessing (`build_preprocessor`) is a single `ColumnTransformer`: numeric
columns get median imputation + standard scaling, low-cardinality categorical
columns get most-frequent imputation + one-hot encoding
(`handle_unknown="ignore"`), and high-cardinality or datetime columns are
dropped and recorded in `column_roles` rather than fed to the model.

Candidates come from the Module 4 recommendation report's `candidates` list
when one exists, else a fixed default set per problem type. Either way,
hyperparameters are copied verbatim from `MODEL_CATALOG` in
`app/agents/recommendation.py` — never LLM-chosen, never randomized.
`random_state=42` is used everywhere; cross-validation is a fixed
`StratifiedKFold` (classification) or `KFold` (regression), `n_splits=5,
shuffle=True, random_state=42`; the held-out split is `test_size=0.2,
random_state=42`.

Metrics: classification gets accuracy, precision/recall/f1 (macro), plus
`roc_auc` for binary targets; regression gets r2, mae, rmse. Every metric is
reported as both a 5-fold CV mean+std (on the train split) and a single
held-out test score per model.

Every fitted pipeline — preprocessing and estimator together — is persisted
with `joblib` to `data/uploads/models/{dataset_id}/<candidate>.joblib`; the
path is recorded on that candidate's `CandidateResult`. The best model is
chosen by the primary metric (the recommendation report's, else
`f1_macro`/`r2`). A candidate that fails to fit is recorded with an `error`
string rather than aborting the run; only if every candidate fails does the
endpoint return `422`. Given the same dataset, a run is fully reproducible —
identical metrics every time.

| `TrainingReport` field | Contents |
| --- | --- |
| `problem_type` / `target` | `"classification"` or `"regression"`, and the target column name. |
| `column_roles` | `numeric` / `categorical` feature lists, plus `dropped_high_cardinality` / `dropped_datetime`. |
| `primary_metric` | The metric used to rank candidates and pick `best_model`. |
| `candidates` | One `CandidateResult` per shortlisted model — params, `cv_metrics`, `test_metrics`, `model_path`, `fit_seconds`, or `error`. |
| `best_model` / `best_score` | The winning candidate's name and its test-set primary-metric score. |
| `leaderboard` | Ranked `LeaderboardRow` list (successes best→worst, failures last). |
| `narrative` | Optional one- to two-sentence summary of the leaderboard; `""` when the LLM provider wasn't available. |
| `warnings` | Notable non-fatal events, e.g. a stratified-split fallback or an unknown candidate name. |

## Module 6: Explainability Agent

Fully deterministic SHAP computation — **one LLM narration call only for
prose**, exactly like Modules 3/4. Defined in `app/agents/explainability.py`.

- **`POST /datasets/{dataset_id}/explain`** — loads the best fitted pipeline
  from Module 5's `training` sidecar, re-derives the exact train/test split
  `run_training` used (no raw arrays are persisted, so the split is
  reproduced from the recorded `target` / `feature_columns` / `problem_type`
  / `stratified` fields with the same `test_size=0.2, random_state=42`),
  and computes SHAP-based global feature importance plus a handful of
  per-prediction explanations. Operates on the **cleaned** dataset when
  Module 2 produced one, else the original — same fallback `/train` uses.
  Persists an `ExplainabilityReport` sidecar. **Provider-gated** (`503` if
  the provider isn't currently usable, following the Module 3/4 pattern,
  not Module 5's optional-narration one); `404` for an unknown dataset;
  `422` if there is no training report yet, `best_model` is null, or the
  SHAP computation itself fails.

**Explainer selection** (`select_explainer`) is by exact `type(estimator).__name__`
against two fixed sets — never a generic "try tree, fall back" chain:

| Model class name | Explainer |
| --- | --- |
| `DecisionTree*`, `RandomForest*`, `XGB*`, `LGBM*`, `CatBoost*` | `shap.TreeExplainer` |
| `LogisticRegression`, `LinearRegression`, `Ridge` | `shap.LinearExplainer` |
| anything else (deliberately including `HistGradientBoosting*`) | capped `shap.KernelExplainer` |

The kernel branch summarizes its background to at most 50 points via
`shap.sample(..., random_state=42)` (chosen over `shap.kmeans` specifically
because it takes an explicit seed, so it's trivially deterministic) and uses
`predict_proba` for classification when available, else `predict`. A model
with neither, or any SHAP failure (e.g. an unsupported estimator/target
combination), raises `ExplainabilityError` rather than an unhandled 500; the
router maps it to `422`.

Deterministic derivations, no LLM involved:

- Rows explained are capped at 50 (`n_rows_explained`), taken as the first
  N rows of the reproduced held-out test split (deterministic — the split
  order is already fixed).
- Feature names come from `pipeline.named_steps["pre"].get_feature_names_out()`
  — the transformed/one-hot-expanded names. With this project's
  `OneHotEncoder` configuration a categorical column `city` with value `NY`
  shows up as `city_NY` (sklearn's default `_` separator, not `col=value`).
- For a multiclass model whose SHAP output carries a per-class dimension,
  each row's top reasons use the SHAP values for the class that row was
  actually predicted as; global importance averages `|shap value|` across
  all classes and all rows. Binary/regression output has no such dimension.
- `feature_importance` is `mean(|shap value|)` per feature over the explained
  rows, sorted descending, rounded to 6 decimals.
- `sample_explanations` covers every explained row when there are 10 or
  fewer, else the first 5, each with its top-3 `|shap_value|`-ranked reasons.

| `ExplainabilityReport` field | Contents |
| --- | --- |
| `model_name` / `explainer_type` | The best model's catalog name and which SHAP branch (`tree`/`linear`/`kernel`) explained it. |
| `n_rows_explained` | How many held-out test rows SHAP actually ran on (≤ 50). |
| `feature_importance` | Ranked `{feature, mean_abs_shap}` list, most important first. |
| `sample_explanations` | Per-row `{row_index, predicted_value, top_reasons}` — up to 3 `{feature, shap_value, feature_value}` reasons each. |
| `narrative` | Optional two-to-three sentence summary naming the top features; `""` when no API key was set or narration failed. |
| `warnings` | Notable non-fatal events, e.g. the kernel branch being used, or explained rows being capped. |

## Module 7: Report Agent

Assembles a single final report from every sidecar the earlier modules
wrote. **Deterministic**: every number, table and metric in a section is
copied verbatim from a sidecar. The LLM writes prose only — the executive
summary, one intro paragraph per section, and the conclusion — in **one**
narration call over a compact digest (counts and names, never full
arrays). Defined in `app/agents/report.py`.

- **`POST /datasets/{dataset_id}/report`** — builds the report and caches
  it as the `final` sidecar. Runs Module 1 first if the dataset was never
  analysed, so it succeeds even when only the upload + `/analyze` have run
  (every other section is marked "not run"). **Provider-gated** — see
  [LLM provider](#llm-provider-all-modules); `503` if the provider isn't
  currently usable.
- **`GET /datasets/{dataset_id}/report`** — returns the cached `FinalReport`
  (`404` if never built). No LLM required; works fully offline.
- **`GET /datasets/{dataset_id}/artifacts`** — `{present: [...], missing: [...]}`
  over the seven upstream sidecar kinds. Used by the frontend project
  overview. No LLM.
- **`GET /datasets/{dataset_id}/report/pdf`** — renders the cached
  `FinalReport.markdown` as a downloadable PDF (`application/pdf`, `404` if
  never built). No LLM required — it only formats an already-built report,
  no LLM call.
- **`GET /datasets/{dataset_id}/report/docx`** — same, as a downloadable
  Word document (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`,
  `404` if never built). No LLM required.

The `FinalReport` carries `sections` (ten structured `ReportSection`s, each
with `status: "ok" | "not_run"`) and `markdown` (the whole rendered
document). Plotly figures are not embedded — the visualizations section
lists chart titles/insights and the frontend reads the `visualization`
sidecar for the interactive charts.

PDF/Word export (`app/agents/report_export.py`) parses `FinalReport.markdown`
— not `sections`, since only `markdown` carries the woven-in LLM section
intros — into a small shared intermediate block representation
(`parse_markdown_blocks`), then two renderers walk that same block list:
`render_pdf` (reportlab flowables) and `render_docx` (python-docx), so the
two formats can't drift out of sync with each other. The parser handles only
the exact Markdown subset this codebase's own section renderers produce
(headers, GitHub-style pipe tables, bullet lists, blockquotes, bold
sub-headings, and `_..._` italic stub/marker lines) — it is not a
general-purpose Markdown parser.

## Design notes

- Plain FastAPI with direct function calls between plain Python classes — no
  LangGraph or other graph-orchestration framework.
- Datasets are stored on local disk — no database.

Both are deliberate choices for a solo build, not oversights.

## Setup

```bash
cd agentds/backend

# 1. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
#   ...or, to also get the test tooling (pytest, httpx):
pip install -r requirements-dev.txt

# 3. Configure environment
cp .env.example .env        # then edit if needed

# 4. Run the API (reload on change)
uvicorn app.main:app --reload
```

The API is then at http://localhost:8000. Open http://localhost:8000/docs for
an interactive API tester (Swagger UI) — you can upload a CSV to
`POST /datasets/upload` straight from the browser.

## LLM provider (all modules)

Every module that calls an LLM goes through one small provider-agnostic
client (`app/agents/_llm_client.py`) instead of calling the Anthropic SDK
directly, so switching provider is a config change, not a rewrite. This
covers both call shapes in the codebase:

- Modules 1 (`/analyze`) and 2 (`/clean`) drive multi-turn agentic
  tool-use loops directly against the client.
- Modules 3, 4, 5, 6, 7 (`/visualize`, `/recommend`, `/train`,
  `/explain`, `/report`) each make exactly one forced-tool narration call
  through `app/agents/_llm.py`'s `narrate()` helper, which uses the same
  client underneath.

`AGENTDS_LLM_PROVIDER` is one global setting for all seven modules — there
is no per-module override.

| Env var                 | Default                  | Purpose |
| ------------------------ | ------------------------ | ------- |
| `AGENTDS_LLM_PROVIDER`   | `ollama`                 | `ollama` or `anthropic`, for every module. |
| `OLLAMA_HOST`            | `http://localhost:11434` | Only read when the provider is `ollama`. |
| `OLLAMA_MODEL`           | `qwen3:8b`               | Only read when the provider is `ollama`. |

**Default (`ollama`) requires a local Ollama server, running, with the model
pulled:**

```bash
ollama pull qwen3:8b   # ~5.2 GB, one-time
ollama serve           # if not already running as a background service
```

If Ollama isn't reachable when a gated endpoint is called, it returns `503`
naming the actual problem (e.g. `"Ollama not reachable at
http://localhost:11434: ..."`) rather than a raw connection-error
traceback — check `ollama ps` / `ollama pull qwen3:8b`. Module 5 (`/train`)
is the one exception: its narration is optional and best-effort, so an
unreachable/failing provider there never fails the request — training
still succeeds, just with an empty `narrative`.

Ollama's tool-calling API has no way to force a specific tool call (unlike
Anthropic's `tool_choice`), so `narrate()` compensates by telling the model
in the prompt that the one available tool must be called — the existing
retry-once-then-deterministic-fallback behavior is unchanged if it doesn't.

**To use Claude instead** (for all modules — the setting is global), set
`AGENTDS_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=<key>` in `.env`.
This restores the original, directly-Anthropic-backed behavior everywhere.

This provider swap was validated first as two standalone, zero-cost probes
(`phase/qwen3-module1-probe/`, `phase/qwen3-module2-probe/`) against the
real tool schemas copied verbatim from `app/agents/data_understanding.py`
and `app/agents/cleaning.py` — 13/13 successful runs, 0 malformed tool
calls across both — before this integration reused that same
request/response handling as its basis, then extended to Modules 3-7's
single-shot `narrate()` calls, live-verified end to end against real
`qwen3:8b` for `/visualize`, `/recommend`, `/train`, `/explain`, and
`/report`.

## Configuration

| Env var            | Default          | Purpose                                  |
| ------------------ | ---------------- | ---------------------------------------- |
| `AGENTDS_DATA_DIR` | `./data/uploads` | Where uploaded datasets are written.     |
| `AGENTDS_LLM_PROVIDER` | `ollama`      | `ollama` or `anthropic`, for every module — see [LLM provider](#llm-provider-all-modules). |
| `ANTHROPIC_API_KEY`| _(unset)_        | Only read/required when `AGENTDS_LLM_PROVIDER=anthropic` (default is `ollama`, which needs no key). Not needed for `/quick-stats`, or for `/train`'s optional narration. |
| `OLLAMA_HOST`      | `http://localhost:11434` | Only read when the provider is `ollama`. |
| `OLLAMA_MODEL`     | `qwen3:8b`       | Only read when the provider is `ollama`. |

## Endpoints

| Method | Path                                  | Description                                      |
| ------ | -------------------------------------- | ------------------------------------------------ |
| GET    | `/`                                     | Health check — returns `{"status": "ok"}`.       |
| POST   | `/datasets/upload`                      | Upload a `.csv`; returns `dataset_id`/`filename`. Rejects non-CSV and empty files with `400`. |
| POST   | `/datasets/{dataset_id}/analyze`        | Agentic Data Understanding report via an LLM tool-use loop. Provider-gated (default `ollama`, see [LLM provider](#llm-provider-all-modules)) — `503` if that provider isn't currently usable. |
| POST   | `/datasets/{dataset_id}/quick-stats`    | Deterministic, LLM-free Data Understanding report. No LLM provider required. |
| POST   | `/datasets/{dataset_id}/clean`          | Agentic cleaning via an LLM tool-use loop. Writes a new cleaned dataset; returns a `CleaningReport`. Provider-gated, same as `/analyze`. |
| POST   | `/datasets/{dataset_id}/visualize`      | Deterministic chart specs + one LLM narration call. Returns a `VisualizationReport`. Provider-gated (`503` if the provider isn't currently usable). |
| POST   | `/datasets/{dataset_id}/recommend`      | Deterministic modeling profile + one LLM reasoning call → 3–4 catalog models, rationales, preprocessing, primary metric. Returns a `RecommendationReport`. Provider-gated (`503` if the provider isn't currently usable). |
| POST   | `/datasets/{dataset_id}/train`          | Deterministic training & model selection over a fixed candidate set. Persists fitted pipelines + a `TrainingReport` sidecar. Not provider-gated (`narrative` is empty if the LLM provider isn't available); `422` if the problem type is unclear or all candidates fail. |
| POST   | `/datasets/{dataset_id}/explain`        | SHAP feature importance + per-prediction reasons for the best trained model. Returns an `ExplainabilityReport`. Provider-gated (`503` if the provider isn't currently usable); `422` if there is no training report yet, `best_model` is null, or SHAP fails to explain the model. |
| POST   | `/datasets/{dataset_id}/report`    | Build + cache the final report (prose via one narration call). Provider-gated (`503` if the provider isn't currently usable). |
| GET    | `/datasets/{dataset_id}/report`    | Return the cached `FinalReport` (`404` if never built). No LLM required. |
| GET    | `/datasets/{dataset_id}/report/pdf`    | Render the cached report as a downloadable PDF (`404` if never built). No LLM required. |
| GET    | `/datasets/{dataset_id}/report/docx`   | Render the cached report as a downloadable Word document (`404` if never built). No LLM required. |
| GET    | `/datasets/{dataset_id}/artifacts` | Which module sidecars exist for a dataset (`{present, missing}`). No LLM required. |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
