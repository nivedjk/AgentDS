"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  ApiError,
  getExplainability,
  runExplain,
  type ExplainabilityReport,
} from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function ExplainabilityPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<ExplainabilityReport | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Show a previously-computed result instantly on load, if one exists.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const cached = await getExplainability(datasetId);
        if (!cancelled && cached) setReport(cached);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.detail : String(err));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const result = await runExplain(datasetId);
      setReport(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setRunning(false);
    }
  }

  const maxAbsShap = report
    ? Math.max(...report.feature_importance.map((f) => f.mean_abs_shap), 0)
    : 0;

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-line pb-4">
        <h1 className="text-lg font-semibold text-text">Explainability</h1>
        <p className="text-sm text-text-muted">
          SHAP-based feature importance and per-prediction explanations for
          the best trained model. Requires an API key server-side, and a
          training report must already exist.
        </p>
      </header>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={running || loading}
          onClick={handleRun}
          className="h-8 rounded-control border border-line bg-surface-2 px-4 text-sm font-medium text-text-secondary transition-colors hover:bg-line disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running
            ? "Explaining…"
            : report
              ? "Re-run Explainability"
              : "Run Explainability"}
        </button>
        {loading && (
          <span className="text-xs text-text-muted">Checking for a saved result…</span>
        )}
        {!loading && report && !running && (
          <span className="text-xs text-text-muted">Showing a saved result.</span>
        )}
      </div>

      {error && (
        <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {error}
        </p>
      )}

      {report && (
        <>
          {/* Identity strip */}
          <section className="flex flex-wrap items-center gap-4 rounded-panel border border-line bg-surface p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Model explained
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.model_name}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Explainer
              </span>
              <StatusBadge variant="done">{report.explainer_type}</StatusBadge>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Rows explained
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.n_rows_explained}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Target
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.target}
              </span>
            </div>
          </section>

          {/* Feature importance */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">
              Feature importance (mean |SHAP|)
            </h2>
            <div className="flex flex-col gap-1.5 rounded-panel border border-line bg-surface p-4">
              {report.feature_importance.map((fi) => (
                <div key={fi.feature} className="flex items-center gap-3">
                  <span className="w-32 shrink-0 truncate text-xs text-text-secondary">
                    {fi.feature}
                  </span>
                  <div className="h-3 flex-1 overflow-hidden rounded-badge bg-surface-2">
                    <div
                      className="h-full bg-info"
                      style={{
                        width:
                          maxAbsShap > 0
                            ? `${(fi.mean_abs_shap / maxAbsShap) * 100}%`
                            : "0%",
                      }}
                    />
                  </div>
                  <span
                    className="w-20 shrink-0 text-right font-mono text-xs text-text"
                    data-mono
                  >
                    {fi.mean_abs_shap}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Sample explanations */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">
              Sample predictions
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {report.sample_explanations.map((sample) => (
                <div
                  key={sample.row_index}
                  className="rounded-panel border border-line bg-surface p-4"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-text-muted" data-mono>
                      row {sample.row_index}
                    </span>
                    <span className="font-mono text-sm font-medium text-text" data-mono>
                      predicted: {sample.predicted_value}
                    </span>
                  </div>
                  <ul className="mt-3 flex flex-col gap-1.5">
                    {sample.top_reasons.map((reason) => {
                      const positive = reason.shap_value >= 0;
                      return (
                        <li
                          key={reason.feature}
                          className="flex items-center justify-between gap-2 text-xs"
                        >
                          <span className="truncate text-text-secondary">
                            {reason.feature} ={" "}
                            <span className="font-mono" data-mono>
                              {reason.feature_value}
                            </span>
                          </span>
                          <span
                            className={`font-mono ${
                              positive ? "text-success" : "text-error"
                            }`}
                            data-mono
                          >
                            {positive ? "▲" : "▼"} {reason.shap_value}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          {/* Warnings */}
          {report.warnings.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="text-sm font-semibold text-text">Warnings</h2>
              <ul className="flex flex-col gap-1.5 rounded-panel border border-warning/30 bg-warning/10 p-4 text-sm text-text-secondary">
                {report.warnings.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-warning">-</span>
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Narrative */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Narrative</h2>
            {report.narrative ? (
              <p className="rounded-panel border border-line bg-surface px-4 py-3 text-sm leading-relaxed text-text-secondary">
                {report.narrative}
              </p>
            ) : (
              <p className="rounded-panel border border-dashed border-line px-4 py-3 text-sm text-text-muted">
                No narrative was produced for this run.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
