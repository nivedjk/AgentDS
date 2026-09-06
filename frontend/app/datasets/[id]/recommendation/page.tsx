"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  ApiError,
  getRecommendation,
  runRecommend,
  type RecommendationReport,
} from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function RecommendationPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<RecommendationReport | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Show a previously-computed result instantly on load, if one exists.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const cached = await getRecommendation(datasetId);
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
      const result = await runRecommend(datasetId);
      setReport(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-line pb-4">
        <h1 className="text-lg font-semibold text-text">
          Model Recommendation
        </h1>
        <p className="text-sm text-text-muted">
          One Claude reasoning call over a deterministic modeling profile —
          picks candidates from the fixed catalog. Requires an API key
          server-side.
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
            ? "Running…"
            : report
              ? "Re-run Recommendation"
              : "Run Recommendation"}
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
                Used cleaned dataset
              </span>
              <StatusBadge
                variant={report.used_cleaned_dataset ? "done" : "not-run"}
              >
                {report.used_cleaned_dataset ? "yes" : "no"}
              </StatusBadge>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Problem type
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.problem_type}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Primary metric
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.primary_metric}
              </span>
            </div>
          </section>

          {/* Candidates */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Candidates</h2>
            <div className="flex flex-col gap-2">
              {report.candidates.map((candidate, i) => (
                <div
                  key={i}
                  className="rounded-panel border border-line bg-surface p-4"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-text">
                      {candidate.name}
                    </span>
                    <StatusBadge variant="not-run">
                      {candidate.library}
                    </StatusBadge>
                  </div>
                  <p className="mt-2 text-sm text-text-secondary">
                    {candidate.rationale}
                  </p>
                  <details className="mt-3">
                    <summary className="cursor-pointer text-xs font-medium text-text-muted">
                      Hyperparameters
                    </summary>
                    <dl className="mt-2 flex flex-col gap-0.5">
                      {Object.entries(candidate.hyperparameters).map(
                        ([key, value]) => (
                          <div key={key} className="flex gap-2 text-xs">
                            <dt className="text-text-muted">{key}:</dt>
                            <dd className="font-mono text-text" data-mono>
                              {typeof value === "object" && value !== null
                                ? JSON.stringify(value)
                                : String(value)}
                            </dd>
                          </div>
                        ),
                      )}
                    </dl>
                  </details>
                </div>
              ))}
            </div>
          </section>

          {/* Preprocessing recommendations */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">
              Preprocessing recommendations
            </h2>
            <ul className="flex flex-col gap-1.5 rounded-panel border border-line bg-surface p-4 text-sm text-text-secondary">
              {report.preprocessing_recommendations.map((line, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-text-muted">-</span>
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Reasoning */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Reasoning</h2>
            <p className="rounded-panel border border-line bg-surface px-4 py-3 text-sm leading-relaxed text-text-secondary">
              {report.reasoning}
            </p>
          </section>

          {/* Modeling profile — collapsed raw JSON */}
          <details className="rounded-panel border border-line bg-surface p-4">
            <summary className="cursor-pointer text-sm font-semibold text-text-secondary">
              Modeling profile (raw)
            </summary>
            <pre className="mt-2 overflow-x-auto font-mono text-xs text-text-muted" data-mono>
              {JSON.stringify(report.modeling_profile, null, 2)}
            </pre>
          </details>
        </>
      )}
    </div>
  );
}
