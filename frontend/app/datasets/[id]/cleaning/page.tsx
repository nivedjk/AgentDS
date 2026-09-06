"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { ApiError, runClean, type CleaningReport } from "@/lib/api";

function KeyValueBlock({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return <span className="text-text-muted">—</span>;
  }
  return (
    <dl className="flex flex-col gap-0.5">
      {entries.map(([key, value]) => (
        <div key={key} className="flex gap-2 text-xs">
          <dt className="text-text-muted">{key}:</dt>
          <dd className="font-mono text-text" data-mono>
            {typeof value === "object" && value !== null
              ? JSON.stringify(value)
              : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export default function CleaningPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<CleaningReport | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const result = await runClean(datasetId);
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
        <h1 className="text-lg font-semibold text-text">Cleaning</h1>
        <p className="text-sm text-text-muted">
          Agentic cleaning driven by Claude tool use. Creates a new derived
          dataset each run — requires an API key server-side.
        </p>
      </header>

      <div>
        <button
          type="button"
          disabled={running}
          onClick={handleRun}
          className="h-8 rounded-control border border-line bg-surface-2 px-4 text-sm font-medium text-text-secondary transition-colors hover:bg-line disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? "Running…" : "Run Cleaning"}
        </button>
      </div>

      {error && (
        <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {error}
        </p>
      )}

      {report && (
        <>
          {/* Before/after summary strip */}
          <section className="flex flex-wrap items-center gap-6 rounded-panel border border-line bg-surface p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Cleaned dataset id
              </span>
              <span className="font-mono text-xs text-text" data-mono>
                {report.cleaned_dataset_id ?? "none produced"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Shape (rows × cols)
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.initial_shape[0]} × {report.initial_shape[1]} →{" "}
                {report.final_shape[0]} × {report.final_shape[1]}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Missing cells
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.initial_missing_cells} → {report.final_missing_cells}
              </span>
            </div>
          </section>

          {/* Step trace */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">
              CleaningAgent step trace
            </h2>
            {report.steps.length === 0 && (
              <p className="rounded-panel border border-dashed border-line px-4 py-6 text-center text-sm text-text-muted">
                No cleaning steps were taken.
              </p>
            )}
            <div className="flex flex-col gap-2">
              {report.steps.map((step) => (
                <div
                  key={step.order}
                  className="rounded-panel border border-line bg-surface p-4"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="inline-flex h-5 w-5 items-center justify-center rounded-badge border border-line bg-surface-2 font-mono text-[10px] text-text-muted"
                      data-mono
                    >
                      {step.order}
                    </span>
                    <span className="font-mono text-xs font-medium text-text" data-mono>
                      {step.tool}
                    </span>
                  </div>
                  <p className="mt-2 text-sm text-text-secondary">
                    {step.reason}
                  </p>
                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <p className="mb-1 text-[10px] uppercase tracking-wide text-text-muted">
                        Params
                      </p>
                      <KeyValueBlock data={step.params} />
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] uppercase tracking-wide text-text-muted">
                        Result
                      </p>
                      <KeyValueBlock data={step.result} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Summary bullets */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Summary</h2>
            <ul className="flex flex-col gap-1.5 rounded-panel border border-line bg-surface p-4 text-sm text-text-secondary">
              {report.summary.map((line, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-text-muted">-</span>
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Remaining issues */}
          {report.remaining_issues.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="text-sm font-semibold text-text">
                Remaining issues
              </h2>
              <ul className="flex flex-col gap-1.5 rounded-panel border border-warning/30 bg-warning/10 p-4 text-sm text-text-secondary">
                {report.remaining_issues.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-warning">-</span>
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
