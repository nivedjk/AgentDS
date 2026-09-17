"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { ApiError, getCleaningReport, runClean, type CleaningReport } from "@/lib/api";
import { useDatasetShell } from "@/lib/dataset-context";

function KeyValueBlock({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return <span className="text-fg-subtle">—</span>;
  }
  return (
    <dl className="flex flex-col gap-0.5">
      {entries.map(([key, value]) => (
        <div key={key} className="flex gap-2 text-xs">
          <dt className="text-fg-subtle">{key}:</dt>
          <dd className="font-mono text-fg" data-mono>
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
  const { pipelineStatus, present, refreshArtifacts } = useDatasetShell();

  const [report, setReport] = useState<CleaningReport | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isPipelineCleaning =
    pipelineStatus?.status === "running" &&
    (pipelineStatus?.current_stage === "cleaning" ||
      pipelineStatus?.stages?.cleaning?.status === "running");

  const isCleaningDone =
    present.includes("cleaning") ||
    pipelineStatus?.stages?.cleaning?.status === "completed" ||
    pipelineStatus?.stages?.cleaning?.status === "done";

  // Load cached report on mount, and re-fetch when pipeline marks cleaning as done
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const cached = await getCleaningReport(datasetId);
        if (!cancelled && cached) {
          setReport(cached);
          setError(null);
        }
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
  }, [datasetId, isCleaningDone]);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const result = await runClean(datasetId);
      setReport(result);
      await refreshArtifacts();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-hairline pb-4">
        <h1 className="text-h2 font-semibold text-fg">Cleaning</h1>
        <p className="text-sm text-fg-subtle">
          Agentic cleaning driven by tool-use. Creates a new derived
          dataset each run — results are cached automatically for downstream modules.
        </p>
      </header>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={running || isPipelineCleaning}
          onClick={handleRun}
          className="h-8 rounded-control border border-hairline bg-panel-raised px-4 text-sm font-medium text-fg-muted transition-colors hover:bg-hairline disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running || isPipelineCleaning ? "Running…" : report ? "Re-run Cleaning" : "Run Cleaning"}
        </button>

        {isPipelineCleaning && (
          <span className="flex items-center gap-2 text-xs text-accent">
            <span className="h-2 w-2 rounded-full bg-accent animate-ping" />
            <span>Multi-Agent pipeline is executing data cleaning…</span>
          </span>
        )}
      </div>

      {error && (
        <p className="rounded-control border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
          {error}
        </p>
      )}

      {loading && !report && (
        <div className="flex items-center gap-2 py-8 text-xs text-fg-subtle">
          <span className="h-2 w-2 rounded-full bg-fg-subtle animate-pulse" />
          <span>Loading cleaning results…</span>
        </div>
      )}

      {!loading && !report && !isPipelineCleaning && (
        <div className="rounded-panel border border-dashed border-hairline p-8 text-center text-sm text-fg-subtle">
          No cleaning report yet. Click &quot;Run Cleaning&quot; or start the full pipeline from the Overview.
        </div>
      )}

      {report && (
        <>
          {/* Before/after summary strip */}
          <section className="flex flex-wrap items-center gap-6 rounded-panel border border-hairline bg-panel p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Cleaned dataset id
              </span>
              <span className="font-mono text-xs text-fg" data-mono>
                {report.cleaned_dataset_id ?? "none produced"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Shape (rows × cols)
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.initial_shape[0]} × {report.initial_shape[1]} →{" "}
                {report.final_shape[0]} × {report.final_shape[1]}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Missing cells
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.initial_missing_cells} → {report.final_missing_cells}
              </span>
            </div>
          </section>

          {/* Step trace */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">
              CleaningAgent step trace
            </h2>
            {report.steps.length === 0 && (
              <p className="rounded-panel border border-dashed border-hairline px-4 py-6 text-center text-sm text-fg-subtle">
                No cleaning steps were taken.
              </p>
            )}
            <div className="flex flex-col gap-2">
              {report.steps.map((step) => (
                <div
                  key={step.order}
                  className="rounded-panel border border-hairline bg-panel p-4"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="inline-flex h-5 w-5 items-center justify-center rounded-badge border border-hairline bg-panel-raised font-mono text-[10px] text-fg-subtle"
                      data-mono
                    >
                      {step.order}
                    </span>
                    <span className="font-mono text-xs font-medium text-fg" data-mono>
                      {step.tool}
                    </span>
                  </div>
                  <p className="mt-2 text-sm text-fg-muted">
                    {step.reason}
                  </p>
                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <p className="mb-1 text-[10px] uppercase tracking-wide text-fg-subtle">
                        Params
                      </p>
                      <KeyValueBlock data={step.params} />
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] uppercase tracking-wide text-fg-subtle">
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
            <h2 className="text-h4 font-semibold text-fg">Summary</h2>
            <ul className="flex flex-col gap-1.5 rounded-panel border border-hairline bg-panel p-4 text-sm text-fg-muted">
              {report.summary.map((line, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-fg-subtle">-</span>
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Remaining issues */}
          {report.remaining_issues.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="text-h4 font-semibold text-fg">
                Remaining issues
              </h2>
              <ul className="flex flex-col gap-1.5 rounded-panel border border-warn/30 bg-warn/10 p-4 text-sm text-fg-muted">
                {report.remaining_issues.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-warn">-</span>
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
