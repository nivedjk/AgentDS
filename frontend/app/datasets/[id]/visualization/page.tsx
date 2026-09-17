"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { useParams } from "next/navigation";
import type { Data, Layout } from "plotly.js";
import {
  ApiError,
  getVisualization,
  runVisualize,
  type VisualizationReport,
} from "@/lib/api";

// Plotly touches `window` on import — must be loaded client-side only.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function VisualizationPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<VisualizationReport | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Show a previously-computed result instantly on load, if one exists.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const cached = await getVisualization(datasetId);
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
      const result = await runVisualize(datasetId);
      setReport(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-hairline pb-4">
        <h1 className="text-h2 font-semibold text-fg">Visualization</h1>
        <p className="text-sm text-fg-subtle">
          Deterministic chart builders plus one Claude narration call.
          Requires an API key server-side.
        </p>
      </header>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={running || loading}
          onClick={handleRun}
          className="h-8 rounded-control border border-hairline bg-panel-raised px-4 text-sm font-medium text-fg-muted transition-colors hover:bg-hairline disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running
            ? "Running…"
            : report
              ? "Re-run Visualization"
              : "Run Visualization"}
        </button>
        {loading && (
          <span className="text-xs text-fg-subtle">Checking for a saved result…</span>
        )}
        {!loading && report && !running && (
          <span className="text-xs text-fg-subtle">Showing a saved result.</span>
        )}
      </div>

      {error && (
        <p className="rounded-control border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
          {error}
        </p>
      )}

      {report && (
        <>
          {/* Identity strip */}
          <section className="flex flex-wrap items-center gap-4 rounded-panel border border-hairline bg-panel p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Source
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.source}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Target column
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.target_column ?? "none identified"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Charts
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.n_charts}
              </span>
            </div>
          </section>

          {/* Charts */}
          <section className="flex flex-col gap-4">
            {report.charts.map((chart, i) => (
              <div
                key={i}
                className="flex flex-col gap-2 rounded-panel border border-hairline bg-panel p-4"
              >
                <Plot
                  data={(chart.plotly.data as Data[] | undefined) ?? []}
                  layout={{
                    ...((chart.plotly.layout as Partial<Layout>) ?? {}),
                    autosize: true,
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    font: { color: "#8b93a2" },
                  }}
                  useResizeHandler
                  style={{ width: "100%", height: "360px" }}
                  config={{ displayModeBar: false, responsive: true }}
                />
                <div className="flex flex-col gap-0.5">
                  <p className="text-sm font-medium text-fg">
                    {chart.title}
                  </p>
                  {chart.insight && (
                    <p className="text-xs text-fg-subtle">{chart.insight}</p>
                  )}
                </div>
              </div>
            ))}
          </section>

          {/* Skipped */}
          {report.skipped.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="text-h4 font-semibold text-fg">Skipped</h2>
              <ul className="flex flex-col gap-1.5 rounded-panel border border-hairline bg-panel p-4 text-sm text-fg-muted">
                {report.skipped.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-fg-subtle">-</span>
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Narrative */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">Narrative</h2>
            <p className="rounded-panel border border-hairline bg-panel px-4 py-3 text-sm leading-relaxed text-fg-muted">
              {report.narrative}
            </p>
          </section>
        </>
      )}
    </div>
  );
}
