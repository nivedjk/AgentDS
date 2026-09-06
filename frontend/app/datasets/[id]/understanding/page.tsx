"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  ApiError,
  getQuickStats,
  type DataUnderstandingReport,
} from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function UnderstandingPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<DataUnderstandingReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const result = await getQuickStats(datasetId);
        if (!cancelled) setReport(result);
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

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-line pb-4">
        <h1 className="text-lg font-semibold text-text">Data Understanding</h1>
        <p className="text-sm text-text-muted">
          Deterministic quick-stats pass over the raw dataset — no API key
          required, safe to re-run.
        </p>
      </header>

      {error && (
        <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {error}
        </p>
      )}

      {loading && !report && (
        <p className="text-sm text-text-muted">Running quick-stats…</p>
      )}

      {report && (
        <>
          {/* Identity strip */}
          <section className="flex flex-wrap items-center gap-4 rounded-panel border border-line bg-surface p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Rows
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.n_rows}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Columns
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.n_columns}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Target candidate
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.target_candidate ?? "none identified"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Problem type
              </span>
              <StatusBadge
                variant={
                  report.problem_type === "unclear" ? "not-run" : "done"
                }
              >
                {report.problem_type}
              </StatusBadge>
            </div>
          </section>

          {/* Columns table */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Columns</h2>
            <div className="overflow-hidden rounded-panel border border-line">
              <table className="w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="h-[26px] bg-surface text-[10px] uppercase tracking-wide text-text-muted">
                    <th className="border-b border-line px-3 font-medium">
                      Name
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Dtype
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Missing
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Missing %
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Unique
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {report.columns.map((col) => (
                    <tr
                      key={col.name}
                      className="h-8 border-b border-line last:border-b-0 hover:bg-surface"
                    >
                      <td className="px-3 text-text">{col.name}</td>
                      <td
                        className="px-3 font-mono text-xs text-text-muted"
                        data-mono
                      >
                        {col.dtype}
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-text"
                        data-mono
                      >
                        {col.missing_count}
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-text"
                        data-mono
                      >
                        {col.missing_pct}%
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-text"
                        data-mono
                      >
                        {col.n_unique}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Correlations table */}
          {report.correlations && report.correlations.length > 0 && (
            <section className="flex flex-col gap-2">
              <h2 className="text-sm font-semibold text-text">
                Correlations (|r| ≥ 0.85)
              </h2>
              <div className="overflow-hidden rounded-panel border border-line">
                <table className="w-full border-collapse text-left text-sm">
                  <thead>
                    <tr className="h-[26px] bg-surface text-[10px] uppercase tracking-wide text-text-muted">
                      <th className="border-b border-line px-3 font-medium">
                        Column A
                      </th>
                      <th className="border-b border-line px-3 font-medium">
                        Column B
                      </th>
                      <th className="border-b border-line px-3 font-medium">
                        Correlation
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.correlations.map((c, i) => (
                      <tr
                        key={`${c.column_a}-${c.column_b}-${i}`}
                        className="h-8 border-b border-line last:border-b-0 hover:bg-surface"
                      >
                        <td className="px-3 text-text">{c.column_a}</td>
                        <td className="px-3 text-text">{c.column_b}</td>
                        <td
                          className="px-3 font-mono text-xs text-text"
                          data-mono
                        >
                          {c.correlation}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Duplicates summary */}
          {report.duplicates && (
            <section className="flex flex-col gap-2">
              <h2 className="text-sm font-semibold text-text">Duplicates</h2>
              <p className="rounded-panel border border-line bg-surface px-4 py-3 text-sm text-text-secondary">
                {report.duplicates.n_duplicate_rows} duplicate row(s) (
                {report.duplicates.percent}% of rows).
              </p>
            </section>
          )}

          {/* Key findings */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Key findings</h2>
            <ul className="flex flex-col gap-1.5 rounded-panel border border-line bg-surface p-4 text-sm text-text-secondary">
              {report.key_findings.map((finding, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-text-muted">-</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Narrative */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Narrative</h2>
            <p className="rounded-panel border border-line bg-surface px-4 py-3 text-sm leading-relaxed text-text-secondary">
              {report.narrative}
            </p>
          </section>

          {/* Reasoning (collapsible/secondary) */}
          <details className="rounded-panel border border-line bg-surface p-4">
            <summary className="cursor-pointer text-sm font-semibold text-text-secondary">
              Reasoning
            </summary>
            <p className="mt-2 text-sm leading-relaxed text-text-muted">
              {report.reasoning}
            </p>
          </details>
        </>
      )}
    </div>
  );
}
