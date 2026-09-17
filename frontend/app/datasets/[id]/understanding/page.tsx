"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  ApiError,
  getQuickStats,
  getUnderstanding,
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
        // Preserve whatever is already cached - a real /analyze (agentic)
        // result included - instead of unconditionally overwriting it with
        // a fresh /quick-stats call on every visit. Only fall back to
        // /quick-stats when nothing has run for this dataset yet.
        const cached = await getUnderstanding(datasetId);
        const result = cached ?? (await getQuickStats(datasetId));
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
      <header className="flex flex-col gap-1 border-b border-hairline pb-4">
        <h1 className="text-h2 font-semibold text-fg">Data Understanding</h1>
        <p className="text-sm text-fg-subtle">
          Deterministic quick-stats pass over the raw dataset — no API key
          required, safe to re-run.
        </p>
      </header>

      {error && (
        <p className="rounded-control border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
          {error}
        </p>
      )}

      {loading && !report && (
        <p className="text-sm text-fg-subtle">Running quick-stats…</p>
      )}

      {report && (
        <>
          {/* Identity strip */}
          <section className="flex flex-wrap items-center gap-4 rounded-panel border border-hairline bg-panel p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Rows
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.n_rows}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Columns
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.n_columns}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Target candidate
              </span>
              <span className="font-mono text-sm text-fg" data-mono>
                {report.target_candidate ?? "none identified"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
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
            <h2 className="text-h4 font-semibold text-fg">Columns</h2>
            <div className="overflow-hidden rounded-panel border border-hairline">
              <table className="w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="h-[26px] bg-panel text-[10px] uppercase tracking-wide text-fg-subtle">
                    <th className="border-b border-hairline px-3 font-medium">
                      Name
                    </th>
                    <th className="border-b border-hairline px-3 font-medium">
                      Dtype
                    </th>
                    <th className="border-b border-hairline px-3 font-medium">
                      Missing
                    </th>
                    <th className="border-b border-hairline px-3 font-medium">
                      Missing %
                    </th>
                    <th className="border-b border-hairline px-3 font-medium">
                      Unique
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {report.columns.map((col) => (
                    <tr
                      key={col.name}
                      className="h-8 border-b border-hairline last:border-b-0 hover:bg-panel"
                    >
                      <td className="px-3 text-fg">{col.name}</td>
                      <td
                        className="px-3 font-mono text-xs text-fg-subtle"
                        data-mono
                      >
                        {col.dtype}
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-fg"
                        data-mono
                      >
                        {col.missing_count}
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-fg"
                        data-mono
                      >
                        {col.missing_pct}%
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-fg"
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
              <h2 className="text-h4 font-semibold text-fg">
                Correlations (|r| ≥ 0.85)
              </h2>
              <div className="overflow-hidden rounded-panel border border-hairline">
                <table className="w-full border-collapse text-left text-sm">
                  <thead>
                    <tr className="h-[26px] bg-panel text-[10px] uppercase tracking-wide text-fg-subtle">
                      <th className="border-b border-hairline px-3 font-medium">
                        Column A
                      </th>
                      <th className="border-b border-hairline px-3 font-medium">
                        Column B
                      </th>
                      <th className="border-b border-hairline px-3 font-medium">
                        Correlation
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.correlations.map((c, i) => (
                      <tr
                        key={`${c.column_a}-${c.column_b}-${i}`}
                        className="h-8 border-b border-hairline last:border-b-0 hover:bg-panel"
                      >
                        <td className="px-3 text-fg">{c.column_a}</td>
                        <td className="px-3 text-fg">{c.column_b}</td>
                        <td
                          className="px-3 font-mono text-xs text-fg"
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
              <h2 className="text-h4 font-semibold text-fg">Duplicates</h2>
              <p className="rounded-panel border border-hairline bg-panel px-4 py-3 text-sm text-fg-muted">
                {report.duplicates.n_duplicate_rows} duplicate row(s) (
                {report.duplicates.percent}% of rows).
              </p>
            </section>
          )}

          {/* Key findings */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">Key findings</h2>
            <ul className="flex flex-col gap-1.5 rounded-panel border border-hairline bg-panel p-4 text-sm text-fg-muted">
              {report.key_findings.map((finding, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-fg-subtle">-</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Narrative */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">Narrative</h2>
            <p className="rounded-panel border border-hairline bg-panel px-4 py-3 text-sm leading-relaxed text-fg-muted">
              {report.narrative}
            </p>
          </section>

          {/* Reasoning (collapsible/secondary) */}
          <details className="rounded-panel border border-hairline bg-panel p-4">
            <summary className="cursor-pointer text-sm font-semibold text-fg-muted">
              Reasoning
            </summary>
            <p className="mt-2 text-sm leading-relaxed text-fg-subtle">
              {report.reasoning}
            </p>
          </details>
        </>
      )}
    </div>
  );
}
