"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  ApiError,
  getTraining,
  runTrain,
  type TrainingReport,
} from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

function RoleList({ label, items }: { label: string; items: string[] }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wide text-text-muted">
        {label} ({items.length})
      </span>
      {items.length === 0 ? (
        <span className="text-xs text-text-muted">—</span>
      ) : (
        <div className="flex flex-wrap gap-1">
          {items.map((name) => (
            <span
              key={name}
              className="rounded-badge border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-text-secondary"
              data-mono
            >
              {name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function TrainingPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<TrainingReport | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Show a previously-computed result instantly on load, if one exists.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const cached = await getTraining(datasetId);
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
      const result = await runTrain(datasetId);
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
          Training &amp; Evaluation
        </h1>
        <p className="text-sm text-text-muted">
          Deterministic AutoML sweep over the fixed model catalog — no API
          key required. Narration prose is best-effort and comes back empty
          if no key is configured server-side.
        </p>
      </header>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={running || loading}
          onClick={handleRun}
          className="h-8 rounded-control border border-line bg-surface-2 px-4 text-sm font-medium text-text-secondary transition-colors hover:bg-line disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? "Training…" : report ? "Re-run Training" : "Run Training"}
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
          {/* Best-model summary strip */}
          <section className="flex flex-wrap items-center gap-6 rounded-panel border border-line bg-surface p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Best model
              </span>
              <span className="font-mono text-sm font-medium text-text" data-mono>
                {report.best_model ?? "none succeeded"}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Best score ({report.primary_metric})
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.best_score ?? "—"}
              </span>
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
                Target
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.target}
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Rows (train / test)
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.n_rows} ({report.train_rows} / {report.test_rows})
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                CV
              </span>
              <span className="font-mono text-sm text-text" data-mono>
                {report.cv_splits}-fold{" "}
                {report.stratified ? "(stratified)" : "(unstratified)"}
              </span>
            </div>
          </section>

          {/* Leaderboard */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Leaderboard</h2>
            <div className="overflow-x-auto rounded-panel border border-line">
              <table className="w-full min-w-[560px] border-collapse text-left text-sm">
                <thead>
                  <tr className="h-[26px] bg-surface text-[10px] uppercase tracking-wide text-text-muted">
                    <th className="border-b border-line px-3 font-medium">
                      Rank
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Model
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      CV mean ± std
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Test ({report.primary_metric})
                    </th>
                    <th className="border-b border-line px-3 font-medium">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {report.leaderboard.map((row) => (
                    <tr
                      key={row.name}
                      className={`h-8 border-b border-line last:border-b-0 ${
                        row.failed
                          ? "bg-error/5 text-text-muted"
                          : row.rank === 1
                            ? "bg-success/10"
                            : "hover:bg-surface"
                      }`}
                    >
                      <td className="px-3 font-mono text-xs text-text" data-mono>
                        {row.rank}
                      </td>
                      <td
                        className={`px-3 font-mono text-xs ${
                          row.failed ? "text-text-muted" : "text-text"
                        }`}
                        data-mono
                      >
                        {row.name}
                        {row.rank === 1 && !row.failed && (
                          <StatusBadge variant="done">
                            <span className="ml-1">best</span>
                          </StatusBadge>
                        )}
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-text"
                        data-mono
                      >
                        {row.cv_mean !== null
                          ? `${row.cv_mean} ± ${row.cv_std}`
                          : "—"}
                      </td>
                      <td
                        className="px-3 font-mono text-xs text-text"
                        data-mono
                      >
                        {row.test_score ?? "—"}
                      </td>
                      <td className="px-3">
                        <StatusBadge variant={row.failed ? "error" : "done"}>
                          {row.failed ? "failed" : "fit"}
                        </StatusBadge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Candidate errors, if any */}
            {report.candidates.some((c) => c.error) && (
              <ul className="flex flex-col gap-1 rounded-panel border border-error/30 bg-error/10 p-3 text-xs text-error">
                {report.candidates
                  .filter((c) => c.error)
                  .map((c) => (
                    <li key={c.name} className="flex gap-2">
                      <span className="font-mono" data-mono>
                        {c.name}:
                      </span>
                      <span>{c.error}</span>
                    </li>
                  ))}
              </ul>
            )}
          </section>

          {/* Column roles */}
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-text">Column roles</h2>
            <div className="grid grid-cols-1 gap-3 rounded-panel border border-line bg-surface p-4 sm:grid-cols-2">
              <RoleList label="Numeric" items={report.column_roles.numeric} />
              <RoleList
                label="Categorical"
                items={report.column_roles.categorical}
              />
              <RoleList
                label="Dropped (high cardinality)"
                items={report.column_roles.dropped_high_cardinality}
              />
              <RoleList
                label="Dropped (datetime)"
                items={report.column_roles.dropped_datetime}
              />
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
                No narrative — ANTHROPIC_API_KEY is not configured server-side
                for this run, so only the deterministic results above were
                produced.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
