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
      <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
        {label} ({items.length})
      </span>
      {items.length === 0 ? (
        <span className="text-xs text-fg-subtle">—</span>
      ) : (
        <div className="flex flex-wrap gap-1">
          {items.map((name) => (
            <span
              key={name}
              className="rounded-badge border border-hairline bg-panel-raised px-1.5 py-0.5 font-mono text-[10px] text-fg-muted"
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

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
        {label}
      </span>
      <span className="font-mono text-sm text-fg" data-mono>
        {value}
      </span>
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

  // Shared scale for the CV ± std whiskers, from the successful test scores.
  const scores = (report?.leaderboard ?? [])
    .filter((r) => !r.failed && r.test_score != null)
    .map((r) => r.test_score as number);
  const sLo = scores.length ? Math.min(...scores) : 0;
  const sHi = scores.length ? Math.max(...scores) : 1;
  const sPad = (sHi - sLo) * 0.18 || Math.abs(sHi) * 0.1 || 1;
  const dLo = sLo - sPad;
  const dHi = sHi + sPad;
  const pct = (v: number) =>
    Math.max(0, Math.min(100, ((v - dLo) / (dHi - dLo)) * 100));

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-hairline pb-4">
        <h1 className="text-h2 font-semibold text-fg">Training &amp; Evaluation</h1>
        <p className="max-w-[70ch] text-sm text-fg-muted">
          Deterministic sweep over the fixed model catalog — no API key required.
          Narration prose is best-effort and comes back empty when no provider is
          configured server-side.
        </p>
      </header>

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={running || loading}
          onClick={handleRun}
          className="h-8 rounded-control border border-hairline bg-panel-raised px-4 text-sm font-medium text-fg-muted transition-colors hover:border-hairline-strong hover:text-fg disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? "Training…" : report ? "Re-run training" : "Run training"}
        </button>
        {loading && (
          <span className="text-xs text-fg-subtle">Checking for a saved result…</span>
        )}
        {!loading && report && !running && (
          <span className="text-xs text-fg-subtle">Showing a saved result.</span>
        )}
      </div>

      {error && (
        <p className="rounded-control border border-danger/30 bg-danger-surface px-3 py-2 text-xs text-danger">
          {error}
        </p>
      )}

      {report && (
        <>
          {/* Best-model summary — the primary score leads */}
          <section className="flex flex-wrap items-stretch gap-5 rounded-panel border border-hairline bg-panel p-5">
            <div className="flex min-w-[180px] flex-col justify-center gap-1 pr-5 sm:border-r sm:border-hairline">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Best · {report.primary_metric}
              </span>
              <span
                className="font-mono text-h1 font-semibold leading-none text-accent tabular-nums"
                data-mono
              >
                {report.best_score ?? "—"}
              </span>
              <span className="font-mono text-xs text-fg-muted" data-mono>
                {report.best_model ?? "none succeeded"}
              </span>
            </div>
            <div className="grid flex-1 grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3">
              <MetaItem label="Problem type" value={report.problem_type} />
              <MetaItem label="Target" value={report.target} />
              <MetaItem
                label="Rows (train / test)"
                value={`${report.n_rows} (${report.train_rows} / ${report.test_rows})`}
              />
              <MetaItem
                label="Cross-validation"
                value={`${report.cv_splits}-fold ${report.stratified ? "stratified" : "unstratified"}`}
              />
              <MetaItem label="Seed" value="random_state=42" />
            </div>
          </section>

          {/* Leaderboard — ranked rows, promoted score, CV whisker */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">Leaderboard</h2>
            <div className="flex flex-col gap-1.5">
              {report.leaderboard.map((row) => {
                const isBest = row.rank === 1 && !row.failed;
                const hasCv = row.cv_mean != null && row.cv_std != null;
                const mean = row.cv_mean as number;
                const std = row.cv_std as number;
                return (
                  <div
                    key={row.name}
                    className={`grid grid-cols-[1.5rem_1fr_auto] items-center gap-x-4 gap-y-1.5 rounded-control border px-3 py-3 ${
                      row.failed
                        ? "border-hairline bg-panel opacity-60"
                        : isBest
                          ? "border-accent-line bg-panel [box-shadow:inset_3px_0_0_var(--accent)]"
                          : "border-hairline bg-panel"
                    }`}
                  >
                    <span
                      className="text-right font-mono text-xs text-fg-subtle"
                      data-mono
                    >
                      {row.failed ? "—" : row.rank}
                    </span>
                    <span
                      className={`font-mono text-[13px] ${row.failed ? "text-fg-subtle" : "text-fg"}`}
                      data-mono
                    >
                      {row.name}
                      {isBest && (
                        <span className="ml-2 align-middle">
                          <StatusBadge variant="done">best</StatusBadge>
                        </span>
                      )}
                    </span>
                    <div className="text-right">
                      {row.failed ? (
                        <span className="font-mono text-xs font-semibold text-danger">
                          FAILED
                        </span>
                      ) : (
                        <>
                          <span
                            className={`block font-mono text-lg font-semibold leading-none tabular-nums ${
                              isBest ? "text-accent" : "text-fg"
                            }`}
                            data-mono
                          >
                            {row.test_score ?? "—"}
                          </span>
                          <span className="text-[9px] uppercase tracking-wide text-fg-subtle">
                            test {report.primary_metric}
                          </span>
                        </>
                      )}
                    </div>

                    {row.failed ? (
                      <p className="col-span-2 col-start-2 font-mono text-[11px] text-danger">
                        {report.candidates.find((c) => c.name === row.name)?.error ??
                          "did not fit"}
                      </p>
                    ) : (
                      <div className="col-span-2 col-start-2 flex flex-col gap-1.5">
                        <span className="font-mono text-[11px] text-fg-subtle">
                          CV {hasCv ? `${row.cv_mean} ± ${row.cv_std}` : "—"}
                        </span>
                        {hasCv && (
                          <div className="relative h-3">
                            <span className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-hairline" />
                            <span
                              className="absolute top-1/2 h-2 -translate-y-1/2 rounded-[1px] border border-accent-line bg-accent-surface"
                              style={{
                                left: `${pct(mean - std)}%`,
                                right: `${100 - pct(mean + std)}%`,
                              }}
                            />
                            <span
                              className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-accent"
                              style={{ left: `${pct(mean)}%` }}
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {/* Column roles */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">Column roles</h2>
            <div className="grid grid-cols-1 gap-3 rounded-panel border border-hairline bg-panel p-4 sm:grid-cols-2">
              <RoleList label="Numeric" items={report.column_roles.numeric} />
              <RoleList label="Categorical" items={report.column_roles.categorical} />
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
              <h2 className="text-h4 font-semibold text-fg">Warnings</h2>
              <ul className="flex flex-col gap-1.5 rounded-panel border border-warn/30 bg-warn-surface p-4 text-sm text-fg-muted">
                {report.warnings.map((line, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-warn">–</span>
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Narrative */}
          <section className="flex flex-col gap-2">
            <h2 className="text-h4 font-semibold text-fg">Narrative</h2>
            {report.narrative ? (
              <p className="rounded-panel border border-hairline bg-panel px-4 py-3 text-sm leading-relaxed text-fg-muted">
                {report.narrative}
              </p>
            ) : (
              <p className="rounded-panel border border-dashed border-hairline px-4 py-3 text-sm text-fg-subtle">
                No narrative — no LLM provider was reachable for this run, so only
                the deterministic results above were produced.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
