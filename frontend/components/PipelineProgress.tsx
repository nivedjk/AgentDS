"use client";

import { useEffect, useState } from "react";
import {
  formatDuration,
  formatElapsed,
  type PipelineStatusResponse,
} from "@/lib/api";
import { PIPELINE_STAGES } from "@/lib/pipeline";

interface PipelineProgressProps {
  pipelineStatus: PipelineStatusResponse | null;
  variant?: "full" | "compact";
  className?: string;
}

export function PipelineProgress({
  pipelineStatus,
  variant = "full",
  className = "",
}: PipelineProgressProps) {
  const [now, setNow] = useState(() => Date.now());

  const isRunning = pipelineStatus?.status === "running";
  const isCompleted = pipelineStatus?.status === "completed";
  const isFailed = pipelineStatus?.status === "failed";

  // Stopwatch ticks once a second only while the pipeline is actively running.
  useEffect(() => {
    if (!isRunning) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [isRunning]);

  const stages = PIPELINE_STAGES;
  const totalStages = pipelineStatus?.total_stages ?? stages.length;

  // Total elapsed is computed from the backend's own start/finish timestamps,
  // never accumulated client-side, so it survives reloads and stays truthful.
  let totalElapsed = 0;
  if (pipelineStatus?.started_at) {
    const startMs = new Date(pipelineStatus.started_at).getTime();
    if (pipelineStatus.finished_at) {
      totalElapsed = Math.max(
        0,
        (new Date(pipelineStatus.finished_at).getTime() - startMs) / 1000,
      );
    } else if (isRunning) {
      totalElapsed = Math.max(0, (now - startMs) / 1000);
    }
  } else if (pipelineStatus?.total_duration_seconds != null) {
    totalElapsed = pipelineStatus.total_duration_seconds;
  }

  let completedCount = 0;
  let activeIndex = -1;
  let activeElapsed = 0;

  stages.forEach((st, idx) => {
    const info = pipelineStatus?.stages?.[st.kind];
    const status = info?.status ?? "pending";
    if (status === "completed" || status === "done" || status === "skipped") {
      completedCount += 1;
    } else if (
      status === "running" ||
      pipelineStatus?.current_stage === st.kind
    ) {
      if (activeIndex === -1) {
        activeIndex = idx;
        const startedAt =
          pipelineStatus?.current_stage_started_at ?? info?.started_at;
        if (startedAt) {
          activeElapsed = Math.max(
            0,
            (now - new Date(startedAt).getTime()) / 1000,
          );
        }
      }
    }
  });

  const activeStage = activeIndex >= 0 ? stages[activeIndex] : null;
  const activeStageInfo = activeStage
    ? pipelineStatus?.stages?.[activeStage.kind]
    : null;
  const currentOperation =
    pipelineStatus?.current_operation ??
    activeStageInfo?.current_operation ??
    null;

  // Truth-based percentage: a running stage never inflates the bar.
  const percentage = isCompleted
    ? 100
    : pipelineStatus?.progress_percent != null
      ? pipelineStatus.progress_percent
      : Math.round((completedCount / totalStages) * 100);

  // Wired for a future backend ETA; nothing is rendered until it exists.
  const etaSeconds =
    (pipelineStatus as { eta_seconds?: number | null } | null)?.eta_seconds ??
    null;

  const barTone = isFailed
    ? "bg-danger"
    : isCompleted
      ? "bg-ok"
      : "bg-accent";

  /* ---------------------------------------------------------------- compact */
  if (variant === "compact") {
    return (
      <div
        className={`flex flex-col gap-2 rounded-panel border border-hairline bg-panel p-3 ${className}`}
      >
        <div className="flex items-center justify-between gap-2 text-xs">
          <div className="flex min-w-0 items-center gap-1.5 font-medium text-fg">
            <StateLamp
              running={isRunning}
              done={isCompleted}
              failed={isFailed}
            />
            <span className="truncate">
              {isRunning
                ? activeStage
                  ? `${activeIndex + 1}/${totalStages} · ${activeStage.label}`
                  : "Running…"
                : isCompleted
                  ? "Pipeline complete"
                  : isFailed
                    ? "Pipeline failed"
                    : "Ready"}
            </span>
          </div>
          <span
            className="shrink-0 font-mono text-[11px] font-semibold text-fg-muted"
            data-mono
          >
            {percentage}%
          </span>
        </div>

        <div className="h-1 w-full overflow-hidden rounded-full bg-panel-raised">
          <div
            className={`h-full transition-[width] duration-300 ease-out ${barTone}`}
            style={{ width: `${percentage}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-[11px] text-fg-subtle">
          <span className="font-mono" data-mono>
            {formatElapsed(totalElapsed)}
          </span>
          {isRunning && currentOperation && (
            <span
              className="truncate pl-2 text-right font-mono text-fg-muted"
              title={currentOperation}
              data-mono
            >
              {currentOperation}
            </span>
          )}
          {isCompleted && (
            <span className="font-medium text-ok">
              {formatDuration(totalElapsed)}
            </span>
          )}
        </div>
      </div>
    );
  }

  /* ------------------------------------------------------------------- full */
  return (
    <div
      className={`rounded-panel border bg-panel p-5 ${
        isRunning
          ? "border-accent-line"
          : isFailed
            ? "border-danger/40"
            : "border-hairline"
      } ${className}`}
    >
      {/* header: state + title + live timers */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <StateLamp running={isRunning} done={isCompleted} failed={isFailed} />
          <div>
            <h2 className="text-h4 font-semibold text-fg">
              {isRunning
                ? "Pipeline running"
                : isCompleted
                  ? "Pipeline complete"
                  : isFailed
                    ? "Pipeline failed"
                    : "Multi-agent pipeline"}
            </h2>
            <p className="mt-0.5 font-mono text-[11px] text-fg-subtle" data-mono>
              {isRunning && activeStage
                ? `stage ${activeIndex + 1}/${totalStages} · ${activeStage.label}${
                    currentOperation ? ` · ${currentOperation}` : ""
                  }`
                : isCompleted
                  ? `all ${totalStages} stages cached`
                  : isFailed
                    ? pipelineStatus?.error || "a stage failed during execution"
                    : `${totalStages} stages · run them in order`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Timer k="Elapsed" v={formatElapsed(totalElapsed)} />
          {isRunning && (
            <Timer
              k="This stage"
              v={formatElapsed(activeElapsed)}
              tone="warn"
            />
          )}
          {etaSeconds != null && (
            <Timer k="ETA" v={`~${formatElapsed(etaSeconds)}`} tone="accent" />
          )}
          {isCompleted && (
            <Timer
              k="Total"
              v={formatDuration(totalElapsed)}
              tone="ok"
            />
          )}
        </div>
      </div>

      {/* progress */}
      <div className="mt-4 space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-fg-subtle">
            {completedCount} of {totalStages} stages complete
          </span>
          <span
            className="font-mono text-sm font-semibold text-fg"
            data-mono
          >
            {percentage}%
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel-raised">
          <div
            className={`h-full rounded-full transition-[width] duration-300 ease-out ${barTone}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      {/* transport — one cell per stage, a signal chain */}
      <div className="mt-4 grid grid-cols-4 overflow-hidden rounded-control border border-hairline sm:grid-cols-7">
        {stages.map((st, idx) => {
          const info = pipelineStatus?.stages?.[st.kind];
          const status = info?.status ?? "pending";
          const isCurrent =
            isRunning &&
            (status === "running" || pipelineStatus?.current_stage === st.kind);
          const isDone =
            status === "completed" ||
            status === "done" ||
            status === "skipped";
          const isErr = status === "failed" || status === "error";
          const duration = info?.duration_seconds;

          return (
            <div
              key={st.kind}
              className={`relative flex min-w-0 flex-col gap-1.5 border-b border-r border-hairline/60 px-2.5 py-2.5 last:border-r-0 [&:nth-child(4n)]:border-r-0 sm:[&:nth-child(4n)]:border-r sm:[&:nth-child(7n)]:border-r-0 ${
                isCurrent ? "bg-warn-surface" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`h-2 w-2 rounded-[1px] ${
                    isDone
                      ? "bg-ok"
                      : isCurrent
                        ? "bg-warn animate-pulse"
                        : isErr
                          ? "bg-danger"
                          : "bg-fg-faint"
                  }`}
                />
                <span
                  className="font-mono text-[10px] text-fg-faint"
                  data-mono
                >
                  {idx + 1}
                </span>
              </div>
              <span
                className={`truncate text-[11px] font-semibold leading-tight ${
                  status === "pending" ? "text-fg-subtle" : "text-fg"
                }`}
                title={st.label}
              >
                {st.label}
              </span>
              <span
                className={`font-mono text-[10px] tabular-nums ${
                  isCurrent ? "text-warn" : "text-fg-subtle"
                }`}
                data-mono
              >
                {isCurrent
                  ? formatElapsed(activeElapsed)
                  : isDone && duration != null
                    ? `${duration.toFixed(1)}s`
                    : isDone
                      ? "done"
                      : isErr
                        ? "failed"
                        : "—"}
              </span>
              {isCurrent && (
                <span className="pointer-events-none absolute inset-x-0 bottom-0 h-[2px] overflow-hidden">
                  <span className="scan absolute inset-y-0 w-2/3 bg-warn" />
                </span>
              )}
            </div>
          );
        })}
      </div>

      <p className="mt-3 border-l-2 border-accent-line bg-accent-surface px-3 py-2 text-[12px] text-fg">
        Progress is <span className="font-semibold">truth-based</span> — a stage
        advances the bar only when the backend confirms it complete.
      </p>

      <style jsx>{`
        .scan {
          animation: scan 1.9s linear infinite;
        }
        @keyframes scan {
          0% {
            transform: translateX(-110%);
          }
          100% {
            transform: translateX(260%);
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .scan {
            animation: none;
            left: 0;
          }
        }
      `}</style>
    </div>
  );
}

function StateLamp({
  running,
  done,
  failed,
}: {
  running: boolean;
  done: boolean;
  failed: boolean;
}) {
  if (running) {
    return (
      <span className="relative flex h-2.5 w-2.5 shrink-0">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-warn opacity-70" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-warn" />
      </span>
    );
  }
  return (
    <span
      className={`h-2.5 w-2.5 shrink-0 rounded-full ${
        done ? "bg-ok" : failed ? "bg-danger" : "bg-fg-faint"
      }`}
    />
  );
}

function Timer({
  k,
  v,
  tone,
}: {
  k: string;
  v: string;
  tone?: "warn" | "accent" | "ok";
}) {
  const toneCls =
    tone === "warn"
      ? "border-warn/30 text-warn"
      : tone === "accent"
        ? "border-accent-line text-accent"
        : tone === "ok"
          ? "border-ok/30 text-ok"
          : "border-hairline text-fg";
  return (
    <div
      className={`flex items-baseline gap-1.5 rounded-control border bg-panel px-2.5 py-1 ${toneCls}`}
    >
      <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
        {k}
      </span>
      <span
        className="font-mono text-[13px] font-semibold tabular-nums"
        data-mono
      >
        {v}
      </span>
    </div>
  );
}
