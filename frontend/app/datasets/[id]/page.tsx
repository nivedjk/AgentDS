"use client";

import Link from "next/link";
import { useDatasetShell } from "@/lib/dataset-context";
import { PIPELINE_STAGES } from "@/lib/pipeline";
import { StatusBadge } from "@/components/StatusBadge";
import { PipelineProgress } from "@/components/PipelineProgress";

export default function DatasetOverviewPage() {
  const { datasetId, dataset, present, reportBuilt, pipelineStatus, loading, triggerPipeline } =
    useDatasetShell();

  const doneCount = present.length + (reportBuilt ? 1 : 0);
  const isRunning = pipelineStatus?.status === "running";

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-h2 font-semibold text-fg">
            {dataset?.filename ?? (loading ? "Loading…" : "Dataset")}
          </h1>
          <p className="font-mono text-xs text-fg-subtle" data-mono>
            {datasetId}
          </p>
        </div>
        <button
          type="button"
          disabled={isRunning}
          onClick={() => triggerPipeline()}
          className="flex h-8 shrink-0 items-center gap-2 rounded-control border border-accent-line bg-accent-surface px-3.5 text-xs font-semibold text-fg transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isRunning ? (
            <>
              <span className="h-1.5 w-1.5 animate-ping rounded-full bg-accent" />
              <span>Running…</span>
            </>
          ) : (
            <span>Run full pipeline</span>
          )}
        </button>
      </header>

      <PipelineProgress pipelineStatus={pipelineStatus} variant="full" />

      <section className="rounded-panel border border-hairline bg-panel p-5">
        <div className="mb-3 flex items-baseline justify-between gap-3">
          <div>
            <h2 className="text-h4 font-semibold text-fg">Stages</h2>
            <p className="text-xs text-fg-subtle">
              {doneCount} of {PIPELINE_STAGES.length} artifacts ready
            </p>
          </div>
          {pipelineStatus?.total_duration_seconds != null && (
            <span className="font-mono text-xs text-fg-subtle" data-mono>
              {pipelineStatus.total_duration_seconds.toFixed(1)}s total
            </span>
          )}
        </div>

        <ul className="flex flex-col gap-1">
          {PIPELINE_STAGES.map((stage, idx) => {
            const href = `/datasets/${datasetId}/${stage.slug}`;
            const stageStatus = pipelineStatus?.stages[stage.kind]?.status;
            const stageDuration = pipelineStatus?.stages[stage.kind]?.duration_seconds;
            const isPresent =
              stage.kind === "report" ? reportBuilt : present.includes(stage.kind);

            let badgeVariant: "done" | "not-run" | "running" | "error" = "not-run";
            let badgeText = "not run";

            if (
              isPresent ||
              stageStatus === "completed" ||
              stageStatus === "done" ||
              stageStatus === "skipped"
            ) {
              badgeVariant = "done";
              badgeText =
                stageStatus === "skipped"
                  ? "skipped"
                  : stageDuration != null
                    ? `done · ${stageDuration.toFixed(1)}s`
                    : "done";
            } else if (
              stageStatus === "running" ||
              pipelineStatus?.current_stage === stage.kind
            ) {
              badgeVariant = "running";
              badgeText = "running";
            } else if (stageStatus === "failed" || stageStatus === "error") {
              badgeVariant = "error";
              badgeText = "failed";
            }

            return (
              <li key={stage.slug}>
                <Link
                  href={href}
                  className="group flex items-center justify-between rounded-control border border-transparent px-3 py-2.5 transition-colors hover:border-hairline hover:bg-panel-raised"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className="w-4 text-right font-mono text-xs text-fg-faint"
                      data-mono
                    >
                      {idx + 1}
                    </span>
                    <span className="text-sm font-medium text-fg-muted group-hover:text-fg">
                      {stage.label}
                    </span>
                  </div>
                  <StatusBadge variant={badgeVariant}>{badgeText}</StatusBadge>
                </Link>
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
