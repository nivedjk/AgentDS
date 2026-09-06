"use client";

import { useDatasetShell } from "@/lib/dataset-context";
import { PIPELINE_STAGES } from "@/lib/pipeline";
import { StatusBadge } from "@/components/StatusBadge";

export default function DatasetOverviewPage() {
  const { datasetId, dataset, present, reportBuilt, loading } =
    useDatasetShell();

  const doneCount =
    present.length + (reportBuilt ? 1 : 0);

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold text-text">
          {dataset?.filename ?? (loading ? "Loading…" : "Dataset")}
        </h1>
        <p className="font-mono text-xs text-text-muted" data-mono>
          {datasetId}
        </p>
      </header>

      <section className="rounded-panel border border-line bg-surface p-5">
        <h2 className="mb-3 text-sm font-semibold text-text">
          Pipeline status
        </h2>
        <p className="mb-4 text-xs text-text-muted">
          {doneCount} of {PIPELINE_STAGES.length} stages complete.
        </p>
        <ul className="flex flex-col gap-1.5">
          {PIPELINE_STAGES.map((stage) => {
            const done =
              stage.kind === "report"
                ? reportBuilt
                : present.includes(stage.kind);
            return (
              <li
                key={stage.slug}
                className="flex items-center justify-between rounded-control border border-line px-3 py-2"
              >
                <span className="text-sm text-text-secondary">
                  {stage.label}
                </span>
                <StatusBadge variant={done ? "done" : "not-run"}>
                  {done ? "done" : "not run"}
                </StatusBadge>
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
