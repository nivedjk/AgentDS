"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import {
  ApiError,
  getArtifactsStatus,
  getPipelineStatus,
  getReportBuilt,
  listDatasets,
  startPipelineRun,
  subscribePipelineEvents,
  type DatasetListEntry,
  type PipelineKind,
  type PipelineStatusResponse,
} from "@/lib/api";
import { PIPELINE_STAGES } from "@/lib/pipeline";
import { DatasetShellContext, type DatasetShellState } from "@/lib/dataset-context";
import { StatusBadge } from "@/components/StatusBadge";
import { PipelineProgress } from "@/components/PipelineProgress";

export default function DatasetLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;
  const pathname = usePathname();

  const [dataset, setDataset] = useState<DatasetListEntry | null>(null);
  const [present, setPresent] = useState<PipelineKind[]>([]);
  const [missing, setMissing] = useState<PipelineKind[]>([]);
  const [reportBuilt, setReportBuilt] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshArtifacts = useCallback(async () => {
    try {
      const [datasets, artifacts, built, pStatus] = await Promise.all([
        listDatasets(),
        getArtifactsStatus(datasetId),
        getReportBuilt(datasetId),
        getPipelineStatus(datasetId).catch(() => null),
      ]);
      setDataset(datasets.find((d) => d.dataset_id === datasetId) ?? null);
      setPresent(artifacts.present);
      setMissing(artifacts.missing);
      setReportBuilt(built);
      if (pStatus) setPipelineStatus(pStatus);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    }
  }, [datasetId]);

  const triggerPipeline = useCallback(async () => {
    try {
      setError(null);
      const res = await startPipelineRun(datasetId);
      setPipelineStatus(res);
      await refreshArtifacts();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    }
  }, [datasetId, refreshArtifacts]);

  useEffect(() => {
    let cancelled = false;

    async function initialLoad() {
      setLoading(true);
      setError(null);
      await refreshArtifacts();
      if (!cancelled) setLoading(false);
    }

    initialLoad();

    return () => {
      cancelled = true;
    };
  }, [refreshArtifacts]);

  // Real-time SSE event-driven streaming with automatic state reconstruction
  useEffect(() => {
    const unsubscribe = subscribePipelineEvents(datasetId, (updatedStatus) => {
      setPipelineStatus(updatedStatus);
      if (
        updatedStatus.status === "completed" ||
        updatedStatus.completed_stages_count !== pipelineStatus?.completed_stages_count
      ) {
        Promise.all([
          getArtifactsStatus(datasetId),
          getReportBuilt(datasetId),
        ])
          .then(([artifacts, built]) => {
            setPresent(artifacts.present);
            setMissing(artifacts.missing);
            setReportBuilt(built);
          })
          .catch(() => {});
      }
    });

    return () => {
      unsubscribe();
    };
  }, [datasetId, pipelineStatus?.completed_stages_count]);

  // Fallback polling loop while the background pipeline is running
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    const isRunning = pipelineStatus?.status === "running";

    if (isRunning) {
      timer = setInterval(async () => {
        try {
          const [artifacts, built, pStatus] = await Promise.all([
            getArtifactsStatus(datasetId),
            getReportBuilt(datasetId),
            getPipelineStatus(datasetId),
          ]);
          setPresent(artifacts.present);
          setMissing(artifacts.missing);
          setReportBuilt(built);
          setPipelineStatus(pStatus);
        } catch {
          // ignore transient poll error
        }
      }, 1500);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [datasetId, pipelineStatus?.status]);

  const shellState: DatasetShellState = {
    datasetId,
    dataset,
    present,
    missing,
    reportBuilt,
    pipelineStatus,
    loading,
    error,
    refreshArtifacts,
    triggerPipeline,
  };

  const isOverviewActive = pathname === `/datasets/${datasetId}`;

  return (
    <DatasetShellContext.Provider value={shellState}>
      <div className="flex min-h-screen">
        <aside className="flex w-sidebar shrink-0 flex-col border-r border-hairline bg-panel">
          <div className="flex h-12 items-center justify-between border-b border-hairline px-4">
            <Link href="/" className="text-sm font-semibold text-fg hover:text-accent transition-colors">
              AgentDS
            </Link>
            <Link
              href="/"
              className="text-[11px] text-fg-subtle hover:text-fg transition-colors"
              title="Return to all datasets"
            >
              ← Datasets
            </Link>
          </div>

          <Link
            href={`/datasets/${datasetId}`}
            className={`border-b border-hairline px-4 py-3 transition-colors block ${
              isOverviewActive ? "bg-panel-raised/60" : "hover:bg-panel-raised/40"
            }`}
            title="Click to view Dataset Landing Page & Full Pipeline Progress"
          >
            <p className="truncate text-sm font-medium text-fg">
              {dataset?.filename ?? (loading ? "Loading…" : "Unknown dataset")}
            </p>
            <p
              className="mt-1 truncate font-mono text-[11px] text-fg-subtle"
              data-mono
            >
              {datasetId}
            </p>
          </Link>

          {/* Compact Pipeline Progress widget — Clickable anytime to jump to Landing Page */}
          <div className="border-b border-hairline p-2">
            <Link
              href={`/datasets/${datasetId}`}
              className="block hover:opacity-90 transition-opacity"
              title="Click to view full pipeline progress & overview"
            >
              <PipelineProgress pipelineStatus={pipelineStatus} variant="compact" />
            </Link>
          </div>

          <nav className="flex flex-1 flex-col gap-0.5 p-2 overflow-y-auto">
            {/* Primary Overview & Pipeline Landing Page Link */}
            <Link
              href={`/datasets/${datasetId}`}
              className={`flex items-center justify-between gap-2 rounded-control px-3 py-2 text-sm transition-colors ${
                isOverviewActive
                  ? "bg-panel-raised text-fg font-semibold shadow-sm"
                  : "text-fg-muted hover:bg-panel-raised hover:text-fg"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-sm">📊</span>
                <span>Overview & Pipeline</span>
              </div>
              {pipelineStatus?.status === "running" && (
                <span className="h-2 w-2 rounded-full bg-accent animate-ping" />
              )}
              {pipelineStatus?.status === "completed" && (
                <span className="text-[11px] font-medium text-ok">✓ 100%</span>
              )}
            </Link>

            <div className="my-1.5 border-t border-hairline" />

            <div className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-fg-subtle">
              Pipeline Stages
            </div>

            {PIPELINE_STAGES.map((stage) => {
              const href = `/datasets/${datasetId}/${stage.slug}`;
              const active = pathname === href;
              
              const stageStatus = pipelineStatus?.stages[stage.kind]?.status;
              const isPresent =
                stage.kind === "report"
                  ? reportBuilt
                  : present.includes(stage.kind);

              let badgeVariant: "done" | "not-run" | "running" | "error" = "not-run";
              let badgeText = "not run";

              if (isPresent || stageStatus === "completed" || stageStatus === "done" || stageStatus === "skipped") {
                badgeVariant = "done";
                badgeText = stageStatus === "skipped" ? "skipped" : "done";
              } else if (stageStatus === "running" || pipelineStatus?.current_stage === stage.kind) {
                badgeVariant = "running";
                badgeText = "running";
              } else if (stageStatus === "failed" || stageStatus === "error") {
                badgeVariant = "error";
                badgeText = "failed";
              }

              return (
                <Link
                  key={stage.slug}
                  href={href}
                  className={`flex items-center justify-between gap-2 rounded-control px-3 py-2 text-sm transition-colors ${
                    active
                      ? "bg-panel-raised text-fg font-medium"
                      : "text-fg-muted hover:bg-panel-raised hover:text-fg"
                  }`}
                >
                  <span className="truncate">{stage.label}</span>
                  <StatusBadge variant={badgeVariant}>
                    {badgeText}
                  </StatusBadge>
                </Link>
              );
            })}
          </nav>

          {error && (
            <div className="border-t border-hairline p-3">
              <p className="rounded-control border border-danger/30 bg-danger/10 px-2 py-1.5 text-[11px] text-danger">
                {error}
              </p>
            </div>
          )}
        </aside>

        <main className="flex-1 overflow-y-auto flex flex-col">
          {/* Top Breadcrumbs & Quick Landing Link Header */}
          <div className="flex h-12 items-center justify-between border-b border-hairline bg-panel/50 px-8 backdrop-blur text-xs shrink-0">
            <div className="flex items-center gap-2 text-fg-subtle">
              <Link href="/" className="hover:text-fg transition-colors">
                Datasets
              </Link>
              <span>/</span>
              <Link
                href={`/datasets/${datasetId}`}
                className={`hover:text-fg transition-colors font-medium ${
                  isOverviewActive ? "text-fg font-semibold" : ""
                }`}
              >
                {dataset?.filename ?? datasetId}
              </Link>
              {!isOverviewActive && (
                <>
                  <span>/</span>
                  <span className="text-fg font-semibold">
                    {PIPELINE_STAGES.find((s) => pathname.endsWith(s.slug))?.label ?? "Stage"}
                  </span>
                </>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Link
                href={`/datasets/${datasetId}`}
                className={`flex items-center gap-1.5 rounded-control px-3 py-1.5 text-xs transition-colors ${
                  isOverviewActive
                    ? "bg-panel-raised text-fg font-semibold border border-hairline"
                    : "text-fg-muted hover:bg-panel-raised hover:text-fg border border-hairline/60"
                }`}
              >
                <span>📊</span>
                <span>Dataset Overview & Pipeline</span>
              </Link>
            </div>
          </div>

          <div className="flex-1">{children}</div>
        </main>
      </div>
    </DatasetShellContext.Provider>
  );
}
