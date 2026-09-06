"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useParams } from "next/navigation";
import {
  ApiError,
  getArtifactsStatus,
  getReportBuilt,
  listDatasets,
  type DatasetListEntry,
  type PipelineKind,
} from "@/lib/api";
import { PIPELINE_STAGES } from "@/lib/pipeline";
import { DatasetShellContext, type DatasetShellState } from "@/lib/dataset-context";
import { StatusBadge } from "@/components/StatusBadge";

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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [datasets, artifacts, built] = await Promise.all([
          listDatasets(),
          getArtifactsStatus(datasetId),
          getReportBuilt(datasetId),
        ]);
        if (cancelled) return;
        setDataset(
          datasets.find((d) => d.dataset_id === datasetId) ?? null,
        );
        setPresent(artifacts.present);
        setMissing(artifacts.missing);
        setReportBuilt(built);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.detail : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const shellState: DatasetShellState = {
    datasetId,
    dataset,
    present,
    missing,
    reportBuilt,
    loading,
    error,
  };

  return (
    <DatasetShellContext.Provider value={shellState}>
      <div className="flex min-h-screen">
        <aside className="flex w-sidebar shrink-0 flex-col border-r border-line bg-surface">
          <div className="flex h-12 items-center border-b border-line px-4">
            <Link href="/" className="text-sm font-semibold text-text">
              AgentDS
            </Link>
          </div>

          <div className="border-b border-line px-4 py-3">
            <p className="truncate text-sm font-medium text-text">
              {dataset?.filename ?? (loading ? "Loading…" : "Unknown dataset")}
            </p>
            <p
              className="mt-1 truncate font-mono text-[11px] text-text-muted"
              data-mono
            >
              {datasetId}
            </p>
          </div>

          <nav className="flex flex-1 flex-col gap-0.5 p-2">
            {PIPELINE_STAGES.map((stage) => {
              const href = `/datasets/${datasetId}/${stage.slug}`;
              const active = pathname === href;
              const done =
                stage.kind === "report"
                  ? reportBuilt
                  : present.includes(stage.kind);
              return (
                <Link
                  key={stage.slug}
                  href={href}
                  className={`flex items-center justify-between gap-2 rounded-control px-3 py-2 text-sm transition-colors ${
                    active
                      ? "bg-surface-2 text-text"
                      : "text-text-secondary hover:bg-surface-2 hover:text-text"
                  }`}
                >
                  <span>{stage.label}</span>
                  <StatusBadge variant={done ? "done" : "not-run"}>
                    {done ? "done" : "not run"}
                  </StatusBadge>
                </Link>
              );
            })}
          </nav>

          {error && (
            <div className="border-t border-line p-3">
              <p className="rounded-control border border-error/30 bg-error/10 px-2 py-1.5 text-[11px] text-error">
                {error}
              </p>
            </div>
          )}
        </aside>

        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </DatasetShellContext.Provider>
  );
}
