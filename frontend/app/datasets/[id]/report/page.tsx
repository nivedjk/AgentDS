"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ApiError,
  getReport,
  runReport,
  reportDocxUrl,
  reportPdfUrl,
  type FinalReport,
} from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function ReportPage() {
  const params = useParams<{ id: string }>();
  const datasetId = params.id;

  const [report, setReport] = useState<FinalReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [buildError, setBuildError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setLoadError(null);
      try {
        const result = await getReport(datasetId);
        if (!cancelled) setReport(result);
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof ApiError ? err.detail : String(err));
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

  async function handleBuild() {
    setBuilding(true);
    setBuildError(null);
    try {
      const result = await runReport(datasetId);
      setReport(result);
    } catch (err) {
      setBuildError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setBuilding(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-8 py-8">
      <header className="flex flex-col gap-1 border-b border-line pb-4">
        <h1 className="text-lg font-semibold text-text">Report</h1>
        <p className="text-sm text-text-muted">
          The single assembled Markdown document over every module that has
          run for this dataset. Building (or rebuilding) requires an API key
          server-side; viewing a cached report does not.
        </p>
      </header>

      <div>
        <button
          type="button"
          disabled={building}
          onClick={handleBuild}
          className="h-8 rounded-control border border-line bg-surface-2 px-4 text-sm font-medium text-text-secondary transition-colors hover:bg-line disabled:cursor-not-allowed disabled:opacity-50"
        >
          {building
            ? "Building…"
            : report
              ? "Rebuild Report"
              : "Build Report"}
        </button>
      </div>

      {loading && !report && (
        <p className="text-sm text-text-muted">Checking for a cached report…</p>
      )}

      {loadError && (
        <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {loadError}
        </p>
      )}

      {buildError && (
        <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {buildError}
        </p>
      )}

      {!loading && !report && !loadError && (
        <p className="rounded-panel border border-dashed border-line px-4 py-6 text-center text-sm text-text-muted">
          No report has been generated for this dataset yet. Click &quot;Build
          Report&quot; above.
        </p>
      )}

      {report && (
        <>
          {/* Status strip */}
          <section className="flex flex-wrap items-center gap-4 rounded-panel border border-line bg-surface p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Generated at
              </span>
              <span className="font-mono text-xs text-text" data-mono>
                {report.generated_at}
              </span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Modules present
              </span>
              <div className="flex flex-wrap gap-1">
                {report.modules_present.length === 0 && (
                  <span className="text-xs text-text-muted">none</span>
                )}
                {report.modules_present.map((m) => (
                  <StatusBadge key={m} variant="done">
                    {m}
                  </StatusBadge>
                ))}
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-text-muted">
                Modules missing
              </span>
              <div className="flex flex-wrap gap-1">
                {report.modules_missing.length === 0 && (
                  <span className="text-xs text-text-muted">none</span>
                )}
                {report.modules_missing.map((m) => (
                  <StatusBadge key={m} variant="not-run">
                    {m}
                  </StatusBadge>
                ))}
              </div>
            </div>
          </section>

          {/* Downloads */}
          <section className="flex items-center gap-3">
            <a
              href={reportPdfUrl(datasetId)}
              className="h-8 rounded-control border border-line bg-surface-2 px-4 text-sm font-medium text-text-secondary transition-colors hover:bg-line inline-flex items-center"
            >
              Download PDF
            </a>
            <a
              href={reportDocxUrl(datasetId)}
              className="h-8 rounded-control border border-line bg-surface-2 px-4 text-sm font-medium text-text-secondary transition-colors hover:bg-line inline-flex items-center"
            >
              Download DOCX
            </a>
          </section>

          {/* Rendered markdown document */}
          <section className="prose-doc rounded-panel border border-line bg-surface p-6">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {report.markdown}
            </ReactMarkdown>
          </section>
        </>
      )}
    </div>
  );
}
