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

const MODULE_LABELS: Record<string, string> = {
  understanding: "Data Understanding",
  cleaning: "Cleaning",
  visualization: "Visualizations",
  recommendation: "Model Recommendation",
  training: "Training & Evaluation",
  explainability: "Explainability",
  whatif: "What-If Analysis",
  report: "Report",
};

function label(kind: string) {
  return MODULE_LABELS[kind] ?? kind.charAt(0).toUpperCase() + kind.slice(1);
}

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
      <header className="flex flex-col gap-1 border-b border-hairline pb-4">
        <h1 className="text-h2 font-semibold text-fg">Report</h1>
        <p className="max-w-[70ch] text-sm text-fg-muted">
          One assembled document over every module that has run for this dataset.
          Every figure is copied verbatim from a module sidecar — the LLM writes
          only the surrounding prose. Building needs a provider; viewing a cached
          report does not.
        </p>
      </header>

      <div>
        <button
          type="button"
          disabled={building}
          onClick={handleBuild}
          className="h-8 rounded-control border border-hairline bg-panel-raised px-4 text-sm font-medium text-fg-muted transition-colors hover:border-hairline-strong hover:text-fg disabled:cursor-not-allowed disabled:opacity-50"
        >
          {building ? "Building…" : report ? "Rebuild report" : "Build report"}
        </button>
      </div>

      {loading && !report && (
        <p className="text-sm text-fg-subtle">Checking for a cached report…</p>
      )}

      {loadError && (
        <p className="rounded-control border border-danger/30 bg-danger-surface px-3 py-2 text-xs text-danger">
          {loadError}
        </p>
      )}

      {buildError && (
        <p className="rounded-control border border-danger/30 bg-danger-surface px-3 py-2 text-xs text-danger">
          {buildError}
        </p>
      )}

      {!loading && !report && !loadError && (
        <p className="rounded-panel border border-dashed border-hairline px-4 py-6 text-center text-sm text-fg-subtle">
          No report has been generated for this dataset yet. Click “Build report”
          above.
        </p>
      )}

      {report && (
        <>
          <section className="flex flex-wrap items-center justify-between gap-4 rounded-panel border border-hairline bg-panel p-4">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] uppercase tracking-wide text-fg-subtle">
                Generated
              </span>
              <span className="font-mono text-xs text-fg" data-mono>
                {report.generated_at}
              </span>
            </div>
            <div className="flex gap-2">
              <a
                href={reportPdfUrl(datasetId)}
                className="inline-flex h-8 items-center rounded-control border border-accent-line bg-accent-surface px-4 text-sm font-semibold text-fg transition-colors hover:border-accent"
              >
                Download PDF
              </a>
              <a
                href={reportDocxUrl(datasetId)}
                className="inline-flex h-8 items-center rounded-control border border-hairline bg-panel-raised px-4 text-sm font-medium text-fg-muted transition-colors hover:border-hairline-strong hover:text-fg"
              >
                Download Word
              </a>
            </div>
          </section>

          <div className="md:grid md:grid-cols-[180px_1fr] md:gap-6">
            {/* Section rail — status of every module the report covers */}
            <nav className="mb-4 self-start overflow-hidden rounded-panel border border-hairline md:sticky md:top-6 md:mb-0">
              {[...report.modules_present.map((m) => [m, true] as const),
                ...report.modules_missing.map((m) => [m, false] as const)].map(
                ([kind, present]) => (
                  <span
                    key={kind}
                    className={`flex items-center gap-2 border-b border-hairline/60 px-3 py-2 text-[11.5px] last:border-b-0 ${
                      present ? "text-fg-muted" : "text-fg-faint"
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-[1px] ${
                        present ? "bg-ok" : "bg-fg-faint"
                      }`}
                    />
                    <span className="truncate">{label(kind)}</span>
                  </span>
                ),
              )}
            </nav>

            <section className="prose-doc rounded-panel border border-hairline bg-panel p-6">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {report.markdown}
              </ReactMarkdown>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
