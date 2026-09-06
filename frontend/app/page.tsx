"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  listDatasets,
  uploadDataset,
  type DatasetListEntry,
} from "@/lib/api";

function formatTimestamp(iso: string | null): string {
  if (!iso) return "unknown time";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export default function HomePage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [datasets, setDatasets] = useState<DatasetListEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await listDatasets();
      setDatasets(list);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.detail : String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleFileChosen(file: File) {
    setUploadError(null);
    setUploading(true);
    try {
      const result = await uploadDataset(file);
      router.push(`/datasets/${result.dataset_id}`);
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.detail : String(err));
      setUploading(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col gap-8 px-6 py-10">
      <header className="flex flex-col gap-1 border-b border-line pb-6">
        <h1 className="text-lg font-semibold tracking-tight text-text">
          AgentDS
        </h1>
        <p className="text-sm text-text-muted">
          Upload a CSV to start a new pipeline run, or open an existing
          dataset below.
        </p>
      </header>

      <section className="flex flex-col gap-3 rounded-panel border border-line bg-surface p-5">
        <h2 className="text-sm font-semibold text-text">Upload dataset</h2>
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
            className="h-7 rounded-control border border-line bg-surface-2 px-3 text-xs font-medium text-text-secondary transition-colors hover:bg-line disabled:cursor-not-allowed disabled:opacity-50"
          >
            {uploading ? "Uploading…" : "Choose CSV file"}
          </button>
          <span className="text-xs text-text-muted">.csv only</span>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) handleFileChosen(file);
          }}
        />
        {uploadError && (
          <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
            {uploadError}
          </p>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold text-text">Datasets</h2>

        {loadError && (
          <p className="rounded-control border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
            Failed to load datasets: {loadError}
          </p>
        )}

        {!loadError && datasets === null && (
          <p className="text-sm text-text-muted">Loading…</p>
        )}

        {!loadError && datasets !== null && datasets.length === 0 && (
          <div className="rounded-panel border border-dashed border-line px-5 py-10 text-center text-sm text-text-muted">
            No datasets yet. Upload a CSV above to get started.
          </div>
        )}

        {!loadError && datasets !== null && datasets.length > 0 && (
          <div className="overflow-hidden rounded-panel border border-line">
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="h-[26px] bg-surface text-[10px] uppercase tracking-wide text-text-muted">
                  <th className="border-b border-line px-3 font-medium">
                    Filename
                  </th>
                  <th className="border-b border-line px-3 font-medium">
                    Dataset ID
                  </th>
                  <th className="border-b border-line px-3 font-medium">
                    Uploaded
                  </th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds) => (
                  <tr
                    key={ds.dataset_id}
                    className="h-8 cursor-pointer border-b border-line last:border-b-0 hover:bg-surface"
                    onClick={() => router.push(`/datasets/${ds.dataset_id}`)}
                  >
                    <td className="px-3 text-text">{ds.filename}</td>
                    <td
                      className="px-3 font-mono text-xs text-text-muted"
                      data-mono
                    >
                      {ds.dataset_id}
                    </td>
                    <td className="px-3 text-xs text-text-muted">
                      {formatTimestamp(ds.uploaded_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
