"use client";

import { createContext, useContext } from "react";
import type { DatasetListEntry, PipelineKind } from "@/lib/api";

export interface DatasetShellState {
  datasetId: string;
  dataset: DatasetListEntry | null;
  present: PipelineKind[];
  missing: PipelineKind[];
  reportBuilt: boolean;
  loading: boolean;
  error: string | null;
}

export const DatasetShellContext = createContext<DatasetShellState | null>(
  null,
);

/** Read the enclosing `/datasets/[id]` shell's live state (dataset identity
 * + per-stage artifact status). Must be used from a page nested under
 * `app/datasets/[id]/layout.tsx`. */
export function useDatasetShell(): DatasetShellState {
  const ctx = useContext(DatasetShellContext);
  if (!ctx) {
    throw new Error(
      "useDatasetShell must be used within the /datasets/[id] layout.",
    );
  }
  return ctx;
}
