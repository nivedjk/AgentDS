import type { PipelineKind } from "@/lib/api";

/** The 7 real pipeline stages, in order, for the dataset sidebar nav.
 * "report" deliberately has no matching `PipelineKind` — it isn't one of
 * the backend's upstream artifact kinds (it's assembled FROM them), so its
 * status is derived separately via `getReportBuilt()`. There is no 8th
 * "whatif" entry here — Module 8 does not exist. */
export interface PipelineStage {
  slug: string;
  kind: PipelineKind | "report";
  label: string;
}

export const PIPELINE_STAGES: PipelineStage[] = [
  { slug: "understanding", kind: "understanding", label: "Data Understanding" },
  { slug: "cleaning", kind: "cleaning", label: "Cleaning" },
  { slug: "visualization", kind: "visualization", label: "Visualization" },
  { slug: "recommendation", kind: "recommendation", label: "Model Recommendation" },
  { slug: "training", kind: "training", label: "Training & Evaluation" },
  { slug: "explainability", kind: "explainability", label: "Explainability" },
  { slug: "report", kind: "report", label: "Report" },
];
