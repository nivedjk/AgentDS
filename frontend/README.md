# AgentDS — Frontend

A Next.js dashboard that walks a dataset through AgentDS's seven-agent pipeline:
upload a CSV, then step through Data Understanding → Cleaning → Visualization →
Model Recommendation → Training → Explainability → Report, with each stage's
status and output rendered as you go.

See the repo root [`README.md`](../README.md) for the project overview and the
[`backend/README.md`](../backend/README.md) for the API.

## Prerequisite

The backend must be running on <http://localhost:8000> — its CORS is pinned to
`http://localhost:3000`, so the frontend only works on that origin during
development.

## Setup

```bash
cd agentds/frontend

npm install
npm run dev
```

Open <http://localhost:3000>.

## Environment

`.env.local`:

| Var | Default | Purpose |
|-----|---------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Backend base URL, consumed in `lib/api.ts` |

## Routes

| Path | Screen |
|------|--------|
| `/` | Upload a CSV + list existing datasets |
| `/datasets/[id]` | Pipeline overview — per-stage status, links onward |
| `/datasets/[id]/understanding` | Module 1 — Data Understanding report |
| `/datasets/[id]/cleaning` | Module 2 — Cleaning step log |
| `/datasets/[id]/visualization` | Module 3 — Plotly charts + insights |
| `/datasets/[id]/recommendation` | Module 4 — Model shortlist + rationale |
| `/datasets/[id]/training` | Module 5 — Leaderboard + CV / test metrics |
| `/datasets/[id]/explainability` | Module 6 — SHAP feature importance |
| `/datasets/[id]/report` | Module 7 — Final report + PDF / Word download |

Stage order and labels are defined once in `lib/pipeline.ts`.

## Stack

- **Next.js 16** (App Router) + **React 19**
- **Tailwind CSS v4** (`@import "tailwindcss"` in `app/globals.css`)
- **Plotly** via `react-plotly.js` for the visualization stage
- **`react-markdown` + `remark-gfm`** for the rendered final report

## Design tokens

Dark, GitHub-flavored palette defined in `app/globals.css` as CSS custom
properties (`--raw-canvas`, `--raw-surface`, `--raw-info`, `--raw-success`, …)
and exposed to Tailwind via `@theme inline`. Fonts: Inter (UI) and JetBrains
Mono (data / telemetry), loaded through `next/font` in `app/layout.tsx`.

## Scripts

| Command | Does |
|---------|------|
| `npm run dev` | Start the dev server on `:3000` |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint |
