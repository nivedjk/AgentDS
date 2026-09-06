"use client";

export function StubStagePage({ label }: { label: string }) {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-3 px-8 py-8">
      <h1 className="text-lg font-semibold text-text">{label}</h1>
      <p className="rounded-panel border border-dashed border-line px-5 py-10 text-center text-sm text-text-muted">
        Coming in phase 2.
      </p>
    </div>
  );
}
