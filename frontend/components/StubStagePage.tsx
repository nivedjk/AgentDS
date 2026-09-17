"use client";

export function StubStagePage({ label }: { label: string }) {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-3 px-8 py-8">
      <h1 className="text-h2 font-semibold text-fg">{label}</h1>
      <p className="rounded-panel border border-dashed border-hairline px-5 py-10 text-center text-sm text-fg-subtle">
        Coming in phase 2.
      </p>
    </div>
  );
}
