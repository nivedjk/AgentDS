const VARIANTS = {
  done: "bg-success/10 text-success border-success/30",
  "not-run": "bg-transparent text-text-muted border-line",
  error: "bg-error/10 text-error border-error/30",
} as const;

export type StatusBadgeVariant = keyof typeof VARIANTS;

export function StatusBadge({
  variant,
  children,
}: {
  variant: StatusBadgeVariant;
  children: React.ReactNode;
}) {
  return (
    <span
      className={`inline-flex h-[18px] items-center rounded-badge border px-1.5 font-mono text-[10px] font-medium uppercase tracking-wide ${VARIANTS[variant]}`}
      data-mono
    >
      {children}
    </span>
  );
}
