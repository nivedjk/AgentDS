const VARIANTS = {
  done: "border-ok/30 bg-ok-surface text-ok",
  "not-run": "border-hairline bg-transparent text-fg-subtle",
  running: "border-warn/30 bg-warn-surface text-warn",
  error: "border-danger/30 bg-danger-surface text-danger",
} as const;

const LAMP = {
  done: "bg-ok",
  "not-run": "bg-fg-faint",
  running: "bg-warn animate-pulse",
  error: "bg-danger",
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
      className={`inline-flex h-[18px] items-center gap-1.5 rounded-badge border px-1.5 font-mono text-[10px] font-medium uppercase tracking-wide ${VARIANTS[variant]}`}
      data-mono
    >
      <span
        className={`h-[5px] w-[5px] shrink-0 rounded-[1px] ${LAMP[variant]}`}
      />
      {children}
    </span>
  );
}
