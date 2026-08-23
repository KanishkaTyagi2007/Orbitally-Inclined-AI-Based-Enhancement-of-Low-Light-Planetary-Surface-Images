import type { ReactNode } from "react";

/** Formatting and small shared pieces used across every card. */

export function fmt(v: unknown): ReactNode {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean")
    return <span className={`chip ${v ? "ok" : "bad"}`}>{v ? "YES" : "NO"}</span>;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    if (Number.isInteger(v)) return String(v);
    return Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)
      ? v.toExponential(3)
      : v.toFixed(4);
  }
  return String(v);
}

export const pct = (v: number | undefined) =>
  v === undefined || v === null ? undefined : `${(v * 100).toFixed(3)} %`;

export type Row = [label: string, value: unknown];

/** A titled card of key/value rows. Rows whose value is absent are dropped, so
 *  a metric that did not run leaves no misleading blank. */
export function StatCard({ title, rows }: { title: string; rows: Row[] }) {
  const present = rows.filter(([, v]) => v !== undefined && v !== null);
  return (
    <section className="card">
      <h2>{title}</h2>
      {present.length === 0 ? (
        <div className="empty">No data.</div>
      ) : (
        present.map(([k, v]) => (
          <div className="krow" key={k}>
            <span className="k">{k}</span>
            <span className="v">{fmt(v)}</span>
          </div>
        ))
      )}
    </section>
  );
}

export function Card({
  title,
  tools,
  className = "",
  children,
}: {
  title: string;
  tools?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`card ${className}`}>
      <h2>
        {title}
        {tools ? <span className="tools">{tools}</span> : null}
      </h2>
      {children}
    </section>
  );
}
