import type { JobResult, Metrics } from "../types";
import { Card } from "./primitives";

const GATED_KEYS = [
  "flux_conservation_passed",
  "structure_guardrail_passed",
  "zero_synthesis_guarantee_held",
] as const;

export function SummaryCard({ metrics }: { metrics: Metrics | null }) {
  const rows: Array<[string, boolean | undefined, string]> = [
    ["Flux conservation", metrics?.flux_conservation_passed, "var(--ok)"],
    ["Structure (SSIM)", metrics?.structure_guardrail_passed, "var(--blue-soft)"],
    ["Zero-synthesis", metrics?.zero_synthesis_guarantee_held, "var(--violet)"],
    ["Trained weights", metrics?.trained_weights_loaded, "var(--pink)"],
  ];
  const drift = metrics?.flux_drift_coarsest_scale;

  return (
    <Card title="Summary">
      <div className="slist">
        {rows.map(([label, value, colour]) => (
          <div className="srow" key={label}>
            <span className="dot" style={{ background: colour }} />
            <span className="nm">{label}</span>
            <span
              className="ct"
              style={{ color: value ? "var(--ok)" : "var(--muted)" }}
            >
              {value === undefined ? "—" : value ? "PASS" : "NO"}
            </span>
          </div>
        ))}
      </div>
      {drift !== undefined && (
        <div className="footnote">
          <span>◆</span>
          <span>
            Coarse-scale flux drift{" "}
            <b style={{ color: "var(--text)" }}>{drift.toFixed(4)}</b> against a{" "}
            {metrics?.flux_conservation_tolerance} tolerance.
          </span>
        </div>
      )}
    </Card>
  );
}

export function ProvenanceCard({
  result,
  noteCount,
}: {
  result: JobResult | null;
  noteCount: number;
}) {
  const m = result?.metrics;
  const known = GATED_KEYS.filter((k) => m?.[k] !== undefined);
  const passed = known.filter((k) => m?.[k]).length;

  const rows: Array<[string, string]> = result
    ? [
        ["Sensor config", result.config || "—"],
        ["Guardrails passed", `${passed}/${known.length || "—"}`],
        ["Run conditions", String(noteCount)],
      ]
    : [
        ["Sensor config", "—"],
        ["Guardrails passed", "—"],
        ["Run conditions", "0"],
      ];

  return (
    <Card title="Provenance" className="accent" tools={<span className="round">↗</span>}>
      <p>
        {m?.radiance_units
          ? `Band 1 is in ${m.radiance_units}. The calibration constants behind it travel in the GeoTIFF header.`
          : "Every run records how it was produced — units, sensor config, and whether any trained weights were involved."}
      </p>
      <div className="stackcards">
        {rows.map(([k, v]) => (
          <div className="mini" key={k}>
            <span className="k">{k}</span>
            <span className="b" title={v}>
              {v}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

/** Notes that qualify how a result should be read get a distinct tag from
 *  purely informational ones, so a crop or a physics-only run is not skimmed
 *  past as neutral context. */
const QUALIFIER = /CROPPED|PHYSICS-ONLY|uniform|not gated/i;

export function NotesCard({ notes, error }: { notes: string[]; error: string | null }) {
  const items = [
    ...(error ? [{ tag: "err" as const, label: "Error", text: error }] : []),
    ...notes.map((n) => ({
      tag: QUALIFIER.test(n) ? ("warn" as const) : ("info" as const),
      label: QUALIFIER.test(n) ? "Qualifies this run" : "Context",
      text: n,
    })),
  ];
  return (
    <Card title="Run conditions">
      {items.length === 0 ? (
        <div className="empty">Nothing to report.</div>
      ) : (
        <div className="notes">
          {items.map((it, i) => (
            <div className="note" key={i}>
              <span className={`tag ${it.tag}`}>{it.label}</span>
              {it.text}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

export function CraterBubbles({ metrics }: { metrics: Metrics | null }) {
  const has = metrics?.craters_detected_enhanced !== undefined;
  const parts: Array<[string, number, string, string]> = [
    ["Matched", metrics?.craters_matched ?? 0, "var(--pink)", "#fff"],
    ["Revealed", metrics?.craters_revealed ?? 0, "var(--violet)", "#fff"],
    ["Lost", metrics?.craters_lost ?? 0, "#3a4150", "var(--text)"],
  ];
  const total = Math.max(1, parts.reduce((s, p) => s + p[1], 0));
  const layout = [
    [0, 26, 62],
    [52, 6, 46],
    [58, 62, 36],
  ];

  return (
    <Card title="Crater detection">
      <div className="bubbles">
        {!has ? (
          <div
            className="empty"
            style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}
          >
            No detections yet
          </div>
        ) : (
          parts.map(([label, value, bg, fg], i) => {
            const share = value / total;
            const size = layout[i][2] * (0.62 + 0.55 * Math.min(1, share * 1.8));
            return (
              <div
                className="bub"
                key={label}
                title={`${label}: ${value}`}
                style={{
                  left: `${layout[i][0]}%`,
                  top: `${layout[i][1]}%`,
                  width: `${size}%`,
                  aspectRatio: "1",
                  background: bg,
                  color: fg,
                }}
              >
                <b>{value}</b>
                <s>{label}</s>
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
}
