import type { Metrics } from "../types";

/** Six pipeline stages mapped onto the percent values app.py reports at each
 *  stage boundary, so each pill fills while its own stage is running. */
export const STAGES: Array<[name: string, from: number, to: number]> = [
  ["Ingest", 5, 20],
  ["Decouple", 20, 35],
  ["Enhance", 35, 60],
  ["Photometry", 60, 78],
  ["Verify", 78, 94],
  ["Export", 94, 100],
];

export function StageBars({ percent }: { percent: number }) {
  return (
    <div className="stages">
      {STAGES.map(([name, from, to]) => {
        const f = Math.max(0, Math.min(1, (percent - from) / (to - from)));
        return (
          <div className="stage" key={name}>
            <b>{name}</b>
            <div
              className="track"
              role="progressbar"
              aria-label={name}
              aria-valuenow={Math.round(f * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <i style={{ width: `${f * 100}%` }} />
              <u>{Math.round(f * 100)}%</u>
            </div>
          </div>
        );
      })}
    </div>
  );
}

type DeltaKind = "up" | "dn" | "flat";
const Delta = ({ kind, text }: { kind: DeltaKind; text: string }) => (
  <span className={`delta ${kind}`}>{text}</span>
);

export function Kpis({ metrics }: { metrics: Metrics | null }) {
  if (!metrics) {
    return (
      <div className="kpis">
        {["Craters revealed", "Craters matched", "Detected total"].map((l) => (
          <div className="kpi" key={l}>
            <div className="n" style={{ color: "var(--dim)" }}>
              —
            </div>
            <div className="l">{l}</div>
          </div>
        ))}
      </div>
    );
  }

  const { detection_gain: gain, ssim, entropy_gain: dh } = metrics;
  const items: Array<[DeltaKind, string, number | string, string]> = [
    [
      gain === undefined ? "flat" : gain > 1 ? "up" : gain < 1 ? "dn" : "flat",
      gain === undefined ? "—" : `${gain >= 1 ? "+" : ""}${((gain - 1) * 100).toFixed(0)}%`,
      metrics.craters_revealed ?? "—",
      "Craters revealed",
    ],
    [
      ssim === undefined ? "flat" : ssim >= 0.6 ? "up" : "dn",
      ssim === undefined ? "—" : `SSIM ${ssim.toFixed(2)}`,
      metrics.craters_matched ?? "—",
      "Craters matched",
    ],
    [
      dh === undefined ? "flat" : dh > 0 ? "up" : dh < 0 ? "dn" : "flat",
      dh === undefined ? "—" : `${dh > 0 ? "+" : ""}${dh.toFixed(2)} bits`,
      metrics.craters_detected_enhanced ?? "—",
      "Detected total",
    ],
  ];

  return (
    <div className="kpis">
      {items.map(([kind, badge, n, label]) => (
        <div className="kpi" key={label}>
          <div className="n">
            <Delta kind={kind} text={badge} />
            {n}
          </div>
          <div className="l">{label}</div>
        </div>
      ))}
    </div>
  );
}
