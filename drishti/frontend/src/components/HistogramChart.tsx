import { useMemo, useRef, useState } from "react";
import type { Metrics } from "../types";
import { Card } from "./primitives";

type Scale = "linear" | "log";
const W = 700;
const H = 220;
const PAD = { l: 34, r: 8, t: 10, b: 22 };

/**
 * Raw vs enhanced tonal distribution, from the 64-bin histograms the metric
 * harness records. This is the chart that shows *how* the histogram moved,
 * which the single-number entropy delta cannot.
 *
 * Drawn as inline SVG rather than pulling in a charting library: two polylines
 * and a crosshair do not justify the dependency, and this keeps the bundle
 * free of anything that would need a CDN at runtime.
 */
export function HistogramChart({ metrics }: { metrics: Metrics | null }) {
  const [scale, setScale] = useState<Scale>("linear");
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const raw = metrics?.histogram_raw;
  const enh = metrics?.histogram_enhanced;

  const model = useMemo(() => {
    if (!raw || !enh || raw.length === 0) return null;
    const tf = (v: number) => (scale === "log" ? Math.log10(1 + v) : v);
    const series = [raw.map(tf), enh.map(tf)];
    const max = Math.max(1, ...series[0], ...series[1]);
    const n = raw.length;
    const X = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
    const Y = (v: number) => H - PAD.b - (v / max) * (H - PAD.t - PAD.b);
    return { series, max, n, X, Y };
  }, [raw, enh, scale]);

  const tools = (
    <>
      {(["linear", "log"] as Scale[]).map((s) => (
        <button
          key={s}
          className="pillbtn"
          aria-selected={scale === s}
          onClick={() => setScale(s)}
        >
          {s === "linear" ? "Linear" : "Log"}
        </button>
      ))}
    </>
  );

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!model || !svgRef.current) return;
    const r = svgRef.current.getBoundingClientRect();
    const frac = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((frac - PAD.l) / (W - PAD.l - PAD.r)) * (model.n - 1));
    setHover(Math.max(0, Math.min(model.n - 1, i)));
  }

  return (
    <Card title="Tonal distribution" tools={tools}>
      <div className="chartwrap">
        <svg
          className="chart"
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          role="img"
          aria-label="Raw versus enhanced intensity histogram"
        >
          {!model ? (
            <text x={W / 2} y={H / 2} fill="#5c6577" fontSize={13} textAnchor="middle">
              Run a scene to see its tonal distribution
            </text>
          ) : (
            <>
              {[0, 1, 2, 3].map((k) => {
                const y = PAD.t + (k * (H - PAD.t - PAD.b)) / 3;
                return (
                  <g key={k}>
                    <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="#212734" />
                    <text x={PAD.l - 7} y={y + 4} fill="#5c6577" fontSize={10} textAnchor="end">
                      {Math.round(model.max * (1 - k / 3))}
                    </text>
                  </g>
                );
              })}
              {["var(--pink)", "var(--blue-soft)"].map((col, si) => (
                <polyline
                  key={col}
                  points={model.series[si].map((v, i) => `${model.X(i)},${model.Y(v)}`).join(" ")}
                  fill="none"
                  stroke={col}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              ))}
              {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                <text
                  key={f}
                  x={model.X(f * (model.n - 1))}
                  y={H - 5}
                  fill="#5c6577"
                  fontSize={10}
                  textAnchor="middle"
                >
                  {f.toFixed(2)}
                </text>
              ))}
              {hover !== null && (
                <line
                  x1={model.X(hover)}
                  y1={PAD.t}
                  x2={model.X(hover)}
                  y2={H - PAD.b}
                  stroke="#8a94a6"
                  strokeDasharray="3 3"
                />
              )}
            </>
          )}
        </svg>
        {model && hover !== null && raw && enh && (
          <div
            className="tip"
            style={{
              opacity: 1,
              left: `${(model.X(hover) / W) * 100}%`,
              top: `${(model.Y(Math.max(model.series[0][hover], model.series[1][hover])) / H) * 100}%`,
            }}
          >
            {(hover / (model.n - 1)).toFixed(2)} · raw {raw[hover]} / enh {enh[hover]}
          </div>
        )}
      </div>
      <div className="legend">
        <span>
          <i style={{ background: "var(--pink)" }} />
          Raw input
        </span>
        <span>
          <i style={{ background: "var(--blue-soft)" }} />
          Enhanced
        </span>
        {model && <span style={{ color: "var(--dim)" }}>{model.n} bins · normalized intensity</span>}
      </div>
    </Card>
  );
}
