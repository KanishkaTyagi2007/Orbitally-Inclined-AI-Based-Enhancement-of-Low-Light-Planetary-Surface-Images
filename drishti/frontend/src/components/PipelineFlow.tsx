import { Card } from "./primitives";
import type { Metrics } from "../types";

/**
 * Stage-by-stage flowchart of what the pipeline actually does to a scene.
 *
 * The diagram earns its place by drawing the one thing prose keeps burying:
 * *which stages have learned weights and which do not*. Half the eight stages
 * are fixed physics -- calibration constants and published photometric laws,
 * with nothing to fit -- and the argument for trusting the output rests on
 * that, so it is the visual organising principle here. Fill distinguishes
 * physics from learned; a badge marks a stage that is constrained beyond its
 * weights, whether by architecture (2.1's half-bin bound, 3A's monotonic
 * curve -- both hold for trained, untrained and adversarial weights alike) or
 * by an explicit gate on the output (stage 5).
 *
 * When a run has finished, the boxes pick up its live values -- the measured
 * curve bound, the flux verdict, the trust mean -- so the chart doubles as a
 * per-scene provenance trace instead of staying a static decoration.
 */

const W = 780;

type Kind = "io" | "physics" | "learned";

interface NodeSpec {
  x: number;
  y: number;
  w: number;
  h: number;
  kind: Kind;
  stage?: string;
  title: string;
  detail: string;
  /** Guarantee that holds for any weights whatsoever. */
  badge?: string;
  tooltip: string;
}

const FILL: Record<Kind, string> = {
  io: "var(--inner)",
  physics: "rgba(47,107,255,.14)",
  learned: "rgba(79,70,229,.22)",
};
const STROKE: Record<Kind, string> = {
  io: "var(--line)",
  physics: "var(--blue)",
  learned: "var(--violet)",
};

function buildNodes(m: Metrics | null): NodeSpec[] {
  const curve = m?.curve_map_abs_max;
  const trust = m?.mean_trust;
  const flux = m?.flux_conservation_passed;
  const units = m?.radiance_units;

  return [
    {
      x: 190, y: 6, w: 400, h: 48, kind: "io",
      title: "RAW SCENE",
      detail: "PDS4 · FITS · GeoTIFF — 16-bit linear DN + geometry",
      tooltip:
        "A PDS4 product is a detached .xml label plus a headerless .img binary. " +
        "The label is what gets opened; it carries the shape, dtype and the " +
        "solar incidence angle stage 4 needs.",
    },
    {
      x: 150, y: 74, w: 480, h: 56, kind: "physics", stage: "1",
      title: "Ingestion & physics front-end",
      detail: "DN → radiance · cosmic-ray scrub · Anscombe VST",
      tooltip:
        "1.1 read + preserve CRS/affine. 1.2 DN → spectral radiance from the " +
        "sensor's gain, bias and exposure. 1.3 L.A.Cosmic edge rejection of SEU " +
        "spikes. 1.4 Anscombe VST maps Poisson-Gaussian noise onto unit-variance " +
        "AWGN so every later stage sees one noise level." +
        (units ? `\n\nThis run: ${units}` : ""),
    },
    {
      x: 150, y: 158, w: 480, h: 56, kind: "learned", stage: "2",
      title: "De-quantization & wavelet decoupling",
      detail: "sub-bin offset field · stationary wavelet transform",
      badge: "≤ ½ bin",
      tooltip:
        "2.1 an implicit field predicts where inside its quantizer bin each " +
        "pixel really sat, undoing contour banding. The offset passes through " +
        "tanh scaled by step/2, so it cannot move a pixel outside its own bin — " +
        "for trained, untrained or adversarial weights alike.\n\n" +
        "2.2 an undecimated (translation-invariant) SWT splits the scene into a " +
        "low-frequency LL field and LH/HL/HH detail bands.",
    },
    {
      x: 24, y: 244, w: 348, h: 68, kind: "learned", stage: "3A",
      title: "Illumination curve (LL)",
      detail: "Zero-DCE++ · multi-order recurrent curves",
      badge: "monotonic",
      tooltip:
        "The network never outputs pixels — it outputs curve parameters A, and " +
        "the only operation applied is LE(x) = x + A·x·(1−x). |A| < 1 makes " +
        "every application strictly increasing, so two pixels of equal input " +
        "radiance always get equal output radiance: brightness can be remapped, " +
        "spatial structure cannot be created." +
        (curve !== undefined ? `\n\nThis run: max|A| = ${curve.toFixed(4)}` : ""),
    },
    {
      x: 408, y: 244, w: 348, h: 68, kind: "learned", stage: "3B",
      title: "Detail restoration (LH/HL/HH)",
      detail: "differentiable PSF deconvolution · NAFNet",
      tooltip:
        "Deconvolution first — it inverts a known physical process, the optics — " +
        "then an activation-free NAFNet subtracts the noise deconvolution " +
        "amplified.\n\nThis is the one unconstrained network in the pipeline, so " +
        "it is the one the training objectives police: its target is the " +
        "measured tile, and it is charged for gradient energy it did not " +
        "receive and for altering an un-degraded tile at all.",
    },
    {
      x: 190, y: 340, w: 400, h: 46, kind: "physics", stage: "3C",
      title: "Frequency recombination",
      detail: "inverse SWT → full-spectrum radiance",
      tooltip: "Inverse stationary wavelet transform, then the inverse VST " +
        "returns the scene to linear radiance — the domain stage 4's photometry " +
        "is defined in.",
    },
    {
      x: 150, y: 408, w: 480, h: 56, kind: "physics", stage: "4",
      title: "Range compression & photometry",
      detail: "bilateral-guided log tone map · Lommel-Seeliger",
      tooltip:
        "4.1 compresses only the bilateral *base* layer, so shadow detail lifts " +
        "without haloing at crater rims. 4.2 divides out the illumination " +
        "geometry D(i,e) = cos i / (cos i + cos e), the standard single-scattering " +
        "model for dark regolith.\n\nThe incidence angle comes from the product's " +
        "own PDS4 label — across these scenes it ranges from 39° to 83°, so a " +
        "shared constant would be wrong for nearly all of them.",
    },
    {
      x: 150, y: 492, w: 480, h: 56, kind: "learned", stage: "5",
      title: "Verification & uncertainty",
      detail: "flux conservation · gradient consistency · trust map",
      badge: "gates",
      tooltip:
        "5.1 checks that mean radiance is conserved across scales — a stage that " +
        "invented structure would not conserve flux. 5.2 correlates Sobel " +
        "magnitudes against the raw scene. 5.3 a heteroscedastic head emits mean " +
        "radiance and a per-pixel log-variance, which becomes the trust map." +
        (flux !== undefined ? `\n\nThis run: flux ${flux ? "PASSED" : "FAILED"}` : "") +
        (trust !== undefined ? `, mean trust ${trust.toFixed(3)}` : ""),
    },
    {
      x: 150, y: 576, w: 480, h: 56, kind: "physics", stage: "6",
      title: "Metrics & lossless export",
      detail: "NIQE · SSIM · frozen crater detector · GeoTIFF",
      tooltip:
        "The crater detector is frozen and identical for the raw and enhanced " +
        "scenes, so a crater it finds only in the enhanced one is a real change " +
        "in detectability, not a retuned threshold. Export is 32-bit float with " +
        "the CRS and affine transform intact.",
    },
    {
      x: 190, y: 660, w: 400, h: 48, kind: "io",
      title: "ENHANCED PRODUCT",
      detail: "band 1 radiance · band 2 trust map · geometry preserved",
      tooltip:
        "Every enhanced pixel ships beside a confidence value, so a reader can " +
        "separate recovered signal from anything the model was unsure of.",
    },
  ];
}

function Node({ n }: { n: NodeSpec }) {
  const cx = n.x + n.w / 2;
  // Title and detail straddle the box's own centre line, so one placement rule
  // covers every node height instead of each kind needing its own offset.
  const twoLine = n.h >= 46;
  const titleY = twoLine ? n.y + n.h / 2 - 9 : n.y + n.h / 2;
  return (
    <g className="flownode">
      <title>{`${n.stage ? `Stage ${n.stage} — ` : ""}${n.title}\n\n${n.tooltip}`}</title>
      <rect
        x={n.x} y={n.y} width={n.w} height={n.h} rx={11}
        fill={FILL[n.kind]} stroke={STROKE[n.kind]} strokeWidth={1.4}
      />
      {n.stage && (
        <text x={n.x + 15} y={n.y + n.h / 2} className="flowstage">
          {n.stage}
        </text>
      )}
      <text x={cx} y={titleY} className={n.kind === "io" ? "flowio" : "flowtitle"}>
        {n.title}
      </text>
      {twoLine && (
        <text x={cx} y={n.y + n.h / 2 + 12} className="flowdetail">
          {n.detail}
        </text>
      )}
      {n.badge && (
        <>
          <rect
            x={n.x + n.w - 14 - n.badge.length * 7.2} y={n.y + 8}
            width={n.badge.length * 7.2 + 4} height={17} rx={8.5}
            fill="rgba(55,211,153,.16)" stroke="var(--ok)" strokeWidth={0.9}
          />
          <text
            x={n.x + n.w - 12 - (n.badge.length * 7.2) / 2} y={n.y + 20}
            className="flowbadge"
          >
            {n.badge}
          </text>
        </>
      )}
    </g>
  );
}

/** Straight vertical connector between two stacked nodes. */
function Down({ from, to, x = W / 2 }: { from: NodeSpec; to: NodeSpec; x?: number }) {
  return (
    <line
      x1={x} y1={from.y + from.h} x2={x} y2={to.y - 7}
      stroke="var(--line)" strokeWidth={1.6} markerEnd="url(#flowarrow)"
    />
  );
}

/** Elbow from a splitting node down into a side branch, or back out of one. */
function Elbow({ y1, y2, x1, x2 }: { y1: number; y2: number; x1: number; x2: number }) {
  const mid = (y1 + y2) / 2;
  return (
    <path
      d={`M ${x1} ${y1} V ${mid} H ${x2} V ${y2 - 7}`}
      fill="none" stroke="var(--line)" strokeWidth={1.6}
      markerEnd="url(#flowarrow)"
    />
  );
}

export function PipelineFlow({ metrics }: { metrics: Metrics | null }) {
  const n = buildNodes(metrics);
  const [input, s1, s2, s3a, s3b, s3c, s4, s5, s6, out] = n;
  const learned = metrics?.trained_weights_loaded;

  return (
    <Card
      title="Implementation flow"
      className="flowcard"
      tools={
        <span className="flowlegend">
          <i className="phys" /> physics
          <i className="lrn" /> learned
          <i className="grt" /> constrained
        </span>
      }
    >
      <svg className="flow" viewBox={`0 0 ${W} 712`} role="img"
           aria-label="AURA-NET pipeline stages, from raw PDS4 scene to enhanced product with trust map">
        <defs>
          <marker id="flowarrow" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--dim)" />
          </marker>
        </defs>

        <Down from={input} to={s1} />
        <Down from={s1} to={s2} />
        <Elbow y1={s2.y + s2.h} y2={s3a.y} x1={W / 2} x2={s3a.x + s3a.w / 2} />
        <Elbow y1={s2.y + s2.h} y2={s3b.y} x1={W / 2} x2={s3b.x + s3b.w / 2} />
        <Elbow y1={s3a.y + s3a.h} y2={s3c.y} x1={s3a.x + s3a.w / 2} x2={W / 2} />
        <Elbow y1={s3b.y + s3b.h} y2={s3c.y} x1={s3b.x + s3b.w / 2} x2={W / 2} />
        <Down from={s3c} to={s4} />
        <Down from={s4} to={s5} />
        <Down from={s5} to={s6} />
        <Down from={s6} to={out} />

        {n.map((node) => (
          <Node key={node.title} n={node} />
        ))}
      </svg>

      <div className="footnote">
        <span>
          {learned === undefined
            ? "Four of the eight stages carry trainable weights; the other four are fixed physics with nothing to fit. The badged constraints hold whether or not a checkpoint is loaded."
            : learned
              ? "Trained weights are loaded, so stages 2.1, 3A, 3B and 5.3 are active. The badged constraints bound them regardless of what those weights are."
              : "No checkpoint: the learned stages are identity pass-throughs, so this run is physics only and the trust map carries just the physical masks."}
        </span>
      </div>
    </Card>
  );
}
