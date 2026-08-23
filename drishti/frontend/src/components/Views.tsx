import { useEffect, useState } from "react";
import { previewUrl, productUrl } from "../api";
import type { ImageKind, JobResult } from "../types";
import { Card, StatCard, pct, type Row } from "./primitives";

const ORDER: ImageKind[] = ["enhanced", "raw", "trust", "comparison"];
const LABEL: Record<ImageKind, string> = {
  enhanced: "Enhanced",
  raw: "Raw",
  trust: "Trust",
  comparison: "Side by side",
};

export function ImageryView({ jobId, result }: { jobId: string | null; result: JobResult | null }) {
  const kinds = ORDER.filter((k) => result?.images?.[k]);
  const [active, setActive] = useState<ImageKind>("enhanced");

  useEffect(() => {
    if (kinds.length && !kinds.includes(active)) setActive(kinds[0]);
  }, [kinds, active]);

  if (!jobId || kinds.length === 0) {
    return (
      <Card title="Imagery">
        <div className="empty">Run a scene to see its imagery.</div>
      </Card>
    );
  }

  return (
    <Card
      title="Imagery"
      tools={kinds.map((k) => (
        <button
          key={k}
          className="pillbtn"
          aria-selected={active === k}
          onClick={() => setActive(k)}
        >
          {LABEL[k]}
        </button>
      ))}
    >
      <div className="viewer">
        <img src={previewUrl(jobId, result!.images[active]!)} alt={`${LABEL[active]} preview`} />
      </div>
      <div className="footnote">
        Previews are display renderings, each independently contrast-stretched.
        Brightness is not proportional to radiance — never use them for photometry.
      </div>
    </Card>
  );
}

export function MetricsView({ jobId, result }: { jobId: string | null; result: JobResult | null }) {
  if (!result) return <Card title="Metrics"><div className="empty">No run yet.</div></Card>;
  const m = result.metrics;

  const quality: Row[] = [
    ["PSNR vs raw", m.psnr],
    ["SSIM vs raw", m.ssim],
    [`NIQE (${m.niqe_backend ?? "n/a"})`, m.niqe],
    ["BRISQUE", m.brisque],
    ["Entropy gain ΔH", m.entropy_gain],
    ["Entropy raw", m.entropy_raw],
    ["Entropy enhanced", m.entropy_enhanced],
  ];
  const craters: Row[] = [
    ["Detected raw", m.craters_detected_raw],
    ["Detected enhanced", m.craters_detected_enhanced],
    ["Matched", m.craters_matched],
    ["Revealed", m.craters_revealed],
    ["Lost", m.craters_lost],
    ["mIoU", m.crater_miou],
    ["Detection gain", m.detection_gain],
    ["Revealed mean trust", m.revealed_crater_mean_trust],
  ];
  const provenance: Row[] = [
    ["Mean trust", m.mean_trust],
    ["Low-trust fraction", pct(m.low_trust_pixel_fraction)],
    ["Trust map informative", m.trust_map_informative],
    ["Trained weights", m.trained_weights_loaded],
    ["Cosmic-ray hits", pct(m.cosmic_ray_hit_fraction)],
    ["Saturated", pct(m.saturated_pixel_fraction)],
    ["Radiance units", m.radiance_units],
    ["Source format", m.source_format],
    ["Georeferenced", m.georeferenced],
    ["Size", m.input_shape ? [...m.input_shape].reverse().join(" × ") : undefined],
    ["Device", m.device],
    ["Runtime (s)", m.runtime_seconds],
    ["Config", result.config],
  ];

  const names: Record<string, string> = {
    enhanced_geotiff: "Enhanced GeoTIFF (32-bit, 2-band)",
    trust_geotiff: "Trust map GeoTIFF",
    metrics_json: "Metrics JSON",
  };
  const links = Object.entries(result.downloads).filter(([, v]) => Boolean(v)) as Array<
    [string, string]
  >;

  return (
    <>
      <div className="mgrid">
        <StatCard title="Image quality" rows={quality} />
        <StatCard title="Crater detection" rows={craters} />
        <StatCard title="Trust & provenance" rows={provenance} />
      </div>
      <div style={{ marginTop: 16 }}>
        <Card title="Products">
          {links.length === 0 || !jobId ? (
            <div className="empty">No products.</div>
          ) : (
            <div className="dls">
              {links.map(([k, v]) => (
                <a key={k} href={productUrl(jobId, v)} download>
                  {names[k] ?? k}
                </a>
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

export function GuardrailsView({ result }: { result: JobResult | null }) {
  if (!result) return <Card title="Guardrails"><div className="empty">No run yet.</div></Card>;
  const m = result.metrics;

  const flux: Row[] = [
    ["Passed", m.flux_conservation_passed],
    ["Drift (coarsest scale)", m.flux_drift_coarsest_scale],
    ["Tolerance", m.flux_conservation_tolerance],
    ...Object.entries(m.flux_drift_per_scale ?? {}).map(
      ([k, v]) => [`Drift @ ${k}px`, v] as Row,
    ),
  ];
  const structure: Row[] = [
    ["SSIM gate passed", m.structure_guardrail_passed],
    ["SSIM threshold", m.structure_guardrail_threshold],
    ["SSIM", m.ssim],
    ["Zero-synthesis held", m.zero_synthesis_guarantee_held],
    ["Curve |A| max", m.curve_map_abs_max],
    ["All gated guardrails", m.all_guardrails_passed],
  ];
  const gradient: Row[] = [
    ["Gated", m.gradient_consistency_gated],
    ["Correlation (Spearman)", m.gradient_correlation],
    ["Baseline correlation", m.gradient_baseline_correlation],
    ["Fidelity vs baseline", m.gradient_fidelity_vs_baseline],
    ["Smoothing sigma", m.gradient_smoothing_sigma],
    ["Threshold (if gated)", m.gradient_correlation_threshold],
  ];
  const photometry: Row[] = [
    ["Lommel-Seeliger applied", m.lommel_seeliger_applied],
    ["Disk function mean", m.disk_function_mean],
    ["Incidence applied (deg)", m.incidence_deg_applied],
    ["Emission applied (deg)", m.emission_deg_applied],
    ["Incidence source", m.incidence_source],
    ["Tone mapping applied", m.tone_mapping_applied],
  ];

  return (
    <div className="mgrid">
      <StatCard title="Gated — flux conservation" rows={flux} />
      <StatCard title="Gated — structure & synthesis" rows={structure} />
      <StatCard title="Reported — gradient consistency" rows={gradient} />
      <StatCard title="Photometry" rows={photometry} />
    </div>
  );
}
