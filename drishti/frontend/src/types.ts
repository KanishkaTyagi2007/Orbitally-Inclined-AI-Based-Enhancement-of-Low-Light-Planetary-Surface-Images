/** Shapes returned by the Flask API in drishti/app.py. */

export type ImageKind = "raw" | "enhanced" | "trust" | "comparison";
export type JobState = "queued" | "running" | "done" | "error";
export type ViewId = "overview" | "pipeline" | "imagery" | "metrics" | "guardrails";

/**
 * The stage-6 metric report. Every field is optional: which metrics exist
 * depends on the config (`evaluation.metrics`), on which guardrails are gated,
 * and on whether optional backends such as `pyiqa` are installed. Reading a
 * missing metric as "absent" rather than "zero" is the whole point of the
 * optionality — a BRISQUE of `null` means no backend, not a perfect score.
 */
export interface Metrics {
  // 6.1 image quality
  psnr?: number;
  ssim?: number;
  niqe?: number;
  niqe_backend?: string;
  brisque?: number | null;
  entropy_gain?: number;
  entropy_raw?: number;
  entropy_enhanced?: number;
  histogram_raw?: number[];
  histogram_enhanced?: number[];
  histogram_bins?: number;

  // 5.1 flux conservation (gated)
  flux_conservation_passed?: boolean;
  flux_drift_coarsest_scale?: number;
  flux_conservation_tolerance?: number;
  flux_drift_per_scale?: Record<string, number>;
  flux_drift_after_tone_mapping?: Record<string, number>;

  // structure + synthesis (gated)
  structure_guardrail_passed?: boolean;
  structure_guardrail_threshold?: number;
  zero_synthesis_guarantee_held?: boolean;
  curve_map_abs_max?: number;
  all_guardrails_passed?: boolean;

  // 5.2 gradient consistency (reported, not gated by default)
  gradient_consistency_gated?: boolean;
  gradient_correlation?: number;
  gradient_baseline_correlation?: number;
  gradient_fidelity_vs_baseline?: number;
  gradient_smoothing_sigma?: number;
  gradient_correlation_threshold?: number;

  // 6.2 frozen crater detector
  craters_detected_raw?: number;
  craters_detected_enhanced?: number;
  craters_matched?: number;
  craters_revealed?: number;
  craters_lost?: number;
  crater_miou?: number;
  detection_gain?: number;
  revealed_crater_mean_trust?: number;
  revealed_crater_min_trust?: number;

  // The trained weights' own held-out audit, copied out of the checkpoint by
  // `AuraNetPipeline.load_checkpoint`. These describe the MODEL, measured on
  // products no training phase saw — not this scene.
  checkpoint_name?: string | null;
  checkpoint_held_out_tiles?: number;
  checkpoint_added_structure_fraction?: number | null;
  checkpoint_detail_reconstruction?: number | null;
  checkpoint_zero_synthesis_guarantee_held?: boolean;
  checkpoint_curve_map_abs_max?: number;
  checkpoint_log_var_spread?: number | null;
  checkpoint_trust_map_informative?: boolean;

  // 5.3 trust + provenance
  mean_trust?: number;
  low_trust_pixel_fraction?: number;
  trust_map_informative?: boolean;
  trained_weights_loaded?: boolean;
  cosmic_ray_hit_fraction?: number;
  saturated_pixel_fraction?: number;
  radiance_units?: string;
  source_format?: string;
  georeferenced?: boolean;
  input_shape?: number[];
  device?: string;
  runtime_seconds?: number;

  // stage 4 photometry
  lommel_seeliger_applied?: boolean;
  tone_mapping_applied?: boolean;
  disk_function_mean?: number;
  incidence_deg_applied?: number;
  emission_deg_applied?: number;
  incidence_source?: string;

  [key: string]: unknown;
}

/** One observational product as `pds4_bundle.Pds4Product.summary` reports it. */
export interface Pds4Summary {
  name: string;
  label_path: string;
  data_path: string | null;
  lines: number;
  samples: number;
  megapixels: number;
  data_type: string;
  processing_level: string;
  instrument: string;
  sensor: string;
  start_time: string;
  incidence_deg: number | null;
  emission_deg: number | null;
  line_exposure_ms: number | null;
  pixel_resolution_m: number | null;
  orbit_number: string | null;
  readable: boolean;
  warnings: string[];
}

/** What a path or upload resolved to, before anything is processed. */
export interface InspectResult {
  primary: string;
  product: Pds4Summary | null;
  candidates: Pds4Summary[];
  notes: string[];
  readable?: boolean;
  shape?: [number, number] | null;
}

export interface JobResult {
  metrics: Metrics;
  config: string;
  checkpoint: string | null;
  images: Partial<Record<ImageKind, string>>;
  downloads: {
    enhanced_geotiff?: string;
    trust_geotiff?: string | null;
    metrics_json?: string;
  };
}

export interface Job {
  job_id: string;
  name: string;
  state: JobState;
  percent: number;
  stage: string;
  elapsed: number;
  error: string | null;
  notes: string[];
  result: JobResult | null;
  source: InspectResult | null;
}

export interface ConfigList {
  configs: string[];
  default: string | null;
  checkpoints: string[];
}

export interface RunOptions {
  config: string;
  /** Checkpoint filename, or "" for a physics-only run. */
  checkpoint: string;
  maxDim: number;
  oversize: "crop" | "reject";
}

/** A run starts from a local path or from uploaded files, never both. */
export type RunSource = { path: string } | { files: File[] };
