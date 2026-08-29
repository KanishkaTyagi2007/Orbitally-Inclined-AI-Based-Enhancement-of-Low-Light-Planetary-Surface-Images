import type { ConfigList, InspectResult, Job, RunOptions, RunSource } from "./types";

/** Thin typed client over the Flask API. Same-origin in production; proxied
 *  through Vite's dev server on 5173. */

/**
 * The server refuses any state-changing or filesystem-reading request without
 * this header. A page on another origin can forge a plain form POST to
 * localhost, but setting a custom header forces a CORS preflight that this
 * server never answers -- so the header is what keeps a stray browser tab from
 * driving the dashboard into reading local paths and rendering them back.
 */
const CLIENT_HEADERS = { "X-Aura-Client": "dashboard" };

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
  }
  return body as T;
}

export function listConfigs(): Promise<ConfigList> {
  return fetch("/api/configs").then((r) => json<ConfigList>(r));
}

export function inspectPath(path: string): Promise<InspectResult> {
  return fetch("/api/inspect", {
    method: "POST",
    headers: { ...CLIENT_HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  }).then((r) => json<InspectResult>(r));
}

export function startRun(
  source: RunSource,
  opts: RunOptions,
): Promise<{ job_id: string }> {
  const fd = new FormData();
  if ("path" in source) {
    fd.append("path", source.path);
  } else {
    // `webkitRelativePath` is set for a folder pick and empty otherwise. Sending
    // it as the part name preserves the bundle's tree on the server, which is
    // what lets a PDS4 label still resolve the .img it references.
    source.files.forEach((f) => fd.append("files", f, f.webkitRelativePath || f.name));
  }
  fd.append("config", opts.config);
  fd.append("checkpoint", opts.checkpoint);
  fd.append("max_dim", String(opts.maxDim));
  fd.append("oversize", opts.oversize);
  return fetch("/api/run", { method: "POST", headers: CLIENT_HEADERS, body: fd }).then(
    (r) => json<{ job_id: string }>(r),
  );
}

export function getJob(jobId: string): Promise<Job> {
  return fetch(`/api/jobs/${jobId}`).then((r) => json<Job>(r));
}

export const previewUrl = (jobId: string, filename: string) =>
  `/api/jobs/${jobId}/preview/${encodeURIComponent(filename)}`;

export const productUrl = (jobId: string, filename: string) =>
  `/api/jobs/${jobId}/product/${encodeURIComponent(filename)}`;
