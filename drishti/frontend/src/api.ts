import type { ConfigList, Job, RunOptions } from "./types";

/** Thin typed client over the Flask API. Same-origin in production; proxied
 *  through Vite's dev server on 5173. */

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

export function startRun(files: File[], opts: RunOptions): Promise<{ job_id: string }> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  fd.append("config", opts.config);
  fd.append("max_dim", String(opts.maxDim));
  fd.append("oversize", opts.oversize);
  return fetch("/api/run", { method: "POST", body: fd }).then((r) =>
    json<{ job_id: string }>(r),
  );
}

export function getJob(jobId: string): Promise<Job> {
  return fetch(`/api/jobs/${jobId}`).then((r) => json<Job>(r));
}

export const previewUrl = (jobId: string, filename: string) =>
  `/api/jobs/${jobId}/preview/${encodeURIComponent(filename)}`;

export const productUrl = (jobId: string, filename: string) =>
  `/api/jobs/${jobId}/product/${encodeURIComponent(filename)}`;
