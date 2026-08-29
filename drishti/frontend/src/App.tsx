import { useCallback, useEffect, useRef, useState } from "react";
import { getJob, listConfigs, startRun } from "./api";
import type { Job, RunOptions, RunSource, ViewId } from "./types";
import { Kpis, StageBars } from "./components/Hero";
import { HistogramChart } from "./components/HistogramChart";
import {
  CraterBubbles,
  NotesCard,
  ProvenanceCard,
  SummaryCard,
} from "./components/OverviewCards";
import { PipelineFlow } from "./components/PipelineFlow";
import { UploadCard } from "./components/UploadCard";
import { GuardrailsView, ImageryView, MetricsView } from "./components/Views";

const VIEWS: Array<[ViewId, string]> = [
  ["overview", "Overview"],
  ["pipeline", "Pipeline"],
  ["imagery", "Imagery"],
  ["metrics", "Metrics"],
  ["guardrails", "Guardrails"],
];

const POLL_MS = 600;

export default function App() {
  const [view, setView] = useState<ViewId>("overview");
  const [configs, setConfigs] = useState<string[]>([]);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [options, setOptions] = useState<RunOptions>({
    config: "default_config.yaml",
    checkpoint: "",
    maxDim: 1536,
    oversize: "crop",
  });
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    listConfigs()
      .then((d) => {
        setConfigs(d.configs);
        setCheckpoints(d.checkpoints ?? []);
        setOptions((o) => ({
          ...o,
          config: d.default ?? o.config,
          // Default to the newest trained checkpoint when one exists. A
          // physics-only run is the honest fallback, not the preferred mode.
          checkpoint: d.checkpoints?.length ? d.checkpoints[d.checkpoints.length - 1] : "",
        }));
      })
      .catch((e: Error) => setError(`Could not reach the API: ${e.message}`));
  }, []);

  // Poll while a job is in flight. Cleared on unmount so a hot reload during
  // development cannot leave an orphaned interval hammering the server.
  useEffect(() => {
    if (!jobId || !busy) return;
    const id = window.setInterval(async () => {
      try {
        const j = await getJob(jobId);
        setJob(j);
        if (j.state === "done" || j.state === "error") {
          setBusy(false);
          if (j.state === "error") setError(j.error);
        }
      } catch {
        /* transient fetch failure: keep polling rather than killing the run */
      }
    }, POLL_MS);
    timer.current = id;
    return () => window.clearInterval(id);
  }, [jobId, busy]);

  const run = useCallback(
    async (source: RunSource) => {
      setError(null);
      setJob(null);
      setBusy(true);
      try {
        const { job_id } = await startRun(source, options);
        setJobId(job_id);
      } catch (e) {
        setBusy(false);
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [options],
  );

  const result = job?.state === "done" ? job.result : null;
  const metrics = result?.metrics ?? null;
  const percent = job?.percent ?? 0;
  const notes = job?.notes ?? [];

  const status = error
    ? "Failed"
    : busy && job
      ? `${job.stage} · ${job.elapsed.toFixed(1)}s`
      : busy
        ? "Uploading…"
        : job?.state === "done"
          ? `Done in ${job.elapsed.toFixed(1)}s`
          : "Idle";

  const title = job?.name
    ? job.state === "done"
      ? "Enhanced, "
      : "Enhancing, "
    : "Enhance a scene, ";

  return (
    <div className="shell">
      <div className="topbar">
        <div className="brand">
          AURA<span>-NET</span>
        </div>
        <nav>
          <div className="navpills" role="tablist">
            {VIEWS.map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={view === id}
                onClick={() => setView(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </nav>
        <div className="icons">
          <div className="devbadge" title="Compute device">
            {(metrics?.device ?? "cpu").toUpperCase().slice(0, 4)}
          </div>
        </div>
      </div>

      <h1>
        {title}
        <em>{job?.name ?? "drop a file to begin"}</em>
      </h1>

      <div className="herorow">
        <StageBars percent={percent} />
        <Kpis metrics={metrics} />
      </div>

      {view === "overview" && (
        <>
          <div className="grid g-a">
            <SummaryCard metrics={metrics} />
            <HistogramChart metrics={metrics} />
            <ProvenanceCard result={result} noteCount={notes.length} />
          </div>
          <div className="grid g-b">
            <NotesCard notes={notes} error={error} />
            <CraterBubbles metrics={metrics} />
            <UploadCard
              configs={configs}
              checkpoints={checkpoints}
              options={options}
              onOptions={setOptions}
              onRun={run}
              busy={busy}
              status={status}
            />
          </div>
        </>
      )}

      {view === "pipeline" && (
        <div className="grid g-flow">
          <PipelineFlow metrics={metrics} />
        </div>
      )}

      {view === "imagery" && <ImageryView jobId={jobId} result={result} />}
      {view === "metrics" && <MetricsView jobId={jobId} result={result} />}
      {view === "guardrails" && <GuardrailsView result={result} />}
    </div>
  );
}
