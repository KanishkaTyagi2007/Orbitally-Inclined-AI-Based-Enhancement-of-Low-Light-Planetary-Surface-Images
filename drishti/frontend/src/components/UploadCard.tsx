import { useRef, useState } from "react";
import type { InspectResult, RunOptions, RunSource } from "../types";
import { inspectPath } from "../api";
import { Card } from "./primitives";

const ACCEPT =
  ".tif,.tiff,.img,.xml,.lbl,.fits,.fit,.fts,.png,.jpg,.jpeg,.zip";

/**
 * Two ways in, because a Chandrayaan-2 product is ~2 GB.
 *
 * "Local folder" is the one that matters: an extracted ISSDC bundle is a
 * directory tree whose science raster is a headerless .img beside its .xml
 * label, and pushing that through a browser upload is neither fast nor
 * necessary for a tool that runs on the same machine as the file. Pasting the
 * path lets the pipeline read the label in place, at full size.
 *
 * "Upload" stays for everything else -- a GeoTIFF, a bundle ZIP, or a
 * hand-picked label/raster pair.
 *
 * Inspecting before running is deliberate. A full-size product takes minutes,
 * and discovering only afterwards that the bundle resolved to the raw fore
 * sensor instead of the calibrated nadir strip is an expensive way to find out.
 */
export function UploadCard({
  configs,
  checkpoints,
  options,
  onOptions,
  onRun,
  busy,
  status,
}: {
  configs: string[];
  checkpoints: string[];
  options: RunOptions;
  onOptions: (o: RunOptions) => void;
  onRun: (source: RunSource) => void;
  busy: boolean;
  status: string;
}) {
  const [mode, setMode] = useState<"path" | "files">("path");
  const [path, setPath] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [over, setOver] = useState(false);
  const [probe, setProbe] = useState<InspectResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);

  const pick = (list: FileList | null) => {
    setFiles(list ? Array.from(list) : []);
    setProbe(null);
    setProbeError(null);
  };

  const inspect = async () => {
    if (!path.trim()) return;
    setProbing(true);
    setProbe(null);
    setProbeError(null);
    try {
      setProbe(await inspectPath(path));
    } catch (e) {
      setProbeError(e instanceof Error ? e.message : String(e));
    } finally {
      setProbing(false);
    }
  };

  const ready = mode === "path" ? path.trim().length > 0 : files.length > 0;
  const run = () =>
    onRun(mode === "path" ? { path: path.trim() } : { files });

  const totalMb = files.reduce((a, f) => a + f.size, 0) / 1048576;

  return (
    <Card
      title="Run a scene"
      tools={
        <span className="tools">
          <button
            className="pillbtn"
            aria-selected={mode === "path"}
            onClick={() => setMode("path")}
          >
            Local folder
          </button>
          <button
            className="pillbtn"
            aria-selected={mode === "files"}
            onClick={() => setMode("files")}
          >
            Upload
          </button>
        </span>
      }
    >
      {mode === "path" ? (
        <>
          <div className="pathrow">
            <input
              className="pathinput"
              type="text"
              spellCheck={false}
              placeholder="C:\Users\you\Downloads\ch2_tmc_ncn_..._d_img_d18"
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                setProbe(null);
                setProbeError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") void inspect();
              }}
            />
            <button className="pillbtn" onClick={() => void inspect()} disabled={probing}>
              {probing ? "…" : "Inspect"}
            </button>
          </div>
          <div className="drop-hint">
            The extracted bundle folder, a bundle .zip, or a single .xml label.
            Nothing is copied — the product is read in place.
          </div>

          {probeError && (
            <div className="probe bad">{probeError}</div>
          )}

          {probe && (
            <div className="probe">
              {probe.product ? (
                <>
                  <b>{probe.product.name}</b>
                  <div className="probemeta">
                    {probe.product.instrument} {probe.product.sensor} ·{" "}
                    {probe.product.processing_level} ·{" "}
                    {probe.product.samples.toLocaleString()} ×{" "}
                    {probe.product.lines.toLocaleString()} px (
                    {probe.product.megapixels} Mpx)
                    {probe.product.incidence_deg != null && (
                      <> · incidence {probe.product.incidence_deg.toFixed(2)}°</>
                    )}
                  </div>
                  {probe.candidates.length > 1 && (
                    <div className="probemeta dim">
                      {probe.candidates.length} products in this bundle; the
                      calibrated nadir scene is preferred.
                    </div>
                  )}
                </>
              ) : (
                <>
                  <b>{probe.primary.split(/[\\/]/).pop()}</b>
                  <div className="probemeta">
                    {probe.shape
                      ? `${probe.shape[1].toLocaleString()} × ${probe.shape[0].toLocaleString()} px`
                      : "shape unknown"}
                  </div>
                </>
              )}
              {probe.notes.map((n) => (
                <div className="probemeta dim" key={n}>
                  {n}
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          <button
            type="button"
            className={`drop${over ? " over" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              pick(e.dataTransfer.files);
            }}
          >
            <div className="t">Drop a scene, or tap to choose</div>
            <div className="h">
              Bundle .zip · GeoTIFF · FITS · PDS4 (.xml + .img together)
            </div>
          </button>
          <button className="pillbtn wide" onClick={() => folderRef.current?.click()}>
            …or pick a whole extracted folder
          </button>

          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            style={{ display: "none" }}
            /* Clearing the value first means re-picking the same file still fires
               `change`; without it a second attempt at the same path is silent. */
            onClick={(e) => {
              (e.target as HTMLInputElement).value = "";
            }}
            onChange={(e) => pick(e.target.files)}
          />
          <input
            ref={folderRef}
            type="file"
            multiple
            /* Non-standard but supported everywhere this tool runs; it is what
               preserves the bundle's tree so a label can still find its .img. */
            {...{ webkitdirectory: "", directory: "" }}
            style={{ display: "none" }}
            onClick={(e) => {
              (e.target as HTMLInputElement).value = "";
            }}
            onChange={(e) => pick(e.target.files)}
          />

          {files.length > 0 && (
            <div className="picked">
              {files.length === 1
                ? `${files[0].name} (${totalMb.toFixed(1)} MB)`
                : `${files.length} files · ${totalMb.toFixed(1)} MB`}
              {totalMb > 1500 && (
                <div className="probemeta dim">
                  That is a large upload. For a full-size product, switch to
                  “Local folder” and paste its path instead.
                </div>
              )}
            </div>
          )}
        </>
      )}

      <div className="tiles">
        <div className="tile">
          <label htmlFor="config">Sensor config</label>
          <select
            id="config"
            value={options.config}
            onChange={(e) => onOptions({ ...options, config: e.target.value })}
          >
            {configs.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="tile">
          <label htmlFor="checkpoint">Weights</label>
          <select
            id="checkpoint"
            value={options.checkpoint}
            onChange={(e) => onOptions({ ...options, checkpoint: e.target.value })}
          >
            <option value="">Physics only</option>
            {checkpoints.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="tile">
          <label htmlFor="maxdim">Max edge px</label>
          <input
            id="maxdim"
            type="number"
            min={64}
            max={16384}
            step={256}
            value={options.maxDim}
            onChange={(e) => onOptions({ ...options, maxDim: Number(e.target.value) })}
          />
        </div>
        <div className="tile">
          <label htmlFor="oversize">If larger</label>
          <select
            id="oversize"
            value={options.oversize}
            onChange={(e) =>
              onOptions({ ...options, oversize: e.target.value as RunOptions["oversize"] })
            }
          >
            <option value="crop">Centre-crop</option>
            <option value="reject">Reject</option>
          </select>
        </div>
      </div>
      <div className="drop-hint">
        Status: <span className="mono">{status}</span>
      </div>

      <button className="go" disabled={busy || !ready} onClick={run}>
        {busy ? "Running…" : "Run pipeline"}
      </button>
    </Card>
  );
}
