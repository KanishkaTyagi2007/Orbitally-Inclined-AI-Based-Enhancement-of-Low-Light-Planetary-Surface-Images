import { useRef, useState } from "react";
import type { RunOptions } from "../types";
import { Card } from "./primitives";

const ACCEPT = ".tif,.tiff,.img,.xml,.lbl,.fits,.fit,.fts,.png,.jpg,.jpeg";

export function UploadCard({
  configs,
  options,
  onOptions,
  onRun,
  busy,
  status,
}: {
  configs: string[];
  options: RunOptions;
  onOptions: (o: RunOptions) => void;
  onRun: (files: File[]) => void;
  busy: boolean;
  status: string;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (list: FileList | null) => setFiles(list ? Array.from(list) : []);

  return (
    <Card title="Run a scene">
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
        <div className="h">GeoTIFF · TIFF · FITS · PNG · PDS4 (.xml + .img together)</div>
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

      {files.length > 0 && (
        <div className="picked">
          {files.map((f) => `${f.name} (${(f.size / 1048576).toFixed(1)} MB)`).join("  ·  ")}
        </div>
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
        <div className="tile">
          <label>Status</label>
          <div style={{ font: "13px var(--mono)", color: "var(--muted)", minHeight: 30 }}>
            {status}
          </div>
        </div>
      </div>

      <button className="go" disabled={busy || files.length === 0} onClick={() => onRun(files)}>
        {busy ? "Running…" : "Run pipeline"}
      </button>
    </Card>
  );
}
