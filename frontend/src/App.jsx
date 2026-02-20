import React, { useEffect, useMemo, useRef, useState } from "react";

function splitClasses(text) {
  return text
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

const MODEL_OPTIONS = [
  "gemini-1.5-flash",
  "gemini-1.5-flash-8b",
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-3.0-flash",
  "gemini-3.0-flash-lite",
  "gemma-3-12b-it",
  "gemma-3-27b-it",
];

export default function App() {
  const [apiBase, setApiBase] = useState("http://localhost:8000");
  const [project, setProject] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [classesText, setClassesText] = useState("");
  const [numAgents, setNumAgents] = useState(4);
  const [epochs, setEpochs] = useState(50);
  const [labelMode, setLabelMode] = useState("gemini");
  const [model, setModel] = useState("gemini-2.5-flash");
  const [localVlmUrl, setLocalVlmUrl] = useState("http://localhost:1234/v1");
  const [localModelName, setLocalModelName] = useState("local-model");
  const [providerAKey, setProviderAKey] = useState("");
  const [providerBKey, setProviderBKey] = useState("");
  const [providerLocalKey, setProviderLocalKey] = useState("");
  const [sourceJobId, setSourceJobId] = useState("");

  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [job, setJob] = useState(null);
  const [messages, setMessages] = useState([]);
  const [nextHint, setNextHint] = useState("");
  const [projectExists, setProjectExists] = useState(null);

  const eventSourceRef = useRef(null);
  const pollTimerRef = useRef(null);
  const terminalNotifyRef = useRef(new Set());
  const seenJobRef = useRef(new Set());

  const endpoint = useMemo(() => apiBase.replace(/\/$/, ""), [apiBase]);

  function addMessage(level, message) {
    setMessages((prev) => {
      const next = [
        ...prev,
        { time: new Date().toLocaleTimeString(), level, message },
      ];
      return next.slice(-500);
    });
  }

  function nextStepHintFor(mode) {
    if (mode === "collect") return "Collect finished. Next: run Label.";
    if (mode === "label") return "Label finished. Next: run Augment.";
    if (mode === "augment") return "Dataset prep finished. Next: start Training.";
    if (mode === "train") return "Training finished. Next: run Evaluation.";
    if (mode === "eval") return "Evaluation finished. You can review metrics or start a new run.";
    return "Job finished. You can start the next step.";
  }

  function isProjectNotFoundError(err) {
    const text = String(err?.message || "").toLowerCase();
    return text.includes("project") && text.includes("not found");
  }

  async function api(path, init) {
    const headers = new Headers(init?.headers || {});
    const hasBody = init && Object.prototype.hasOwnProperty.call(init, "body") && init.body != null;
    if (hasBody && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const res = await fetch(`${endpoint}${path}`, {
      ...init,
      headers,
    });

    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body?.detail) detail = body.detail;
      } catch {
        // ignore
      }
      throw new Error(detail);
    }

    if (res.status === 204) return null;
    return res.json();
  }

  function clearEventsStream() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }

  function startEventsStream(nextJobId) {
    clearEventsStream();
    const url = `${endpoint}/api/projects/${encodeURIComponent(
      project
    )}/events?job_id=${encodeURIComponent(nextJobId)}`;

    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener("info", (ev) => {
      const payload = JSON.parse(ev.data);
      addMessage("info", payload.message);
    });

    es.addEventListener("log", (ev) => {
      const payload = JSON.parse(ev.data);
      addMessage("log", payload.message);
    });

    es.addEventListener("error", (ev) => {
      const payload = JSON.parse(ev.data);
      addMessage("error", payload.message);
    });

    es.onerror = () => {
      addMessage("meta", "Event stream closed.");
      clearEventsStream();
    };
  }

  async function autoDetectLocalModel() {
    try {
      const endpoint = localVlmUrl.replace(/\/v1\/?$/, ""); // strip tail
      // Standard OpenAI compatible list
      const res = await fetch(`${endpoint}/v1/models`);
      if (!res.ok) {
        throw new Error(`Failed to list models: ${res.status}`);
      }
      const data = await res.json();
      // data.data is the list. 
      const list = data.data || [];
      if (list.length > 0) {
        // Pick the first one as a reasonable default
        // Often list[0].id is the model name
        setLocalModelName(list[0].id);
        addMessage("ok", `Detected active model: ${list[0].id}`);
      } else {
        addMessage("meta", "No models returned by local server.");
      }
    } catch (e) {
      addMessage("error", `Could not detect model: ${e.message}`);
    }
  }

  async function upsertProject() {
    try {
      setBusy(true);
      const payload = {
        project,
        video_url: videoUrl,
        classes: splitClasses(classesText),
        label_mode: labelMode,
        model: labelMode === "local" ? localModelName : model,
        gemini_model: labelMode === "gemini" ? model : undefined,
        openai_model: labelMode === "gpt" ? "gpt-4o" : undefined,
        local_model_name: labelMode === "local" ? localModelName : undefined,
        local_vlm_url: labelMode === "local" ? localVlmUrl : undefined,
        num_agents: Number(numAgents),
        epochs: Number(epochs),
      };

      await api("/api/projects", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setProjectExists(true);
      addMessage("ok", `Project '${project}' saved.`);
      await refreshStatus();
      await locateDataset();
    } catch (err) {
      addMessage("error", err.message);
    } finally {
      setBusy(false);
    }
  }

  async function startRun(mode) {
    try {
      setBusy(true);
      setNextHint("");
      const runPayload = { mode };
      if (sourceJobId.trim()) {
        runPayload.source_job_id = sourceJobId.trim();
      }
      if (providerAKey.trim()) {
        runPayload.gemini_api_key = providerAKey.trim();
      }
      if (providerBKey.trim()) {
        runPayload.openai_api_key = providerBKey.trim();
      }
      if (providerLocalKey.trim()) {
        runPayload.local_vlm_api_key = providerLocalKey.trim();
      }
      const run = await api(`/api/projects/${encodeURIComponent(project)}/run`, {
        method: "POST",
        body: JSON.stringify(runPayload),
      });
      setJobId(run.job_id);
      addMessage("ok", `Process started: ${mode} (job ${run.job_id})`);
      startEventsStream(run.job_id);
      await refreshStatus();
    } catch (err) {
      addMessage("error", err.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadJob(currentJobId) {
    if (!currentJobId) return;
    try {
      const data = await api(
        `/api/projects/${encodeURIComponent(project)}/jobs/${encodeURIComponent(currentJobId)}`
      );
      seenJobRef.current.add(currentJobId);
      setJob(data);
      const terminalKey = `${currentJobId}:${data.status}`;
      if (data.status === "completed" && !terminalNotifyRef.current.has(terminalKey)) {
        terminalNotifyRef.current.add(terminalKey);
        addMessage("ok", `Job ${currentJobId} completed. Output is ready.`);
        const hint = nextStepHintFor(data.requested_mode);
        setNextHint(hint);
        addMessage("meta", hint);
        await locateDataset();
      }
      if (data.status === "failed" && !terminalNotifyRef.current.has(terminalKey)) {
        terminalNotifyRef.current.add(terminalKey);
        addMessage("error", `Job ${currentJobId} failed: ${data.error || "unknown"}`);
        if (data.error_code) {
          addMessage("meta", `Error code: ${data.error_code}`);
        }
        if (data.error_hint) {
          addMessage("meta", `Hint: ${data.error_hint}`);
        }
        if (typeof data.retryable === "boolean") {
          addMessage("meta", data.retryable ? "This failure is retryable." : "This failure needs setup/config fix before retry.");
        }
        if ((data.error || "").includes("PROVIDER_A_KEY") || (data.error || "").includes("PROVIDER_B_KEY")) {
          addMessage("meta", "Set Provider Keys in the backend terminal, then run Label again.");
        }
      }
    } catch (err) {
      addMessage("error", err.message);
    }
  }

  async function refreshStatus() {
    try {
      const data = await api(`/api/projects/${encodeURIComponent(project)}/status`);
      setStatus(data);
      setProjectExists(true);
      if (data.latest_job_id) {
        setJobId(data.latest_job_id);
      }
      return data;
    } catch (err) {
      if (isProjectNotFoundError(err)) {
        setProjectExists(false);
      } else {
        addMessage("error", err.message);
      }
      return null;
    }
  }

  async function locateDataset() {
    try {
      const data = await api(`/api/projects/${encodeURIComponent(project)}/artifacts`);
      setArtifacts(data);
      setProjectExists(true);
      addMessage("meta", `Dataset location: ${data.output_dir}`);
    } catch (err) {
      if (isProjectNotFoundError(err)) {
        setProjectExists(false);
      } else {
        addMessage("error", err.message);
      }
    }
  }

  async function cancelJob() {
    try {
      setBusy(true);

      const globalJobId = status?.active_job_id;
      const localJobId = status?.running_job_id;
      
      if (globalJobId && !localJobId) {
          // We are blocked by another project's job.
          if (!confirm(`A job from another project (${globalJobId.slice(0,8)}) is running. Stop it?`)) {
              return;
          }
           await api(`/api/system/cancel_active`, { method: "POST" });
      } else {
           await api(`/api/projects/${encodeURIComponent(project)}/cancel`, {
            method: "POST",
          });
      }

      addMessage("ok", "Cancellation requested.");
      await refreshStatus();
    } catch (err) {
      addMessage("error", err.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    (async () => {
      const current = await refreshStatus();
      if (current) {
        await locateDataset();
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    clearInterval(pollTimerRef.current);
    pollTimerRef.current = setInterval(async () => {
      if (projectExists === false) {
        return;
      }
      const current = await refreshStatus();
      if (!current) return;
      if (current.running_job_id) {
        await loadJob(current.running_job_id);
      } else if (current.latest_job_id && !seenJobRef.current.has(current.latest_job_id)) {
        await loadJob(current.latest_job_id);
      }
      if (!current.running_job_id && eventSourceRef.current) {
        clearEventsStream();
      }
    }, 3000);

    return () => {
      clearInterval(pollTimerRef.current);
      clearEventsStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, endpoint, projectExists]);

  return (
    <div className="app">
      <h1>Yolodex Frontend</h1>
      <p className="subtitle">
        GUI for API input + process notifications (start, progress logs, done).
      </p>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>API Keys</h3>
        <div className="grid">
          {labelMode === "gemini" && (
            <div>
              <label>Primary Provider Key (Gemini)</label>
              <input
                type="password"
                value={providerAKey}
                onChange={(e) => setProviderAKey(e.target.value)}
                placeholder="Key A..."
              />
            </div>
          )}
          {labelMode === "gpt" && (
            <div>
              <label>OpenAI Key (GPT-4o)</label>
              <input
                type="password"
                value={providerBKey}
                onChange={(e) => setProviderBKey(e.target.value)}
                placeholder="sk-..."
              />
            </div>
          )}
          {labelMode === "local" && (
            <div>
              <label>Local VLM Key</label>
              <input
                type="password"
                value={providerLocalKey}
                onChange={(e) => setProviderLocalKey(e.target.value)}
                placeholder="(Optional, e.g. lm-studio)"
              />
            </div>
          )}
        </div>
        <p className="meta" style={{ marginTop: 10 }}>
          Keys are sent only when you click a run button.
        </p>
      </div>

      <div className="card">
        <div className="grid">
          <div>
            <label>API Base URL</label>
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
          </div>
          <div>
            <label>Project</label>
            <input value={project} onChange={(e) => setProject(e.target.value)} />
          </div>
          <div>
            <label>YouTube Link</label>
            <input value={videoUrl} onChange={(e) => setVideoUrl(e.target.value)} />
          </div>
          <div>
            <label>Classes (comma-separated)</label>
            <input value={classesText} onChange={(e) => setClassesText(e.target.value)} />
          </div>
          <div>
            <label>Label Agents</label>
            <input
              type="number"
              min="1"
              value={numAgents}
              onChange={(e) => setNumAgents(e.target.value)}
            />
          </div>
          <div>
            <label>Epochs</label>
            <input
              type="number"
              min="1"
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
            />
          </div>
          <div>
            <label>Label Mode</label>
            <select value={labelMode} onChange={(e) => setLabelMode(e.target.value)}>
              <option value="gemini">Gemini (Google)</option>
              <option value="gpt">GPT-4o (OpenAI)</option>
              <option value="local">Local VLM (LM Studio/vLLM)</option>
            </select>
          </div>
          {labelMode === "gemini" && (
            <div>
              <label>Gemini Model</label>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {MODEL_OPTIONS.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))}
              </select>
            </div>
          )}
          {labelMode === "local" && (
            <>
              <div>
                <label>Local VLM URL</label>
                <input
                  value={localVlmUrl}
                  onChange={(e) => setLocalVlmUrl(e.target.value)}
                  placeholder="http://localhost:1234/v1"
                />
              </div>
              <div style={{ display: "flex", flexDirection: "column" }}>
                <label>Local Model Name</label>
                <div style={{ display: "flex", gap: "8px" }}>
                  <input
                    style={{ flex: 1 }}
                    value={localModelName}
                    onChange={(e) => setLocalModelName(e.target.value)}
                    placeholder="gemma-2-27b-it"
                  />
                  <button 
                  className="secondary" 
                  style={{ padding: "0 10px", fontSize: "0.8rem", whiteSpace: "nowrap" }}
                  onClick={autoDetectLocalModel}
                  disabled={busy}
                  >
                    Detect
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <button onClick={upsertProject} disabled={busy}>Save / Update Project</button>
          <button className="secondary" onClick={refreshStatus} disabled={busy}>Refresh Status</button>
          <button className="secondary" onClick={locateDataset} disabled={busy}>Locate Dataset</button>
        </div>
      </div>

      <div className="card">
        <div style={{ marginBottom: 12 }}>
          <label>Source Job ID (Optional Source Override)</label>
          <input
            value={sourceJobId}
            onChange={(e) => setSourceJobId(e.target.value)}
            placeholder="e.g. 4ae8dbf9d53448faa7e9fc261d8f7f69 (leave blank for auto)"
            style={{ width: "100%", marginTop: 4 }}
          />
        </div>
        <div className="row">
          <button onClick={() => startRun("collect")} disabled={busy}>Prepare Dataset: Collect</button>
          <button onClick={() => startRun("label")} disabled={busy}>Prepare Dataset: Label</button>
          <button onClick={() => startRun("augment")} disabled={busy}>Prepare Dataset: Augment</button>
          <button onClick={() => startRun("train")} disabled={busy}>Start Training</button>
          <button onClick={() => startRun("eval")} disabled={busy}>Run Evaluation</button>
          <button className="secondary" onClick={cancelJob} disabled={!status?.running_job_id}>Cancel Job</button>
        </div>
        <p className="meta" style={{ marginTop: 10 }}>
          Only one job can run at a time. API returns 409 if another job is active.
        </p>
      </div>

      <div className="card">
        <span className="status-pill">Latest Job: {jobId || "-"}</span>
        <span className="status-pill">Running Job: {status?.running_job_id || "none"}</span>
        <span className="status-pill">Global Active: {status?.active_job_id || "none"}</span>
        <p className="meta" style={{ marginTop: 10 }}>
          Dataset Dir: {artifacts?.output_dir || "unknown"}
        </p>
        <p className="meta">
          Frames: {artifacts?.frame_count ?? "-"} | Labels: {artifacts?.label_count ?? "-"} | Previews: {artifacts?.preview_count ?? "-"}
        </p>
        {job && (
          <p className="meta">
            Job Status: {job.status} {job.current_phase ? `| Phase: ${job.current_phase}` : ""}
          </p>
        )}
        {!status?.active_job_id && nextHint && <p className="meta">Next Action: {nextHint}</p>}
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Notifications & Logs</h3>
        <div className="log">
          {messages.length === 0 && <div className="log-line meta">No events yet.</div>}
          {messages.map((line, idx) => (
            <div key={`${line.time}-${idx}`} className={`log-line ${line.level === "error" ? "err" : line.level === "ok" ? "ok" : ""}`}>
              [{line.time}] {line.message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
