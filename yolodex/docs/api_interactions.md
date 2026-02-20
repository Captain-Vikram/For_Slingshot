# API System Design & Interactions

The Yolodex API system manages long-running jobs (collect, label, train, etc.) via a job queue system. To ensure reliability and performance, specifically regarding job status monitoring and log streaming, the following design patterns are implemented:

## Job State Management

### In-Memory Caching with Disk Persistence

- **Primary Source of Truth**: Active jobs are maintained in an in-memory cache (`JobManager`). This eliminates disk I/O for high-frequency operations like status polling and log appending.
- **Debounced Persistence**: Job state is persisted to disk (`runs/_jobs/*.json`) only when:
  - Critical state changes occur (status transitions: `queued` -> `running` -> `completed/failed`).
  - A significant time interval has passed (throttled log updates), reducing disk write operations by >90% during high-output phases.
- **Crash Recovery**: On startup or cache miss, job state is lazy-loaded from disk.

## API Interactions

### Efficient Status Polling & Event Streaming

- **Status Endpoint (`GET /api/projects/{project}/jobs/{job_id}`)**:
  - Serves data directly from memory for active jobs, responding in sub-millisecond time.
  - Falls back to disk only for archived/cold jobs.
- **Event Stream (`GET /api/projects/{project}/events`)**:
  - Uses Server-Sent Events (SSE) for real-time log streaming.
  - **Optimization**: The event generator reads from the in-memory cache instead of reloading the file from disk on every loop iteration. This prevents "continuous GET request output" issues caused by blocking I/O and ensures smooth log flow.

### Concurrency

- **Thread Safety**: All state mutations are protected by a reentrant lock within the `JobManager`, ensuring safe access from the API threads and the background job runner thread.
- **Non-Blocking Execution**: Jobs run in a dedicated background thread, decoupled from API request handling.
