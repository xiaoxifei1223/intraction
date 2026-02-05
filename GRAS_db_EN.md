# GRAS Data Layer Database Design (GRAS_db)

> Objectives: Supporting the **Data Layer** in GRAS four-tier architecture, specifically:
> - Accommodating both Java rule-based calculations and AI algorithm inputs/outputs
> - Compatible with manual Excel upload metric sources
> - Supporting real-time queries, historical analysis, and lineage governance for Decision/Presentation layers

---

## I. Overall Architecture and Layering

The data layer adopts a three-tier storage architecture:

1. **Hot Storage (Online State Storage)**
   - Stores:
     - Latest metric calculated values (current value)
     - Latest forecast values (forecast)
     - Current dynamic baseline / threshold
     - Current risk status
   - Characteristics:
     - Frequent read/write, latency-sensitive
     - Key-Value / document model preferred
   - Typical Implementations:
     - AWS: DynamoDB (can overlay Redis cache)
     - GCP: Firestore / Bigtable

2. **Warm Storage (PostgreSQL)**
   - Stores:
     - Metric historical snapshots (metric value time series)
     - Historical forecast results (for post-evaluation and comparison)
     - Manually uploaded Excel raw info + structured content
     - API access logs, job execution records and other audit data
   - Characteristics:
     - Strong schema, supports complex queries / JOIN / audit
     - Medium data volume, mainly recent data (historical data can be archived to data lake)

3. **Metadata Management Storage (Metadata/Lineage)**
   - Stores:
     - Metric definitions (metric definition)
     - Dataset and job metadata
     - Lineage graph
   - Typical Implementations:
     - Start with PostgreSQL tables to build "graph structure"; migrate to graph database (e.g., Neptune/Neo4j) if needed later

> Note: Cold data layer (data lake S3/GCS + Parquet/Athena/BigQuery) is already covered in GRAS.md, here we mainly supplement database-side design.

---

## II. Hot Storage Design (Online Runtime State)

### 2.1 Design Objectives

- Provide "one-hop fetch all" interface for Java decision layer / API layer:
  - Given `(appId, metricId)`, retrieve in one query:
    - Latest metric actual value
    - Latest forecast result (can be from Java rules or AI model or fusion result)
    - Current baseline and threshold
    - Current Risk status

- Simultaneously ensure:
  - Simple write operations for compute tasks (Java / AI Batch)
  - Unified structure for decision engine and visualization reads

### 2.2 Logical Structure: `metric_runtime_state`

Using DynamoDB as example (document/Key-Value model), logical field design:

**Primary Key**

- `pk = appId` (partition key)
- `sk = metricId` (sort key, e.g., `"G.error_rate"`, `"A.availability"`)

**Fields**

- Basic Identifiers:
  - `app_id`: Application ID
  - `metric_id`: Metric ID

- Current Metric Value:
  - `current_value`: Latest actual value
  - `current_value_time`: Timestamp corresponding to latest actual value
  - `period_type`: `REALTIME/DAILY/WEEKLY/MONTHLY` (optional, for period distinction)

- Current baseline / threshold:
  - `current_baseline`: Current dynamic baseline value
  - `baseline_lower` / `baseline_upper`: Baseline range
  - `current_threshold`: Current decision threshold (may come from SLO / dynamic rules)
  - `threshold_type`: `STATIC/DYNAMIC/SLO`

- Latest Prediction Result (combining Java rules + AI model):
  - `last_prediction_value`
  - `last_prediction_lower` / `last_prediction_upper`
  - `last_prediction_target_time`: Future time point corresponding to prediction
  - `last_prediction_horizon`: Prediction lead time (e.g., `"2h"`, `"24h"`)
  - `last_prediction_source`: `"JAVA_RULE" / "AI_MODEL" / "FUSION"`
  - `last_prediction_model_version`

- Current Risk Status:
  - `risk_score`: 0–100
  - `risk_level`: `INFO/WARN/MINOR/MAJOR/CRITICAL`
  - `status_updated_at`: Status update time

- Other Context:
  - `extra_params`: JSON, for storing computation context parameters (e.g., holiday flags, current deployment version, etc.)
  - `updated_at`: Last update time of this record

> Java existing programs can evolve to:  
> - After real-time calculation / scheduled aggregation, update `current_value` related fields  
> - If rule-based prediction exists, also update `last_prediction_*`, `source="JAVA_RULE"`  
> When AI Job updates, write `source="AI_MODEL"` or fused `"FUSION"`.

---

## III. Warm Storage (PostgreSQL) Design

### 3.1 Metric History: `metric_value_history`

For storing historical metric values of various periods (including from Java rule calculations, AI algorithm outputs, Excel, etc.).

```sql
CREATE TABLE metric_value_history (
    id              BIGSERIAL PRIMARY KEY,
    app_id          VARCHAR(100) NOT NULL,
    metric_id       VARCHAR(100) NOT NULL,
    period_type     VARCHAR(20)  NOT NULL,  -- 'REALTIME'/'DAILY'/'WEEKLY'/'MONTHLY'
    period_start    TIMESTAMPTZ  NOT NULL,
    period_end      TIMESTAMPTZ  NOT NULL,
    value           DOUBLE PRECISION NOT NULL,
    source          VARCHAR(50)  NOT NULL,  -- 'JAVA_RULE'/'AI_MODEL'/'EXCEL'/'MANUAL'
    source_ref_id   VARCHAR(200),          -- Corresponding to excel_file_id/job_run_id etc.
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_metric_value_hist_app_metric_time
    ON metric_value_history (app_id, metric_id, period_start);
```

Usage Recommendations:

- Monthly/Weekly metrics: `period_type='MONTHLY'/'WEEKLY'`, `period_start/period_end` covers the period.
- For real-time metrics with large data volume, can keep only minute/hour-level aggregations, details sink to data lake.

### 3.2 Prediction History: `metric_prediction_history`

For recording "prediction result history" of Java rules + AI models, facilitating post-evaluation and comparison.

```sql
CREATE TABLE metric_prediction_history (
    id               BIGSERIAL PRIMARY KEY,
    app_id           VARCHAR(100) NOT NULL,
    metric_id        VARCHAR(100) NOT NULL,
    target_time      TIMESTAMPTZ  NOT NULL,  -- Predicted time point
    forecast_value   DOUBLE PRECISION NOT NULL,
    lower_bound      DOUBLE PRECISION,
    upper_bound      DOUBLE PRECISION,
    horizon          INTERVAL,               -- Prediction lead time, e.g., '2 hours'
    source           VARCHAR(50) NOT NULL,   -- 'JAVA_RULE'/'AI_MODEL'
    model_version    VARCHAR(50),
    confidence       DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_metric_pred_hist_app_metric_target
    ON metric_prediction_history (app_id, metric_id, target_time);
```

Combined with `metric_value_history`, can perform:

- Actual vs Forecast comparison
- Prediction error analysis (MAPE, deviation changes over time)

### 3.3 Excel Upload: `excel_file` + `excel_metric_value`

#### 3.3.1 File Level: `excel_file`

Records each uploaded Excel file and its storage location for traceability.

```sql
CREATE TABLE excel_file (
    id              BIGSERIAL PRIMARY KEY,
    storage_uri     TEXT NOT NULL,      -- File path in S3/GCS/internal storage
    original_name   TEXT NOT NULL,
    uploader        VARCHAR(100),
    upload_time     TIMESTAMPTZ NOT NULL,
    sheet_name      TEXT,               -- Main sheet
    checksum        TEXT,               -- e.g., MD5/sha256, for deduplication
    status          VARCHAR(20) NOT NULL DEFAULT 'IMPORTED',  -- 'IMPORTED'/'FAILED'/...
    remark          TEXT
);
```

#### 3.3.2 Row Level (Structured Data): `excel_metric_value`

When row data in Excel is essentially metric values, can be parsed into structured records for further writing to `metric_value_history`.

```sql
CREATE TABLE excel_metric_value (
    id              BIGSERIAL PRIMARY KEY,
    excel_file_id   BIGINT NOT NULL REFERENCES excel_file(id),
    app_id          VARCHAR(100),
    metric_id       VARCHAR(100),
    period_type     VARCHAR(20),
    period_start    TIMESTAMPTZ,
    period_end      TIMESTAMPTZ,
    value           DOUBLE PRECISION,
    raw_row_data    JSONB,     -- Original row content (for reference)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Subsequent ETL:

- From `excel_metric_value` → write to `metric_value_history`, `source='EXCEL'`, `source_ref_id=excel_file_id`  
- Can then participate in unified calculations of Java/AI/Decision layers.

### 3.4 API Access Log: `api_access_log`

For auditing and behavior analysis (which metrics are most frequently queried, access patterns, etc.).

```sql
CREATE TABLE api_access_log (
    id              BIGSERIAL PRIMARY KEY,
    api_name        VARCHAR(200) NOT NULL,
    http_method     VARCHAR(10)  NOT NULL,
    caller_id       VARCHAR(100),        -- Caller user/system ID
    app_id          VARCHAR(100),
    metric_id       VARCHAR(100),
    request_time    TIMESTAMPTZ NOT NULL,
    status_code     INT,
    latency_ms      INT,
    trace_id        VARCHAR(100),
    client_ip       INET,
    extra           JSONB
);

CREATE INDEX idx_api_log_time ON api_access_log (request_time);
CREATE INDEX idx_api_log_app_metric ON api_access_log (app_id, metric_id);
```

When data volume is too large, can keep only recent period in Postgres, rest sink to data lake.

### 3.5 Job Execution Records (Optional): `job_run`

For recording execution status of Java scheduled tasks, AI Batch, ETL jobs, etc., linking with lineage management tables.

```sql
CREATE TABLE job_run (
    id              BIGSERIAL PRIMARY KEY,
    job_name        TEXT NOT NULL,         -- e.g., 'java_metric_job', 'ai_forecast_batch'
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    status          VARCHAR(20),           -- 'SUCCESS'/'FAILED'/...
    params          JSONB,                 -- Execution parameters
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## IV. Metadata & Lineage Design

Metadata and lineage information can also be expressed as graph structures using PostgreSQL tables initially.

### 4.1 Metric Definition: `metric_definition`

For unified management of various GRAS metric definitions, units, granularity, etc.

```sql
CREATE TABLE metric_definition (
    metric_id        VARCHAR(100) PRIMARY KEY,   -- Unique ID, aligned with runtime/history tables
    name             TEXT NOT NULL,              -- Metric name
    description      TEXT,
    dimension        VARCHAR(20) NOT NULL,       -- 'G'/'R'/'A'/'S'
    unit             VARCHAR(50),
    granularity      VARCHAR(20),                -- 'REALTIME'/'DAILY'/'WEEKLY'/'MONTHLY'
    owner_team       VARCHAR(100),
    sla_target       DOUBLE PRECISION,           -- e.g., 99.9
    calc_logic       TEXT,                       -- Natural language description
    calc_expression  TEXT,                       -- Formula or DSL (optional)
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 4.2 Dataset Node: `dataset`

In lineage graph, each "data entity" serves as a node:

- Such as: Excel file, DB table, S3 Parquet, metric itself, model artifacts, etc.

```sql
CREATE TABLE dataset (
    id              BIGSERIAL PRIMARY KEY,
    dataset_type    VARCHAR(50) NOT NULL,  -- 'EXCEL_FILE'/'S3_PARQUET'/'DB_TABLE'/'METRIC'/'MODEL'
    name            TEXT NOT NULL,
    ref_key         TEXT,                  -- Corresponding to excel_file_id / S3 key / table name / metric_id / model path
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 4.3 Lineage Edge: `lineage_edge`

Expresses "from which datasets, through which Job, another dataset was produced".

```sql
CREATE TABLE lineage_edge (
    id              BIGSERIAL PRIMARY KEY,
    from_dataset_id BIGINT NOT NULL REFERENCES dataset(id),
    to_dataset_id   BIGINT NOT NULL REFERENCES dataset(id),
    job_run_id      BIGINT REFERENCES job_run(id),
    transformation  TEXT,         -- e.g., 'aggregate_monthly', 'java_rule_calc', 'ai_forecast'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Examples:

- `EXCEL_FILE` → `metric_value_history` (from manual spreadsheet)  
- `S3 raw details` → `feature table` → `model` → `prediction results` → `metric_runtime_state`  

All can form a chain.

---

## V. Java Rules / AI Algorithms / Excel Landing Approach in Three Tiers

Comprehensive explanation of how three sources flow through the three tiers:

### 5.1 Java Rule Calculation

- Real-time / scheduled calculation results:
  - Write to `metric_value_history` (source=`'JAVA_RULE'`)
  - Update `metric_runtime_state.current_value` related fields in Hot Storage
- Java rule prediction:
  - Write to `metric_prediction_history` (source=`'JAVA_RULE'`)
  - Update `last_prediction_*` in Hot Storage, `last_prediction_source='JAVA_RULE'`
- Corresponding job execution info:
  - Write to `job_run`, and register in `dataset + lineage_edge` from which data source to which result table.

### 5.2 AI Algorithm (Python Batch / Cloud Run / Fargate)

- Pull historical data from data lake / Postgres:
  - Training/inference generates prediction results.
- Output:
  - Large volume details sink to S3 (cold tier)
  - Aggregated/aligned prediction results write to `metric_prediction_history` (source=`'AI_MODEL'`)
  - Update `last_prediction_*` and `current_baseline/current_threshold` in Hot Storage
- Also record lineage through `job_run + dataset + lineage_edge`.

### 5.3 Manual Excel Upload

- Upload behavior:
  - File placed in S3/GCS; write one `excel_file` record
- Parsing:
  - Each row written to `excel_metric_value`, can attach `raw_row_data`
- ETL:
  - Convert rows that can serve as metrics to `metric_value_history` records, `source='EXCEL'`, `source_ref_id=excel_file_id`
- Upper layer usage:
  - When Java / AI reads `metric_value_history`, can uniformly access Excel and system-generated metric data.

---

## VI. Summary

Through:

- Hot Storage: `metric_runtime_state` provides single-point real-time view  
- Warm Storage (PostgreSQL): `metric_value_history`, `metric_prediction_history`, `excel_*`, `api_access_log`, `job_run`  
- Metadata & Lineage: `metric_definition`, `dataset`, `lineage_edge`

Enables:

- **analysis layer** (Java rules + AI) — unified data landing pattern  
- **decision layer** — can read unified real-time/historical state, distinguish sources and perform fusion  
- **presentation layer** — can do real-time display, replay history, and explain "where this metric came from"
