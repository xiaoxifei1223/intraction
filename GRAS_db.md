# GRAS 数据层数据库设计（GRAS_db）

> 目标：支撑 GRAS 四层架构中的 **Data Layer**，尤其是：
> - 同时承载 Java 规则计算 + AI 算法的输入输出
> - 兼容手工 Excel 上传的指标来源
> - 支撑 Decision / Presentation 层的实时查询、历史分析和溯源治理

---

## 一、整体架构与分层

数据层采用三层存储架构：

1. **Hot Storage（在线状态存储）**
   - 存放：
     - 最新指标计算值（current value）
     - 最新预测值（forecast）
     - 当前动态 baseline / threshold
     - 当前风险状态
   - 特点：
     - 读写频繁，延迟敏感
     - Key-Value / 文档模型更合适
   - 典型实现：
     - AWS：DynamoDB（可叠加 Redis 缓存）
     - GCP：Firestore / Bigtable

2. **Warm Storage（PostgreSQL）**
   - 存放：
     - 指标历史快照（指标值时间序列）
     - 历史预测结果（用于事后评估和对比）
     - 手工上传 Excel 的原始信息 + 结构化内容
     - API 访问日志、Job 运行记录等审计数据
   - 特点：
     - 强 schema，支持复杂查询 / JOIN / 审计
     - 数据量中等，主要近期数据（远期可归档到数据湖）

3. **Metadata Management Storage（元数据/血缘）**
   - 存放：
     - 指标定义（metric definition）
     - 数据集（dataset）和作业（job）元数据
     - 血缘图谱（lineage graph）
   - 典型实现：
     - 先用 PostgreSQL 表建"图结构"；后续如需要可迁移到图数据库（如 Neptune/Neo4j）

> 说明：冷数据层（数据湖 S3/GCS + Parquet/Athena/BigQuery）在 GRAS.md 已有，这里主要补数据库侧的设计。

---

## 二、Hot Storage 设计（在线运行状态）

### 2.1 设计目标

- 为 Java 决策层 / API 层提供 "一跳拿全" 的接口：
  - 给定 `(appId, metricId)`，一次查询得到：
    - 最新指标实测值
    - 最新预测结果（可以来自 Java 规则或 AI 模型或融合结果）
    - 当前 baseline 和 threshold
    - 当前 Risk 状态

- 同时保证：
  - 计算任务（Java / AI Batch）写入简单
  - 决策引擎和可视化读起来结构统一

### 2.2 逻辑结构：`metric_runtime_state`

以 DynamoDB 为例（文档/Key-Value 模型），逻辑字段设计如下：

**主键**

- `pk = appId`（分区键）
- `sk = metricId`（排序键，例如：`"G.error_rate"`、`"A.availability"`）

**字段**

- 基本标识：
  - `app_id`：应用 ID
  - `metric_id`：指标 ID

- 当前指标值：
  - `current_value`：最新实测值
  - `current_value_time`：最新实测值对应时间戳
  - `period_type`：`REALTIME/DAILY/WEEKLY/MONTHLY`（可选，用于区分周期）

- 当前 baseline / threshold：
  - `current_baseline`：当前动态基线值
  - `baseline_lower` / `baseline_upper`：基线区间
  - `current_threshold`：当前决策阈值（可能来自 SLO / 动态规则）
  - `threshold_type`：`STATIC/DYNAMIC/SLO`

- 最新预测结果（结合 Java 规则 + AI 模型）：
  - `last_prediction_value`
  - `last_prediction_lower` / `last_prediction_upper`
  - `last_prediction_target_time`：预测对应未来时间点
  - `last_prediction_horizon`：预测提前量（例如 `"2h"`, `"24h"`）
  - `last_prediction_source`：`"JAVA_RULE" / "AI_MODEL" / "FUSION"`
  - `last_prediction_model_version`

- 当前风险状态：
  - `risk_score`：0–100
  - `risk_level`：`INFO/WARN/MINOR/MAJOR/CRITICAL`
  - `status_updated_at`：状态更新时间

- 其它上下文：
  - `extra_params`：JSON，用于存放计算时的上下文参数（如节假日标记、当前部署版本等）
  - `updated_at`：本记录最后更新时间

> Java 现有程序可演进为：  
> - 实时计算 / 定时汇总后，更新 `current_value` 相关字段  
> - 如果有规则预测，也更新 `last_prediction_*`，`source="JAVA_RULE"`  
> AI Job 更新时则写 `source="AI_MODEL"` 或融合后的 `"FUSION"`。

---

## 三、Warm Storage（PostgreSQL）设计

### 3.1 指标历史：`metric_value_history`

用于存放各类周期的历史指标值（包括来自 Java 规则计算、AI 算法输出、Excel 等）。

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
    source_ref_id   VARCHAR(200),          -- 对应 excel_file_id/job_run_id 等
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_metric_value_hist_app_metric_time
    ON metric_value_history (app_id, metric_id, period_start);
```

使用建议：

- 月度/周度指标：`period_type='MONTHLY'/'WEEKLY'`，`period_start/period_end` 覆盖该周期。
- 实时指标若数据量大，可只保留分钟/小时级聚合，明细沉到数据湖。

### 3.2 预测历史：`metric_prediction_history`

用于记录 Java 规则 + AI 模型的"预测结果历史"，方便事后评估与对比。

```sql
CREATE TABLE metric_prediction_history (
    id               BIGSERIAL PRIMARY KEY,
    app_id           VARCHAR(100) NOT NULL,
    metric_id        VARCHAR(100) NOT NULL,
    target_time      TIMESTAMPTZ  NOT NULL,  -- 预测的时间点
    forecast_value   DOUBLE PRECISION NOT NULL,
    lower_bound      DOUBLE PRECISION,
    upper_bound      DOUBLE PRECISION,
    horizon          INTERVAL,               -- 预测提前量，如 '2 hours'
    source           VARCHAR(50) NOT NULL,   -- 'JAVA_RULE'/'AI_MODEL'
    model_version    VARCHAR(50),
    confidence       DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_metric_pred_hist_app_metric_target
    ON metric_prediction_history (app_id, metric_id, target_time);
```

配合 `metric_value_history`，可以做：

- Actual vs Forecast 对比
- 预测误差分析（MAPE、偏差随时间的变化）

### 3.3 Excel 上传：`excel_file` + `excel_metric_value`

#### 3.3.1 文件级：`excel_file`

记录每次上传的 Excel 文件及其存储位置，方便溯源。

```sql
CREATE TABLE excel_file (
    id              BIGSERIAL PRIMARY KEY,
    storage_uri     TEXT NOT NULL,      -- 原始文件在 S3/GCS/内部存储的路径
    original_name   TEXT NOT NULL,
    uploader        VARCHAR(100),
    upload_time     TIMESTAMPTZ NOT NULL,
    sheet_name      TEXT,               -- 主要 sheet
    checksum        TEXT,               -- 如 MD5/sha256，用于去重
    status          VARCHAR(20) NOT NULL DEFAULT 'IMPORTED',  -- 'IMPORTED'/'FAILED'/...
    remark          TEXT
);
```

#### 3.3.2 行级（结构化数据）：`excel_metric_value`

当 Excel 中行数据本质上是指标值时，可以解析为结构化记录，以便进一步写入 `metric_value_history`。

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
    raw_row_data    JSONB,     -- 原始行内容（备查）
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

后续 ETL：

- 从 `excel_metric_value` → 写入 `metric_value_history`，`source='EXCEL'`，`source_ref_id=excel_file_id`  
- 随即可参与 Java/AI/Decision 层的统一计算。

### 3.4 API 访问日志：`api_access_log`

用于审计和行为分析（哪些指标最常被查询、访问模式如何）。

```sql
CREATE TABLE api_access_log (
    id              BIGSERIAL PRIMARY KEY,
    api_name        VARCHAR(200) NOT NULL,
    http_method     VARCHAR(10)  NOT NULL,
    caller_id       VARCHAR(100),        -- 调用方用户/系统ID
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

数据量太大时，可只保留近一段时间在 Postgres，其余落数据湖。

### 3.5 Job 运行记录（可选）：`job_run`

用于记录 Java 定时任务、AI Batch、ETL 作业等运行情况，和血缘管理表联动。

```sql
CREATE TABLE job_run (
    id              BIGSERIAL PRIMARY KEY,
    job_name        TEXT NOT NULL,         -- 如 'java_metric_job', 'ai_forecast_batch'
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    status          VARCHAR(20),           -- 'SUCCESS'/'FAILED'/...
    params          JSONB,                 -- 运行参数
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 四、Metadata & Lineage 设计

元数据和血缘信息同样可以先用 PostgreSQL 表来表达图结构。

### 4.1 指标定义：`metric_definition`

用于统一管理各类 GRAS 指标的口径、单位、粒度等。

```sql
CREATE TABLE metric_definition (
    metric_id        VARCHAR(100) PRIMARY KEY,   -- 唯一 ID，与 runtime/history 表对齐
    name             TEXT NOT NULL,              -- 中文名称
    description      TEXT,
    dimension        VARCHAR(20) NOT NULL,       -- 'G'/'R'/'A'/'S'
    unit             VARCHAR(50),
    granularity      VARCHAR(20),                -- 'REALTIME'/'DAILY'/'WEEKLY'/'MONTHLY'
    owner_team       VARCHAR(100),
    sla_target       DOUBLE PRECISION,           -- 如 99.9
    calc_logic       TEXT,                       -- 自然语言描述
    calc_expression  TEXT,                       -- 公式或 DSL（可选）
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 4.2 数据集节点：`dataset`

在血缘图里，每个"数据实体"作为一个节点：

- 如：Excel 文件、DB 表、S3 Parquet、指标本身、模型产物等。

```sql
CREATE TABLE dataset (
    id              BIGSERIAL PRIMARY KEY,
    dataset_type    VARCHAR(50) NOT NULL,  -- 'EXCEL_FILE'/'S3_PARQUET'/'DB_TABLE'/'METRIC'/'MODEL'
    name            TEXT NOT NULL,
    ref_key         TEXT,                  -- 对应 excel_file_id / S3 key / 表名 / metric_id / 模型路径
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 4.3 血缘边：`lineage_edge`

表达 "从哪些数据集，通过哪个 Job，产生了另一个数据集"。

```sql
CREATE TABLE lineage_edge (
    id              BIGSERIAL PRIMARY KEY,
    from_dataset_id BIGINT NOT NULL REFERENCES dataset(id),
    to_dataset_id   BIGINT NOT NULL REFERENCES dataset(id),
    job_run_id      BIGINT REFERENCES job_run(id),
    transformation  TEXT,         -- 如 'aggregate_monthly', 'java_rule_calc', 'ai_forecast'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

例子：

- `EXCEL_FILE` → `metric_value_history`（源自手工表）  
- `S3 原始明细` → `特征表` → `模型` → `预测结果` → `metric_runtime_state`  

都可以建成一条链。

---

## 五、Java 规则 / AI 算法 / Excel 在三层中的落地方式

综合说明一下三种来源在三层的走向：

### 5.1 Java 规则计算

- 实时 / 定时计算结果：
  - 写入 `metric_value_history`（source=`'JAVA_RULE'`）
  - 更新 Hot Storage 中 `metric_runtime_state.current_value` 相关字段
- Java 规则预测：
  - 写入 `metric_prediction_history`（source=`'JAVA_RULE'`）
  - 更新 Hot Storage 中 `last_prediction_*`，`last_prediction_source='JAVA_RULE'`
- 对应 Job 执行信息：
  - 写入 `job_run`，并在 `dataset + lineage_edge` 中登记从哪个数据源到哪个结果表。

### 5.2 AI 算法（Python Batch / Cloud Run / Fargate）

- 从数据湖 / Postgres 拉历史数据：
  - 训练/推理生成预测结果。
- 输出：
  - 大体量明细落 S3（冷层）
  - 汇总/对齐后的预测结果写入 `metric_prediction_history`（source=`'AI_MODEL'`）
  - 更新 Hot Storage 中 `last_prediction_*` 和 `current_baseline/current_threshold`
- 同样通过 `job_run + dataset + lineage_edge` 记录血缘。

### 5.3 手工 Excel 上传

- 上传行为：
  - 文件放 S3/GCS；写一条 `excel_file`
- 解析：
  - 每一行写入 `excel_metric_value`，可附带 `raw_row_data`
- ETL：
  - 将可作为指标的行转换为 `metric_value_history` 记录，`source='EXCEL'`，`source_ref_id=excel_file_id`
- 上层使用：
  - Java / AI 读取 `metric_value_history` 时即可统一访问 Excel 和系统产出的指标数据。

---

## 六、小结

通过：

- Hot Storage：`metric_runtime_state` 提供单点实时视图  
- Warm Storage（PostgreSQL）：`metric_value_history`、`metric_prediction_history`、`excel_*`、`api_access_log`、`job_run`  
- Metadata & Lineage：`metric_definition`、`dataset`、`lineage_edge`

可以让：

- **analysis 层**（Java 规则 + AI）——有统一的数据落地模式  
- **decision 层**——可以读取统一的实时/历史状态、区分来源并做融合  
- **presentation 层**——既能做实时展示，又可以回放历史、解释"这个指标是从哪来的"
