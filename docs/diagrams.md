# Diagrams for the report (all match the code that was actually built)

How to use: open https://mermaid.live , paste ONE diagram's code (without the triple backticks), then
**Actions → PNG** (or SVG) and insert the image in your report. Use 2x or 3x scale for sharp text.

---

## Figure A. System architecture

```mermaid
flowchart LR
    subgraph SRC["Data sources (Azure PdM, CSV)"]
        S1["PdM_telemetry"]
        S2["PdM_errors"]
        S3["PdM_maint"]
        S4["PdM_failures"]
        S5["PdM_machines"]
    end
    subgraph PREP["Python data layer (pandas)"]
        C["clean_data.py<br/>validate, type, dedupe,<br/>quarantine, log"]
        F["features.py<br/>rolling stats, error counts,<br/>time since maintenance, labels"]
    end
    subgraph WH["DuckDB warehouse (pdm.duckdb)"]
        D["Star schema<br/>3 dims + date dim<br/>4 fact tables"]
        O["Operational tables<br/>predictions, batch_log,<br/>drift_log, model_registry"]
    end
    subgraph ML["Machine learning (scikit-learn)"]
        T["train_models.py<br/>Dummy, LogReg,<br/>Random Forest, HistGB"]
        E["evaluate_models.py<br/>PR-AUC, threshold, test once"]
    end
    subgraph AD["Adaptive engine (adapt.py)"]
        V["Validate and ingest batch"]
        SC["Score with champion<br/>risk bands"]
        DR["Drift check (PSI)"]
        FB["Outcome feedback<br/>and performance check"]
        RT["Challenger training<br/>promote / rollback"]
    end
    UI["Streamlit dashboard<br/>6 pages (Plotly)"]
    USER(("Maintenance<br/>manager"))

    S1 --> C
    S2 --> C
    S3 --> C
    S4 --> C
    S5 --> C
    C --> F
    F --> T
    T --> E
    C --> D
    E -->|"champion v1,<br/>thresholds"| O
    V --> D
    V --> SC
    SC --> DR
    SC --> O
    DR --> O
    FB --> O
    O --> RT
    RT --> O
    D --> UI
    O --> UI
    USER <--> UI
    UI -->|"upload batch /<br/>outcomes / retrain"| V
    UI --> FB
    UI --> RT
```

---

## Figure B. Data cleaning and feature-engineering workflow

```mermaid
flowchart TD
    A["Raw CSV files<br/>data/raw (never edited)"] --> B["Read everything as text"]
    B --> C["Standardise column names"]
    C --> D["Convert types<br/>datetime, numbers"]
    D --> E{"Key fields valid?<br/>machine, time, category"}
    E -- "no" --> Q["Quarantine file<br/>(kept for audit)"]
    E -- "yes" --> F{"Machine ID exists in<br/>machines table?"}
    F -- "no" --> Q
    F -- "yes" --> G["Remove exact duplicates<br/>check business-key conflicts"]
    G --> H["Report-only checks<br/>missing values, outliers, gaps"]
    H --> I["Save clean parquet files<br/>+ cleaning log + quality report"]
    I --> J["Build features per machine-hour<br/>using past data only"]
    J --> K["Create label y:<br/>failure in next 24 h"]
    K --> L["Drop warm-up rows and<br/>rows without 24 h of future data"]
    L --> M["Chronological split<br/>train / validation / test<br/>with 24 h purge gaps"]
    M --> N["Leakage audit<br/>and automated tests"]
```

---

## Figure C. Program flow chart: model training and evaluation

```mermaid
flowchart TD
    A(["Start"]) --> B["Load feature table"]
    B --> C["Split by time<br/>train Jan-Aug, val Sep-Oct, test Nov-Dec"]
    C --> D["Fit pipeline on TRAIN only<br/>impute, scale or encode, model"]
    D --> E["Train 4 models<br/>Dummy, Logistic, Random Forest, HistGB"]
    E --> F["Score validation and test periods"]
    F --> G["Pick best model by<br/>validation PR-AUC"]
    G --> H["Choose alert threshold<br/>that maximises validation F1"]
    H --> I["Report test results ONCE<br/>with fixed model and threshold"]
    I --> J["Save model, thresholds,<br/>metadata, baseline statistics"]
    J --> K["Register as champion v1<br/>in model_registry"]
    K --> L(["End"])
```

---

## Figure D. Batch prediction workflow

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Dashboard
    participant AD as Adaptive engine
    participant WH as DuckDB warehouse
    participant M as Champion model
    U->>UI: Upload weekly CSV files
    UI->>AD: process_batch()
    AD->>AD: Validate schema, types, IDs, duplicates
    alt validation fails
        AD-->>UI: Clear error messages, nothing stored
    else validation passes
        AD->>WH: BEGIN transaction, insert facts
        AD->>WH: Read last 168 h of history
        AD->>AD: Build features (same code as training)
        AD->>M: predict_proba()
        M-->>AD: scores
        AD->>AD: Risk bands from thresholds, PSI drift check
        AD->>WH: Insert predictions, drift_log, batch_log, COMMIT
        AD-->>UI: Summary, updated priority list
    end
    UI-->>U: Dashboard refreshed
```

---

## Figure E. Adaptive feedback loop

```mermaid
flowchart TD
    A["New batch arrives"] --> B["Validate and store"]
    B --> C["Score with champion model"]
    C --> D["Compute PSI drift<br/>(no labels needed)"]
    D --> E{"Drift alert?"}
    E -- "yes" --> F["Flag on dashboard,<br/>investigate"]
    E -- "no" --> G["Continue"]
    F --> H["Wait for outcomes<br/>(24 h horizon must pass)"]
    G --> H
    H --> I["Record actual failures,<br/>declare known-until time"]
    I --> J["Mature predictions:<br/>attach actual outcome"]
    J --> K{"Performance degraded?<br/>F1 below validation F1 minus 0.05"}
    K -- "no" --> A
    K -- "yes" --> L["Train challenger on<br/>recent labelled window"]
    L --> M{"Beats champion by margin<br/>on newest validation block<br/>and F1 not worse?"}
    M -- "no" --> N["Reject, keep champion"]
    M -- "yes" --> O["Promote challenger<br/>old champion kept for rollback"]
    N --> A
    O --> A
    O -.->|"if problems appear"| P["Rollback to previous champion"]
    P --> A
```

---

## Figure F. Warehouse schema (star schema plus operational tables)

```mermaid
erDiagram
    dim_machine ||--o{ fact_telemetry : "has"
    dim_machine ||--o{ fact_error : "has"
    dim_machine ||--o{ fact_maintenance : "has"
    dim_machine ||--o{ fact_failure : "has"
    dim_date ||--o{ fact_telemetry : "on"
    dim_date ||--o{ fact_error : "on"
    dim_date ||--o{ fact_maintenance : "on"
    dim_date ||--o{ fact_failure : "on"
    dim_error_type ||--o{ fact_error : "type"
    dim_component ||--o{ fact_maintenance : "component"
    dim_component ||--o{ fact_failure : "component"
    dim_machine ||--o{ predictions : "scored"
    model_registry ||--o{ predictions : "made by"

    dim_machine {
        int machine_id PK
        string model
        int age
    }
    dim_date {
        int date_key PK
        date full_date
        int year
        int month
        string day_name
    }
    dim_component {
        string component_id PK
    }
    dim_error_type {
        string error_type_id PK
    }
    fact_telemetry {
        int machine_id PK
        datetime reading_time PK
        int date_key FK
        float volt
        float rotate
        float pressure
        float vibration
    }
    fact_error {
        int machine_id PK
        datetime event_time PK
        string error_type_id PK
        int date_key FK
    }
    fact_maintenance {
        int machine_id PK
        datetime event_time PK
        string component_id PK
        int date_key FK
    }
    fact_failure {
        int machine_id PK
        datetime event_time PK
        string component_id PK
        int date_key FK
    }
    predictions {
        string batch_id
        int machine_id
        datetime prediction_time
        float score
        string risk_band
        string model_version
        int actual_outcome
        string outcome_status
    }
    model_registry {
        string version PK
        string status
        float threshold_high
        float val_pr_auc
    }
```

---

## Figure G. User flow chart (dashboard)

```mermaid
flowchart TD
    A(["Open dashboard"]) --> B["1. Fleet overview<br/>KPIs, risk bands, priority list"]
    B --> C{"High-risk machine<br/>on the list?"}
    C -- "yes" --> D["2. Machine drill-down<br/>sensors, errors, maintenance, score"]
    C -- "no" --> E["3. Failure and error analysis<br/>trends by component, model, month"]
    D --> F["Decide: schedule inspection<br/>or maintenance"]
    E --> G["4. Model performance<br/>adjust alert threshold, see trade-off"]
    F --> G
    G --> H["5. Monitoring and model management<br/>drift, live performance"]
    H --> I{"Drift or degradation alert?"}
    I -- "yes" --> J["Train challenger, review criteria,<br/>promote or reject, rollback if needed"]
    I -- "no" --> K["6. New data and feedback<br/>load next batch, record outcomes"]
    J --> K
    K --> B
```

---

## Figure H. Dashboard page map

```mermaid
flowchart LR
    W[("DuckDB<br/>warehouse")]
    subgraph PAGES["Streamlit pages"]
        P1["1 Fleet overview"]
        P2["2 Machine drill-down"]
        P3["3 Failure and error analysis"]
        P4["4 Model performance"]
        P5["5 Monitoring and model management"]
        P6["6 New data and feedback"]
    end
    W -->|"predictions, dim_machine"| P1
    W -->|"fact_telemetry, fact_failure,<br/>fact_error, fact_maintenance,<br/>predictions"| P2
    W -->|"star-schema queries"| P3
    W -->|"matured predictions"| P4
    W -->|"drift_log, batch_log,<br/>model_registry, model_events"| P5
    P6 -->|"validated batches and outcomes"| W
```
