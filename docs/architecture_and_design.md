# System Architecture & Technical Design Document

## Project Title
**Design and Development of a Working Prototype for Energy-consumption big-data analytics Using A distributed Spark analytics pipeline**

| Academic Specification | Project Detail |
| :--- | :--- |
| **Author / Candidate** | **YALAMANCHI VENU GOPAL** |
| **Faculty Guide / Mentor** | **Ms. G. Hema Sri** |
| **Domain Specialization** | **Big Data Analytics, Data Warehousing & Visualization** |
| **Final Deliverable** | **Distributed Spark Analytics Pipeline & Utility-Operator Anomaly Dashboard** |
| **Keywords** | `energy`, `consumption`, `big`, `data`, `analytics`, `distributed` |

---

## Abstract
*This work presents a distributed Spark analytics pipeline to identify consumption anomalies across a smart grid, addressing the problem of energy-consumption big-data analytics through a data-driven approach. The system is trained and validated using utility smart-meter big-data streams, with pre-processing and feature engineering steps tailored to the domain. The final deliverable is a utility-operator anomaly dashboard, aimed at making the solution usable by real stakeholders.*

---

## 1. Executive Summary & Problem Formulation
Modern electrical power grids are rapidly transitioning into **Smart Grids** equipped with **Advanced Metering Infrastructure (AMI)**. Smart meters continuously capture fine-grained energy consumption readings (typically at 15-minute to 1-hour intervals) alongside power quality variables (voltage, power factor, reactive power).

### Core Challenges:
1. **Big Data Volume & Velocity**: Ingesting and analyzing continuous time-series data from hundreds of thousands of smart meters generates gigabytes to terabytes of operational telemetry that overwhelm traditional single-node databases.
2. **Non-Technical Losses (NTL)**: Power theft, unmetered tapping, and physical meter tampering result in billions of dollars in global annual utility losses.
3. **Equipment Faults & Power Quality Outages**: Sudden surges, zero-consumption meter dropouts, and voltage sags degrade grid reliability and asset longevity.
4. **Lack of Operational Decision Support**: Utility control room operators need real-time visualization and prioritized anomaly alerting to dispatch field crews effectively.

---

## 2. Distributed Architecture Overview

The prototype is organized into a four-tier distributed data pipeline:

```
[ Tier 1: Data Ingestion & Simulation Layer ]
       |
       v  (High-Velocity Telemetry Streams: kWh, V, PF, kVAR)
[ Tier 2: Distributed Spark Analytics Engine ]
       * Pre-processing & Imputation
       * PySpark Windowed Feature Engineering (24h Rolling Mean/Std, PAR, Z-Score)
       * PySpark MLlib K-Means Behavioral Clustering & Centroid Distance Outlier Scoring
       * Feeder Mass-Energy Balance Auditing
       |
       v  (Optimized Columnar Fact & Dimension Tables)
[ Tier 3: Star-Schema Data Warehouse (Parquet Lakehouse) ]
       * dim_meters, dim_feeders
       * fact_meter_readings, fact_anomaly_alerts, fact_feeder_losses
       |
       v  (Sub-second OLAP Querying & Filtering)
[ Tier 4: Utility-Operator Streamlit Anomaly Dashboard ]
       * Executive KPIs, Alert Center, Meter Deep-Dive, Feeder Theft Locator
```

---

## 3. Mathematical & Algorithmic Formulations

### 3.1 Dynamic Sliding Window Statistics
Static thresholds fail in energy analytics because consumption naturally follows diurnal cycles (morning peaks, evening peaks, weekday vs. weekend shifts).
To capture local context, we compute dynamic rolling statistics over a 24-hour sliding window W:

- Rolling 24-hour Mean:
  mu_{t, 24h} = (1 / |W|) * sum_{i in W} x_{t-i}

- Rolling 24-hour Standard Deviation:
  sigma_{t, 24h} = sqrt((1 / |W|) * sum_{i in W} (x_{t-i} - mu_{t, 24h})^2)

### 3.2 Dynamic Rolling Z-Score
The Z-Score measures standard deviations from the local temporal baseline:

  Z_t = (x_t - mu_{t, 24h}) / (sigma_{t, 24h} + epsilon)

Where epsilon = 1e-4 prevents division-by-zero during continuous low-load states. Readings with |Z_t| > 3.2 are flagged as statistical spikes or drops.

### 3.3 Peak-to-Average Ratio (PAR)
PAR quantifies the peakiness of consumer load profiles:

  PAR_t = max_{i in W}(x_{t-i}) / (mu_{t, 24h} + epsilon)

A sudden spike results in an elevated PAR, while an illegal meter bypass significantly depresses overall consumption while keeping PAR artificially flat.

### 3.4 Unsupervised Behavioral Anomaly Scoring (MLlib K-Means)
To capture multivariate behavioral divergence without manual labels, the pipeline constructs feature vectors:

  v = [kWh, mu_{24h}, PAR, Z, |V - V_{nominal}|, sin(omega * t), cos(omega * t)]

After z-score standardization, samples are clustered using K-Means (k=5). The anomaly score is defined as the Euclidean distance to the assigned cluster centroid c_k:

  D(v) = || v - c_k ||_2

Data points situated in the upper 98th percentile (D > D_98) are classified as behavioral outliers.

### 3.5 Feeder Energy Balance & Non-Technical Loss (NTL) Audit
Under the conservation of energy, the energy supplied by a distribution substation feeder must equal the aggregate metered consumption plus technical line losses (TL):

  E_{feeder}(t) = sum_{m in F} E_m(t) + TL(t) + NTL(t)

  Loss Rate % = ((E_{feeder}(t) - sum_{m in F} E_m(t)) / E_{feeder}(t)) * 100%

- Normal Technical Line Loss: 4.0% - 6.5% (resistive I^2R heating in conductors and transformers).
- Non-Technical Loss / Theft Threshold: > 8.5%. Any feeder segment exceeding this threshold is automatically flagged for field inspection.

---

## 4. Star-Schema Data Warehouse Architecture

The project implements a dimensional Star Schema optimized for OLAP analytics:

- **dim_meters**: meter_id, customer_type, feeder_id, substation, tariff_plan, latitude, longitude
- **dim_feeders**: feeder_id, substation, capacity_kva, region
- **fact_meter_readings**: reading_id, timestamp, meter_id, feeder_id, kwh, voltage, power_factor, rolling_mean_24h, z_score, par
- **fact_anomaly_alerts**: alert_id, timestamp, meter_id, feeder_id, anomaly_type, severity, confidence_score, actual_kwh, baseline_kwh, deviation_pct
- **fact_feeder_losses**: loss_id, timestamp, feeder_id, feeder_supplied_kwh, consumer_metered_kwh, total_loss_kwh, loss_percentage, loss_category

### Storage Format:
- Columnar Apache Parquet with Snappy compression.
- Partitioned by feeder_id for efficient partition pruning during high-speed grid queries.
