# Energy-Consumption Big-Data Analytics Prototype

### Title
**Design and Development of a Working Prototype for Energy-consumption big-data analytics Using A distributed Spark analytics pipeline**

| Academic Information | Details |
| :--- | :--- |
| **Candidate Name** | **YALAMANCHI VENU GOPAL** |
| **Supervisor / Faculty Guide** | **Ms. G. Hema Sri** |
| **Department / Domain** | **Big Data Analytics, Data Warehousing & Visualization** |
| **Project Deliverable** | **Working Prototype & Real-time Utility Anomaly Dashboard** |
| **Keywords** | `energy`, `consumption`, `big`, `data`, `analytics`, `distributed` |

---

## 📋 Abstract
> *This work presents a distributed Spark analytics pipeline to identify consumption anomalies across a smart grid, addressing the problem of energy-consumption big-data analytics through a data-driven approach. The system is trained and validated using utility smart-meter big-data streams, with pre-processing and feature engineering steps tailored to the domain. The final deliverable is a utility-operator anomaly dashboard, aimed at making the solution usable by real stakeholders.*

---

## ⚡ Key Features

1. **Smart Meter Big Data Stream Generator (`src/generator.py`)**:
   - Simulates multi-class smart meter telemetry (Residential, Commercial, Industrial) across distribution feeders and substations.
   - Injects realistic, domain-specific anomalies:
     - **Meter Tampering / Power Theft (`THEFT_BYPASS`)**
     - **Sudden Consumption Spikes (`CONSUMPTION_SPIKE`)**
     - **Zero Consumption Faults (`ZERO_CONSUMPTION_FAULT`)**
     - **Voltage Sags / Power Quality Swells (`VOLTAGE_SAG`)**
   - Simulates primary feeder master meter checkpoints for substation-level mass energy balance auditing.

2. **Distributed Spark Pipeline (`src/spark_pipeline.py`)**:
   - Distributed ingestion and schema validation using PySpark SQL.
   - PySpark Windowed feature engineering: 24-hour rolling averages, moving standard deviation, dynamic Z-Scores, and Peak-to-Average Ratio (PAR).
   - PySpark MLlib (`VectorAssembler`, `StandardScaler`, `KMeans`) for behavioral pattern clustering and centroid distance anomaly scoring.
   - Feeder-level mass-energy balance conservation calculations to pinpoint Non-Technical Losses (NTL).

3. **Dual Execution Engine (`src/local_engine.py`)**:
   - Zero-dependency local vectorized engine executing the exact same mathematical models and transformations using Pandas/NumPy/SciPy for seamless execution on any standard laptop without requiring Java/Hadoop configurations.

4. **Star-Schema Data Warehouse (`src/warehouse.py`)**:
   - Partitioned columnar storage using Apache Parquet (Snappy compressed).
   - **Dimension Tables**: `dim_meters`, `dim_feeders`.
   - **Fact Tables**: `fact_meter_readings`, `fact_anomaly_alerts`, `fact_feeder_losses`.

5. **Utility-Operator Anomaly Dashboard (`dashboard/app.py`)**:
   - Modern Streamlit web application tailored for grid control room operators:
     - **Tab 1: Grid Telemetry & Real-Time KPIs**: Total MWh delivered, active meters, anomaly counts, feeder loss rates, and hourly load curves.
     - **Tab 2: Anomaly Alert Center & Dispatch**: Filterable alert log with severity badges, deviation percentages, confidence scores, and field crew dispatch actions.
     - **Tab 3: Smart Meter Deep-Dive**: Time-series visualizer showing actual consumption vs. dynamic rolling baseline, 3-sigma upper/lower control limits, and anomaly markers.
     - **Tab 4: Feeder Balance & Non-Technical Loss (Theft Locator)**: Substation injected energy vs. aggregate consumer metered readings, highlighting feeders with >8.5% loss indicating illegal line taps.
     - **Tab 5: Live Stream Simulator & Anomaly Injector**: Interactive testbench to inject synthetic anomalies on the fly and verify sub-second detection.
     - **Tab 6: Academic Viva & Architecture Guide**: Complete architectural and algorithmic documentation built right into the UI for live academic defense.

6. **Academic Documentation & Viva Prep**:
   - Detailed System Architecture & Design Document (`docs/architecture_and_design.md`).
   - 30-Question Viva Voce Preparation Guide (`docs/viva_questions_and_guide.md`).

---

## ?? Project Directory Structure

```
energy_spark_analytics/
?
??? dashboard/
?   ??? app.py                      # Interactive Streamlit Utility Operator Dashboard
?
??? data/
?   ??? raw_meter_readings.parquet  # Ingested raw telemetry
?   ??? feeder_master_readings.parquet
?   ??? smart_meters_meta.csv
?   ??? warehouse/                  # Star-Schema Data Warehouse (Parquet)
?       ??? dim_meters/
?       ??? dim_feeders/
?       ??? fact_meter_readings/
?       ??? fact_anomaly_alerts/
?       ??? fact_feeder_losses/
?
??? docs/
?   ??? architecture_and_design.md  # Detailed System Architecture & Equations
?   ??? viva_questions_and_guide.md # 30+ Viva Voce questions & model answers
?
??? src/
?   ??? generator.py                # Smart meter stream generator & anomaly injector
?   ??? local_engine.py             # Vectorized local analytics & ML engine
?   ??? spark_pipeline.py           # Production Apache Spark distributed pipeline
?   ??? warehouse.py                # Star-schema Parquet storage manager
?
??? requirements.txt                # Python dependencies
??? run_dashboard.bat               # Windows one-click launcher for the dashboard
??? run_pipeline.py                 # CLI script to execute end-to-end pipeline
??? README.md                       # Main project documentation
```

---

## ? Quickstart Guide

### 1. Install Dependencies
Open your command terminal in the project directory and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Big Data Analytics Pipeline
Execute the end-to-end data generation, distributed feature engineering, ML anomaly detection, and data warehouse persistence:
```bash
python run_pipeline.py
```
*Output Summary:*
- Ingests **20,160+ readings** across 60 smart meters and 4 distribution feeders.
- Calculates rolling statistics, Z-scores, and Peak-to-Average Ratios.
- Flags and categorizes anomalies (Theft, Spikes, Zero Dropouts, Voltage Sags).
- Populates the Parquet Star-Schema Data Warehouse in under 2 seconds.

### 3. Launch the Utility-Operator Dashboard
Launch the interactive web application using:
```bash
streamlit run dashboard/app.py
```
*(Or simply double-click `run_dashboard.bat` on Windows)*

Open your browser to:
?? **`http://localhost:8501`**

---

## ?? Examination & Viva Voce Ready
Refer to the comprehensive guides in `docs/`:
- `docs/architecture_and_design.md`: Explains the math (Z-score, PAR, KMeans distance, Mass-energy balance).
- `docs/viva_questions_and_guide.md`: Contains 30+ commonly asked viva questions on Apache Spark, Big Data architectures, ML anomaly detection, and data warehousing.
