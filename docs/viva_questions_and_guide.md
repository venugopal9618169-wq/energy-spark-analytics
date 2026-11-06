# B.Tech Project - Comprehensive Viva Voce Preparation Guide

## Title
**Design and Development of a Working Prototype for Energy-consumption big-data analytics Using A distributed Spark analytics pipeline**

- **Candidate Name:** **YALAMANCHI VENU GOPAL**
- **Faculty Guide:** **Ms. G. Hema Sri**
- **Domain:** **Big Data Analytics, Data Warehousing & Visualization**
- **Keywords:** `energy`, `consumption`, `big`, `data`, `analytics`, `distributed`

---

## Category 1: Project Overview & Smart Grid Domain

### Q1: What is the primary objective of this project?
**Answer:** The objective is to design, develop, and evaluate an end-to-end distributed Big Data analytics pipeline using Apache Spark to process high-velocity smart-meter telemetry streams, engineer time-series features, detect consumption anomalies (power theft, surges, meter faults, voltage sags), store results in a star-schema Parquet data warehouse, and provide a real-time utility-operator decision support dashboard.

### Q2: What is the difference between Technical Losses (TL) and Non-Technical Losses (NTL)?
**Answer:**
- **Technical Losses (TL)**: Inherent physical dissipation of electrical energy as heat in conductors, cables, and transformers due to Joule heating (I^2 * R). These typically account for 4% to 7% of total generation and cannot be completely eliminated.
- **Non-Technical Losses (NTL)**: Losses caused by external human actions, primarily electricity theft (illegal line tapping, meter tampering, bypassing current transformers), billing inaccuracies, and defective unmetered connections. NTL represents massive financial losses for utilities.

### Q3: Why is energy consumption data considered "Big Data"?
**Answer:** Smart grids deploy Advanced Metering Infrastructure (AMI). If a utility has 1,000,000 smart meters reporting 15-minute readings:
- **Volume**: 96 readings/day/meter = 96 million records per day (~10 GB/day uncompressed). Over a year, this reaches multiple terabytes.
- **Velocity**: Continuous streaming telemetry arriving at sub-second intervals across thousands of distribution feeders.
- **Variety**: Mixed electrical telemetry including active energy (kWh), reactive energy (kVAR), voltage (V), power factor, frequency, and event logs.
- **Veracity**: Sensor noise, communication dropouts, and corrupted packets require distributed cleaning and imputation.

### Q4: What types of anomalies does the prototype detect?
**Answer:**
1. **Meter Tampering / Power Theft (`THEFT_BYPASS`)**: Sudden, sustained 50-80% drop in measured consumption while neighbouring feeder load remains high.
2. **Consumption Spikes (`CONSUMPTION_SPIKE`)**: Abnormal surges (3x to 6x normal load) caused by equipment short-circuits or unauthorized heavy machinery.
3. **Zero-Consumption Dropouts (`ZERO_CONSUMPTION_FAULT`)**: Meter recording 0.00 kWh while grid voltage is fully normal (indicates dead meter, tripped breaker, or disconnected CT).
4. **Voltage Sags / Swells (`VOLTAGE_SAG`)**: Voltage dipping below 205V or surging above 255V, violating power quality standards and risking equipment damage.
5. **Behavioral Outliers (`BEHAVIORAL_ANOMALY`)**: Unusual consumption profiles discovered via K-Means clustering distance scoring.

---

## Category 2: Apache Spark & Big Data Architecture

### Q5: Why did you choose Apache Spark over Hadoop MapReduce?
**Answer:**
1. **In-Memory Computing**: Spark caches intermediate RDDs/DataFrames in RAM, making iterative algorithms (like K-Means) and rolling time-series calculations up to 100x faster than MapReduce (which writes intermediate results to disk).
2. **Unified API**: Spark combines Spark SQL, MLlib (machine learning), and Structured Streaming in a single unified distributed framework.
3. **Rich Windowing Support**: Spark SQL provides native Window functions for calculating 24-hour moving averages and standard deviations across millions of partitioned meter streams.

### Q6: Explain the internal architecture of Apache Spark.
**Answer:**
- **Driver Program**: Runs the main process, creates the `SparkSession`, parses user code, generates the logical execution plan, translates it into a physical Directed Acyclic Graph (DAG) of stages, and coordinates task scheduling.
- **Cluster Manager**: Allocates compute resources (can be Standalone, Apache YARN, Mesos, or Kubernetes).
- **Worker Nodes / Executors**: Distributed worker processes that store cached data partitions in RAM/disk and execute tasks assigned by the Driver.

### Q7: What is Lazy Evaluation in Apache Spark? Why is it useful?
**Answer:**
Spark distinguishes between **Transformations** (e.g., `filter`, `select`, `withColumn`, `groupBy`) and **Actions** (e.g., `count`, `collect`, `write`, `show`).
Spark does not execute transformations immediately; instead, it records them as a lineage graph (DAG). Execution only triggers when an action is called.
**Benefits**: The Catalyst Optimizer can inspect the entire pipeline, apply filter pushdown, reorder operations, combine adjacent projections, and eliminate redundant calculations before execution.

### Q8: What are Narrow vs. Wide Transformations? Which one causes a shuffle?
**Answer:**
- **Narrow Transformations**: Each input partition contributes to at most one output partition (e.g., `map`, `filter`, `withColumn`). No network shuffle is required; tasks run entirely locally in executor memory.
- **Wide Transformations**: Multiple input partitions contribute to an output partition (e.g., `groupByKey`, `join`, `repartition`, window functions with `partitionBy`). These require a **network shuffle**, transferring data across cluster nodes, which is the most resource-intensive phase in distributed computing.

### Q9: How did you implement Window Functions in PySpark?
**Answer:**
We defined a window specification partitioned by `meter_id` and ordered by timestamp:
```python
window_spec = Window.partitionBy("meter_id")     .orderBy(F.col("timestamp").cast("long"))     .rangeBetween(-86400, 0) # 86400 seconds = 24 hours
```
This enables distributed calculation of rolling mean, rolling standard deviation, and rolling maximum per meter without collecting all data to the driver node.

---

## Category 3: Feature Engineering & Machine Learning

### Q10: Why use dynamic rolling statistics instead of static thresholds?
**Answer:** Electricity consumption has pronounced diurnal (day/night) cycles, seasonal shifts, and varies dramatically by customer category (Residential vs Commercial vs Industrial).
A static threshold like "flag if kWh > 10" would trigger hundreds of false alarms for an industrial plant while missing a severe spike in a small residential household. Dynamic rolling statistics evaluate each reading relative to that specific meter's recent 24-hour baseline.

### Q11: What is the Peak-to-Average Ratio (PAR)?
**Answer:**
PAR is defined as:
`PAR = max(kwh in 24h) / (mean(kwh in 24h) + epsilon)`
It measures how "spiky" the load profile is. High PAR indicates intermittent heavy loads (e.g., electric vehicle charging or arc welders). An artificially depressed PAR accompanied by a dropping baseline indicates an illegal shunt/bypass.

### Q12: How does the dynamic rolling Z-Score work? What is epsilon?
**Answer:**
`Z = (kwh_t - rolling_mean_24h) / (rolling_std_24h + epsilon)`
The Z-Score indicates how many standard deviations the current reading deviates from the moving average.
`epsilon` (1e-4) is a small regularization constant added to the denominator. If a meter is idle or turned off, `rolling_std` can become 0. Adding epsilon prevents division-by-zero errors.

### Q13: Why did you use Unsupervised Learning (K-Means) instead of Supervised Learning?
**Answer:** In real-world utility operations:
1. Ground-truth labels of power theft are scarce, confidential, and labor-intensive to obtain (requiring physical site raids).
2. The nature of theft and equipment malfunctions continually evolves (new bypass methods, inverter faults).
Unsupervised learning models normal multidimensional behavioral patterns and flags anything that lies far from established cluster centroids without needing prior training labels.

### Q14: How are cyclical time features engineered?
**Answer:** Hour of day ranges from 0 to 23. If fed directly as a numeric feature, hour 23 and hour 0 appear mathematically distant (difference of 23), whereas they are adjacent hours in real life.
We apply trigonometric cyclical transformations:
- `sin_hour = sin(2 * pi * hour / 24)`
- `cos_hour = cos(2 * pi * hour / 24)`
This maps hours onto a continuous 2D unit circle where hour 23 smoothly transitions into hour 0.

---

## Category 4: Data Warehousing & Storage

### Q15: Explain your Star Schema design.
**Answer:**
The Data Warehouse adopts a Star Schema to facilitate fast analytical (OLAP) aggregations:
- **Dimension Tables**:
  - `dim_meters`: Contains descriptive attributes (meter_id, customer_type, feeder_id, substation, tariff_plan, latitude, longitude).
  - `dim_feeders`: Contains grid infrastructure metadata (feeder_id, substation, capacity_kva, region).
- **Fact Tables**:
  - `fact_meter_readings`: Numerical telemetry (kwh, voltage, power factor, rolling_mean, z_score, par).
  - `fact_anomaly_alerts`: Alert logs with severity, anomaly category, confidence score, and deviation %.
  - `fact_feeder_losses`: Energy balance records (feeder supplied vs consumer metered).

### Q16: Why did you choose Apache Parquet format over CSV or JSON?
**Answer:**
1. **Columnar Storage**: In OLAP queries, an operator often queries only 2 columns (`timestamp`, `kwh`) out of 15. Parquet reads only the requested columns from disk, skipping unnecessary fields.
2. **High Compression**: Parquet organizes data of the same type together, enabling Snappy or GZIP compression algorithms to compress files by 70-85% compared to CSV.
3. **Embedded Schema & Statistics**: Parquet embeds column schemas and dictionary metadata (min, max per row group), enabling pushdown predicate skipping without full table scans.

### Q17: What is Partition Pruning?
**Answer:**
In our data warehouse, tables are partitioned by `feeder_id`. When a dashboard operator filters for `feeder_id = 'FDR-NORTH-01'`, the query engine reads only the directory containing that feeder's files, completely skipping all other feeder directories. This reduces I/O by 75% across 4 feeders.

---

## Category 5: Practical Implementation, Scalability & Defense

### Q18: How does the Feeder Balance Audit pinpoint power theft?
**Answer:**
By the Law of Conservation of Energy:
`E_feeder(t) = sum(E_metered(t)) + Technical_Losses(t) + Theft(t)`
If a consumer tampers with their meter to report lower kWh, their individual meter reading drops. However, the physical power is still drawn from the substation feeder!
The feeder master meter measures the true power flowing into that line segment. When `Feeder_Supplied - Sum(Meter_Readings)` exceeds normal technical loss (6%), the energy gap pinpointed on that specific feeder proves the existence of non-technical theft in that neighborhood.

### Q19: How did you implement dual-engine execution?
**Answer:**
To guarantee reproducibility across any presentation device:
1. `src/spark_pipeline.py`: Full distributed Apache Spark 3.x/4.x implementation using SparkSession, PySpark SQL, Window functions, and MLlib VectorAssembler/KMeans (intended for Spark clusters, Databricks, or cloud platforms).
2. `src/local_engine.py`: A vectorized Python/NumPy/Pandas implementation executing the identical mathematical formulas, rolling windows, and K-Means clustering logic for zero-friction local execution on Windows laptops without requiring Java/Hadoop configurations.

### Q20: How would this system scale to a city with 2 million smart meters?
**Answer:**
1. **Streaming Ingestion**: Deploy Apache Kafka or AWS Kinesis to buffer incoming high-velocity meter telemetry streams.
2. **Spark Structured Streaming**: Run PySpark Structured Streaming with micro-batches (e.g., 10-second intervals) across a Kubernetes-managed Spark cluster (e.g., 20 worker nodes with 16 cores and 64 GB RAM each).
3. **Delta Lake / Iceberg**: Use Delta Lake with ACID transactions, time travel, and auto-compaction over object storage (S3/GCS/HDFS).
4. **Caching & Redis**: Cache high-priority alerts in Redis for sub-millisecond retrieval by the utility operator dashboard.
