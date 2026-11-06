"""
Energy Big Data Warehouse Module
Implements a Star Schema Data Warehouse stored in partitioned Parquet format.
Tables:
- dim_meters
- dim_feeders
- fact_meter_readings
- fact_anomaly_alerts
- fact_feeder_losses
"""

import os
import pandas as pd

WAREHOUSE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "warehouse")

def init_warehouse_directories():
    os.makedirs(WAREHOUSE_DIR, exist_ok=True)
    for table in ["dim_meters", "dim_feeders", "fact_meter_readings", "fact_anomaly_alerts", "fact_feeder_losses"]:
        os.makedirs(os.path.join(WAREHOUSE_DIR, table), exist_ok=True)

def save_dim_meters(meters_df):
    init_warehouse_directories()
    cols = ["meter_id", "customer_type", "feeder_id", "substation", "tariff_plan", "latitude", "longitude"]
    df = meters_df[[c for c in cols if c in meters_df.columns]].drop_duplicates()
    out_path = os.path.join(WAREHOUSE_DIR, "dim_meters", "dim_meters.parquet")
    df.to_parquet(out_path, index=False)
    return out_path

def save_dim_feeders(feeder_configs):
    init_warehouse_directories()
    df = pd.DataFrame(feeder_configs).drop_duplicates()
    out_path = os.path.join(WAREHOUSE_DIR, "dim_feeders", "dim_feeders.parquet")
    df.to_parquet(out_path, index=False)
    return out_path

def save_fact_meter_readings(readings_df):
    init_warehouse_directories()
    out_path = os.path.join(WAREHOUSE_DIR, "fact_meter_readings", "readings.parquet")
    readings_df.to_parquet(out_path, index=False)
    return out_path

def save_fact_anomaly_alerts(alerts_df):
    init_warehouse_directories()
    out_path = os.path.join(WAREHOUSE_DIR, "fact_anomaly_alerts", "alerts.parquet")
    alerts_df.to_parquet(out_path, index=False)
    return out_path

def save_fact_feeder_losses(losses_df):
    init_warehouse_directories()
    out_path = os.path.join(WAREHOUSE_DIR, "fact_feeder_losses", "losses.parquet")
    losses_df.to_parquet(out_path, index=False)
    return out_path

def load_warehouse_table(table_name):
    table_path = os.path.join(WAREHOUSE_DIR, table_name)
    if not os.path.exists(table_path):
        return pd.DataFrame()
    files = [os.path.join(table_path, f) for f in os.listdir(table_path) if f.endswith(".parquet")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
