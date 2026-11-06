"""
Energy Big Data Analytics Engine (Vectorized Local Pipeline)
Executes distributed-equivalent feature engineering, statistical anomaly detection,
K-Means clustering distance scoring, and Non-Technical Loss (NTL) grid audits.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

def run_feature_engineering(readings_df):
    """
    Applies time-series feature engineering and rolling window statistics per meter.
    """
    df = readings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["meter_id", "timestamp"]).reset_index(drop=True)
    
    # Extract temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Cyclical hour encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    
    # Rolling 24-hour window aggregations grouped by meter
    # (Assuming hourly intervals, 24 periods = 24 hours)
    grouped = df.groupby("meter_id")["kwh"]
    
    df["rolling_mean_24h"] = grouped.transform(lambda x: x.rolling(window=24, min_periods=4).mean()).bfill()
    df["rolling_std_24h"] = grouped.transform(lambda x: x.rolling(window=24, min_periods=4).std()).fillna(0.05)
    df["rolling_max_24h"] = grouped.transform(lambda x: x.rolling(window=24, min_periods=4).max()).bfill()
    df["rolling_min_24h"] = grouped.transform(lambda x: x.rolling(window=24, min_periods=4).min()).bfill()
    
    # Peak-to-Average Ratio (PAR)
    df["par"] = df["rolling_max_24h"] / (df["rolling_mean_24h"] + 1e-4)
    
    # Dynamic Rolling Z-Score
    df["z_score"] = (df["kwh"] - df["rolling_mean_24h"]) / (df["rolling_std_24h"] + 1e-4)
    
    # Voltage deviation from standard (230V or 400V depending on customer type)
    v_target = np.where(df["customer_type"] == "INDUSTRIAL", 400.0, 230.0)
    df["voltage_dev"] = np.abs(df["voltage"] - v_target)
    
    return df

def run_kmeans_anomaly_detection(features_df, k=5, threshold_percentile=98.0):
    """
    K-Means behavioral clustering and distance-to-centroid anomaly scoring.
    """
    from scipy.cluster.vq import kmeans2
    
    df = features_df.copy()
    feature_cols = ["kwh", "rolling_mean_24h", "par", "z_score", "voltage_dev", "sin_hour", "cos_hour"]
    
    X = df[feature_cols].values
    # Robust normalization
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0) + 1e-5
    X_norm = (X - means) / stds
    X_norm = np.nan_to_num(X_norm)
    
    # Fit K-Means
    centroids, labels = kmeans2(X_norm, k=k, minit='points', seed=42)
    
    # Compute Euclidean distance to assigned cluster centroid
    assigned_centroids = centroids[labels]
    distances = np.linalg.norm(X_norm - assigned_centroids, axis=1)
    
    df["cluster_label"] = labels
    df["cluster_dist"] = distances
    
    # Outlier threshold based on distance percentile
    cutoff = np.percentile(distances, threshold_percentile)
    df["is_cluster_anomaly"] = distances > cutoff
    
    return df

def detect_anomalies_and_generate_alerts(df):
    """
    Multi-layer rule-based + statistical + cluster anomaly classification.
    """
    alerts = []
    df_classified = df.copy()
    
    df_classified["detected_anomaly"] = "NORMAL"
    df_classified["anomaly_severity"] = "NORMAL"
    df_classified["confidence_score"] = 0.0
    df_classified["deviation_pct"] = 0.0
    
    alert_counter = 1000
    
    for idx, row in df_classified.iterrows():
        mid = row["meter_id"]
        ts = row["timestamp"]
        feeder = row["feeder_id"]
        kwh = row["kwh"]
        base_kwh = row["rolling_mean_24h"]
        z = row["z_score"]
        v = row["voltage"]
        v_dev = row["voltage_dev"]
        is_cluster_outlier = row.get("is_cluster_anomaly", False)
        
        detected = "NORMAL"
        severity = "NORMAL"
        conf = 0.0
        
        dev_pct = round(((kwh - base_kwh) / (base_kwh + 1e-4)) * 100.0, 1)
        
        # 1. Voltage sag / power quality violation
        is_voltage_sag = (row["customer_type"] == "INDUSTRIAL" and (v < 360.0 or v > 440.0)) or                          (row["customer_type"] != "INDUSTRIAL" and (v < 205.0 or v > 255.0))
        if is_voltage_sag:
            detected = "VOLTAGE_SAG"
            severity = "CRITICAL" if (v < 195.0 or (row["customer_type"] == "INDUSTRIAL" and v < 340.0)) else "HIGH"
            conf = min(0.99, 0.75 + (v_dev / 80.0))
            
        # 2. Zero drop fault / dead meter
        elif kwh < 0.01 and base_kwh > 0.8:
            detected = "ZERO_CONSUMPTION_FAULT"
            severity = "CRITICAL"
            conf = 0.95
            
        # 3. Sudden consumption spike (surges)
        elif z > 3.2 or dev_pct > 250.0:
            detected = "CONSUMPTION_SPIKE"
            severity = "CRITICAL" if z > 4.5 else "HIGH"
            conf = min(0.98, 0.75 + (z / 15.0))
            
        # 4. Theft / Meter Bypass (sustained drop below baseline while active)
        elif (dev_pct < -55.0 and base_kwh > 1.2 and kwh > 0.05) or (is_cluster_outlier and dev_pct < -45.0):
            detected = "THEFT_BYPASS"
            severity = "CRITICAL"
            conf = min(0.97, 0.80 + abs(dev_pct) / 200.0)
            
        # 5. Cluster Behavioral Outlier (moderate divergence)
        elif is_cluster_outlier and abs(z) > 2.2:
            detected = "BEHAVIORAL_ANOMALY"
            severity = "MEDIUM"
            conf = 0.78
            
        if detected != "NORMAL":
            df_classified.at[idx, "detected_anomaly"] = detected
            df_classified.at[idx, "anomaly_severity"] = severity
            df_classified.at[idx, "confidence_score"] = round(conf, 3)
            df_classified.at[idx, "deviation_pct"] = dev_pct
            
            alerts.append({
                "alert_id": f"ALT-{alert_counter}",
                "timestamp": str(ts),
                "meter_id": mid,
                "feeder_id": feeder,
                "customer_type": row["customer_type"],
                "anomaly_type": detected,
                "severity": severity,
                "confidence_score": round(conf, 3),
                "actual_kwh": round(kwh, 3),
                "baseline_kwh": round(base_kwh, 3),
                "deviation_pct": dev_pct,
                "voltage": row["voltage"],
                "status": "OPEN"
            })
            alert_counter += 1
            
    alerts_df = pd.DataFrame(alerts)
    return df_classified, alerts_df

def calculate_feeder_losses(readings_df, feeder_master_df):
    """
    Audits mass energy balance at each feeder substation to pinpoint Non-Technical Losses (Theft).
    """
    consumer_hourly = readings_df.groupby(["feeder_id", "timestamp"])["kwh"].sum().reset_index()
    consumer_hourly.rename(columns={"kwh": "consumer_metered_kwh"}, inplace=True)
    
    consumer_hourly["timestamp"] = pd.to_datetime(consumer_hourly["timestamp"]).astype(str)
    feeder_master_df["timestamp"] = pd.to_datetime(feeder_master_df["timestamp"]).astype(str)
    
    merged = pd.merge(feeder_master_df, consumer_hourly, on=["feeder_id", "timestamp"], how="inner")
    
    merged["total_loss_kwh"] = merged["feeder_supplied_kwh"] - merged["consumer_metered_kwh"]
    merged["loss_percentage"] = (merged["total_loss_kwh"] / (merged["feeder_supplied_kwh"] + 1e-4)) * 100.0
    merged["loss_percentage"] = merged["loss_percentage"].round(2)
    
    # Classify technical vs non-technical losses
    # Expected technical loss ~ 4-6%. Anything above 8.5% implies theft / unmetered tapping
    merged["loss_category"] = np.where(
        merged["loss_percentage"] > 8.5,
        "SUSPECTED_THEFT_NTL",
        "NORMAL_TECHNICAL_LOSS"
    )
    
    merged["loss_id"] = [f"FLOSS-{1000 + i}" for i in range(len(merged))]
    
    return merged
