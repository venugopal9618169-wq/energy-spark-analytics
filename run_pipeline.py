"""
Main Pipeline Runner for Energy Consumption Big Data Analytics
Orchestrates:
1. Synthetic Smart Meter Big Data Generation
2. Pre-processing & Distributed Feature Engineering
3. Multi-layer Machine Learning & Statistical Anomaly Detection
4. Star-Schema Data Warehouse Population (Partitioned Parquet)
5. Executive Summary & Verification Metrics
"""

import os
import sys
import time

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generator import generate_smart_meter_streams, FEEDER_CONFIGS
from src.local_engine import (
    run_feature_engineering,
    run_kmeans_anomaly_detection,
    detect_anomalies_and_generate_alerts,
    calculate_feeder_losses
)
from src.warehouse import (
    save_dim_meters,
    save_dim_feeders,
    save_fact_meter_readings,
    save_fact_anomaly_alerts,
    save_fact_feeder_losses
)

def main():
    print("="*75)
    print(" ENERGY-CONSUMPTION BIG-DATA ANALYTICS PIPELINE ")
    print(" Design & Development of a Working Prototype ")
    print("="*75)
    
    start_time = time.time()
    
    # -------------------------------------------------------------
    # STEP 1: Smart Meter Big Data Stream Generation
    # -------------------------------------------------------------
    print("\n[1/4] Ingesting & Simulating Smart Meter Big Data Streams...")
    meters_df, readings_df, feeder_master_df = generate_smart_meter_streams(
        days=14,
        interval_hours=1.0,
        num_meters=60
    )
    print(f"  -> Total Smart Meters Provisioned: {len(meters_df)}")
    print(f"  -> Total Telemetry Readings Ingested: {len(readings_df):,}")
    print(f"  -> Feeder Master Checkpoints: {len(feeder_master_df):,}")
    
    # -------------------------------------------------------------
    # STEP 2: Distributed Feature Engineering & ML Clustering
    # -------------------------------------------------------------
    print("\n[2/4] Executing Distributed Feature Engineering & MLlib Clustering...")
    print("  -> Computing 24-hour rolling averages, stddev, PAR, and dynamic Z-Scores...")
    features_df = run_feature_engineering(readings_df)
    
    print("  -> Fitting K-Means behavioral clustering (k=5) and computing centroid distances...")
    clustered_df = run_kmeans_anomaly_detection(features_df, k=5, threshold_percentile=98.0)
    
    # -------------------------------------------------------------
    # STEP 3: Multi-Layer Anomaly Detection & Feeder Loss Audits
    # -------------------------------------------------------------
    print("\n[3/4] Running Multi-Tier Anomaly Classifier & Feeder Energy Balance...")
    classified_df, alerts_df = detect_anomalies_and_generate_alerts(clustered_df)
    losses_df = calculate_feeder_losses(readings_df, feeder_master_df)
    
    print(f"  -> Total Anomalous Events Flagged: {len(alerts_df):,}")
    critical_cnt = len(alerts_df[alerts_df['severity'] == 'CRITICAL'])
    high_cnt = len(alerts_df[alerts_df['severity'] == 'HIGH'])
    medium_cnt = len(alerts_df[alerts_df['severity'] == 'MEDIUM'])
    print(f"     * CRITICAL Severity: {critical_cnt}")
    print(f"     * HIGH Severity:     {high_cnt}")
    print(f"     * MEDIUM Severity:   {medium_cnt}")
    
    theft_cases = len(alerts_df[alerts_df['anomaly_type'] == 'THEFT_BYPASS'])
    surge_cases = len(alerts_df[alerts_df['anomaly_type'] == 'CONSUMPTION_SPIKE'])
    fault_cases = len(alerts_df[alerts_df['anomaly_type'] == 'ZERO_CONSUMPTION_FAULT'])
    sag_cases = len(alerts_df[alerts_df['anomaly_type'] == 'VOLTAGE_SAG'])
    print(f"  -> Anomaly Breakdown:")
    print(f"     * Meter Tampering / Power Theft: {theft_cases}")
    print(f"     * Sudden Consumption Spikes:    {surge_cases}")
    print(f"     * Zero Consumption Dropouts:    {fault_cases}")
    print(f"     * Voltage Sags / Swells:         {sag_cases}")
    
    # -------------------------------------------------------------
    # STEP 4: Persisting to Star-Schema Data Warehouse (Parquet)
    # -------------------------------------------------------------
    print("\n[4/4] Persisting tables to Star-Schema Data Warehouse (Parquet Lakehouse)...")
    dim_meters_p = save_dim_meters(meters_df)
    dim_feeders_p = save_dim_feeders(FEEDER_CONFIGS)
    fact_readings_p = save_fact_meter_readings(classified_df)
    fact_alerts_p = save_fact_anomaly_alerts(alerts_df)
    fact_losses_p = save_fact_feeder_losses(losses_df)
    
    print(f"  -> dim_meters:          {dim_meters_p}")
    print(f"  -> dim_feeders:         {dim_feeders_p}")
    print(f"  -> fact_meter_readings: {fact_readings_p}")
    print(f"  -> fact_anomaly_alerts: {fact_alerts_p}")
    print(f"  -> fact_feeder_losses:  {fact_losses_p}")
    
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "="*75)
    print(f" PIPELINE EXECUTION COMPLETED IN {elapsed}s ")
    print(" Data Warehouse is ready for Utility Operator Dashboard visualization! ")
    print("="*75)

if __name__ == "__main__":
    main()
