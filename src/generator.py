"""
Energy Consumption Big Data Simulator
Simulates smart meter readings across a smart grid distribution network with 
realistic diurnal patterns, noise, and injected ground-truth anomalies.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

FEEDER_CONFIGS = [
    {"feeder_id": "FDR-NORTH-01", "substation": "Substation-Alpha", "capacity_kva": 2500, "region": "North Metro"},
    {"feeder_id": "FDR-NORTH-02", "substation": "Substation-Alpha", "capacity_kva": 2000, "region": "North Metro"},
    {"feeder_id": "FDR-EAST-01", "substation": "Substation-Beta", "capacity_kva": 3000, "region": "East Industrial"},
    {"feeder_id": "FDR-WEST-01", "substation": "Substation-Gamma", "capacity_kva": 1800, "region": "West Suburban"},
]

CUSTOMER_TYPES = {
    "RESIDENTIAL": {"base_mean": 1.8, "base_std": 0.6, "peak_morning": 8, "peak_evening": 20, "v_nominal": 230.0},
    "COMMERCIAL": {"base_mean": 15.0, "base_std": 4.0, "peak_morning": 11, "peak_evening": 15, "v_nominal": 230.0},
    "INDUSTRIAL": {"base_mean": 65.0, "base_std": 15.0, "peak_morning": 14, "peak_evening": 14, "v_nominal": 400.0}
}

def generate_meter_metadata(num_meters=60):
    meters = []
    types_pool = ["RESIDENTIAL"] * 40 + ["COMMERCIAL"] * 14 + ["INDUSTRIAL"] * 6
    
    # Pre-select anomalous meters for realistic case studies
    anomaly_assignments = {
        "MTR-RES-007": "THEFT_BYPASS",
        "MTR-RES-019": "ZERO_CONSUMPTION_FAULT",
        "MTR-COM-043": "CONSUMPTION_SPIKE",
        "MTR-IND-056": "VOLTAGE_SAG",
        "MTR-RES-031": "THEFT_BYPASS",
        "MTR-COM-048": "CONSUMPTION_SPIKE",
    }
    
    for i in range(num_meters):
        ctype = types_pool[i % len(types_pool)]
        feeder = FEEDER_CONFIGS[i % len(FEEDER_CONFIGS)]
        meter_id = f"MTR-{ctype[:3]}-{i+1:03d}"
        
        assigned_anomaly = anomaly_assignments.get(meter_id, "NONE")
        
        meters.append({
            "meter_id": meter_id,
            "customer_type": ctype,
            "feeder_id": feeder["feeder_id"],
            "substation": feeder["substation"],
            "tariff_plan": "Flat-Rate" if ctype == "RESIDENTIAL" else "Time-of-Use",
            "latitude": round(17.3850 + random.uniform(-0.08, 0.08), 6),
            "longitude": round(78.4867 + random.uniform(-0.08, 0.08), 6),
            "assigned_anomaly": assigned_anomaly
        })
    return pd.DataFrame(meters)

def generate_smart_meter_streams(days=14, interval_hours=1.0, num_meters=60):
    """
    Generates time-series telemetry streams for all meters with realistic load curves
    and injected anomalies.
    """
    meters_df = generate_meter_metadata(num_meters)
    start_time = datetime.now() - timedelta(days=days)
    total_steps = int(days * 24 / interval_hours)
    timestamps = [start_time + timedelta(hours=i * interval_hours) for i in range(total_steps)]
    
    records = []
    
    for _, meter in meters_df.iterrows():
        mid = meter["meter_id"]
        ctype = meter["customer_type"]
        feeder = meter["feeder_id"]
        anomaly_type = meter["assigned_anomaly"]
        cfg = CUSTOMER_TYPES[ctype]
        
        # Anomaly start point (e.g., occurs during the 2nd week)
        anomaly_start_idx = int(total_steps * 0.45)
        anomaly_duration = int(total_steps * 0.35)
        
        for idx, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.weekday()
            is_weekend = 1 if dow >= 5 else 0
            
            # Base diurnal profile
            if ctype == "RESIDENTIAL":
                # Morning peak and evening peak
                diurnal = 0.5 * np.exp(-0.5 * ((hour - 8)/2.2)**2) + 1.2 * np.exp(-0.5 * ((hour - 20)/2.5)**2)
                base = cfg["base_mean"] + diurnal + np.random.normal(0, cfg["base_std"])
                if is_weekend:
                    base *= 1.15  # higher daytime usage on weekends
            elif ctype == "COMMERCIAL":
                # Business hours peak (9am to 6pm)
                business_active = 1.0 if 9 <= hour <= 18 else 0.25
                if is_weekend:
                    business_active *= 0.3
                base = cfg["base_mean"] * business_active + np.random.normal(0, cfg["base_std"] * 0.5)
            else: # INDUSTRIAL
                # 2-shift or 3-shift constant load with minor dips
                shift_factor = 1.0 if 6 <= hour <= 22 else 0.7
                base = cfg["base_mean"] * shift_factor + np.random.normal(0, cfg["base_std"])
                
            base = max(0.05, float(base))
            
            # Voltage & Power factor base
            v_nom = cfg["v_nominal"]
            voltage = v_nom + np.random.normal(0, 2.5)
            pf = float(np.clip(np.random.normal(0.94, 0.02), 0.75, 0.99))
            
            # Ground truth anomaly label
            ground_truth_anomaly = "NORMAL"
            
            # Anomaly injection logic
            if anomaly_type != "NONE" and anomaly_start_idx <= idx <= (anomaly_start_idx + anomaly_duration):
                if anomaly_type == "THEFT_BYPASS":
                    # Under-reporting by 60% - 75%
                    base = base * random.uniform(0.25, 0.40)
                    ground_truth_anomaly = "THEFT_BYPASS"
                elif anomaly_type == "CONSUMPTION_SPIKE":
                    # Sudden sharp spike
                    base = base * random.uniform(3.5, 6.0)
                    ground_truth_anomaly = "CONSUMPTION_SPIKE"
                elif anomaly_type == "ZERO_CONSUMPTION_FAULT":
                    # Meter dead-drop
                    base = 0.0
                    ground_truth_anomaly = "ZERO_CONSUMPTION_FAULT"
                elif anomaly_type == "VOLTAGE_SAG":
                    # Voltage drops drastically
                    voltage = voltage * random.uniform(0.78, 0.85)
                    ground_truth_anomaly = "VOLTAGE_SAG"
            
            # Derived electrical metrics
            kwh = round(base, 4)
            reactive_kvar = round(kwh * np.tan(np.arccos(pf)), 4)
            
            records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "meter_id": mid,
                "feeder_id": feeder,
                "customer_type": ctype,
                "kwh": kwh,
                "voltage": round(voltage, 2),
                "power_factor": round(pf, 3),
                "reactive_kvar": reactive_kvar,
                "ground_truth_anomaly": ground_truth_anomaly
            })
            
    readings_df = pd.DataFrame(records)
    
    # Generate Feeder Master Meter Readings (Energy Injected at Feeder Substation)
    feeder_master_records = []
    feeder_grp = readings_df.groupby(["feeder_id", "timestamp"])["kwh"].sum().reset_index()
    
    for _, row in feeder_grp.iterrows():
        fdr = row["feeder_id"]
        ts = row["timestamp"]
        consumer_total = row["kwh"]
        # Normal technical loss is 4% - 6%
        technical_loss_pct = 0.05
        # If feeder contains THEFT_BYPASS, technical + theft loss occurs
        injected_loss = consumer_total * technical_loss_pct
        
        # In meters with theft, the unmeasured energy is still supplied by the feeder!
        # So feeder supplied energy = sum(meter readings) + stolen energy + technical loss
        theft_in_feeder = 0.0
        # If FDR-NORTH-01 or FDR-WEST-01 (which contain theft meters), add realistic theft gap
        if fdr in ["FDR-NORTH-01", "FDR-WEST-01"]:
            theft_in_feeder = consumer_total * 0.12  # 12% non-technical loss
            
        feeder_supplied_kwh = round(consumer_total + injected_loss + theft_in_feeder + np.random.normal(0, consumer_total*0.01), 3)
        
        feeder_master_records.append({
            "timestamp": ts,
            "feeder_id": fdr,
            "feeder_supplied_kwh": feeder_supplied_kwh,
            "consumer_total_kwh": round(consumer_total, 3)
        })
        
    feeder_master_df = pd.DataFrame(feeder_master_records)
    
    return meters_df, readings_df, feeder_master_df

if __name__ == "__main__":
    print("[*] Generating synthetic smart meter big data streams...")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    
    meters_df, readings_df, feeder_master_df = generate_smart_meter_streams(days=14, interval_hours=1.0, num_meters=60)
    
    meters_path = os.path.join(out_dir, "smart_meters_meta.csv")
    readings_path = os.path.join(out_dir, "raw_meter_readings.parquet")
    feeder_path = os.path.join(out_dir, "feeder_master_readings.parquet")
    
    meters_df.to_csv(meters_path, index=False)
    readings_df.to_parquet(readings_path, index=False)
    feeder_master_df.to_parquet(feeder_path, index=False)
    
    print(f"[+] Successfully saved {len(meters_df)} smart meters to {meters_path}")
    print(f"[+] Successfully saved {len(readings_df)} telemetry readings to {readings_path}")
    print(f"[+] Successfully saved {len(feeder_master_df)} feeder master records to {feeder_path}")
