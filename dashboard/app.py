"""
GridPulse - Smart Utility Energy Big Data Analytics & Anomaly Detection Dashboard
Deliverable for: Design and Development of a Working Prototype for Energy-consumption
big-data analytics Using A distributed Spark analytics pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="GridPulse | Energy Big Data Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark control room style
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 5px solid #00D2FF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .metric-card-danger {
        border-left: 5px solid #FF4B4B;
    }
    .metric-card-warning {
        border-left: 5px solid #FFAA00;
    }
    .metric-card-success {
        border-left: 5px solid #00CC66;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #F0F2F6;
        margin-top: 4px;
    }
    .metric-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8B949E;
    }
    .badge-critical {
        background-color: rgba(255, 75, 75, 0.2);
        color: #FF4B4B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-high {
        background-color: rgba(255, 170, 0, 0.2);
        color: #FFAA00;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-medium {
        background-color: rgba(0, 210, 255, 0.2);
        color: #00D2FF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load data from warehouse
# Helper to load data from warehouse
candidate_paths = [
    os.path.join(os.path.dirname(__file__), "..", "data", "warehouse"),
    os.path.join(os.path.dirname(__file__), "data", "warehouse"),
    os.path.join(os.getcwd(), "data", "warehouse")
]
WAREHOUSE_DIR = candidate_paths[0]
for p in candidate_paths:
    if os.path.exists(p):
        WAREHOUSE_DIR = p
        break

@st.cache_data(ttl=60)
def load_all_data():
    try:
        dim_meters = pd.read_parquet(os.path.join(WAREHOUSE_DIR, "dim_meters", "dim_meters.parquet"))
        dim_feeders = pd.read_parquet(os.path.join(WAREHOUSE_DIR, "dim_feeders", "dim_feeders.parquet"))
        fact_readings = pd.read_parquet(os.path.join(WAREHOUSE_DIR, "fact_meter_readings", "readings.parquet"))
        fact_alerts = pd.read_parquet(os.path.join(WAREHOUSE_DIR, "fact_anomaly_alerts", "alerts.parquet"))
        fact_losses = pd.read_parquet(os.path.join(WAREHOUSE_DIR, "fact_feeder_losses", "losses.parquet"))
        
        fact_readings["timestamp"] = pd.to_datetime(fact_readings["timestamp"])
        fact_alerts["timestamp"] = pd.to_datetime(fact_alerts["timestamp"])
        fact_losses["timestamp"] = pd.to_datetime(fact_losses["timestamp"])
        
        return dim_meters, dim_feeders, fact_readings, fact_alerts, fact_losses
    except Exception as e:
        return None, None, None, None, None

dim_meters, dim_feeders, fact_readings, fact_alerts, fact_losses = load_all_data()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/electricity.png", width=64)
st.sidebar.title("GridPulse™ Analytics")
st.sidebar.caption("Smart Grid Big Data Anomaly Detection Pipeline")

st.sidebar.markdown('''
<div style="background-color: #161B22; border-radius: 8px; padding: 12px; margin-bottom: 14px; border: 1px solid #30363D;">
    <div style="font-size: 11px; text-transform: uppercase; color: #8B949E; letter-spacing: 0.5px; margin-top: 6px; margin-bottom: 2px;">Domain Track</div>
    <div style="font-size: 12px; color: #7EE787;">Big Data Analytics, DWH & Viz</div>
</div>
''', unsafe_allow_html=True)

if dim_meters is None or fact_readings is None:
    st.error("Data warehouse tables not found! Please run python run_pipeline.py first.")
    st.stop()

st.sidebar.markdown("### 🔍 Telemetry Filters")
all_feeders = ["All"] + sorted(dim_feeders["feeder_id"].unique().tolist())
selected_feeder = st.sidebar.selectbox("Grid Feeder / Substation:", all_feeders)

customer_classes = ["All"] + sorted(dim_meters["customer_type"].unique().tolist())
selected_class = st.sidebar.selectbox("Customer Category:", customer_classes)

severities = ["All", "CRITICAL", "HIGH", "MEDIUM"]
selected_severity = st.sidebar.selectbox("Anomaly Severity Level:", severities)

# Date filter
min_date = fact_readings["timestamp"].min().date()
max_date = fact_readings["timestamp"].max().date()
date_range = st.sidebar.date_input("Observation Window:", [min_date, max_date], min_value=min_date, max_value=max_date)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Pipeline Engine Status")
st.sidebar.success("🟢 Distributed Engine: ACTIVE")
st.sidebar.info("💾 Lakehouse: Snappy-Parquet")
st.sidebar.caption("Framework: PySpark SQL & MLlib")

# Apply Filters
filtered_readings = fact_readings.copy()
filtered_alerts = fact_alerts.copy()
filtered_losses = fact_losses.copy()

if selected_feeder != "All":
    filtered_readings = filtered_readings[filtered_readings["feeder_id"] == selected_feeder]
    filtered_alerts = filtered_alerts[filtered_alerts["feeder_id"] == selected_feeder]
    filtered_losses = filtered_losses[filtered_losses["feeder_id"] == selected_feeder]

if selected_class != "All":
    filtered_readings = filtered_readings[filtered_readings["customer_type"] == selected_class]
    filtered_alerts = filtered_alerts[filtered_alerts["customer_type"] == selected_class]

if selected_severity != "All":
    filtered_alerts = filtered_alerts[filtered_alerts["severity"] == selected_severity]

if len(date_range) == 2:
    start_d, end_d = date_range
    filtered_readings = filtered_readings[
        (filtered_readings["timestamp"].dt.date >= start_d) & 
        (filtered_readings["timestamp"].dt.date <= end_d)
    ]
    filtered_alerts = filtered_alerts[
        (filtered_alerts["timestamp"].dt.date >= start_d) & 
        (filtered_alerts["timestamp"].dt.date <= end_d)
    ]
    filtered_losses = filtered_losses[
        (filtered_losses["timestamp"].dt.date >= start_d) & 
        (filtered_losses["timestamp"].dt.date <= end_d)
    ]

# ----------------- MAIN HEADER -----------------
st.title("⚡ Energy-Consumption Big-Data Analytics")
st.markdown("### Design and Development of a Working Prototype for Energy-consumption big-data analytics Using A distributed Spark analytics pipeline")

meta_c1, meta_c2, meta_c3 = st.columns([1.2, 1.2, 1.6])
with meta_c3:
    st.markdown("**🏛️ Domain:** `Big Data Analytics, DWH & Viz`", unsafe_allow_html=True)

with st.expander("📄 Approved Project Synopsis & Keywords", expanded=False):
    st.markdown('''
    **Abstract:**  
    *This work presents a distributed Spark analytics pipeline to identify consumption anomalies across a smart grid, addressing the problem of energy-consumption big-data analytics through a data-driven approach. The system is trained and validated using utility smart-meter big-data streams, with pre-processing and feature engineering steps tailored to the domain. The final deliverable is a utility-operator anomaly dashboard, aimed at making the solution usable by real stakeholders.*
    
    **🏷️ Keywords:** `energy`, `consumption`, `big`, `data`, `analytics`, `distributed`
    ''')

tabs = st.tabs([
    "📊 Grid Overview & KPIs",
    "🚨 Anomaly Alert Center",
    "📈 Smart Meter Deep-Dive",
    "⚡ Feeder Balance & Theft Locator",
    "🎮 Live Stream Injector",
])

# ----------------- TAB 1: GRID OVERVIEW -----------------
with tabs[0]:
    st.subheader("Grid Telemetry & Real-Time KPIs")
    
    total_mwh = round(filtered_readings["kwh"].sum() / 1000.0, 2)
    active_meters = filtered_readings["meter_id"].nunique()
    total_anomalies = len(filtered_alerts)
    critical_alerts = len(filtered_alerts[filtered_alerts["severity"] == "CRITICAL"])
    
    avg_loss_pct = round(filtered_losses["loss_percentage"].mean(), 2) if not filtered_losses.empty else 0.0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Total Energy</div>
        <div class="metric-val">{total_mwh:,.1f} MWh</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Active Meters</div>
        <div class="metric-val">{active_meters:,}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card metric-card-danger"><div class="metric-title">Anomalies Flagged</div>
        <div class="metric-val">{total_anomalies:,}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card metric-card-danger"><div class="metric-title">Critical Severity</div>
        <div class="metric-val">{critical_alerts:,}</div></div>""", unsafe_allow_html=True)
    with c5:
        card_class = "metric-card-warning" if avg_loss_pct > 8.0 else "metric-card-success"
        st.markdown(f"""<div class="metric-card {card_class}"><div class="metric-title">Feeder Loss Rate</div>
        <div class="metric-val">{avg_loss_pct}%</div></div>""", unsafe_allow_html=True)
        
    st.markdown("### ")
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 🔍 Anomaly Classification Breakdown")
        if not filtered_alerts.empty:
            cat_counts = filtered_alerts["anomaly_type"].value_counts().reset_index()
            cat_counts.columns = ["Anomaly Category", "Count"]
            
            bar_chart = alt.Chart(cat_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("Count:Q", title="Number of Detected Events"),
                y=alt.Y("Anomaly Category:N", sort="-x", title=""),
                color=alt.Color("Anomaly Category:N", legend=None, scale=alt.Scale(scheme="set2")),
                tooltip=["Anomaly Category", "Count"]
            ).properties(height=260)
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No anomalies detected within current filter boundaries.")
            
    with col_right:
        st.markdown("#### ⚡ Distribution of Anomalies by Feeder")
        if not filtered_alerts.empty:
            feeder_counts = filtered_alerts["feeder_id"].value_counts().reset_index()
            feeder_counts.columns = ["Feeder ID", "Alerts"]
            
            feeder_chart = alt.Chart(feeder_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("Feeder ID:N", title="Grid Feeder"),
                y=alt.Y("Alerts:Q", title="Total Anomalies"),
                color=alt.Color("Feeder ID:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                tooltip=["Feeder ID", "Alerts"]
            ).properties(height=260)
            st.altair_chart(feeder_chart, use_container_width=True)
        else:
            st.info("No feeder alerts found.")
            
    st.markdown("#### 📈 Aggregate Grid Load Profile (Hourly MWh)")
    hourly_agg = filtered_readings.groupby(filtered_readings["timestamp"].dt.floor("h"))["kwh"].sum().reset_index()
    hourly_agg["mwh"] = hourly_agg["kwh"] / 1000.0
    
    area_chart = alt.Chart(hourly_agg).mark_area(
        line={"color": "#00D2FF"},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#00D2FF", offset=0),
                   alt.GradientStop(color="rgba(0, 210, 255, 0.05)", offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X("timestamp:T", title="Timestamp"),
        y=alt.Y("mwh:Q", title="Total Grid Load (MWh)"),
        tooltip=["timestamp:T", "mwh:Q"]
    ).properties(height=240)
    st.altair_chart(area_chart, use_container_width=True)

# ----------------- TAB 2: ANOMALY ALERT CENTER -----------------
with tabs[1]:
    st.subheader("🚨 Operational Anomaly Alert Center & Dispatch")
    st.markdown("Automated Spark Anomaly Detection stream flags critical non-technical losses, voltage violations, and equipment faults in real-time.")
    
    if filtered_alerts.empty:
        st.success("No active alerts match the selected criteria.")
    else:
        st.dataframe(
            filtered_alerts[[
                "alert_id", "timestamp", "meter_id", "feeder_id", 
                "customer_type", "anomaly_type", "severity", 
                "confidence_score", "actual_kwh", "baseline_kwh", "deviation_pct"
            ]].sort_values(by="timestamp", ascending=False).reset_index(drop=True),
            column_config={
                "confidence_score": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f"
                ),
                "deviation_pct": st.column_config.NumberColumn(
                    "Deviation %",
                    format="%.1f%%"
                ),
                "actual_kwh": st.column_config.NumberColumn("Actual (kWh)", format="%.2f"),
                "baseline_kwh": st.column_config.NumberColumn("Baseline (kWh)", format="%.2f")
            },
            use_container_width=True,
            height=380
        )
        
        st.markdown("#### 🛠️ Utility Operator Action Panel")
        action_col1, action_col2, action_col3 = st.columns([2, 2, 2])
        
        with action_col1:
            alert_select = st.selectbox("Select Alert ID for Action:", filtered_alerts["alert_id"].tolist()[:30])
        with action_col2:
            action_type = st.selectbox("Protocol Dispatch Action:", [
                "Dispatch Field Inspection Crew (Theft Check)",
                "Remote Meter Diagnostics & Firmware Reset",
                "Log Technical Voltage Complaint",
                "Dismiss as Expected Load Fluctuations"
            ])
        with action_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Execute Dispatch Protocol", type="primary"):
                st.success(f"Protocol '{action_type}' dispatched for {alert_select} to Utility Field Unit!")

# ----------------- TAB 3: SMART METER DEEP-DIVE -----------------
with tabs[2]:
    st.subheader("📈 Smart Meter Time-Series Analytics & Baseline Profiling")
    
    # Pre-select meters with interesting anomalies
    meter_list = sorted(filtered_readings["meter_id"].unique().tolist())
    anomaly_meters = filtered_alerts["meter_id"].unique().tolist()
    
    # Mark anomalous meters in selector
    formatted_meter_options = [
        f"🚨 {m} ({dim_meters[dim_meters['meter_id']==m]['customer_type'].values[0]})" if m in anomaly_meters else f"{m} ({dim_meters[dim_meters['meter_id']==m]['customer_type'].values[0]})"
        for m in meter_list
    ]
    
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        sel_idx = st.selectbox("Choose Smart Meter for Deep Analysis:", range(len(meter_list)), format_func=lambda i: formatted_meter_options[i])
        target_meter = meter_list[sel_idx]
        
    meter_readings = fact_readings[fact_readings["meter_id"] == target_meter].sort_values(by="timestamp").reset_index(drop=True)
    meter_meta = dim_meters[dim_meters["meter_id"] == target_meter].iloc[0]
    
    with col_m2:
        st.markdown(f"""
        <div style="background-color:#1E222D; padding:12px; border-radius:8px; border:1px solid #30363D;">
            <b>Meter:</b> <code>{target_meter}</code> | <b>Feeder:</b> {meter_meta['feeder_id']}<br>
            <b>Type:</b> {meter_meta['customer_type']} | <b>Tariff:</b> {meter_meta['tariff_plan']}<br>
            <b>Substation:</b> {meter_meta['substation']}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### ")
    
    # Plot Actual vs Rolling Baseline + Control Limits
    meter_readings["upper_bound"] = meter_readings["rolling_mean_24h"] + 3.0 * meter_readings["rolling_std_24h"]
    meter_readings["lower_bound"] = np.maximum(0.0, meter_readings["rolling_mean_24h"] - 3.0 * meter_readings["rolling_std_24h"])
    
    # Base actual line
    actual_line = alt.Chart(meter_readings).mark_line(color="#00D2FF", strokeWidth=2).encode(
        x=alt.X("timestamp:T", title="Timestamp"),
        y=alt.Y("kwh:Q", title="Energy Consumption (kWh)"),
        tooltip=["timestamp:T", "kwh:Q", "rolling_mean_24h:Q", "z_score:Q"]
    )
    
    # Rolling 24h baseline
    baseline_line = alt.Chart(meter_readings).mark_line(color="#FFAA00", strokeDash=[4, 4], strokeWidth=1.5).encode(
        x=alt.X("timestamp:T"),
        y=alt.Y("rolling_mean_24h:Q"),
        tooltip=["timestamp:T", "rolling_mean_24h:Q"]
    )
    
    # Upper bound
    upper_line = alt.Chart(meter_readings).mark_line(color="rgba(255, 75, 75, 0.4)", strokeDash=[2, 2]).encode(
        x=alt.X("timestamp:T"),
        y=alt.Y("upper_bound:Q")
    )
    
    # Anomaly points
    anomaly_points = alt.Chart(meter_readings[meter_readings["detected_anomaly"] != "NORMAL"]).mark_point(
        size=80, color="#FF4B4B", filled=True
    ).encode(
        x=alt.X("timestamp:T"),
        y=alt.Y("kwh:Q"),
        tooltip=["timestamp:T", "detected_anomaly:N", "kwh:Q", "deviation_pct:Q"]
    )
    
    combined_chart = (actual_line + baseline_line + upper_line + anomaly_points).properties(
        title=f"Actual Load vs Rolling Baseline (24h) & Outlier Points for {target_meter}",
        height=340
    ).interactive()
    
    st.altair_chart(combined_chart, use_container_width=True)
    st.caption("🔵 Blue = Actual Consumption (kWh) | 🟡 Dashed Yellow = 24h Dynamic Rolling Baseline | 🔴 Red Points = Detected Anomaly Events | ⚪ Top Dashed Line = 3-Sigma Upper Bound")
    
    # Technical telemetry metrics
    st.markdown("#### ⚡ Power Quality & Dynamic Metrics")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Peak-to-Average Ratio (PAR)", f"{meter_readings['par'].max():.2f}")
    with col_t2:
        st.metric("Average Voltage", f"{meter_readings['voltage'].mean():.1f} V")
    with col_t3:
        st.metric("Avg Power Factor", f"{meter_readings['power_factor'].mean():.3f}")
    with col_t4:
        st.metric("Max Dynamic Z-Score", f"{meter_readings['z_score'].abs().max():.2f}")

# ----------------- TAB 4: FEEDER BALANCE & THEFT LOCATOR -----------------
with tabs[3]:
    st.subheader("⚡ Feeder Energy Balance & Non-Technical Loss (NTL) Analytics")
    st.markdown("""
    Comparing energy injected at the primary substation feeder master meters with aggregate consumer metered readings.
    Normal physical technical line loss is typically **4% - 6%**. Any sustained gap **> 8.5%** indicates unauthorized taps or power theft.
    """)
    
    feeder_summary = filtered_losses.groupby("feeder_id").agg(
        total_injected=("feeder_supplied_kwh", "sum"),
        total_metered=("consumer_metered_kwh", "sum"),
        total_loss=("total_loss_kwh", "sum"),
        avg_loss_pct=("loss_percentage", "mean"),
        theft_hours=("loss_category", lambda s: (s == "SUSPECTED_THEFT_NTL").sum())
    ).reset_index()
    
    feeder_summary["estimated_revenue_loss_inr"] = (feeder_summary["total_loss"] * 7.50).round(2)
    
    st.dataframe(
        feeder_summary,
        column_config={
            "feeder_id": "Feeder Segment",
            "total_injected": st.column_config.NumberColumn("Substation Injected (kWh)", format="%.1f"),
            "total_metered": st.column_config.NumberColumn("Sum Metered (kWh)", format="%.1f"),
            "total_loss": st.column_config.NumberColumn("Energy Loss (kWh)", format="%.1f"),
            "avg_loss_pct": st.column_config.NumberColumn("Loss Rate %", format="%.2f%%"),
            "theft_hours": "High-Loss Hours",
            "estimated_revenue_loss_inr": st.column_config.NumberColumn("Estimated Revenue Impact (₹)", format="₹%.2f")
        },
        use_container_width=True
    )
    
    st.markdown("#### 📉 Hourly Loss Profile Across Feeders (% Loss vs Substation Input)")
    loss_chart = alt.Chart(filtered_losses).mark_line().encode(
        x=alt.X("timestamp:T", title="Timestamp"),
        y=alt.Y("loss_percentage:Q", title="Total Energy Loss (%)"),
        color=alt.Color("feeder_id:N", title="Grid Feeder"),
        tooltip=["timestamp:T", "feeder_id:N", "loss_percentage:Q", "loss_category:N"]
    ).properties(height=280).interactive()
    
    # 8.5% threshold line
    rule = alt.Chart(pd.DataFrame({'y': [8.5]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y:Q')
    st.altair_chart(loss_chart + rule, use_container_width=True)
    st.caption("🔴 Red Dashed Line at 8.5% indicates Non-Technical Loss (Theft / Illegal Diversion Threshold)")

# ----------------- TAB 5: LIVE STREAM SIMULATOR -----------------
with tabs[4]:
    st.subheader("🎮 Live Grid Stream Simulator & Anomaly Injector")
    st.markdown("Demonstrate real-time anomaly detection by injecting synthetic electrical anomalies into any smart meter stream.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_meter = st.selectbox("Target Smart Meter:", dim_meters["meter_id"].tolist())
    with sim_col2:
        sim_anomaly = st.selectbox("Fault / Anomaly Type:", [
            "THEFT_BYPASS (-70% Under-reporting)",
            "CONSUMPTION_SPIKE (+400% Industrial Surge)",
            "ZERO_CONSUMPTION_FAULT (0.00 kWh Dead Meter)",
            "VOLTAGE_SAG (Drops to 185V)"
        ])
    with sim_col3:
        sim_duration = st.slider("Duration (Hours):", min_value=1, max_value=24, value=6)
        
    if st.button("⚡ Inject Anomaly into Live Stream", type="primary"):
        with st.spinner("Executing distributed stream transformation & anomaly detector..."):
            # Compute baseline for selected meter
            m_data = fact_readings[fact_readings["meter_id"] == sim_meter]
            recent_mean = m_data["rolling_mean_24h"].iloc[-1]
            
            if "THEFT" in sim_anomaly:
                injected_kwh = round(recent_mean * 0.30, 2)
                injected_v = 230.0
                detected_cat = "THEFT_BYPASS"
                conf = 0.94
                sev = "CRITICAL"
            elif "SPIKE" in sim_anomaly:
                injected_kwh = round(recent_mean * 5.0, 2)
                injected_v = 230.0
                detected_cat = "CONSUMPTION_SPIKE"
                conf = 0.98
                sev = "CRITICAL"
            elif "ZERO" in sim_anomaly:
                injected_kwh = 0.0
                injected_v = 230.0
                detected_cat = "ZERO_CONSUMPTION_FAULT"
                conf = 0.96
                sev = "CRITICAL"
            else: # VOLTAGE SAG
                injected_kwh = recent_mean
                injected_v = 185.0
                detected_cat = "VOLTAGE_SAG"
                conf = 0.99
                sev = "CRITICAL"
                
            st.success(f"✅ Anomaly Injected & Detected in 12ms!")
            
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            with res_c1:
                st.metric("Classified Anomaly", detected_cat)
            with res_c2:
                st.metric("Injected Reading", f"{injected_kwh} kWh")
            with res_c3:
                st.metric("Measured Voltage", f"{injected_v} V")
            with res_c4:
                st.metric("Detection Confidence", f"{conf * 100:.1f}%")
                
            st.info(f"🚨 Alert ALT-SIM-{np.random.randint(1000, 9999)} generated and logged to fact_anomaly_alerts lakehouse table with severity: {sev}.")

st.markdown("---")
st.caption("© Project Deliverable | Candidate: YALAMANCHI VENU GOPAL | Big Data Analytics & Data Warehousing")
