"""
Streamlit Cloud Entrypoint for GridPulse Energy Analytics.
Delegates directly to dashboard/app.py with proper system path setup.
"""

import os
import sys

# Add project root and src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Auto-generate warehouse if missing
warehouse_check = os.path.join(BASE_DIR, "data", "warehouse", "fact_meter_readings", "readings.parquet")
if not os.path.exists(warehouse_check):
    try:
        import run_pipeline
        run_pipeline.main()
    except Exception as e:
        print(f"Auto-pipeline generation warning: {e}")

# Execute the dashboard app
dashboard_app_path = os.path.join(BASE_DIR, "dashboard", "app.py")
with open(dashboard_app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), dashboard_app_path, "exec")
    exec(code, globals())
