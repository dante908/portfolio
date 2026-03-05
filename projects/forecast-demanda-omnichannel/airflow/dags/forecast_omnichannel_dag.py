from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with DAG(
    dag_id="forecast_demanda_omnichannel_weekly",
    start_date=datetime(2026, 1, 1),
    schedule="0 7 * * 1",
    catchup=False,
    tags=["forecast", "omnichannel", "portfolio"],
) as dag:
    run_pipeline = BashOperator(
        task_id="run_forecast_pipeline",
        bash_command=f"cd '{PROJECT_ROOT}' && python3 src/main.py",
    )

    run_pipeline
