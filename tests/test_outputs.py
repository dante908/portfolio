from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"


def test_all_projects_have_required_folders() -> None:
    for project in [p for p in PROJECTS.iterdir() if p.is_dir()]:
        assert (project / "data").exists(), f"missing data folder: {project.name}"
        assert (project / "models").exists(), f"missing models folder: {project.name}"
        assert (project / "notebooks").exists(), f"missing notebooks folder: {project.name}"


def test_metrics_jsons_are_valid() -> None:
    for project in [p for p in PROJECTS.iterdir() if p.is_dir()]:
        metrics_path = project / "models" / "metrics.json"
        if metrics_path.exists():
            obj = json.loads(metrics_path.read_text())
            assert isinstance(obj, dict), f"invalid metrics.json in {project.name}"
            assert len(obj) > 0, f"empty metrics.json in {project.name}"


def test_forecast_backtest_schema() -> None:
    p = PROJECTS / "forecast-demanda-omnichannel" / "data" / "forecast_backtest.csv"
    df = pd.read_csv(p)
    expected = {"date", "channel", "sku", "actual_demand", "forecast_demand", "abs_error"}
    assert expected.issubset(df.columns)


def test_churn_scored_schema() -> None:
    p = PROJECTS / "churn-saas-b2b" / "data" / "churn_test_scored.csv"
    df = pd.read_csv(p)
    expected = {"churn", "churn_score", "churn_pred"}
    assert expected.issubset(df.columns)
