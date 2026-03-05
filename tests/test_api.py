from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover
    pytest.skip(f"FastAPI TestClient unavailable in this environment: {exc}", allow_module_level=True)

from api.main import app


client = TestClient(app)


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_projects_list() -> None:
    res = client.get("/projects")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert "forecast-demanda-omnichannel" in data


def test_metrics_endpoint() -> None:
    res = client.get("/projects/churn-saas-b2b/metrics")
    assert res.status_code == 200
    assert "roc_auc" in res.json()
