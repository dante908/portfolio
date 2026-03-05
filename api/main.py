from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"

app = FastAPI(title="Portfolio Data API", version="1.0.0")


def _project_dir(project: str) -> Path:
    p = PROJECTS / project
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    return p


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/projects")
def list_projects() -> list[str]:
    return sorted([d.name for d in PROJECTS.iterdir() if d.is_dir()])


@app.get("/projects/{project}/metrics")
def project_metrics(project: str) -> dict[str, Any]:
    p = _project_dir(project) / "models" / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    return json.loads(p.read_text())


@app.get("/projects/{project}/analysis-notes")
def project_notes(project: str) -> dict[str, str]:
    p = _project_dir(project) / "notebooks" / "analysis_notes.md"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Analysis notes not found")
    return {"notes": p.read_text()}
