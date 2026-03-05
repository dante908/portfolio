from __future__ import annotations

import json
from pathlib import Path

try:
    import mlflow
except Exception as exc:  # pragma: no cover
    raise SystemExit("mlflow nao instalado. Rode: python3 -m pip install mlflow") from exc

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
TRACKING_DIR = ROOT / "mlruns"


def flatten(prefix: str, value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(f"{prefix}.{k}" if prefix else str(k), v)
    elif isinstance(value, (int, float)):
        yield prefix, float(value)


def log_project(project_dir: Path) -> None:
    metrics_path = project_dir / "models" / "metrics.json"
    if not metrics_path.exists():
        print(f"[skip] sem metrics: {project_dir.name}")
        return

    metrics = json.loads(metrics_path.read_text())

    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("portfolio-projects")
    with mlflow.start_run(run_name=project_dir.name):
        mlflow.log_param("project", project_dir.name)

        for k, v in flatten("", metrics):
            if len(k) <= 250:
                mlflow.log_metric(k, v)

        for folder in ["models", "reports", "notebooks"]:
            d = project_dir / folder
            if d.exists():
                mlflow.log_artifacts(str(d), artifact_path=folder)

    print(f"[ok] mlflow logged: {project_dir.name}")


def main() -> None:
    for d in sorted(PROJECTS.iterdir()):
        if d.is_dir():
            log_project(d)


if __name__ == "__main__":
    main()
