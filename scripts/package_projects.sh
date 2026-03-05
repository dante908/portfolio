#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECTS_DIR="$ROOT_DIR/projects"
PROJECTS=(
  "churn-saas-b2b"
  "forecast-demanda-omnichannel"
  "fraude-pagamentos-rtr"
  "people-analytics-turnover"
  "recomendacao-ecommerce"
  "segmentacao-rfm-clustering"
)

cd "$PROJECTS_DIR"
for project in "${PROJECTS[@]}"; do
  rm -f "$project.zip"
  zip -rq "$project.zip" "$project" -x "*/.DS_Store" "*/__pycache__/*" "*/.venv/*"
  echo "Pacote atualizado: $project.zip"
done
