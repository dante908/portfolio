#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECTS=(
  "churn-saas-b2b"
  "forecast-demanda-omnichannel"
  "fraude-pagamentos-rtr"
  "people-analytics-turnover"
  "recomendacao-ecommerce"
  "segmentacao-rfm-clustering"
)

check_python_deps() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy
import pandas
PY
}

run_project() {
  local name="$1"
  local project_dir="$ROOT_DIR/projects/$name"
  echo ""
  echo "=== $name ==="

  if ! check_python_deps; then
    echo "Dependencias ausentes para $name no interpretador $PYTHON_BIN."
    echo "Instale com: $PYTHON_BIN -m pip install -r \"$project_dir/requirements.txt\""
    return 1
  fi
  "$PYTHON_BIN" -m py_compile "$project_dir/src/"*.py
  (cd "$project_dir" && "$PYTHON_BIN" src/main.py)
}

validate_project() {
  local name="$1"
  local project_dir="$ROOT_DIR/projects/$name"
  local required_files=()

  "$PYTHON_BIN" -m py_compile "$project_dir/src/"*.py
  [[ -d "$project_dir/data" ]] || return 1
  [[ -d "$project_dir/models" ]] || return 1
  [[ -d "$project_dir/notebooks" ]] || return 1

  case "$name" in
    churn-saas-b2b)
      required_files=(
        "$project_dir/data/churn_saas_synthetic.csv"
        "$project_dir/data/churn_test_scored.csv"
        "$project_dir/models/churn_model.json"
        "$project_dir/models/metrics.json"
        "$project_dir/notebooks/analysis_notes.md"
      )
      ;;
    forecast-demanda-omnichannel)
      required_files=(
        "$project_dir/data/demand_history_synthetic.csv"
        "$project_dir/data/forecast_backtest.csv"
        "$project_dir/models/model_info.json"
        "$project_dir/models/metrics.json"
        "$project_dir/notebooks/analysis_notes.md"
      )
      ;;
    fraude-pagamentos-rtr)
      required_files=(
        "$project_dir/data/transactions_synthetic.csv"
        "$project_dir/data/transactions_test_scored.csv"
        "$project_dir/models/model_info.json"
        "$project_dir/models/metrics.json"
        "$project_dir/notebooks/analysis_notes.md"
      )
      ;;
    people-analytics-turnover)
      required_files=(
        "$project_dir/data/employees_synthetic.csv"
        "$project_dir/data/employees_test_scored.csv"
        "$project_dir/models/model.json"
        "$project_dir/models/metrics.json"
        "$project_dir/notebooks/analysis_notes.md"
      )
      ;;
    recomendacao-ecommerce)
      required_files=(
        "$project_dir/data/interactions_synthetic.csv"
        "$project_dir/data/recommendations_top10.csv"
        "$project_dir/models/model_info.json"
        "$project_dir/models/metrics.json"
        "$project_dir/notebooks/analysis_notes.md"
      )
      ;;
    segmentacao-rfm-clustering)
      required_files=(
        "$project_dir/data/transactions_synthetic.csv"
        "$project_dir/data/rfm_clusters.csv"
        "$project_dir/models/cluster_summary.csv"
        "$project_dir/models/model_info.json"
        "$project_dir/notebooks/analysis_notes.md"
      )
      ;;
  esac

  for file in "${required_files[@]}"; do
    [[ -f "$file" ]] || return 1
  done
}

cmd="${1:-run}"

case "$cmd" in
  run)
    for project in "${PROJECTS[@]}"; do
      run_project "$project"
    done
    ;;
  validate)
    for project in "${PROJECTS[@]}"; do
      validate_project "$project"
    done
    echo "Validacao concluida."
    ;;
  *)
    echo "Uso: $0 [run|validate]"
    exit 1
    ;;
esac
