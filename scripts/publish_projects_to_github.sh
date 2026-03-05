#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GITHUB_USER="${GITHUB_USER:-dante908}"
MODE="${1:-check}"

PROJECTS=(
  "churn-saas-b2b"
  "forecast-demanda-omnichannel"
  "fraude-pagamentos-rtr"
  "people-analytics-turnover"
  "recomendacao-ecommerce"
  "segmentacao-rfm-clustering"
)

check_repo_exists() {
  local project="$1"
  local repo_url="https://github.com/${GITHUB_USER}/${project}.git"
  git ls-remote --heads "$repo_url" >/dev/null 2>&1
}

publish_project() {
  local project="$1"
  local project_dir="$ROOT_DIR/projects/$project"
  local repo_url="https://github.com/${GITHUB_USER}/${project}.git"
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  mkdir -p "$tmp_dir/$project"
  rsync -a \
    --exclude ".DS_Store" \
    --exclude "__pycache__" \
    --exclude ".venv" \
    "$project_dir/" "$tmp_dir/$project/"

  (
    cd "$tmp_dir/$project"
    git init -q
    git add .
    git commit -q -m "Initial project commit"
    git branch -M main
    git remote add origin "$repo_url"
    git push -u origin main
  )

  rm -rf "$tmp_dir"
}

missing_count=0
for project in "${PROJECTS[@]}"; do
  if check_repo_exists "$project"; then
    echo "[ok] repo exists: ${GITHUB_USER}/${project}"
  else
    echo "[missing] repo not found: ${GITHUB_USER}/${project}"
    missing_count=$((missing_count + 1))
  fi
done

if [[ "$MODE" == "check" ]]; then
  if [[ $missing_count -gt 0 ]]; then
    echo ""
    echo "Create the missing repositories on GitHub first, then run:"
    echo "  GITHUB_USER=${GITHUB_USER} $0 push"
    exit 1
  fi
  echo "All repositories exist."
  exit 0
fi

if [[ "$MODE" != "push" ]]; then
  echo "Usage: $0 [check|push]"
  exit 1
fi

if [[ $missing_count -gt 0 ]]; then
  echo "Cannot push: there are missing repositories."
  exit 1
fi

for project in "${PROJECTS[@]}"; do
  echo "Publishing ${project}..."
  publish_project "$project"
done

echo "All projects published."
