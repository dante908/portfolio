#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GITHUB_USER="${GITHUB_USER:-dante908}"
TAG_NAME="${TAG_NAME:-v1.0.0}"

PROJECTS=(
  "churn-saas-b2b"
  "forecast-demanda-omnichannel"
  "fraude-pagamentos-rtr"
  "people-analytics-turnover"
  "recomendacao-ecommerce"
  "segmentacao-rfm-clustering"
)

for project in "${PROJECTS[@]}"; do
  echo "Tagging ${project} (${TAG_NAME})..."
  tmp_dir="$(mktemp -d)"
  rsync -a --exclude ".DS_Store" --exclude "__pycache__" --exclude ".venv" "$ROOT_DIR/projects/$project/" "$tmp_dir/$project/"

  (
    set -e
    cd "$tmp_dir/$project"
    git init -q
    git add .
    git commit -q -m "Release ${TAG_NAME}"
    git branch -M main
    git remote add origin "https://github.com/${GITHUB_USER}/${project}.git"
    git push -u -f origin main
    git tag -a "$TAG_NAME" -m "Release ${TAG_NAME}"
    git push -f origin "$TAG_NAME"
  )

  rm -rf "$tmp_dir"
  echo "[ok] ${project} tagged ${TAG_NAME}"
done

echo "All repositories tagged."
