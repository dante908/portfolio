SHELL := /bin/zsh
.PHONY: bootstrap run-all validate quality test log-experiments serve-api package-all publish-check publish-projects tag-projects clean

PYTHON ?= python3
ROOT := $(CURDIR)
SCRIPT := scripts/run_all_projects.sh
PACKAGE_SCRIPT := scripts/package_projects.sh
PUBLISH_SCRIPT := scripts/publish_projects_to_github.sh
TAG_SCRIPT := scripts/tag_project_releases.sh

bootstrap:
	@chmod +x $(SCRIPT)
	@echo "Bootstrap concluido."

run-all: bootstrap
	@PYTHON_BIN=$(PYTHON) $(SCRIPT) run

validate: bootstrap
	@PYTHON_BIN=$(PYTHON) $(SCRIPT) validate

quality: bootstrap
	@$(PYTHON) scripts/data_quality_checks.py

test: bootstrap
	@$(PYTHON) -m pytest -q

log-experiments: bootstrap
	@$(PYTHON) scripts/log_experiments_mlflow.py

serve-api: bootstrap
	@$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

package-all: bootstrap
	@chmod +x $(PACKAGE_SCRIPT)
	@$(PACKAGE_SCRIPT)

publish-check: bootstrap
	@chmod +x $(PUBLISH_SCRIPT)
	@GITHUB_USER=dante908 $(PUBLISH_SCRIPT) check

publish-projects: bootstrap
	@chmod +x $(PUBLISH_SCRIPT)
	@GITHUB_USER=dante908 $(PUBLISH_SCRIPT) push

tag-projects: bootstrap
	@chmod +x $(TAG_SCRIPT)
	@GITHUB_USER=dante908 TAG_NAME=v1.0.0 $(TAG_SCRIPT)

clean:
	@find projects -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find projects -type f -name ".DS_Store" -delete
	@echo "Caches removidos."
