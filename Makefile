SHELL := /bin/zsh
.PHONY: bootstrap run-all validate package-all clean

PYTHON ?= python3
ROOT := $(CURDIR)
SCRIPT := scripts/run_all_projects.sh
PACKAGE_SCRIPT := scripts/package_projects.sh

bootstrap:
	@chmod +x $(SCRIPT)
	@echo "Bootstrap concluido."

run-all: bootstrap
	@PYTHON_BIN=$(PYTHON) $(SCRIPT) run

validate: bootstrap
	@PYTHON_BIN=$(PYTHON) $(SCRIPT) validate

package-all: bootstrap
	@chmod +x $(PACKAGE_SCRIPT)
	@$(PACKAGE_SCRIPT)

clean:
	@find projects -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find projects -type f -name ".DS_Store" -delete
	@echo "Caches removidos."
