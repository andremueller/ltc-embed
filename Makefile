.PHONY: help venv install run clean check-deps

.DEFAULT_GOAL := help

VENV       ?= .venv
PYTHON     = $(VENV)/bin/python3
PIP        = $(VENV)/bin/pip

INPUT_DIR  ?= .
SUFFIX     ?= _tc
FPS        ?=
DURATION   ?= 10
OVERWRITE  ?=
OFFSET     ?= 0

ARGS = $(if $(FPS),--fps $(FPS)) --suffix $(SUFFIX) --duration $(DURATION) --offset $(OFFSET) $(if $(OVERWRITE),--overwrite) --verbose

help: ## Show this help
	@awk 'BEGIN {FS = "## "} /^[a-zA-Z_-]+:.*## / {split($$1, t, ":"); printf "  \033[36m%-16s\033[0m %s\n", t[1], $$2}' $(MAKEFILE_LIST)

venv: ## Create virtualenv and install Python deps
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install numpy

install: venv ## Install system deps (brew) + venv
	@echo "=== Installing system dependencies ==="
	brew install ffmpeg libltc || true

check-deps: ## Verify required tools are available
	@echo "ffmpeg:  $$(ffmpeg -version 2>&1 | head -1 || echo 'MISSING')"
	@echo "ltcdecode:  $$(ltcdecode 2>&1 | head -1 || echo 'MISSING')"
	@[ -x $(PYTHON) ] && $(PYTHON) -c "import numpy; print('numpy:    OK (' + numpy.__version__ + ')')" || echo "venv:     MISSING — run: make venv"

run: venv ## Process video files (INPUT_DIR, FPS, SUFFIX, DURATION, OFFSET, OVERWRITE)
	$(PYTHON) ltc_embed.py $(ARGS) $(INPUT_DIR)

clean: ## Remove all *_tc output files from current directory
	@find . -maxdepth 1 \( -iname '*_tc.mp4' -o -iname '*_tc.mov' -o -iname '*_tc.m4v' -o -iname '*_tc.mkv' -o -iname '*_tc.mts' \) -exec rm -v {} \;
