SHELL=/bin/bash
# Edit SRC_NAME to use a memorable name for imports. Must match name used in setup.py, since that determines name of *.egg-info
SRC_NAME=multi_label_emg
SCRIPTS_NAME=scripts
# Edit SYSTEM_PYTHON to use another python version
SYSTEM_PYTHON=python3.11
VENV_NAME=venv
PYTHON=$(VENV_NAME)/bin/python3
EGG=$(SRC_NAME).egg-info

all: setup lint

test: setup
	$(VENV_NAME)/bin/pytest

lint: setup
	$(VENV_NAME)/bin/pre-commit run --all-files

# python environment setup
.PHONY: setup
setup: $(EGG)

$(EGG): $(PYTHON) setup.py requirements.txt requirements-dev.txt
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -Ue .
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(VENV_NAME)/bin/pre-commit install

$(PYTHON):
	$(SYSTEM_PYTHON) -m venv $(VENV_NAME)

# For rebuilding python environment from scratch
destroy-setup: confirm
	rm -rf $(EGG)
	rm -rf $(VENV_NAME)

.PHONY: confirm
confirm:
	@( read -p "Confirm? [y/n]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
