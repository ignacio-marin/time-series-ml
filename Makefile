VENV := venv
PYTHON := $(VENV)/bin/python

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV) requirements.txt
	$(PYTHON) -m pip install -r requirements.txt
