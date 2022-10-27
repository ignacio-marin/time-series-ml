VENV := venv
PYTHON := $(VENV)/bin/python

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV) requirements.txt
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install wheel
	$(PYTHON) -m pip install -r requirements.txt

data-folder: 
	mkdir -p data/bike_sharing/raw
	mkdir -p data/bike_sharing/processed
	mkdir -p data/bike_sharing/models
	mkdir -p data/walmart/raw
	mkdir -p data/walmart/processed
	mkdir -p data/walmart/models