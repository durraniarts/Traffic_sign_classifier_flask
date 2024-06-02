.PHONY: setup 

setup:
	python -m venv .venv
	$(if $(filter $(OS),Windows_NT),\
	.venv\Scripts\activate && pip install -r requirements.txt,\
	source .venv/bin/activate && pip install -r requirements.txt)