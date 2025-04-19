# Variables
MODEL_NAME=triplet_model
MLSERVER_URL=http://127.0.0.1:5001/predict

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  push-drawings       Push new drawings to the GitHub repository"
	@echo "  evaluate-model      Evaluate the model"
	@echo "  test-e2e            Run end-to-end tests"
	@echo "  run-mlflow          Start the MLflow UI"
	@echo "  run-mlserver        Start the ML server"
	@echo "  run-webapp          Start the web app"
	@echo "  run-app             Start both the ML server and web app"

# Push new drawings to GitHub
.PHONY: push-drawings
push-drawings:
	@echo "Pushing new drawings to GitHub..."
	@cd ~/GitHub/country-guess || exit
	@if git status --porcelain | grep -q "data/drawings/"; then \
	  git add data/drawings/*; \
	  git commit -m "New drawings"; \
	  git push origin main:$(DRAWINGS_BRANCH); \
	else \
	  echo "No new drawings to push."; \
	fi

# Evaluate the model
.PHONY: evaluate-model
evaluate-model:
	@echo "Evaluating the model..."
	python -m tests.evaluation --model_name $(MODEL_NAME)

# Run end-to-end tests
.PHONY: test-e2e
test-e2e:
	@echo "Running end-to-end tests..."
	python -m unittest tests/test_e2e.py -v

# Start the MLflow UI
.PHONY: run-mlflow
run-mlflow:
	@echo "Starting MLflow UI on port 5002..."
	mlflow ui --port 5002

# Start the ML server
.PHONY: run-mlserver
run-mlserver:
	@echo "Starting ML server..."
	MODEL_NAME=$(MODEL_NAME) python -m mlserver.serve

# Start the web app
.PHONY: run-webapp
run-webapp:
	@echo "Starting web app..."
	MLSERVER_URL=$(MLSERVER_URL) python -m webapp.app

# Start both the ML server and web app
.PHONY: run-app
run-app:
	@echo "Starting both ML server and web app..."
	MODEL_NAME=$(MODEL_NAME) python -m mlserver.serve &
	MLSERVER_URL=$(MLSERVER_URL) python -m webapp.app
