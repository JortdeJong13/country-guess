# Variables
MODEL_NAME=triplet_model
MLSERVER_URL=http://127.0.0.1:5001/predict
DEBUG=1

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  push-drawings       Push new drawings to the GitHub repository"
	@echo "  evaluate-model      Evaluate the model"
	@echo "  test-e2e            Run end-to-end test"
	@echo "  run-mlserver        Start the ML server"
	@echo "  run-webapp          Start the web app"
	@echo "  run-app             Start both the ML server and web app"

# Push new drawings to GitHub
.PHONY: push-drawings
push-drawings:
	@echo "Pushing new drawings to GitHub..."
	@if git status --porcelain | grep -q "data/drawings/"; then \
	  git add data/drawings/*; \
	  git commit -m "New drawings"; \
	  git push origin HEAD:drawings; \
	else \
	  echo "No new drawings to push."; \
	fi

# Evaluate the model
.PHONY: evaluate-model
evaluate-model:
	@echo "Evaluating the model..."
	python -m tests.evaluation --model_name $(MODEL_NAME)

# Run end-to-end test
.PHONY: test-e2e
test-e2e:
	@echo "Running end-to-end tests..."
	python -m unittest tests/test_e2e.py -v

# Start the ML server
.PHONY: run-mlserver
run-mlserver:
	@echo "Starting ML server..."
	DEBUG=$(DEBUG) MODEL_NAME=$(MODEL_NAME) python -m mlserver.serve

# Start the web app
.PHONY: run-webapp
run-webapp:
	@echo "Starting web app..."
	DEBUG=$(DEBUG) MLSERVER_URL=$(MLSERVER_URL) python -m webapp.app

# Start both the ML server and web app
.PHONY: run-app
run-app:
	@echo "Starting both ML server and web app..."
	( \
		DEBUG=$(DEBUG) MODEL_NAME=$(MODEL_NAME) python -m mlserver.serve & \
		ML_PID=$$!; \
		DEBUG=$(DEBUG) MLSERVER_URL=$(MLSERVER_URL) python -m webapp.app; \
		kill $$ML_PID; \
	)
