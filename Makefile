# Variables
MODEL_NAME=triplet_model
MLSERVER_URL=http://127.0.0.1:5001
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
	@echo "  run-admin           Start the admin app"

# Push new drawings to GitHub
.PHONY: push-drawings
push-drawings:
	@echo "Checking for new drawings..."
	@git fetch origin main
	@! git merge --ff--only origin/main >/dev/null 2>&1 || true
	@if [ $$? -eq 0 ]; then \
		echo "Already up-to-date or fast-forwarded to origin/main."; \
	fi
	@if git status --porcelain -- data/drawings | grep -q .; then \
		echo "Pushing new drawings to GitHub..."; \
		git add data/drawings/*; \
		git commit -m "New drawings $$(date -I)"; \
		git push; \
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
	@python -m webbrowser http://127.0.0.1:5002
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

# Start the admin app
.PHONY: run-admin
run-admin:
	@python -m webbrowser http://127.0.0.1:5003
	@echo "Starting admin app..."
	python -m webapp.admin
