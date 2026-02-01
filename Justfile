# --- Variables ---

MODEL_NAME := "triplet_model"
MLSERVER_URL := "http://127.0.0.1:5001"
DEBUG := "1"

# --- Recipes ---

# Push new drawings to GitHub
push-drawings:
    @echo "Checking for new drawings..."
    @git fetch origin main
    @git merge --ff-only origin/main >/dev/null 2>&1 || true
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
evaluate-model:
    @echo "Evaluating the model..."
    python -m tests.evaluation --model_name {{ MODEL_NAME }}

# Run end-to-end test
test-e2e:
    @echo "Running end-to-end tests..."
    python -m unittest discover tests -v

# Start the ML server
run-mlserver:
    @echo "Starting ML server..."
    DEBUG={{ DEBUG }} MODEL_NAME={{ MODEL_NAME }} python -m mlserver.serve

# Start the web app
run-webapp:
    @echo "Starting web app..."
    DEBUG={{ DEBUG }} MLSERVER_URL={{ MLSERVER_URL }} python -m webapp.app

# Start both the ML server and web app
run-app:
    @echo "Starting both ML server and web app..."
    DEBUG={{ DEBUG }} MODEL_NAME={{ MODEL_NAME }} python -m mlserver.serve &
    ML_PID=$$!
    @echo "ML server started in background with PID: $$ML_PID"
    bash -c '
    trap "echo \"Terminating ML server (PID: $$ML_PID)...\"; kill $$ML_PID 2>/dev/null || true" EXIT TERM INT
    DEBUG={{ DEBUG }} MLSERVER_URL={{ MLSERVER_URL }} python -m webapp.app
    '

# Start the admin app
run-admin:
    @echo "Starting admin app..."
    DEBUG={{ DEBUG }} python -m webapp.admin
