# --- Variables ---

MODEL_NAME := "triplet_model"
MLSERVER_URL := "http://127.0.0.1:5001"
DRAWING_STORE_URL := "http://home-server:8080"
DATABASE_URL := "postgres://country_guess:country_guess_dev@127.0.0.1:5432/country_guess?sslmode=disable"
DEBUG := "1"

# --- Recipes ---

# Evaluate the model
evaluate-model:
    @echo "Evaluating the model..."
    python -m tests.evaluation --model_name {{ MODEL_NAME }} --drawing_store_url {{ DRAWING_STORE_URL }}

# Run end-to-end test against native PostgreSQL. Run `just setup-local-db` first
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
    DEBUG={{ DEBUG }} MLSERVER_URL={{ MLSERVER_URL }} DRAWING_STORE_URL={{ DRAWING_STORE_URL }} python -m webapp.app

# Prepare the native PostgreSQL database used by the local app mode
setup-local-db:
    @echo "Checking native PostgreSQL..."
    @pg_isready -h 127.0.0.1 -p 5432
    @if ! psql -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname = 'country_guess'" | grep -q 1; then \
        createuser --login --createdb country_guess; \
    fi
    @psql -d postgres -v ON_ERROR_STOP=1 -c "ALTER ROLE country_guess WITH LOGIN CREATEDB PASSWORD 'country_guess_dev';"
    @if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'country_guess'" | grep -q 1; then \
        createdb --owner=country_guess country_guess; \
    fi
    @echo "Native PostgreSQL database is ready."

# Start the drawings API locally. PostgreSQL must be running separately
run-drawingstore:
    @echo "Starting drawingstore..."
    DATABASE_URL={{ DATABASE_URL }} go -C drawingstore run .

# Start the complete app
run-app:
    @DEBUG="{{ DEBUG }}" MODEL_NAME="{{ MODEL_NAME }}" MLSERVER_URL="{{ MLSERVER_URL }}" DRAWING_STORE_URL="{{ DRAWING_STORE_URL }}" DATABASE_URL="{{ DATABASE_URL }}" bash scripts/run-app.sh

# Start the admin app
run-admin:
    @echo "Starting admin app..."
    DEBUG={{ DEBUG }} DRAWING_STORE_URL={{ DRAWING_STORE_URL }} python -m webapp.admin

# Print the total unique authors based on author_id
unique-authors:
    @echo "Counting total unique authors..."
    @curl --fail --silent --show-error "{{ DRAWING_STORE_URL }}/drawings/summary" | jq -r '.unique_authors'
