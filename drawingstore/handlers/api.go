package handlers

import (
	"encoding/json"
	"net/http"

	"log/slog"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// API encapsulates all application handlers and their dependencies.
type API struct {
	Pool   *pgxpool.Pool
	Logger *slog.Logger
}

// NewAPI creates a new API instance with injected dependencies.
func NewAPI(pool *pgxpool.Pool, logger *slog.Logger) *API {
	return &API{
		Pool:   pool,
		Logger: logger,
	}
}

// RegisterRoutes registers all drawingstore HTTP handlers on the provided router.
func (api *API) RegisterRoutes(r chi.Router) {
	// Health endpoint
	r.Get("/health", api.GetHealth)

	// Drawings endpoints
	r.Post("/drawings", api.CreateDrawing)
	r.Get("/drawings/{id}", api.GetDrawing)
	r.Delete("/drawings/{id}", api.DeleteDrawing)
}

// writeJSON is a helper to write a JSON response.
func (api *API) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		api.Logger.Error("failed to write JSON response", "error", err, "status", status, "payload", v)
		// Fallback to a generic error if JSON encoding fails for some reason
		http.Error(w, `{"message":"internal server error"}`, http.StatusInternalServerError)
	}
}

// writeError is a helper to write a JSON error response.
func (api *API) writeError(w http.ResponseWriter, status int, message, err string) {
	api.writeJSON(w, status, models.ErrorResponse{Message: message, Error: err})
}
