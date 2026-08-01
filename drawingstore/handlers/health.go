package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// GetHealth handles the GET /health endpoint.
func (api *API) GetHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := api.Pool.Ping(ctx); err != nil {
		api.Logger.Error("health check database ping failed", "error", err)
		api.writeError(w, http.StatusServiceUnavailable, "database unavailable", "")
		return
	}
	api.writeJSON(w, http.StatusOK, models.HealthResponse{Status: "healthy"})
}
