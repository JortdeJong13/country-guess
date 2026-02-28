package handlers

import (
	"net/http"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// GetHealth handles the GET /health endpoint.
func (api *API) GetHealth(w http.ResponseWriter, r *http.Request) {
	api.writeJSON(w, http.StatusOK, models.HealthResponse{Status: "healthy"})
}
