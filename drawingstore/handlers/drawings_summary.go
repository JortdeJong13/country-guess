package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// GetDrawingSummary returns lightweight counts for the drawing collection.
func (api *API) GetDrawingSummary(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var summary models.DrawingSummaryResponse
	err := api.Pool.QueryRow(ctx, `
		SELECT
			count(*),
			count(*) FILTER (WHERE country IS NOT NULL),
			count(*) FILTER (WHERE validated = true)
		FROM drawings`).Scan(
		&summary.Total,
		&summary.WithFeedback,
		&summary.Validated,
	)
	if err != nil {
		api.Logger.Error("summarize drawings failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to summarize drawings", "")
		return
	}

	api.writeJSON(w, http.StatusOK, summary)
}
