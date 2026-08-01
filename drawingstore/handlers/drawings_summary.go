package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// GetDrawingSummary returns lightweight counts for the drawing collection.
func (api *API) GetDrawingSummary(w http.ResponseWriter, r *http.Request) {
	authorID := r.URL.Query().Get("author_id")
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	query := `
		SELECT
			count(*),
			count(*) FILTER (WHERE country IS NOT NULL),
			count(*) FILTER (WHERE validated = true),
			count(*) FILTER (WHERE country IS NOT NULL AND validated = false),
			count(DISTINCT author_id) FILTER (WHERE author_id IS NOT NULL AND author_id <> '')`
	args := []any{}
	if authorID != "" {
		query += `,
			count(*) FILTER (WHERE validated = true AND author_id = $1)`
		args = append(args, authorID)
	}
	query += ` FROM drawings`

	var summary models.DrawingSummaryResponse
	scanTargets := []any{
		&summary.Total,
		&summary.WithFeedback,
		&summary.Validated,
		&summary.Unvalidated,
		&summary.UniqueAuthors,
	}
	var validatedByAuthor int
	if authorID != "" {
		scanTargets = append(scanTargets, &validatedByAuthor)
	}
	err := api.Pool.QueryRow(ctx, query, args...).Scan(scanTargets...)
	if err != nil {
		api.Logger.Error("summarize drawings failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to summarize drawings", "")
		return
	}
	if authorID != "" {
		summary.ValidatedByAuthor = &validatedByAuthor
	}

	api.writeJSON(w, http.StatusOK, summary)
}
