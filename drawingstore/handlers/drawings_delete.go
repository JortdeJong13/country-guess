package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

// DeleteDrawing handles the DELETE /drawings/{id} endpoint.
func (api *API) DeleteDrawing(w http.ResponseWriter, r *http.Request) {
	drawingIDStr := chi.URLParam(r, "id")
	id, err := uuid.Parse(drawingIDStr)
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid drawing ID", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// Execute the DELETE query
	cmdTag, err := api.Pool.Exec(ctx, "DELETE FROM drawings WHERE id = $1", id)
	if err != nil {
		api.Logger.Error("delete drawing failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to delete drawing", err.Error())
		return
	}

	// Check if any row was actually deleted.
	if cmdTag.RowsAffected() == 0 {
		api.writeError(w, http.StatusNotFound, "drawing not found", "no drawing with the given ID")
		return
	}

	// Successfully deleted
	w.WriteHeader(http.StatusNoContent)
}
