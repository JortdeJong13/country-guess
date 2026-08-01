package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// GetDrawing handles the GET /drawings/{id} endpoint.
func (api *API) GetDrawing(w http.ResponseWriter, r *http.Request) {
	drawingIDStr := chi.URLParam(r, "id")
	id, err := uuid.Parse(drawingIDStr)
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid drawing ID", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	drawing, err := scanDrawing(api.Pool.QueryRow(ctx,
		`SELECT`+drawingFields+` FROM drawings WHERE id = $1`,
		id,
	))

	if err != nil {
		if err == pgx.ErrNoRows {
			api.writeError(w, http.StatusNotFound, "drawing not found", "no drawing with the given ID")
			return
		}
		api.Logger.Error("query drawing failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to retrieve drawing", err.Error())
		return
	}

	api.writeJSON(w, http.StatusOK, drawing)
}
