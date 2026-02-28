package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jortdejong13/country-guess/drawingstore/models"
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

	var drawing models.Drawing
	var rankingJSON json.RawMessage // To read JSONB from DB

	err = api.Pool.QueryRow(ctx,
		`SELECT
			id, geometry, country, author, author_id, validated,
			ranking, country_score, country_guess, guess_score, normalized_score,
			created_at, updated_at
		 FROM drawings WHERE id = $1`,
		id,
	).Scan(
		&drawing.ID,
		&drawing.Geometry,
		&drawing.Country,
		&drawing.Author,
		&drawing.AuthorID,
		&drawing.Validated,
		&rankingJSON,
		&drawing.CountryScore,
		&drawing.CountryGuess,
		&drawing.GuessScore,
		&drawing.NormalizedScore,
		&drawing.CreatedAt,
		&drawing.UpdatedAt,
	)

	if err != nil {
		if err == pgx.ErrNoRows {
			api.writeError(w, http.StatusNotFound, "drawing not found", "no drawing with the given ID")
			return
		}
		api.Logger.Error("query drawing failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to retrieve drawing", err.Error())
		return
	}

	// Unmarshal the ranking JSONB into the struct
	if err := json.Unmarshal(rankingJSON, &drawing.Ranking); err != nil {
		api.Logger.Error("unmarshal ranking failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to parse ranking data", err.Error())
		return
	}

	api.writeJSON(w, http.StatusOK, drawing)
}
