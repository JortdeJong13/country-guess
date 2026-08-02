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

// UpdateDrawing handles feedback and admin validation updates.
func (api *API) UpdateDrawing(w http.ResponseWriter, r *http.Request) {
	id, err := uuid.Parse(chi.URLParam(r, "id"))
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid drawing ID", err.Error())
		return
	}

	var req models.UpdateDrawingRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid request body", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	tx, err := api.Pool.Begin(ctx)
	if err != nil {
		api.Logger.Error("begin update transaction failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to update drawing", "")
		return
	}
	defer tx.Rollback(ctx)

	drawing, err := scanDrawing(tx.QueryRow(ctx,
		`SELECT`+drawingFields+` FROM drawings WHERE id = $1 FOR UPDATE`,
		id,
	))
	if err != nil {
		if err == pgx.ErrNoRows {
			api.writeError(w, http.StatusNotFound, "drawing not found", "no drawing with the given ID")
			return
		}
		api.Logger.Error("load drawing for update failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to update drawing", "")
		return
	}

	if req.Country != nil {
		drawing.Country = req.Country
	}
	if req.Author != nil {
		drawing.Author = req.Author
	}
	if req.Report != nil && *req.Report {
		drawing.ReportCount++
	}
	if req.Validated != nil {
		drawing.Validated = *req.Validated
		if *req.Validated {
			drawing.ReportCount = 0
		}
	}
	drawing.CalculateDerivedFields()

	updated, err := tx.Exec(ctx,
		`UPDATE drawings
		 SET country = $1, author = $2, validated = $3,
		     country_score = $4, country_guess = $5, guess_score = $6,
		     report_count = $7
		 WHERE id = $8`,
		drawing.Country, drawing.Author, drawing.Validated,
		drawing.CountryScore, drawing.CountryGuess, drawing.GuessScore,
		drawing.ReportCount, id,
	)
	if err != nil {
		api.Logger.Error("update drawing failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to update drawing", "")
		return
	}
	if updated.RowsAffected() != 1 {
		api.writeError(w, http.StatusNotFound, "drawing not found", "no drawing with the given ID")
		return
	}

	if err := tx.Commit(ctx); err != nil {
		api.Logger.Error("commit drawing update failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to update drawing", "")
		return
	}

	drawing, err = scanDrawing(api.Pool.QueryRow(ctx,
		`SELECT`+drawingFields+` FROM drawings WHERE id = $1`,
		id,
	))
	if err != nil {
		api.Logger.Error("reload updated drawing failed", "error", err, "id", id)
		api.writeError(w, http.StatusInternalServerError, "failed to retrieve updated drawing", "")
		return
	}

	api.writeJSON(w, http.StatusOK, drawing)
}
