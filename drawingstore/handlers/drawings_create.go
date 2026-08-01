package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// CreateDrawing handles the POST /drawings endpoint.
func (api *API) CreateDrawing(w http.ResponseWriter, r *http.Request) {
	var req models.CreateDrawingRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid request body", err.Error())
		return
	}

	if len(req.Ranking) == 0 {
		api.writeError(w, http.StatusBadRequest, "ranking must not be empty", "")
		return
	}

	pointCount, err := models.PointCount(req.Geometry)
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid geometry", err.Error())
		return
	}

	drawing := models.Drawing{
		Country:    req.Country,
		Author:     req.Author,
		AuthorID:   req.AuthorID,
		Ranking:    req.Ranking,
		PointCount: &pointCount,
	}
	drawing.CalculateDerivedFields()

	rankingJSON, err := json.Marshal(req.Ranking)
	if err != nil {
		api.Logger.Error("marshal ranking failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to marshal ranking", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var id string
	err = api.Pool.QueryRow(ctx,
		`INSERT INTO drawings (
			geometry, country, author, author_id, ranking, point_count,
			country_score, country_guess, guess_score, normalized_score
		)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id`,
		req.Geometry, drawing.Country, drawing.Author, drawing.AuthorID, rankingJSON,
		pointCount, drawing.CountryScore, drawing.CountryGuess, drawing.GuessScore,
		drawing.NormalizedScore,
	).Scan(&id)
	if err != nil {
		api.Logger.Error("insert drawing failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to insert drawing", err.Error())
		return
	}

	// Return created id
	api.writeJSON(w, http.StatusCreated, models.CreateDrawingResponse{ID: id})
}
