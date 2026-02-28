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
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid request body", err.Error())
		return
	}

	// Ensure ranking is non-nil for marshalling; server will treat nil as empty.
	if req.Ranking == nil {
		req.Ranking = []models.RankingItem{}
	}

	// Derive top guess (if any)
	var countryGuess *string
	var guessScore *float64
	if len(req.Ranking) > 0 {
		top := req.Ranking[0]
		cg := top.Country
		cs := top.Score
		countryGuess = &cg
		guessScore = &cs
	}

	// At creation, the 'country' field (the validated country) and 'country_score'
	// are typically not provided by the client, and are expected to be set later
	// (e.g., by an admin validation step).
	// Therefore, we will insert nil for 'country' and 'country_score' as per original logic.
	var country *string = nil       // 'country' field from the DB schema
	var countryScore *float64 = nil // 'country_score' field from the DB schema

	// Derive normalized score
	var normalizedScore *float64
	if guessScore != nil && len(req.Geometry) > 0 {
		type MultiLineString struct {
			Coordinates [][][]float64 `json:"coordinates"`
		}
		var geom MultiLineString
		if err := json.Unmarshal(req.Geometry, &geom); err == nil {
			// Compute geometry size
			size := 0
			for _, line := range geom.Coordinates {
				size += len(line)
			}
			if size > 0 {
				penalty := 200.0
				sizeF := float64(size)
				sizeFactor := sizeF / (sizeF + penalty)
				ns := (*guessScore) * sizeFactor
				normalizedScore = &ns
			}
		}
	}

	// Marshal ranking to JSON for JSONB storage
	rankingJSON, err := json.Marshal(req.Ranking)
	if err != nil {
		api.Logger.Error("marshal ranking failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to marshal ranking", err.Error())
		return
	}

	// Insert row
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var id string
	// Notice: passing `country` and `countryScore` (both nil) to the insert query.
	err = api.Pool.QueryRow(ctx,
		`INSERT INTO drawings (geometry, country, author, author_id, ranking, country_score, country_guess, guess_score, normalized_score)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id`,
		req.Geometry, country, req.Author, req.AuthorID, rankingJSON, countryScore, countryGuess, guessScore, normalizedScore,
	).Scan(&id)
	if err != nil {
		api.Logger.Error("insert drawing failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to insert drawing", err.Error())
		return
	}

	// Return created id
	api.writeJSON(w, http.StatusCreated, models.CreateDrawingResponse{ID: id})
}
