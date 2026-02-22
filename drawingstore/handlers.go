package main

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"log/slog"

	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// RegisterRoutes registers drawingstore HTTP handlers on the provided router.
// It expects a pgxpool for DB access and an slog logger for structured logs.
//
// Current handlers:
// - GET  /health    : health check
// - POST /drawings  : create a new drawing
func RegisterRoutes(r chi.Router, pool *pgxpool.Pool, logger *slog.Logger) {
	// Create a JSON helper scoped to this file.
	writeJSON := func(w http.ResponseWriter, status int, v any) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(v)
	}

	writeError := func(w http.ResponseWriter, status int, message, err string) {
		writeJSON(w, status, models.ErrorResponse{Message: message, Error: err})
	}

	// Health endpoint
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		// Simple health response - keep fast and dependency-free here.
		writeJSON(w, http.StatusOK, models.HealthResponse{Status: "healthy"})
	})

	// POST /drawings - insert a new drawing
	r.Post("/drawings", func(w http.ResponseWriter, r *http.Request) {
		// Minimal validation by design: assume caller sends correct payload.
		var req models.CreateDrawingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid request body", err.Error())
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

		// Marshal ranking to JSON for JSONB storage
		rankingJSON, err := json.Marshal(req.Ranking)
		if err != nil {
			logger.Error("marshal ranking failed", "error", err)
			writeError(w, http.StatusInternalServerError, "failed to marshal ranking", err.Error())
			return
		}

		// Insert row
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		var id string
		err = pool.QueryRow(ctx,
			`INSERT INTO drawings (geometry, country, author, hashed_ip, ranking, country_guess, guess_score)
			 VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
			req.Geometry, nil, req.Author, req.HashedIP, rankingJSON, countryGuess, guessScore,
		).Scan(&id)
		if err != nil {
			logger.Error("insert drawing failed", "error", err)
			writeError(w, http.StatusInternalServerError, "failed to insert drawing", err.Error())
			return
		}

		// Return created id
		writeJSON(w, http.StatusCreated, models.CreateDrawingResponse{ID: id})
	})
}

// writeError is a helper to write a JSON error response.
func writeError(w http.ResponseWriter, status int, message, err string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(models.ErrorResponse{Message: message, Error: err})
}
