package handlers

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jortdejong13/country-guess/drawingstore/models"
)

const leaderboardMinimumPointCount = 100

// ListDrawings handles filtered and bounded drawing collection queries.
func (api *API) ListDrawings(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	queue := query.Get("queue")

	validated, err := optionalBool(query.Get("validated"))
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid validated filter", err.Error())
		return
	}
	excludeOther, err := optionalBool(query.Get("exclude_other"))
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid exclude_other filter", err.Error())
		return
	}
	if queue != "" && queue != "validation" {
		api.writeError(w, http.StatusBadRequest, "invalid queue", "supported queue: validation")
		return
	}

	defaultLimit := 100
	if queue == "validation" {
		defaultLimit = 1
	}
	limit, err := nonNegativeInt(query.Get("limit"), defaultLimit)
	if err != nil || limit == 0 || limit > 1000 {
		api.writeError(w, http.StatusBadRequest, "invalid limit", "limit must be between 1 and 1000")
		return
	}
	offset, err := nonNegativeInt(query.Get("offset"), 0)
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid offset", "offset must be non-negative")
		return
	}

	where, args := drawingFilters(queue, validated, excludeOther)
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var total int
	if err := api.Pool.QueryRow(ctx,
		"SELECT count(*) FROM drawings WHERE "+where,
		args...,
	).Scan(&total); err != nil {
		api.Logger.Error("count drawings failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to list drawings", "")
		return
	}

	listArgs := append(append([]any{}, args...), limit, offset)
	rows, err := api.Pool.Query(ctx,
		`SELECT`+drawingFields+` FROM drawings WHERE `+where+
			` ORDER BY created_at ASC, id ASC LIMIT $`+strconv.Itoa(len(args)+1)+
			` OFFSET $`+strconv.Itoa(len(args)+2),
		listArgs...,
	)
	if err != nil {
		api.Logger.Error("list drawings failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to list drawings", "")
		return
	}
	defer rows.Close()

	drawings := make([]*models.Drawing, 0)
	for rows.Next() {
		drawing, err := scanDrawing(rows)
		if err != nil {
			api.Logger.Error("scan drawing failed", "error", err)
			api.writeError(w, http.StatusInternalServerError, "failed to list drawings", "")
			return
		}
		drawings = append(drawings, drawing)
	}
	if err := rows.Err(); err != nil {
		api.Logger.Error("iterate drawings failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to list drawings", "")
		return
	}

	api.writeJSON(w, http.StatusOK, models.ListDrawingsResponse{
		Drawings: drawings,
		Total:    total,
	})
}

// GetLeaderboard returns a single stable rank for efficient next/previous navigation.
func (api *API) GetLeaderboard(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	rank, err := nonNegativeInt(query.Get("rank"), 0)
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid rank", "rank must be non-negative")
		return
	}
	validated, err := optionalBool(query.Get("validated"))
	if err != nil {
		api.writeError(w, http.StatusBadRequest, "invalid validated filter", err.Error())
		return
	}

	where, args := leaderboardFilters(validated)
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var total int
	if err := api.Pool.QueryRow(ctx,
		"SELECT count(*) FROM drawings WHERE "+where,
		args...,
	).Scan(&total); err != nil {
		api.Logger.Error("count leaderboard drawings failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to load leaderboard", "")
		return
	}
	if rank >= total {
		api.writeError(w, http.StatusNotFound, "leaderboard rank not found", fmt.Sprintf("rank %d is outside 0-%d", rank, total-1))
		return
	}

	rowArgs := append(append([]any{}, args...), 1, rank)
	drawing, err := scanDrawing(api.Pool.QueryRow(ctx,
		`SELECT`+drawingFields+` FROM drawings WHERE `+where+
			` ORDER BY country_score DESC, created_at ASC, id ASC LIMIT $`+
			strconv.Itoa(len(args)+1)+` OFFSET $`+strconv.Itoa(len(args)+2),
		rowArgs...,
	))
	if err != nil {
		if err == pgx.ErrNoRows {
			api.writeError(w, http.StatusNotFound, "leaderboard rank not found", "")
			return
		}
		api.Logger.Error("load leaderboard drawing failed", "error", err)
		api.writeError(w, http.StatusInternalServerError, "failed to load leaderboard", "")
		return
	}

	api.writeJSON(w, http.StatusOK, models.LeaderboardResponse{
		Rank:    rank,
		Total:   total,
		Drawing: drawing,
	})
}

func drawingFilters(queue string, validated, excludeOther *bool) (string, []any) {
	filters := []string{"1 = 1"}
	args := make([]any, 0, 1)

	if queue == "validation" {
		// Pending rows have no feedback and must not enter the admin queue.
		filters = append(filters, "country IS NOT NULL", "validated = false")
	}
	if excludeOther != nil && *excludeOther {
		filters = append(filters, "country IS NOT NULL", "country <> 'Other'")
	}
	if validated != nil && queue != "validation" {
		args = append(args, *validated)
		filters = append(filters, fmt.Sprintf("validated = $%d", len(args)))
	}

	return strings.Join(filters, " AND "), args
}

func leaderboardFilters(validated *bool) (string, []any) {
	filters := []string{
		"country IS NOT NULL",
		"country = country_guess",
		"country <> 'Other'",
		"country_score IS NOT NULL",
		fmt.Sprintf("point_count >= %d", leaderboardMinimumPointCount),
	}
	args := make([]any, 0, 1)
	if validated != nil {
		args = append(args, *validated)
		filters = append(filters, fmt.Sprintf("validated = $%d", len(args)))
	}
	return strings.Join(filters, " AND "), args
}

func optionalBool(value string) (*bool, error) {
	if value == "" {
		return nil, nil
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return nil, fmt.Errorf("expected true or false")
	}
	return &parsed, nil
}

func nonNegativeInt(value string, defaultValue int) (int, error) {
	if value == "" {
		return defaultValue, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		return 0, fmt.Errorf("expected a non-negative integer")
	}
	return parsed, nil
}
