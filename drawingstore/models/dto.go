package models

import "encoding/json"

// CreateDrawingRequest is the payload accepted by POST /drawings.
type CreateDrawingRequest struct {
	Geometry json.RawMessage `json:"geometry"`
	Ranking  []RankingItem   `json:"ranking"`
	Country  *string         `json:"country,omitempty"`
	Author   *string         `json:"author,omitempty"`
	AuthorID *string         `json:"author_id,omitempty"`
}

// CreateDrawingResponse is returned after successfully creating a drawing.
type CreateDrawingResponse struct {
	ID string `json:"id"`
}

// UpdateDrawingRequest is the allowed partial-update payload for PATCH /drawings/{id}.
type UpdateDrawingRequest struct {
	Country   *string `json:"country,omitempty"`
	Author    *string `json:"author,omitempty"`
	Validated *bool   `json:"validated,omitempty"`
}

// ListDrawingsResponse is returned by filtered drawing collections.
type ListDrawingsResponse struct {
	Drawings []*Drawing `json:"drawings"`
	Total    int        `json:"total"`
}

// DrawingSummaryResponse contains lightweight collection counts.
type DrawingSummaryResponse struct {
	Total             int  `json:"total"`
	WithFeedback      int  `json:"with_feedback"`
	Validated         int  `json:"validated"`
	Unvalidated       int  `json:"unvalidated"`
	UniqueAuthors     int  `json:"unique_authors"`
	ValidatedByAuthor *int `json:"validated_by_author,omitempty"`
}

// LeaderboardResponse contains one stable leaderboard position.
type LeaderboardResponse struct {
	Rank    int      `json:"rank"`
	Total   int      `json:"total"`
	Drawing *Drawing `json:"drawing"`
}

// ErrorResponse is a simple JSON error shape used by handlers.
type ErrorResponse struct {
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}
