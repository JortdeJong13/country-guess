package models

import "encoding/json"

// CreateDrawingRequest is the payload accepted by POST /drawings.
type CreateDrawingRequest struct {
	Geometry json.RawMessage `json:"geometry"`
	Ranking  []RankingItem   `json:"ranking"`
	Author   *string         `json:"author,omitempty"`
	AuthorID *string         `json:"author_id,omitempty"`
}

// CreateDrawingResponse is returned after successfully creating a drawing.
type CreateDrawingResponse struct {
	ID string `json:"id"`
}

// UpdateDrawingRequest is the allowed partial-update payload for PUT /drawings/{id}.
type UpdateDrawingRequest struct {
	Country   *string        `json:"country,omitempty"`
	Author    *string        `json:"author,omitempty"`
	AuthorID  *string        `json:"author_id,omitempty"`
	Validated *bool          `json:"validated,omitempty"`
	Ranking   *[]RankingItem `json:"ranking,omitempty"`
}

// GetDrawingResponse represents the response shape for GET /drawings which
type GetDrawingResponse struct {
	Drawing *Drawing `json:"drawing"`
	Total   int      `json:"total"`
}

// ErrorResponse is a simple JSON error shape used by handlers.
type ErrorResponse struct {
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}
