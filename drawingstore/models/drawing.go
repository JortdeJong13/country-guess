package models

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

// RankingItem is one model prediction in descending score order.
type RankingItem struct {
	Country string  `json:"country"`
	Score   float64 `json:"score"`
}

// Drawing is the canonical representation of a stored drawing for the service.
type Drawing struct {
	ID           uuid.UUID       `json:"id"`
	Geometry     json.RawMessage `json:"geometry"` // GeoJSON geometry object
	Country      *string         `json:"country,omitempty"`
	Author       *string         `json:"author,omitempty"`
	AuthorID     *string         `json:"author_id,omitempty"`
	Validated    bool            `json:"validated"`
	Ranking      []RankingItem   `json:"ranking"`
	CountryScore *float64        `json:"country_score,omitempty"`
	CountryGuess *string         `json:"country_guess,omitempty"`
	GuessScore   *float64        `json:"guess_score,omitempty"`
	PointCount   *int            `json:"point_count,omitempty"`
	CreatedAt    time.Time       `json:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at"`
}
