package models

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// RankingItem represents a single (country, score) tuple produced by the ML model.
// [ ["France", 0.95], ["Germany", 0.03], ... ]
type RankingItem struct {
	Country string  `json:"country"`
	Score   float64 `json:"score"`
}

// Drawing is the canonical representation of a stored drawing for the service.
type Drawing struct {
	ID        uuid.UUID       `json:"id"`
	Geometry  json.RawMessage `json:"geometry"` // GeoJSON geometry object
	Country   *string         `json:"country_name,omitempty"`
	Author    *string         `json:"author,omitempty"`
	HashedIP  *string         `json:"hashed_ip,omitempty"`
	Validated bool            `json:"validated"`
	Ranking   json.RawMessage `json:"ranking"` // JSON array of [country, score] pairs

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`

	// Derived fields (computed by the service)
	CountryScore *float64 `json:"country_score,omitempty"`
	CountryGuess *string  `json:"country_guess,omitempty"`
	GuessScore   *float64 `json:"guess_score,omitempty"`
}

// ParseRanking strictly parses the Drawing.Ranking JSON into a slice of
func (d *Drawing) ParseRanking() ([]RankingItem, error) {
	if len(d.Ranking) == 0 {
		return nil, nil
	}

	var raw []any
	if err := json.Unmarshal(d.Ranking, &raw); err != nil {
		return nil, fmt.Errorf("invalid ranking JSON: %w", err)
	}

	out := make([]RankingItem, 0, len(raw))
	for i, elem := range raw {
		pair, ok := elem.([]any)
		if !ok || len(pair) != 2 {
			return nil, fmt.Errorf("ranking[%d] is not a two-element array", i)
		}

		// First element must be a string (country name)
		name, ok := pair[0].(string)
		if !ok {
			return nil, fmt.Errorf("ranking[%d][0] is not a string", i)
		}

		// Second element must be a float64 (default JSON number decoding)
		scoreVal, ok := pair[1].(float64)
		if !ok {
			return nil, fmt.Errorf("ranking[%d][1] is not a numeric value (float64 expected)", i)
		}

		out = append(out, RankingItem{
			Country: name,
			Score:   scoreVal,
		})
	}

	return out, nil
}

// ComputeDerived computes and sets the derived fields on the Drawing instance:
func (d *Drawing) ComputeDerived() error {
	items, err := d.ParseRanking()
	if err != nil {
		return err
	}
	if len(items) == 0 {
		// nothing to compute
		d.CountryGuess = nil
		d.GuessScore = nil
		d.CountryScore = nil
		return nil
	}

	// Top guess is the first item
	top := items[0]
	d.CountryGuess = &top.Country
	d.GuessScore = &top.Score

	// If a canonical country is set, find its score in the ranking
	if d.Country != nil {
		for _, it := range items {
			if it.Country == *d.Country {
				s := it.Score
				d.CountryScore = &s
				break
			}
		}
	}

	return nil
}
