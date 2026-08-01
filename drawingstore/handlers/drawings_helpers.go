package handlers

import (
	"encoding/json"
	"fmt"

	"github.com/jortdejong13/country-guess/drawingstore/models"
)

const drawingFields = `
	id, geometry, country, author, author_id, validated,
	ranking, country_score, country_guess, guess_score, point_count,
	created_at, updated_at`

type rowScanner interface {
	Scan(dest ...any) error
}

func scanDrawing(row rowScanner) (*models.Drawing, error) {
	var drawing models.Drawing
	var geometryJSON []byte
	var rankingJSON []byte

	err := row.Scan(
		&drawing.ID,
		&geometryJSON,
		&drawing.Country,
		&drawing.Author,
		&drawing.AuthorID,
		&drawing.Validated,
		&rankingJSON,
		&drawing.CountryScore,
		&drawing.CountryGuess,
		&drawing.GuessScore,
		&drawing.PointCount,
		&drawing.CreatedAt,
		&drawing.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	drawing.Geometry = json.RawMessage(geometryJSON)
	if err := json.Unmarshal(rankingJSON, &drawing.Ranking); err != nil {
		return nil, fmt.Errorf("decode ranking: %w", err)
	}
	return &drawing, nil
}
