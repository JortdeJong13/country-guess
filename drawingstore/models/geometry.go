package models

import (
	"encoding/json"
	"errors"
	"fmt"
)

type multiLineString struct {
	Type        string        `json:"type"`
	Coordinates [][][]float64 `json:"coordinates"`
}

// PointCount validates the supported GeoJSON geometry and returns its number
// of points.
func PointCount(raw json.RawMessage) (int, error) {
	if len(raw) == 0 || !json.Valid(raw) {
		return 0, errors.New("geometry must be valid JSON")
	}

	var geometry multiLineString
	if err := json.Unmarshal(raw, &geometry); err != nil {
		return 0, fmt.Errorf("invalid geometry: %w", err)
	}
	if geometry.Type != "MultiLineString" {
		return 0, errors.New("geometry must be a MultiLineString")
	}

	pointCount := 0
	for _, line := range geometry.Coordinates {
		if len(line) < 2 {
			return 0, errors.New("each line must contain at least two points")
		}
		pointCount += len(line)
	}
	if pointCount == 0 {
		return 0, errors.New("geometry must contain at least one line")
	}

	return pointCount, nil
}
