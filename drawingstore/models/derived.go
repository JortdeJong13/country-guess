package models

// CalculateDerivedFields updates the values that can be calculated from a
// drawing's ranking, feedback, and geometry complexity.
func (d *Drawing) CalculateDerivedFields() {
	d.CountryGuess = nil
	d.GuessScore = nil
	d.CountryScore = nil
	d.NormalizedScore = nil

	if len(d.Ranking) == 0 {
		return
	}

	guess := d.Ranking[0]
	d.CountryGuess = stringPointer(guess.Country)
	d.GuessScore = floatPointer(guess.Score)

	// Pending drawings have no feedback yet, so they intentionally have no
	// country score or leaderboard score.
	if d.Country == nil || d.PointCount == nil {
		return
	}

	for _, item := range d.Ranking {
		if item.Country != *d.Country {
			continue
		}

		d.CountryScore = floatPointer(item.Score)
		if item.Country == guess.Country && item.Country != "Other" {
			const complexityPenalty = 200.0
			factor := float64(*d.PointCount) /
				(float64(*d.PointCount) + complexityPenalty)
			d.NormalizedScore = floatPointer(item.Score * factor)
		}
		return
	}
}

func stringPointer(value string) *string {
	return &value
}

func floatPointer(value float64) *float64 {
	return &value
}
