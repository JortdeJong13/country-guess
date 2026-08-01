package models

// CalculateDerivedFields updates the values that can be calculated from a
// drawing's ranking and feedback.
func (d *Drawing) CalculateDerivedFields() {
	d.CountryGuess = nil
	d.GuessScore = nil
	d.CountryScore = nil

	if len(d.Ranking) == 0 {
		return
	}

	guess := d.Ranking[0]
	d.CountryGuess = stringPointer(guess.Country)
	d.GuessScore = floatPointer(guess.Score)

	// Pending drawings have no feedback yet, so they intentionally have no
	// country score.
	if d.Country == nil {
		return
	}

	for _, item := range d.Ranking {
		if item.Country != *d.Country {
			continue
		}

		d.CountryScore = floatPointer(item.Score)
		return
	}
}

func stringPointer(value string) *string {
	return &value
}

func floatPointer(value float64) *float64 {
	return &value
}
