-- +goose Up
-- The leaderboard now uses the model score directly. Very simple drawings
-- remain stored, but need at least 100 points to enter the leaderboard.
DROP INDEX IF EXISTS idx_drawings_normalized_score;
ALTER TABLE drawings DROP COLUMN IF EXISTS normalized_score;

CREATE INDEX IF NOT EXISTS idx_drawings_leaderboard
  ON drawings (country_score DESC, created_at ASC, id ASC)
  WHERE country IS NOT NULL
    AND country = country_guess
    AND country <> 'Other'
    AND country_score IS NOT NULL
    AND point_count >= 100;

-- +goose Down
DROP INDEX IF EXISTS idx_drawings_leaderboard;

ALTER TABLE drawings ADD COLUMN IF NOT EXISTS normalized_score DOUBLE PRECISION;
UPDATE drawings
SET normalized_score = country_score * point_count / (point_count + 200.0)
WHERE country IS NOT NULL
  AND country = country_guess
  AND country <> 'Other'
  AND country_score IS NOT NULL
  AND point_count IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_drawings_normalized_score
  ON drawings (normalized_score DESC, created_at ASC, id ASC)
  WHERE country IS NOT NULL
    AND country = country_guess
    AND country <> 'Other'
    AND normalized_score IS NOT NULL;
