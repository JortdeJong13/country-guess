-- +goose Up
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS drawings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    geometry JSONB NOT NULL,
    country TEXT,
    author TEXT,
    author_id TEXT,
    validated BOOLEAN NOT NULL DEFAULT FALSE,
    ranking JSONB NOT NULL DEFAULT '[]'::jsonb,
    country_score DOUBLE PRECISION,
    country_guess TEXT,
    guess_score DOUBLE PRECISION,
    normalized_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Leaderboard index:
CREATE INDEX IF NOT EXISTS idx_drawings_normalized_score
  ON drawings (normalized_score DESC)
  WHERE country = country_guess AND normalized_score IS NOT NULL;

-- Admin index for unvalidated drawings:
CREATE INDEX IF NOT EXISTS idx_drawings_unvalidated_created_at
  ON drawings (created_at ASC)
  WHERE validated = false;

-- Trigger function to keep updated_at current on updates.
-- +goose StatementBegin
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd

DROP TRIGGER IF EXISTS set_updated_at ON drawings;
CREATE TRIGGER set_updated_at
BEFORE UPDATE ON drawings
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- +goose Down
-- Rollback: drop indexes, trigger, function, and table.
DROP INDEX IF EXISTS idx_drawings_unvalidated_id;
DROP INDEX IF EXISTS idx_drawings_normalized_score;

DROP TRIGGER IF EXISTS set_updated_at ON drawings;
DROP FUNCTION IF EXISTS update_updated_at_column();

DROP TABLE IF EXISTS drawings;
