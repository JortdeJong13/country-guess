-- +goose Up
ALTER TABLE drawings
ADD COLUMN IF NOT EXISTS report_count INTEGER NOT NULL DEFAULT 0;

DROP INDEX IF EXISTS idx_drawings_unvalidated_created_at;
CREATE INDEX IF NOT EXISTS idx_drawings_validation_queue
  ON drawings (
    report_count DESC,
    country_score DESC NULLS LAST,
    created_at ASC,
    id ASC
  )
  WHERE country IS NOT NULL
    AND (validated = false OR report_count > 0);

-- +goose Down
DROP INDEX IF EXISTS idx_drawings_validation_queue;

CREATE INDEX IF NOT EXISTS idx_drawings_unvalidated_created_at
  ON drawings (created_at ASC, id ASC)
  WHERE country IS NOT NULL AND validated = false;

ALTER TABLE drawings DROP COLUMN IF EXISTS report_count;
