// Command import-legacy-drawings imports the committed GeoJSON corpus into
// drawingstore's PostgreSQL database. It is intentionally a one-shot command
// rather than part of the drawingstore server startup path.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jortdejong13/country-guess/drawingstore/migrations"
	"github.com/jortdejong13/country-guess/drawingstore/models"
)

const defaultDatabaseURL = "postgres://country_guess:country_guess_dev@127.0.0.1:5432/country_guess?sslmode=disable"

type legacyFeatureCollection struct {
	Type     string          `json:"type"`
	Features []legacyFeature `json:"features"`
}

type legacyFeature struct {
	Properties legacyProperties `json:"properties"`
	Geometry   json.RawMessage  `json:"geometry"`
}

type legacyProperties struct {
	Timestamp  string          `json:"timestamp"`
	Country    *string         `json:"country_name"` // nil for drawings without feedback.
	Author     *string         `json:"author"`
	Validated  bool            `json:"validated"` // Missing values remain false.
	RawRanking json.RawMessage `json:"ranking"`
}

type importOptions struct {
	directory string
	dryRun    bool
	location  *time.Location
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{}))

	directory := flag.String("dir", "../data/drawings", "directory containing legacy GeoJSON drawings")
	dryRun := flag.Bool("dry-run", false, "validate and report without connecting to PostgreSQL")
	timezone := flag.String("legacy-timezone", "Europe/Amsterdam", "timezone for legacy timestamps without an offset")
	flag.Parse()

	location, err := time.LoadLocation(*timezone)
	if err != nil {
		logger.Error("invalid legacy timezone", "timezone", *timezone, "error", err)
		os.Exit(1)
	}

	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		databaseURL = defaultDatabaseURL
	}

	if err := run(logger, databaseURL, importOptions{
		directory: *directory,
		dryRun:    *dryRun,
		location:  location,
	}); err != nil {
		logger.Error("legacy drawing import failed", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger, databaseURL string, options importOptions) error {
	paths, err := filepath.Glob(filepath.Join(options.directory, "*.geojson"))
	if err != nil {
		return fmt.Errorf("find legacy drawings: %w", err)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return fmt.Errorf("no GeoJSON drawings found in %s", options.directory)
	}

	drawings := make([]models.Drawing, 0, len(paths))
	var invalid []string
	unvalidated := 0
	for _, path := range paths {
		drawing, err := loadLegacyDrawing(path, options.location)
		if err != nil {
			invalid = append(invalid, fmt.Sprintf("%s: %v", filepath.Base(path), err))
			continue
		}
		if !drawing.Validated {
			unvalidated++
		}
		drawings = append(drawings, *drawing)
	}

	logger.Info("legacy drawings scanned",
		"directory", options.directory,
		"files", len(paths),
		"valid", len(drawings),
		"invalid", len(invalid),
		"unvalidated", unvalidated,
	)
	for _, message := range invalid {
		logger.Error("invalid legacy drawing", "drawing", message)
	}
	if len(invalid) > 0 {
		return fmt.Errorf("%d legacy drawings could not be imported", len(invalid))
	}

	if options.dryRun {
		logger.Info("dry run complete", "drawings", len(drawings))
		return nil
	}

	if err := migrations.RunURL(databaseURL); err != nil {
		return fmt.Errorf("run database migrations: %w", err)
	}

	pool, err := pgxpool.New(context.Background(), databaseURL)
	if err != nil {
		return fmt.Errorf("connect to database: %w", err)
	}
	defer pool.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := pool.Ping(ctx); err != nil {
		return fmt.Errorf("ping database: %w", err)
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin import transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	inserted, skipped, err := insertDrawings(ctx, tx, drawings)
	if err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit import transaction: %w", err)
	}

	logger.Info("legacy drawing import complete",
		"scanned", len(drawings),
		"inserted", inserted,
		"skipped_existing", skipped,
	)
	return nil
}

func loadLegacyDrawing(path string, location *time.Location) (*models.Drawing, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var collection legacyFeatureCollection
	if err := json.Unmarshal(contents, &collection); err != nil {
		return nil, fmt.Errorf("decode GeoJSON: %w", err)
	}
	if collection.Type != "FeatureCollection" {
		return nil, fmt.Errorf("expected a FeatureCollection, found %q", collection.Type)
	}
	if len(collection.Features) != 1 {
		return nil, fmt.Errorf("expected one feature, found %d", len(collection.Features))
	}

	feature := collection.Features[0]
	pointCount, err := models.PointCount(feature.Geometry)
	if err != nil {
		return nil, err
	}
	ranking, err := parseRanking(feature.Properties.RawRanking)
	if err != nil {
		return nil, err
	}
	createdAt, err := parseLegacyTimestamp(feature.Properties.Timestamp, location)
	if err != nil {
		return nil, err
	}

	drawing := &models.Drawing{
		ID:         legacyID(filepath.Base(path)),
		Geometry:   feature.Geometry,
		Country:    feature.Properties.Country,
		Author:     feature.Properties.Author,
		Validated:  feature.Properties.Validated,
		Ranking:    ranking,
		PointCount: &pointCount,
		CreatedAt:  createdAt,
		UpdatedAt:  createdAt,
	}
	drawing.CalculateDerivedFields()
	return drawing, nil
}

func insertDrawings(ctx context.Context, tx pgx.Tx, drawings []models.Drawing) (int, int, error) {
	const query = `
INSERT INTO drawings (
	id, geometry, country, author, author_id, validated, ranking, point_count,
	country_score, country_guess, guess_score, normalized_score, created_at, updated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (id) DO NOTHING`

	inserted := 0
	skipped := 0
	for _, drawing := range drawings {
		rankingJSON, err := json.Marshal(drawing.Ranking)
		if err != nil {
			return 0, 0, fmt.Errorf("encode ranking for %s: %w", drawing.ID, err)
		}

		result, err := tx.Exec(ctx, query,
			drawing.ID,
			drawing.Geometry,
			drawing.Country,
			drawing.Author,
			drawing.AuthorID,
			drawing.Validated,
			rankingJSON,
			drawing.PointCount,
			drawing.CountryScore,
			drawing.CountryGuess,
			drawing.GuessScore,
			drawing.NormalizedScore,
			drawing.CreatedAt,
			drawing.UpdatedAt,
		)
		if err != nil {
			return 0, 0, fmt.Errorf("insert drawing %s: %w", drawing.ID, err)
		}
		if result.RowsAffected() == 1 {
			inserted++
		} else {
			skipped++
		}
	}
	return inserted, skipped, nil
}

func parseRanking(raw json.RawMessage) ([]models.RankingItem, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, errors.New("ranking is missing")
	}

	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("decode ranking: %w", err)
	}
	if len(items) == 0 {
		return nil, errors.New("ranking must not be empty")
	}

	ranking := make([]models.RankingItem, 0, len(items))
	for index, rawItem := range items {
		var tuple []json.RawMessage
		trimmedItem := bytes.TrimSpace(rawItem)
		if bytes.HasPrefix(trimmedItem, []byte("[")) {
			if err := json.Unmarshal(rawItem, &tuple); err != nil || len(tuple) != 2 {
				return nil, fmt.Errorf("ranking item %d must contain country and score", index)
			}
			var country string
			var score float64
			if err := json.Unmarshal(tuple[0], &country); err != nil {
				return nil, fmt.Errorf("ranking item %d has invalid country", index)
			}
			if country == "" {
				return nil, fmt.Errorf("ranking item %d has an empty country", index)
			}
			if err := json.Unmarshal(tuple[1], &score); err != nil {
				return nil, fmt.Errorf("ranking item %d has invalid score", index)
			}
			ranking = append(ranking, models.RankingItem{Country: country, Score: score})
			continue
		}

		var item models.RankingItem
		if err := json.Unmarshal(rawItem, &item); err != nil || item.Country == "" {
			return nil, fmt.Errorf("ranking item %d is invalid", index)
		}
		ranking = append(ranking, item)
	}
	return ranking, nil
}

func parseLegacyTimestamp(value string, location *time.Location) (time.Time, error) {
	if value == "" {
		return time.Time{}, errors.New("timestamp is missing")
	}
	if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return parsed, nil
	}
	for _, layout := range []string{
		"2006-01-02T15:04:05.999999999",
		"2006-01-02T15:04:05",
	} {
		if parsed, err := time.ParseInLocation(layout, value, location); err == nil {
			return parsed, nil
		}
	}
	return time.Time{}, fmt.Errorf("invalid timestamp %q", value)
}

func legacyID(filename string) uuid.UUID {
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte("country-guess:legacy:"+filename))
}
