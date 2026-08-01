// Command backfill-legacy-author-ids restores author IDs from legacy GeoJSON
// files after the one-time migration to PostgreSQL.
//
// It deliberately reads only author_id. Legacy hashed IP values are never
// copied. Existing database values are never overwritten.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const legacyIDPrefix = "country-guess:legacy:"

type legacyFeatureCollection struct {
	Features []legacyFeature `json:"features"`
}

type legacyFeature struct {
	Properties legacyProperties `json:"properties"`
}

type legacyProperties struct {
	AuthorID string `json:"author_id"`
}

type backfillEntry struct {
	Filename string
	ID       uuid.UUID
	AuthorID string
}

type backfillResult struct {
	Candidates      int
	Updated         int
	WouldUpdate     int
	AlreadyMatching int
	Missing         int
	Conflicts       int
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	directory := flag.String("dir", "../data/drawings", "directory containing legacy GeoJSON drawings")
	dryRun := flag.Bool("dry-run", false, "report changes without updating PostgreSQL")
	flag.Parse()

	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		logger.Error("DATABASE_URL is required")
		os.Exit(1)
	}

	entries, err := scanDirectory(*directory)
	if err != nil {
		logger.Error("scan failed", "error", err)
		os.Exit(1)
	}
	logger.Info("legacy author IDs scanned", "directory", *directory, "candidates", len(entries))

	pool, err := pgxpool.New(context.Background(), databaseURL)
	if err != nil {
		logger.Error("connect to database failed", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := pool.Ping(ctx); err != nil {
		logger.Error("ping database failed", "error", err)
		os.Exit(1)
	}

	result, err := backfill(ctx, pool, entries, *dryRun)
	if err != nil {
		logger.Error("backfill failed", "error", err)
		os.Exit(1)
	}

	logger.Info("legacy author ID backfill complete",
		"dry_run", *dryRun,
		"candidates", result.Candidates,
		"updated", result.Updated,
		"would_update", result.WouldUpdate,
		"already_matching", result.AlreadyMatching,
		"missing", result.Missing,
		"conflicts", result.Conflicts,
	)
}

func scanDirectory(directory string) ([]backfillEntry, error) {
	paths, err := filepath.Glob(filepath.Join(directory, "*.geojson"))
	if err != nil {
		return nil, fmt.Errorf("find GeoJSON drawings: %w", err)
	}
	sort.Strings(paths)

	entries := make([]backfillEntry, 0)
	for _, path := range paths {
		contents, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}

		var collection legacyFeatureCollection
		if err := json.Unmarshal(contents, &collection); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		if len(collection.Features) != 1 {
			return nil, fmt.Errorf("%s: expected one feature, found %d", path, len(collection.Features))
		}

		authorID := strings.TrimSpace(collection.Features[0].Properties.AuthorID)
		if authorID == "" {
			continue
		}

		filename := filepath.Base(path)
		entries = append(entries, backfillEntry{
			Filename: filename,
			ID:       uuid.NewSHA1(uuid.NameSpaceURL, []byte(legacyIDPrefix+filename)),
			AuthorID: authorID,
		})
	}

	return entries, nil
}

func backfill(ctx context.Context, pool *pgxpool.Pool, entries []backfillEntry, dryRun bool) (backfillResult, error) {
	result := backfillResult{Candidates: len(entries)}
	tx, err := pool.Begin(ctx)
	if err != nil {
		return result, fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	for _, entry := range entries {
		var isNull bool
		var existing string
		err := tx.QueryRow(ctx,
			`SELECT author_id IS NULL, COALESCE(author_id, '') FROM drawings WHERE id = $1`,
			entry.ID,
		).Scan(&isNull, &existing)
		if errors.Is(err, pgx.ErrNoRows) {
			result.Missing++
			continue
		}
		if err != nil {
			return result, fmt.Errorf("inspect %s (%s): %w", entry.Filename, entry.ID, err)
		}

		if isNull {
			if dryRun {
				result.WouldUpdate++
				continue
			}

			commandTag, err := tx.Exec(ctx,
				`UPDATE drawings SET author_id = $1 WHERE id = $2 AND author_id IS NULL`,
				entry.AuthorID, entry.ID,
			)
			if err != nil {
				return result, fmt.Errorf("update %s (%s): %w", entry.Filename, entry.ID, err)
			}
			if commandTag.RowsAffected() == 1 {
				result.Updated++
			}
			continue
		}

		if existing == entry.AuthorID {
			result.AlreadyMatching++
		} else {
			result.Conflicts++
		}
	}

	if result.Conflicts > 0 {
		return result, fmt.Errorf("found %d existing author_id conflicts; no changes committed", result.Conflicts)
	}
	if dryRun {
		return result, nil
	}
	if err := tx.Commit(ctx); err != nil {
		return result, fmt.Errorf("commit transaction: %w", err)
	}
	return result, nil
}
