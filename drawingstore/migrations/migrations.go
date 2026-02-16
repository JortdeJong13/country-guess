package migrations

import (
	"context"
	"database/sql"
	"embed"
	"fmt"
	"time"

	// register pgx driver for database/sql when opening by DSN
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
)

//go:embed *.sql
var embeddedMigrations embed.FS

// Run applies embedded migrations using the provided *sql.DB.
func Run(db *sql.DB) error {
	// Configure goose to read migrations from the embedded filesystem.
	goose.SetBaseFS(embeddedMigrations)

	// Ensure goose is using the postgres dialect.
	if err := goose.SetDialect("postgres"); err != nil {
		return fmt.Errorf("set goose dialect: %w", err)
	}

	// Run all up migrations found at the root of the embedded FS (".").
	if err := goose.Up(db, "."); err != nil {
		return fmt.Errorf("goose up: %w", err)
	}

	return nil
}

// RunURL opens a database/sql connection using the provided DSN (driver name is pgx)
// and runs embedded migrations. It is a convenience wrapper around Run.
func RunURL(dsn string) error {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return fmt.Errorf("open db: %w", err)
	}
	defer db.Close()

	// Ping with timeout to fail fast if the DB is unreachable.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("ping db: %w", err)
	}

	return Run(db)
}
