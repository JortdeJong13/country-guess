package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jortdejong13/country-guess/drawingstore/migrations"
	"github.com/jortdejong13/country-guess/drawingstore/models"
)

// Temporarily local development DB URL
const databaseURL = "postgres://db_user:db_password@drawings-db:5432/db_name?sslmode=disable"

func main() {
	// Structured logger
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{}))

	// Run embedded migrations via the migrations package.
	logger.Info("running embedded database migrations")
	if err := migrations.RunURL(databaseURL); err != nil {
		logger.Error("migrations failed", "error", err)
		// Fail fast; migrations must succeed before serving traffic.
		os.Exit(1)
	}
	logger.Info("migrations applied")

	// Setup pgx connection pool for application use.
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		logger.Error("failed to create pgxpool", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	// Basic router and middleware (chi)
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Recoverer)
	// Simple structured request logger using slog
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			next.ServeHTTP(w, r)
			logger.Info("http_request",
				"method", r.Method,
				"path", r.URL.Path,
				"remote", r.RemoteAddr,
				"request_id", middleware.GetReqID(r.Context()),
				"duration", time.Since(start).String(),
			)
		})
	})

	// Health endpoint
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(models.HealthResponse{Status: "healthy"})
	})

	srv := &http.Server{
		Addr:         ":8080",
		Handler:      r,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server
	logger.Info("starting drawingstore server", "addr", srv.Addr)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()

	// Graceful shutdown on signals
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	sig := <-stop
	logger.Info("shutting down", "signal", sig.String())

	ctxShutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctxShutdown); err != nil {
		logger.Error("graceful shutdown failed", "error", err)
	} else {
		logger.Info("server stopped")
	}
}
