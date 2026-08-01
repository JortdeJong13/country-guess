package main

import (
	"context"
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
	"github.com/jortdejong13/country-guess/drawingstore/handlers"
	"github.com/jortdejong13/country-guess/drawingstore/migrations"
)

func main() {
	// Structured logger
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{}))
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		logger.Error("DATABASE_URL is required")
		os.Exit(1)
	}
	serverAddr := os.Getenv("HTTP_ADDR")
	if serverAddr == "" {
		serverAddr = ":8080"
	}

	// Run embedded migrations via the migrations package.
	logger.Info("running embedded database migrations")
	if err := migrations.RunURL(databaseURL); err != nil {
		logger.Error("migrations failed", "error", err)
		// Migrations must succeed before serving traffic.
		os.Exit(1)
	}
	logger.Info("migrations applied")

	// Setup pgx connection pool for application use.
	poolConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		logger.Error("failed to parse database URL", "error", err)
		os.Exit(1)
	}
	poolConfig.MaxConns = 10
	poolConfig.MinConns = 0
	poolConfig.MaxConnLifetime = time.Hour

	ctx := context.Background()
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		logger.Error("failed to create pgxpool", "error", err)
		os.Exit(1)
	}
	defer pool.Close()
	pingCtx, cancelPing := context.WithTimeout(ctx, 5*time.Second)
	if err := pool.Ping(pingCtx); err != nil {
		cancelPing()
		logger.Error("database ping failed", "error", err)
		os.Exit(1)
	}
	cancelPing()

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

	// Initialize the handlers API struct with its dependencies
	api := handlers.NewAPI(pool, logger)

	// Register all routes using the API instance's method
	api.RegisterRoutes(r)

	srv := &http.Server{
		Addr:         serverAddr,
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
