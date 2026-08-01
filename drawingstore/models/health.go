package models

// HealthResponse is the JSON shape returned by the /health endpoint.
type HealthResponse struct {
	Status string `json:"status"`
}
