FROM golang:1.26-alpine AS builder

WORKDIR /app

COPY drawingstore/go.mod .
COPY drawingstore/go.sum .
RUN go mod download

COPY drawingstore/. .

ENV CGO_ENABLED=0
RUN go build -o drawingstore -ldflags="-s -w" .

FROM gcr.io/distroless/static

WORKDIR /app

COPY --from=builder /app/drawingstore /app/drawingstore

ENTRYPOINT ["/app/drawingstore"]
