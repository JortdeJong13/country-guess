FROM golang:1.26-alpine AS builder

WORKDIR /app

COPY drawingstore/go.mod .
COPY drawingstore/go.sum .
RUN go mod download

COPY drawingstore/. .

RUN go build -o drawingstore -ldflags="-s -w" ./main.go

FROM alpine:latest

WORKDIR /app

RUN apk add --no-cache ca-certificates

COPY --from=builder /app/drawingstore .

ENTRYPOINT ["./drawingstore"]
