# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

RUN apk add --no-cache git ca-certificates

COPY go.mod go.sum* ./
RUN go mod download || true

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /bin/api-gateway ./cmd/api

# Production minimal distroless/scratch stage
FROM alpine:3.19

RUN apk --no-cache add ca-certificates

WORKDIR /root/
COPY --from=builder /bin/api-gateway /bin/api-gateway

EXPOSE 8080

ENTRYPOINT ["/bin/api-gateway"]
