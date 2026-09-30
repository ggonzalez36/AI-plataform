# ADR-0002: API Gateway Implemented in Go

## Status
**Accepted**

## Context
The platform requires an edge API Gateway responsible for:
- Reverse proxying client requests to heterogeneous downstream microservices (FastAPI, ONNX, Qdrant).
- Enforcing JWT authentication, API Key verification, and RBAC policies.
- Distributed Token-Bucket Rate Limiting backed by Redis to prevent service degradation and abuse.
- Injecting W3C `traceparent` headers to establish OpenTelemetry trace roots for all requests.

Candidates considered:
1. **Kong / Traefik / Envoy (Pre-built Off-the-Shelf)**: Excellent for generic routing, but limits programmatic custom business logic and demonstrates less custom software engineering depth.
2. **Java (Spring Cloud Gateway)**: Strong enterprise adoption, but heavy baseline memory consumption (~350MB+ RAM per container) and slower startup times.
3. **Go (Custom Gateway via Gin/Chi & Go-Redis)**: Ultra-low latency, native goroutine concurrency, sub-25MB container memory footprint, and instant cold starts (<100ms).

## Decision
We implement a custom, cloud-native API Gateway in **Go (1.22+)**.

## Consequences
### Positive
- **High Concurrency & Low Latency**: Goroutines handle thousands of concurrent I/O operations with microsecond-level overhead.
- **Resource Efficiency**: Can run on small Kubernetes pods (`cpu: 100m, memory: 64Mi`), drastically reducing cloud operating costs.
- **Cloud-Native Synergy**: First-class integration with Kubernetes tools, Docker distroless images, and OpenTelemetry Go SDK.

### Negative / Trade-offs
- Teams must maintain Go code alongside Python AI code (multi-language monorepo).
