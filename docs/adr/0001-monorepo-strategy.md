# ADR-0001: Monorepo Strategy for Unified Platform

## Status
**Accepted**

## Context
When architecting an enterprise-grade AI and ML system, engineering teams often face the dilemma of choosing between **multi-repo** (separate repositories for Gateway, RAG, MLOps, IaC, and Monitoring) versus a **modular monorepo**.

While multi-repo creates hard isolation, it introduces severe friction:
- Disjointed CI/CD workflows and version drifting between edge gateways and downstream microservices.
- Cumbersome local testing requiring manual orchestration across 5+ independent repositories.
- Disconnected observability and distributed tracing verification.
- Fragmented documentation and diminished visibility for technical evaluations and recruitment portfolios.

## Decision
We adopt a **modular monorepo** layout hosting:
- Ingress API Gateway (`services/api-gateway`)
- AI RAG Engine (`services/hybrid-rag-engine`)
- MLOps Inference Engine (`services/mlops-inference`)
- Cloud-Native Infrastructure & GitOps (`k8s/`, `terraform/`)
- Unified SRE Observability (`monitoring/`)

## Consequences
### Positive
- **Atomic Changes**: Changes to API contracts (e.g. payload schemas or tracing headers) can be updated across Gateway and downstream services in a single Pull Request.
- **Single-Command Local Environment**: Developers can spin up the full platform with `docker compose up -d` or `make up`.
- **Unified CI/CD & Security Gates**: Shared scanning tools (Trivy, Gitleaks, Dependabot) run in a consolidated GitHub Actions environment.
- **High Showcase Value**: Presents a comprehensive end-to-end software architecture in one place.

### Negative / Trade-offs
- CI pipelines must use path filtering (`paths-ignore` / `paths`) to prevent unnecessary builds of untouched services.
- Requires strict internal service boundaries to avoid tight coupling.
