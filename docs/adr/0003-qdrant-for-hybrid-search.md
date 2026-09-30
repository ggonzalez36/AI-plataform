# ADR-0003: Qdrant as Primary Vector Store for Hybrid Search

## Status
**Accepted**

## Context
For enterprise document intelligence, plain dense vector retrieval (cosine similarity over dense embeddings) suffers from critical limitations:
- Poor retrieval accuracy on exact alphanumeric matches (e.g. policy numbers, SKU IDs, legal clauses, acronyms).
- Sub-optimal performance when queries require keyword-specific precision alongside semantic matching.

Candidates considered:
1. **pgvector (PostgreSQL extension)**: Convenient if Postgres is already used, but scales poorly under high-dimensional vector workloads and lacks native first-class sparse BM25 fusion out-of-the-box.
2. **Pinecone**: Fully managed, but closed-source and creates vendor lock-in; impossible to run hermetically in local Docker Compose or air-gapped Kubernetes environments.
3. **Qdrant**: Written in Rust, open-source, highly performant HNSW indexing, and native support for dense + sparse vectors in a single query with Reciprocal Rank Fusion (RRF).

## Decision
We select **Qdrant** as the primary vector search database.

## Consequences
### Positive
- **Native Hybrid Search**: Single query can combine dense embeddings with sparse lexical tokens (BM25-like) natively on the server side.
- **Payload-Based Filtering**: Rich JSON payloads can be indexed and filtered in sub-milliseconds without table joins.
- **Local & Production Parity**: Runs identical containerized engines in local `docker-compose.yml` and distributed Kubernetes StatefulSets.
