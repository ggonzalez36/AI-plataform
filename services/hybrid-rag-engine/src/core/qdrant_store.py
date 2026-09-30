import logging
import uuid
import math
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.config import config
from src.core.embeddings import embedding_engine
from src.core.reranker import reranker

logger = logging.getLogger("hybrid-rag.qdrant")

SEED_DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Corporate Compliance & Data Retention Manual",
        "snippet": "All audit logs, financial records, and operational access trails must be securely retained for 7 years according to SOX guidelines and encrypted at rest with AES-256-GCM.",
        "category": "compliance",
    },
    {
        "id": "doc-002",
        "title": "API Gateway Security & Rate Limiting Standard",
        "snippet": "Edge gateways must enforce token-bucket rate limiting at 50 req/sec per tenant with distributed Redis backing, and reject excessive requests with HTTP 429.",
        "category": "security",
    },
    {
        "id": "doc-003",
        "title": "Distributed Tracing & W3C Context Injection Policy",
        "snippet": "Every HTTP and gRPC ingress hop must propagate W3C traceparent headers to OpenTelemetry collectors, establishing root-to-leaf distributed latency visibility.",
        "category": "observability",
    },
    {
        "id": "doc-004",
        "title": "MLOps Model Governance & Data Drift Thresholds",
        "snippet": "Production machine learning models executed in ONNX Runtime must trigger automated retrain workflows if Kolmogorov-Smirnov data drift p-value drops below 0.05.",
        "category": "mlops",
    },
]

class QdrantStore:
    def __init__(self):
        self.client = None
        self.connected = False
        self.collection_name = config.collection_name
        # Resilient in-memory fallback store
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._init_memory_store()
        self.try_connect()

    def _init_memory_store(self):
        for doc in SEED_DOCUMENTS:
            dense = embedding_engine.generate_dense(doc["snippet"])
            indices, values = embedding_engine.generate_sparse(doc["snippet"])
            self._memory_store[doc["id"]] = {
                "id": doc["id"],
                "title": doc["title"],
                "snippet": doc["snippet"],
                "category": doc["category"],
                "dense": dense,
                "sparse_indices": indices,
                "sparse_values": values,
                "created_at": datetime.utcnow().isoformat(),
            }

    def try_connect(self) -> bool:
        """Attempts connection to Qdrant cluster"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels

            self.client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port,
                timeout=2.0,
            )
            # Health ping
            self.client.get_collections()
            self.connected = True
            logger.info("🟢 Successfully connected to Qdrant at %s:%d", config.qdrant_host, config.qdrant_port)
            self._ensure_collection()
            return True
        except Exception as e:
            self.connected = False
            logger.warning("⚠️ Qdrant unavailable (%s). Running with resilient in-memory hybrid store.", str(e))
            return False

    def _ensure_collection(self):
        """Ensures the hybrid collection exists with dense and sparse vector configurations"""
        if not self.connected or not self.client:
            return

        try:
            from qdrant_client.http import models as qmodels

            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info("Creating Qdrant hybrid collection '%s'...", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": qmodels.VectorParams(
                            size=config.dense_vector_dim,
                            distance=qmodels.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": qmodels.SparseVectorParams(
                            index=qmodels.SparseIndexParams(on_disk=False)
                        )
                    },
                )
                self._seed_qdrant()
        except Exception as e:
            logger.error("Failed to configure Qdrant collection: %v", e)

    def _seed_qdrant(self):
        """Populates initial seed documents into Qdrant"""
        for doc in SEED_DOCUMENTS:
            self.ingest(
                title=doc["title"],
                content=doc["snippet"],
                category=doc["category"],
                doc_id=doc["id"],
            )

    def ingest(self, title: str, content: str, category: str = "compliance", metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> str:
        """Indexes a document with dense and sparse vector representations"""
        point_id = doc_id or str(uuid.uuid4())
        dense_vec = embedding_engine.generate_dense(content)
        sparse_indices, sparse_values = embedding_engine.generate_sparse(content)

        payload = {
            "title": title,
            "snippet": content,
            "category": category,
            "metadata": metadata or {},
            "ingested_at": datetime.utcnow().isoformat(),
        }

        # Store in in-memory fallback
        self._memory_store[point_id] = {
            "id": point_id,
            "title": title,
            "snippet": content,
            "category": category,
            "dense": dense_vec,
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "created_at": payload["ingested_at"],
        }

        # Upsert into Qdrant if active
        if self.connected and self.client:
            try:
                from qdrant_client.http import models as qmodels
                point = qmodels.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": qmodels.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload=payload,
                )
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
            except Exception as e:
                logger.warning("Failed to upsert to Qdrant: %s. Document retained in memory.", str(e))
                self.connected = False

        return point_id

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Executes Hybrid Search (Dense + Sparse) with Reciprocal Rank Fusion"""
        query_dense = embedding_engine.generate_dense(query)
        sparse_indices, sparse_values = embedding_engine.generate_sparse(query)

        # Check Qdrant connectivity
        if self.connected and self.client:
            try:
                from qdrant_client.http import models as qmodels

                filter_condition = None
                if category_filter:
                    filter_condition = qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="category",
                                match=qmodels.MatchValue(value=category_filter),
                            )
                        ]
                    )

                # Dense Search
                dense_hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", query_dense),
                    query_filter=filter_condition,
                    limit=top_k * 2,
                )
                dense_ranked = [
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "title": hit.payload.get("title", ""),
                        "snippet": hit.payload.get("snippet", ""),
                        "category": hit.payload.get("category", ""),
                    }
                    for hit in dense_hits
                ]

                # Sparse Search
                sparse_ranked = []
                if sparse_indices:
                    sparse_hits = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=qmodels.NamedSparseVector(
                            name="sparse",
                            vector=qmodels.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            ),
                        ),
                        query_filter=filter_condition,
                        limit=top_k * 2,
                    )
                    sparse_ranked = [
                        {
                            "id": hit.id,
                            "score": hit.score,
                            "title": hit.payload.get("title", ""),
                            "snippet": hit.payload.get("snippet", ""),
                            "category": hit.payload.get("category", ""),
                        }
                        for hit in sparse_hits
                    ]

                # Fuse and Re-rank
                fused = reranker.reciprocal_rank_fusion(
                    dense_ranked=dense_ranked,
                    sparse_ranked=sparse_ranked,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    top_k=top_k * 2,
                )
                return reranker.cross_encoder_rerank(query, fused, top_k=top_k)

            except Exception as e:
                logger.warning("Qdrant search error (%s). Falling back to in-memory search.", str(e))
                self.connected = False

        # In-Memory Fallback Hybrid Search
        return self._in_memory_hybrid_search(query_dense, sparse_indices, sparse_values, query, top_k, category_filter, dense_weight, sparse_weight)

    def _in_memory_hybrid_search(
        self,
        query_dense: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        raw_query: str,
        top_k: int,
        category_filter: Optional[str],
        dense_weight: float,
        sparse_weight: float,
    ) -> List[Dict[str, Any]]:
        dense_candidates = []
        sparse_candidates = []
        sparse_query_dict = dict(zip(sparse_indices, sparse_values))

        for doc_id, doc in self._memory_store.items():
            if category_filter and doc.get("category") != category_filter:
                continue

            # Dense cosine similarity
            doc_dense = doc["dense"]
            dot_product = sum(a * b for a, b in zip(query_dense, doc_dense))
            dense_candidates.append({
                "id": doc_id,
                "score": round(max(0.0, dot_product), 4),
                "title": doc["title"],
                "snippet": doc["snippet"],
                "category": doc["category"],
            })

            # Sparse dot product
            doc_sparse_indices = doc.get("sparse_indices", [])
            doc_sparse_values = doc.get("sparse_values", [])
            doc_sparse_dict = dict(zip(doc_sparse_indices, doc_sparse_values))

            sparse_score = 0.0
            for idx, q_weight in sparse_query_dict.items():
                if idx in doc_sparse_dict:
                    sparse_score += q_weight * doc_sparse_dict[idx]

            sparse_candidates.append({
                "id": doc_id,
                "score": round(sparse_score, 4),
                "title": doc["title"],
                "snippet": doc["snippet"],
                "category": doc["category"],
            })

        dense_candidates.sort(key=lambda x: x["score"], reverse=True)
        sparse_candidates.sort(key=lambda x: x["score"], reverse=True)

        fused = reranker.reciprocal_rank_fusion(
            dense_ranked=dense_candidates[:top_k * 2],
            sparse_ranked=sparse_candidates[:top_k * 2],
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=top_k * 2,
        )

        return reranker.cross_encoder_rerank(raw_query, fused, top_k=top_k)

    def count(self) -> int:
        return len(self._memory_store)

# Singleton instance
vector_store = QdrantStore()
