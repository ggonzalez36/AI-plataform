import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    port: int = int(os.getenv("PORT", "8001"))
    qdrant_host: str = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("COLLECTION_NAME", "enterprise_knowledge")
    dense_vector_dim: int = 384
    rrf_k: int = 60
    top_k_default: int = 3
    environment: str = os.getenv("ENVIRONMENT", "development")

config = RAGConfig()
