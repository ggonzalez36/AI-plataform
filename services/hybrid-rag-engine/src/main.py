import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.core.qdrant_store import vector_store
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("hybrid-rag")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info("🚀 Starting Enterprise Hybrid RAG Engine...")
    connected = vector_store.try_connect()
    if connected:
        logger.info("✅ Qdrant cluster connected. Ready for high-throughput hybrid retrieval.")
    else:
        logger.info("⚡ In-memory vector store initialized with compliance seed documents.")
    yield
    # Shutdown sequence
    logger.info("🛑 Shutting down Hybrid RAG Engine cleanly.")

app = FastAPI(
    title="Enterprise Hybrid RAG Engine",
    description="Dense & Sparse document retrieval powered by Qdrant, RRF fusion, and automated Ragas evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for corporate integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=config.port, reload=False)
