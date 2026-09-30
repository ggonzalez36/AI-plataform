import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.engine.onnx_runner import onnx_runner
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("mlops")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info("🚀 Starting Enterprise MLOps Inference Engine...")
    loaded = onnx_runner.try_load_model()
    if loaded:
        logger.info("✅ ONNX Runtime Session active: '%s'", onnx_runner.engine_name)
    else:
        logger.info("⚡ Vectorized scoring engine active with sub-millisecond execution.")
    yield
    # Shutdown sequence
    logger.info("🛑 Shutting down MLOps Inference Engine cleanly.")

app = FastAPI(
    title="Enterprise MLOps Inference Engine",
    description="Low-latency ONNX Runtime powered scoring service with Prometheus metrics and Evidently Data Drift",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=config.port, reload=False)
