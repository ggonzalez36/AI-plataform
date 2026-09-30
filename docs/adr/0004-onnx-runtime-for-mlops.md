# ADR-0004: ONNX Runtime for Low-Latency MLOps Inference

## Status
**Accepted**

## Context
Deploying trained machine learning models (e.g. Scikit-learn, XGBoost, PyTorch) directly into production API containers introduces heavy bloat:
- Docker images balloon to 2GB-5GB+ when packaging full frameworks like PyTorch or TensorFlow.
- Python GIL (Global Interpreter Lock) contention degrades throughput under concurrent HTTP loads.
- Serving raw Python model objects leads to unpredictability in CPU cache utilization and memory leaks.

Candidates considered:
1. **Raw Scikit-learn / PyTorch in FastAPI**: High memory usage, slow cold starts, heavy container images.
2. **Triton Inference Server**: Powerful for multi-GPU clusters, but excessive operational overhead for lightweight tabular/classification CPU inference.
3. **ONNX Runtime (Open Neural Network Exchange)**: Cross-platform, hardware-accelerated execution engine with minimal runtime footprint (<100MB container layers) and multi-threaded C++ execution core.

## Decision
We standardize the real-time inference microservice on **ONNX Runtime**. Models trained in Scikit-learn, XGBoost, or PyTorch are exported to `.onnx` and tracked through MLflow.

## Consequences
### Positive
- **Dramatic Latency Reduction**: Inference times typically drop to <5ms on standard CPU pods.
- **Slim Containers**: Removes heavy training libraries from the runtime image, reducing image size by ~80%.
- **Portability**: The same model graph can execute on CPU, Intel OpenVINO, or NVIDIA CUDA without altering the scoring code.
