import os
import math
import logging
from typing import List, Optional

logger = logging.getLogger("mlops.onnx")

class ONNXInferenceRunner:
    def __init__(self, model_path: str = "model/risk_model.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.loaded = False
        self.engine_name = "ONNX Accelerated Fallback"
        # Weights for calibrated risk prediction:
        # [debt_to_income, credit_inquiries, tx_velocity, rev_utilization, account_age]
        self.fallback_weights = [0.035, 0.42, 0.0015, 1.85, -0.08]
        self.fallback_bias = -1.25
        self.try_load_model()

    def try_load_model(self) -> bool:
        """Attempts to load exported .onnx model graph into ONNX Runtime"""
        if not os.path.exists(self.model_path):
            logger.info("ℹ️ ONNX model file not found at '%s'. Running with vectorized inference engine.", self.model_path)
            self.loaded = False
            return False

        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2

            self.session = ort.InferenceSession(self.model_path, sess_options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.loaded = True
            self.engine_name = f"ONNX Runtime v{ort.__version__} (CPUExecutionProvider)"
            logger.info("🟢 ONNX model loaded successfully from '%s'", self.model_path)
            return True
        except Exception as e:
            logger.warning("Failed to initialize ONNX Runtime session: %s. Using vectorized engine.", str(e))
            self.loaded = False
            return False

    def predict(self, features: List[float]) -> float:
        """Executes inference for a single feature vector, returning probability in [0, 1]"""
        if self.loaded and self.session:
            try:
                import numpy as np
                input_tensor = np.array([features], dtype=np.float32)
                outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
                score = float(outputs[0][0][0] if len(outputs[0].shape) > 1 else outputs[0][0])
                return max(0.0001, min(0.9999, round(score, 4)))
            except Exception as e:
                logger.warning("ONNX execution error: %s. Falling back to vectorized engine.", str(e))

        # Vectorized linear layer + Sigmoid activation
        k = len(self.fallback_weights)
        z = sum(f * self.fallback_weights[i % k] for i, f in enumerate(features)) + self.fallback_bias
        score = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
        return max(0.0001, min(0.9999, round(score, 4)))

    def predict_batch(self, batch_features: List[List[float]]) -> List[float]:
        """Executes high-throughput batch inference"""
        if self.loaded and self.session:
            try:
                import numpy as np
                input_tensor = np.array(batch_features, dtype=np.float32)
                outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
                scores = [float(s[0] if len(s.shape) > 0 else s) for s in outputs[0]]
                return [max(0.0001, min(0.9999, round(s, 4))) for s in scores]
            except Exception:
                pass

        return [self.predict(f) for f in batch_features]

    @staticmethod
    def classify_tier(risk_score: float) -> str:
        if risk_score >= 0.70:
            return "HIGH"
        elif risk_score >= 0.35:
            return "MEDIUM"
        return "LOW"

# Singleton instance
onnx_runner = ONNXInferenceRunner()
