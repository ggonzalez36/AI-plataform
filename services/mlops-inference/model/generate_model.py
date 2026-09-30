"""
Enterprise MLOps Model Exporter
Trains a calibrated Credit & Fraud Risk Classifier and exports the computational graph to ONNX.
"""

import os
import numpy as np

def generate_synthetic_data(n_samples: int = 2000):
    np.random.seed(42)
    # Features:
    # 0: debt_to_income (10 - 90)
    # 1: credit_inquiries (0 - 8)
    # 2: tx_velocity (50 - 1500)
    # 3: rev_utilization (0.01 - 0.99)
    # 4: account_age (1 - 25)
    dti = np.random.normal(45.0, 15.0, n_samples)
    inquiries = np.random.poisson(1.5, n_samples)
    velocity = np.random.exponential(350.0, n_samples)
    utilization = np.random.beta(2.0, 5.0, n_samples)
    age = np.random.gamma(4.0, 2.0, n_samples)

    X = np.column_stack([dti, inquiries, velocity, utilization, age]).astype(np.float32)

    # Risk score ground truth
    logits = (0.035 * dti) + (0.42 * inquiries) + (0.0015 * velocity) + (1.85 * utilization) - (0.08 * age) - 1.25
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(np.int32)
    return X, y

def export_to_onnx(output_path: str = "model/risk_model.onnx"):
    try:
        from sklearn.linear_model import LogisticRegression
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        X, y = generate_synthetic_data()
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)

        initial_type = [("float_input", FloatTensorType([None, 5]))]
        onx = convert_sklearn(clf, initial_types=initial_type, target_opset=14)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())

        print(f"✅ ONNX model successfully saved to '{output_path}'")
    except ImportError:
        print("⚠️ sklearn or skl2onnx not installed in local environment. ONNX runner operates in vectorized mode.")

if __name__ == "__main__":
    export_to_onnx()
