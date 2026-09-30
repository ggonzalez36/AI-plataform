from src.engine.onnx_runner import onnx_runner

def test_predict_single_feature_vector():
    features = [45.0, 1.0, 300.0, 0.20, 10.0]
    score = onnx_runner.predict(features)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_predict_batch():
    batch = [
        [30.0, 0.0, 150.0, 0.10, 15.0],
        [85.0, 5.0, 1200.0, 0.90, 1.0],
    ]
    scores = onnx_runner.predict_batch(batch)
    
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    # The high risk borrower should have higher risk score than the low risk borrower
    assert scores[1] > scores[0]

def test_tier_classification():
    assert onnx_runner.classify_tier(0.85) == "HIGH"
    assert onnx_runner.classify_tier(0.45) == "MEDIUM"
    assert onnx_runner.classify_tier(0.15) == "LOW"
