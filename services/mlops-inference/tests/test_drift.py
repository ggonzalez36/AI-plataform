from src.monitoring.drift import DataDriftDetector

def test_drift_detector_initial_state():
    detector = DataDriftDetector()
    report = detector.compute_drift_report()
    assert report.sample_count == 0
    assert not report.overall_drift_detected

def test_drift_detected_on_outliers():
    detector = DataDriftDetector(max_buffer_size=100)
    # Inject 30 extreme anomaly data points that deviate strongly from baseline
    for _ in range(30):
        detector.record_observation([120.0, 10.0, 5000.0, 0.99, 0.1])
    
    report = detector.compute_drift_report()
    assert report.sample_count == 30
    assert report.overall_drift_detected
    assert report.status == "DRIFT_DETECTED"
