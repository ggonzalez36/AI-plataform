import math
from typing import List, Dict, Tuple
from collections import deque
from src.api.models import DriftFeatureStat, DriftReportResponse

FEATURE_NAMES = [
    "debt_to_income",
    "credit_inquiries",
    "tx_velocity",
    "rev_utilization",
    "account_age",
]

# Baseline training distribution reference (mean, std)
FEATURE_BASELINES: Dict[str, Tuple[float, float]] = {
    "debt_to_income": (45.0, 15.0),
    "credit_inquiries": (1.5, 1.2),
    "tx_velocity": (350.0, 120.0),
    "rev_utilization": (0.30, 0.20),
    "account_age": (8.0, 3.5),
}

class DataDriftDetector:
    def __init__(self, max_buffer_size: int = 500, p_value_threshold: float = 0.05):
        self.max_buffer_size = max_buffer_size
        self.p_value_threshold = p_value_threshold
        self.buffers: Dict[str, deque] = {name: deque(maxlen=max_buffer_size) for name in FEATURE_NAMES}
        self.total_observations = 0

    def record_observation(self, features: List[float]):
        """Records an incoming inference feature vector into monitoring buffers"""
        self.total_observations += 1
        for i, val in enumerate(features[:len(FEATURE_NAMES)]):
            self.buffers[FEATURE_NAMES[i]].append(val)

    def _approximate_normal_cdf(self, x: float, mean: float, std: float) -> float:
        """Standard normal error-function approximation for cumulative probability"""
        if std <= 1e-6:
            return 1.0 if x >= mean else 0.0
        z = (x - mean) / (std * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))

    def _calculate_ks_stat(self, sample: List[float], baseline_mean: float, baseline_std: float) -> Tuple[float, float]:
        """Calculates Kolmogorov-Smirnov distance (D) and asymptotic p-value"""
        n = len(sample)
        if n < 5:
            return 0.0, 1.0

        sorted_sample = sorted(sample)
        max_d = 0.0

        for i, x in enumerate(sorted_sample):
            empirical_cdf = (i + 1) / n
            theoretical_cdf = self._approximate_normal_cdf(x, baseline_mean, baseline_std)
            d = abs(empirical_cdf - theoretical_cdf)
            if d > max_d:
                max_d = d

        # Asymptotic p-value approximation: P(D > d) ≈ 2 * exp(-2 * n * d^2)
        p_val = 2.0 * math.exp(-2.0 * n * (max_d ** 2))
        p_val = max(0.0001, min(1.0, p_val))
        return round(max_d, 4), round(p_val, 4)

    def compute_drift_report(self) -> DriftReportResponse:
        """Generates comprehensive data drift metrics across all monitored features"""
        report: Dict[str, DriftFeatureStat] = {}
        overall_drift = False

        for name in FEATURE_NAMES:
            sample = list(self.buffers[name])
            mean_b, std_b = FEATURE_BASELINES[name]

            d_stat, p_val = self._calculate_ks_stat(sample, mean_b, std_b)
            drift_detected = p_val < self.p_value_threshold if len(sample) >= 10 else False

            if drift_detected:
                overall_drift = True

            report[name] = DriftFeatureStat(
                ks_statistic=d_stat,
                p_value=p_val,
                drift_detected=drift_detected,
            )

        status_str = "DRIFT_DETECTED" if overall_drift else "STABLE"

        return DriftReportResponse(
            feature_count=len(FEATURE_NAMES),
            sample_count=min(len(b) for b in self.buffers.values()) if self.buffers else 0,
            overall_drift_detected=overall_drift,
            features=report,
            status=status_str,
        )

    def has_drift(self) -> bool:
        report = self.compute_drift_report()
        return report.overall_drift_detected

# Singleton instance
drift_detector = DataDriftDetector()
