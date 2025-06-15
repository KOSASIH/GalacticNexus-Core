# ai/security/streaming_anomaly_detector.py

from river import anomaly, preprocessing, drift
import numpy as np
from typing import Dict, Any, Callable, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamingAnomalyDetector")

class StreamingAnomalyDetector:
    SUPPORTED_MODELS = {
        "half_space_trees": anomaly.HalfSpaceTrees,
        "iforest": anomaly.IForest,
        "oneclass_svm": anomaly.OneClassSVM,
        "knn": anomaly.KNNAnomalyDetector
    }

    def __init__(
        self,
        algorithm: str = "half_space_trees",
        seed: int = 42,
        adaptive_threshold: bool = True,
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        algorithm: which streaming anomaly detector to use
        adaptive_threshold: if True, adapt threshold based on recent scores
        alert_callback: function to call on anomaly detection
        """
        if algorithm not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown algorithm '{algorithm}'.")
        self.model = self.SUPPORTED_MODELS[algorithm](seed=seed)
        self.scaler = preprocessing.StandardScaler()
        self.drift_detector = drift.ADWIN()
        self.scores: List[float] = []
        self.threshold = 3.0  # Default threshold
        self.adaptive_threshold = adaptive_threshold
        self.alert_callback = alert_callback
        logger.info(f"Initialized StreamingAnomalyDetector with {algorithm}")

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update the model with a new data point."""
        x_scaled = self.scaler.learn_one(x).transform_one(x)
        self.model = self.model.learn_one(x_scaled)

    def score_one(self, x: Dict[str, float]) -> float:
        """Return anomaly score for a single data point."""
        x_scaled = self.scaler.transform_one(x)
        score = self.model.score_one(x_scaled)
        self.scores.append(score)
        self.drift_detector.update(score)
        if self.adaptive_threshold and len(self.scores) > 30:
            recent_scores = self.scores[-30:]
            self.threshold = np.mean(recent_scores) + 3 * np.std(recent_scores)
        return score

    def is_anomaly(self, x: Dict[str, float], threshold: Optional[float] = None) -> bool:
        """Return True if the data point is an anomaly. Calls alert_callback if anomaly."""
        actual_threshold = threshold if threshold is not None else self.threshold
        score = self.score_one(x)
        is_anom = score > actual_threshold
        if is_anom and self.alert_callback:
            alert = {
                "input": x,
                "score": score,
                "threshold": actual_threshold,
                "model": type(self.model).__name__,
                "explanation": "Score exceeded threshold"
            }
            self.alert_callback(alert)
        return is_anom

    def explain(self, x: Dict[str, float]) -> Dict[str, Any]:
        """Return a detailed explanation for the anomaly score."""
        score = self.score_one(x)
        return {
            "input": x,
            "score": score,
            "threshold": self.threshold,
            "model": type(self.model).__name__,
            "drift": self.drift_detector.change_detected,
            "explanation": (
                "Anomaly detected" if score > self.threshold else "Normal"
            )
        }

    def get_score_history(self) -> List[float]:
        """Returns the list of anomaly scores for monitoring."""
        return self.scores

# Example usage:
# def my_alert(alert): print("ALERT:", alert)
# sad = StreamingAnomalyDetector(algorithm="half_space_trees", alert_callback=my_alert)
# stream = [{"a": np.random.rand(), "b": np.random.rand()} for _ in range(100)]
# for data in stream:
#     sad.learn_one(data)
#     if sad.is_anomaly(data): print("Anomaly detected!", sad.explain(data))
