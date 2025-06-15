# ai/assistants/tabular_insight.py

import pandas as pd
from river import anomaly
from typing import Dict

class TabularInsight:
    def __init__(self):
        self.anomaly_detector = anomaly.HalfSpaceTrees(seed=42)

    def analyze(self, csv_path: str) -> Dict:
        df = pd.read_csv(csv_path)
        anomalies = []
        for idx, row in df.iterrows():
            score = self.anomaly_detector.score_one(row.to_dict())
            if score > 3.0:
                anomalies.append({"index": idx, "row": row.to_dict(), "score": score})
            self.anomaly_detector = self.anomaly_detector.learn_one(row.to_dict())
        return {
            "num_rows": len(df),
            "anomalies_found": len(anomalies),
            "anomalies": anomalies[:10]  # show top 10
        }

# Example:
# ti = TabularInsight()
# print(ti.analyze("data.csv"))
