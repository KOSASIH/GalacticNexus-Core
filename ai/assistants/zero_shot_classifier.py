# ai/assistants/zero_shot_classifier.py

from transformers import pipeline
from typing import List, Dict, Any

class ZeroShotClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli", device=-1):
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    
    def classify(self, text: str, candidate_labels: List[str], multi_label: bool = False) -> Dict[str, Any]:
        result = self.classifier(text, candidate_labels, multi_label=multi_label)
        return {
            "sequence": result["sequence"],
            "labels": result["labels"],
            "scores": result["scores"],
            "top_label": result["labels"][0],
            "top_score": result["scores"][0]
        }

# Example:
# zsc = ZeroShotClassifier()
# print(zsc.classify("Deploy this project to Kubernetes.", ["DevOps", "Frontend", "Backend", "Cloud"]))
