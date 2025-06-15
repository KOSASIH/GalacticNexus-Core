# ai/assistants/multimodal_doc_classifier.py

from transformers import pipeline
from typing import Dict, Any, List

class MultiModalDocClassifier:
    def __init__(self, model_name="microsoft/layoutlmv3-base", device=-1):
        self.classifier = pipeline("document-classification", model=model_name, device=device)

    def classify(self, doc_path: str) -> Dict[str, Any]:
        result = self.classifier(doc_path)
        return {
            "labels": [res['label'] for res in result],
            "scores": [res['score'] for res in result]
        }

# Example:
# mmdc = MultiModalDocClassifier()
# print(mmdc.classify("sample.pdf"))
