# ai/assistants/issue_classifier.py

import logging
from typing import List, Union, Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IssueClassifier")

class IssueClassifier:
    SUPPORTED_MODELS = {
        "en": "bhadresh-savani/bert-base-uncased-emotion",
        "multi": "joeddav/xlm-roberta-large-xnli",
        "issue_type": "microsoft/codebert-base-issue-classifier"
    }
    SUPPORTED_LABELS = [
        "bug", "feature", "documentation", "question", "enhancement",
        "performance", "security", "refactor", "emotion"
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        classification_type: str = "issue_type"  # "issue_type", "emotion", or "multi"
    ):
        self.device = device
        self.classification_type = classification_type

        if not model_name:
            if classification_type == "issue_type":
                self.model_name = self.SUPPORTED_MODELS["issue_type"]
            elif classification_type == "emotion":
                self.model_name = self.SUPPORTED_MODELS["en"]
            else:
                self.model_name = self.SUPPORTED_MODELS["multi"]
        else:
            self.model_name = model_name

        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            device=device,
            return_all_scores=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        logger.info(f"Loaded classifier model: {self.model_name}")

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            logger.info(f"Detected language: {lang}")
            return lang
        except LangDetectException:
            logger.warning("Could not detect English.")
            return "en"

    def set_model_for_language(self, lang: str):
        if lang != "en":
            self.model_name = self.SUPPORTED_MODELS["multi"]
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                return_all_scores=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info(f"Model switched for language '{lang}': {self.model_name}")

    def classify(
        self,
        issue_text return_metadata: If True, includes metadata.
        """
        if isinstance(issue_text, str):
            texts = [issue_text]
        else:
            texts = issue_text

        lang = self.detect_language(texts[0])
        if lang != "en":
            self.set_model_for_language(lang)

        if classification_type and classification_type != self.classification_type:
            self.__init__(model_name=None, device=self.device, classification_type=classification_type)

        results = self.classifier(texts)
        if isinstance(issue_text, str):
            results = [results] if isinstance(results, dict) else results

        processed_results = []
        for idx, res_group in enumerate(results):
            # Sort scores descending and pick top
            sorted_labels = sorted(res_group, key=lambda x: x['score'], reverse=True)
            best_label = sorted_labels[0]['label']
            best_score = sorted_labels[0]['score']
            entry = {
                "text": texts[idx],
                "predicted_label": best_label,
                "confidence": best_score,
                "all_scores": sorted_labels
            }
            if explain or return_metadata:
                entry["model"] = self.model_name
                entry["language"] = lang
                entry["classification_type"] = classification_type or self.classification_type
            processed_results.append(entry)

        if isinstance(issue_text, str):
            processed_results = processed_results[0]
        if return_metadata:
            return {
                "results": processed_results,
                "meta": {
                    "model": self.model_name,
                    "classification_type": classification_type or self.classification_type,
                    "language": lang
                }
            }
        return processed_results

    def batch_classify(self, texts: List[str], **kwargs) -> List[Dict]:
        """Batch classify a list of issue texts."""
        return self.classify(texts, **kwargs)

# Example usage:
# ic = IssueClassifier(classification_type="issue_type")
# print(ic.classify("App crashes when uploading files.", explain=True))
# ic = IssueClassifier(classification_type="emotion")
# print(ic.classify("I am frustrated with this error.", explain=True))
# print(ic.batch_classify(["Add dark mode", "Fix login bug", "Improve docs"], return_metadata=True))
