# ai/assistants/sentiment_analyzer.py

import logging
from typing import List, Dict, Union, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentimentAnalyzer")

class SentimentAnalyzer:
    SUPPORTED_MODELS = {
        "en": "distilbert-base-uncased-finetuned-sst-2-english",
        "multi": "nlptown/bert-base-multilingual-uncased-sentiment",
        "emotion": "j-hartmann/emotion-english-distilroberta-base",
        "intent": "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    }

    def __init__(
        self, 
        model_name: Optional[str] = None, 
        device: int = -1,
        analysis_type: str = "sentiment"  # "sentiment", "emotion", or "intent"
    ):
        self.device = device
        self.analysis_type = analysis_type

        if not model_name:
            if analysis_type == "sentiment":
                self.model_name = self.SUPPORTED_MODELS["en"]
            elif analysis_type == "emotion":
                self.model_name = self.SUPPORTED_MODELS["emotion"]
            elif analysis_type == "intent":
                self.model_name = self.SUPPORTED_MODELS["intent"]
            else:
                self.model_name = self.SUPPORTED_MODELS["multi"]
        else:
            self.model_name = model_name

        if analysis_type == "sentiment":
            self.analyzer = pipeline("sentiment-analysis", model=self.model_name, device=device)
        elif analysis_type == "emotion":
            self.analyzer = pipeline("text-classification", model=self.model_name, device=device)
        elif analysis_type == "intent":
            self.analyzer = pipeline("text-classification", model=self.model_name, device=device)
        else:
            self.analyzer = pipeline("sentiment-analysis", model=self.SUPPORTED_MODELS["multi"], device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        logger.info(f"Loaded analyzer model: {self.model_name}")

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            logger.info(f"Detected language: {lang}")
            return lang
        except LangDetectException:
            logger.warning("Could not detect language, defaulting to English.")
            return "en"

    def set_model_for_language(self, lang: str):
        if lang != "en":
            self.model_name = self.SUPPORTED_MODELS["multi"]
            self.analyzer = pipeline("sentiment-analysis", model=self.model_name, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info(f"Model switched for language '{lang}': {self.model_name}")

    def analyze(
        self, 
        text: Union[str, List[str]], 
        explain: bool = False,
        analysis_type: Optional[str] = None,
        return_metadata: bool = False
    ) -> Union[List[Dict], Dict]:
        """
        Analyze sentiment, emotion, or intent of input text(s).
        - explain: If True, includes model details and confidence.
        - analysis_type: Override instance analysis_type.
        - return_metadata: If True, includes metadata.
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        lang = self.detect_language(texts[0])
        if lang != "en":
            self.set_model_for_language(lang)

        if analysis_type:
            # dynamically switch analysis type
            if analysis_type != self.analysis_type:
                self.__init__(model_name=None, device=self.device, analysis_type=analysis_type)

        results = self.analyzer(texts)
        if isinstance(text, str):
            results = [results] if isinstance(results, dict) else results

        # Add confidence and explainability
        processed_results = []
        for idx, res in enumerate(results):
            label = res.get("label", "N/A")
            score = res.get("score", 0.0)
            entry = {
                "text": texts[idx],
                "label": label,
                "confidence": score
            }
            if explain or return_metadata:
                entry["model"] = self.model_name
                entry["language"] = lang
            processed_results.append(entry)

        if isinstance(text, str):
            processed_results = processed_results[0]
        if return_metadata:
            return {
                "results": processed_results,
                "meta": {
                    "model": self.model_name,
                    "analysis_type": analysis_type or self.analysis_type,
                    "language": lang
                }
            }
        return processed_results

    def batch_analyze(self, texts: List[str], **kwargs) -> List[Dict]:
        """Batch sentiment analysis for a list of texts."""
        return self.analyze(texts, **kwargs)

# Example usage:
# sent_an = SentimentAnalyzer()
# print(sent_an.analyze("I love using GalacticNexus!"))
# sent_an = SentimentAnalyzer(analysis_type="emotion")
# print(sent_an.analyze("I am so excited and happy about this project!", explain=True))
# sent_an = SentimentAnalyzer()
# print(sent_an.batch_analyze(["I love this!", "This is bad.", "Meh."]))
