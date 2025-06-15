# ai/assistants/document_summarizer.py

import os
import math
import logging
from typing import List, Dict, Union, Optional

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocumentSummarizer")

class DocumentSummarizer:
    SUPPORTED_MODELS = {
        "en": [
            "facebook/bart-large-cnn",
            "google/pegasus-xsum",
            "t5-base"
        ],
        "multi": [
            "csebuetnlp/mT5_multilingual_XLSum"
        ]
    }

    def __init__(
        self, 
        model_name: Optional[str] = None, 
        device: int = -1,
        summary_mode: str = "abstractive"  # "abstractive", "extractive", or "hybrid"
    ):
        """
        model_name: Optional custom model name
        device: -1 for CPU, or CUDA device id
        summary_mode: "abstractive", "extractive", or "hybrid"
        """
        self.summary_mode = summary_mode
        self.device = device

        # Try to auto-select model if not given
        if not model_name:
            self.model_name = self.SUPPORTED_MODELS["en"][0]
        else:
            self.model_name = model_name

        self.summarizer = pipeline("summarization", model=self.model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        logger.info(f"Loaded summarization model: {self.model_name}")

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            logger.info(f"Detected language: {lang}")
            return lang
        except LangDetectException:
            logger.warning("Could not detect language, defaulting to English.")
            return "en"

    def set_model_for_language(self, lang: str):
        """Switch to a multilingual model if needed."""
        if lang != "en":
            # Use a multilingual model if available
            self.model_name = self.SUPPORTED_MODELS["multi"][0]
            self.summarizer = pipeline("summarization", model=self.model_name, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info(f"Model switched for language '{lang}': {self.model_name}")

    def chunk_text(self, text: str, max_tokens: int = 850) -> List[str]:
        """Split long text into manageable chunks for the model."""
        sentences = text.replace('\n', ' ').split('. ')
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(self.tokenizer.encode(current_chunk + sentence)) < max_tokens:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        logger.info(f"Text split into {len(chunks."""
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']

    def summarize(
        self, 
        text: str, 
        max_length: int = 200, 
        min_length: int = 50, 
        do_sample: bool = False,
        explain: bool = False,
        return_metadata: bool = False
    ) -> Union[str, Dict]:
        """
        Summarizes the given document.
        - Supports long documents, multilingual, and multiple summary modes.
        - explain: If True, includes reasoning and scoring.
        - return_metadata: If True, includes summary metadata.
        """
        lang = self.detect_language(text)
        if lang != "en":
            self.set_model_for_language(lang)
        else:
            # Reset to default if switched earlier
            if self.model_name not in self.SUPPORTED_MODELS["en"]:
                self.model_name = self.SUPPORTED_MODELS["en"][0]
                self.summarizer = pipeline("summarization", model=self.model_name, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Handle long documents
        max_input_tokens = 850
        if len(self.tokenizer.encode(text)) > max_input_tokens:
            chunks = self.chunk_text(text, max_tokens=max_input_tokens)
            summaries = [
                self.summarize_chunk(chunk, max_length, min_length, do_sample)
                for chunk in chunks
            ]
            combined_summary = " ".join(summaries)
        else:
            combined_summary = self.summarize_chunk(text, max_length, min_length, do_sample)

        # Extraction mode (simple, for demonstration)
        extractive_sentences = ""
        if self.summary_mode in ["extractive", "hybrid"]:
            # Simple extractive: select longest and most relevant sentences
            sentences = text.split('. ')
            ranked = sorted(sentences, key=lambda s: len(s), reverse=True)
            extractive_sentences = '. '.join(ranked[:3])
        
        # Hybrid: combine both
        if self.summary_mode == "hybrid":
            final_summary = f"{combined_summary}\n\nKey Sentences:\n{extractive_sentences}"
        elif self.summary_mode == "extractive":
            final_summary = extractive_sentences
        else:
            final_summary = combined_summary

        if explain or return_metadata:
            metadata = {
                "model": self.model_name,
                "language": lang,
                "length_original": len(text),
                "length_summary": len(final_summary),
                "confidence": min(1.0, len(final_summary) / max(1, len(text))),
                "mode": self.summary_mode
            }
            explanation = f"Summary generated using {self.model_name} in {self.summary_mode} mode. Detected language: {lang}."
            result = {
                "summary": final_summary,
                "metadata": metadata,
                "explanation": explanation if explain else None
            }
            return result
        return final_summary

    def batch_summarize(self, texts: List[str], **kwargs) -> List[str]:
        """Summarize a batch of documents (parallelized if possible)."""
        return [self.summarize(t, **kwargs) for t in texts]

# Example usage:
# doc_sum = DocumentSummarizer(summary_mode="hybrid")
# result = doc_sum.summarize("Very long or multilingual document text ...", explain=True, return_metadata=True)
# print(result)
