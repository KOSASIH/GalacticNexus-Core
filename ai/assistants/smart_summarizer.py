# ai/assistants/smart_summarizer.py

from transformers import pipeline

class SmartSummarizer:
    def __init__(self, abstractive_model="facebook/bart-large-cnn", extractive_model="sshleifer/distilbart-cnn-12-6", device=-1):
        self.abstractive = pipeline("summarization", model=abstractive_model, device=device)
        self.extractive = pipeline("summarization", model=extractive_model, device=device)

    def summarize(self, text: str, mode: str = "auto") -> str:
        if mode == "extractive":
            result = self.extractive(text, max_length=130, min_length=30)
        elif mode == "abstractive":
            result = self.abstractive(text, max_length=130, min_length=30)
        else:
            # Auto mode: use extractive for short, abstractive for long
            if len(text.split()) < 100:
                result = self.extractive(text, max_length=60, min_length=20)
            else:
                result = self.abstractive(text, max_length=130, min_length=30)
        return result[0]['summary_text']

# Example:
# ss = SmartSummarizer()
# print(ss.summarize("Long or short document text ..."))
