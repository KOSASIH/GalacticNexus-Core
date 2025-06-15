# ai/assistants/document_qa.py

from transformers import pipeline
from typing import Dict, Any

class DocumentQnA:
    def __init__(self, model_name="deepset/roberta-base-squad2", device=-1):
        self.qa = pipeline("question-answering", model=model_name, device=device)
    
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        result = self.qa(question=question, context=context)
        return {
            "answer": result["answer"],
            "score": result["score"],
            "start": result["start"],
            "end": result["end"]
        }

# Example:
# doc_qa = DocumentQnA()
# print(doc_qa.answer("Who developed GalacticNexus?", "GalacticNexus was created by KOSASIH in 2024."))
