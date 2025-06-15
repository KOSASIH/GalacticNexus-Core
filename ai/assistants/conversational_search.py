# ai/assistants/conversational_search.py

from transformers import pipeline

class ConversationalSearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = pipeline("feature-extraction", model=model_name)
        # Add your own vector DB or search backend

    def search(self, question: str, documents: list) -> dict:
        # Create embeddings for the question and each document
        q_vec = self.embedder(question)[0][0]
        doc_vecs = [self.embedder(doc)[0][0] for doc in documents]
        # Compute similarities (dot product, cosine, etc.)
        sims = [sum(q * d for q, d in zip(q_vec, doc_vec)) for doc_vec in doc_vecs]
        best_idx = sims.index(max(sims))
        return {
            "best_match "to install run Z"]
# print(cs.search("How do I install the system?", docs))
