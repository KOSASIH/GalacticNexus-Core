# ai/assistants/recommender.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdaptiveRecommender")

class AdaptiveRecommender:
    def __init__(
        self,
        n_neighbors: int = 5,
        algorithm: str = "auto",  # 'auto', 'ball_tree', 'kd_tree', 'brute'
        metric: str = "euclidean",  # 'euclidean', 'manhattan', 'cosine'
        use_cosine: bool = False,
        return_scores: bool = False
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.use_cosine = use_cosine
        self.return_scores = return_scores
        self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        self.scaler = StandardScaler()
        self.user_vectors = None
        self.user_metadata: List[Dict[str, Any]] = []
        self.fitted = False

    def fit(self, user_vectors: Union[np.ndarray, List[List[float]]], user_metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Fit the recommender with user vectors and optional metadata.
        """
        user_vectors = np.array(user_vectors)
        self.user_vectors = self.scaler.fit_transform(user_vectors)
        self.model.fit(self.user_vectors)
        self.fitted = True
        if user_metadata:
            self.user_metadata = user_metadata
        logger.info("AdaptiveRecommender fitted with %d users.", len(user_vectors))

    def recommend(self, user_vector: List[float], top_k: Optional[int] = None, explain: bool = False) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Recommend the most similar user(s) or item(s) for a given user vector.
        Returns indices or enriched results if metadata is available.
        """
        if not self.fitted:
            raise Exception("Model not fitted yet. Call fit() first.")
        user_vec = self.scaler.transform([user_vector])
        n_neighbors = top_k if top_k else self.n_neighbors

        if self.use_cosine:
            sims = cosine_similarity(user_vec, self.user_vectors)[0]
            indices = np.argsort(sims)[::-1][:n_neighbors]
            scores = sims[indices]
        else:
            distances, indices = self.model.kneighbors(user_vec, n_neighbors=n_neighbors)
            scores = 1 / (1 + distances[0])  # higher score = closer

        recs = []
        for i, idx in enumerate(indices if not self.use_cosine else indices):
            rec = {"index": int(idx)}
            if self.user_metadata:
                rec["metadata"] = self.user_metadata[idx]
            rec["score"] = float(scores[i])
            if explain:
                rec["explanation"] = f"Similarity score: {scores[i]:.4f} using {'cosine' if self.use_cosine else self.metric} metric."
            recs.append(rec)
        if self.return_scores or explain or self.user_metadata:
            return recs
        else:
            return [int(idx) for idx in (indices if not self.use_cosine else indices)]

    def recommend_batch(self, user_vectors: List[List[float]], top_k: Optional[int] = None, explain: bool = False) -> List[Any]:
        """
        Recommend for a batch of user vectors.
        """
        return [self.recommend(vec, top_k=top_k, explain=explain) for vec in user_vectors]

    def cold_start(self, new_vector: List[float], strategy: str = "mean") -> List[int]:
        """
        Handle new users/items with no history ("cold start").
        """
        if not self.fitted:
            raise Exception("Model not fitted yet. Call fit() first.")
        if strategy == "mean":
            avg_vec = np.mean(self.user_vectors, axis=0)
            return self.recommend(avg_vec)
        elif strategy == "random":
            idxs = np.random.choice(len(self.user_vectors), self.n_neighbors, replace=False)
            return [int(idx) for idx in idxs]
        else:
            raise ValueError("Unknown cold start strategy.")

# Example usage:
# rec = AdaptiveRecommender(n_neighbors=3, use_cosine=True, return_scores=True)
# user_vectors = np.random.rand(10, 5)
# user_metadata = [{"user_id": i, "name": f"User{i}"} for i in range(10)]
# rec.fit(user_vectors, user_metadata)
# print(rec.recommend(np.random.rand(5), explain=True))
# print(rec.recommend_batch([np.random.rand(5) for _ in range(2)], explain=True))
# print(rec.cold_start(np.random.rand(5)))
