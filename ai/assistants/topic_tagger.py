# ai/assistants/topic_tagger.py

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicTagger:
    def __init__(self, n_topics=5):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.lda = LatentDirichletAllocation(n_components=n_topics)

    def fit(self, documents):
        X = self.vectorizer.fit_transform(documents)
        self.lda.fit(X)

    def tag(self, document):
        X = self.vectorizer.transform([document])
        topic_dist = self.lda.transform(X)[0]
        top_topic = topic_dist.argmax()
        return {
            "topic": int(top_topic),
            "distribution": topic_dist.tolist()
        }

# Example:
# docs = ["AI is transforming industry.", "Space exploration is advancing.", ...]
# tt = TopicTagger(n_topics=3)
# tt.fit(docs)
# print(tt.tag("New AI breakthrough announced."))
