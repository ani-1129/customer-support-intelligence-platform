import numpy as np
from typing import List, Dict, Any

class TicketReranker:
    def __init__(self, use_transformer: bool = False, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.use_transformer = use_transformer
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None and self.use_transformer:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                print("sentence-transformers not installed or failed to load. Falling back to TF-IDF reranker.")
                self.use_transformer = False
        return self._model

    def rerank(self, query: str, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of tickets relative to the query string.
        Each ticket is expected to be a dictionary containing at least a "text" key.
        """
        if not tickets:
            return []

        # If using deep learning Cross-Encoder
        if self.use_transformer and self.model is not None:
            pairs = [[query, t.get("text", "")] for t in tickets]
            scores = self.model.predict(pairs)
            # Normalize or map scores if needed, but relative order is what matters
            for idx, score in enumerate(scores):
                tickets[idx]["rerank_score"] = float(score)
        else:
            # Fallback to TF-IDF cosine-similarity cross scoring
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            texts = [t.get("text", "") for t in tickets]
            corpus = [query] + texts
            
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(corpus)
                
                # First row is the query
                query_vector = tfidf_matrix[0:1]
                doc_vectors = tfidf_matrix[1:]
                
                similarities = cosine_similarity(query_vector, doc_vectors).flatten()
                for idx, similarity in enumerate(similarities):
                    tickets[idx]["rerank_score"] = float(similarity)
            except Exception as e:
                # If fitting fails (e.g. empty inputs or vocabulary), copy original retrieval score
                for t in tickets:
                    t["rerank_score"] = t.get("score", 0.0)

        # Sort tickets descending by rerank_score
        sorted_tickets = sorted(tickets, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return sorted_tickets
