import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class MockEmbeddingModel(BaseEmbeddingModel):
    """
    Deterministic Mock Embedding Model. Produces reproducible unit-normalized vectors
    based on the hash of the text. Dimension is configurable.
    """
    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        # Hash text to seed RNG for deterministic generation
        hasher = hashlib.md5(text.encode('utf-8'))
        seed = int(hasher.hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        
        # Generate random vector and normalize to unit sphere
        vec = rng.standard_normal(self._dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Local SentenceTransformer embedding model. Uses all-MiniLM-L6-v2 by default.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # Lazy loading

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(text, convert_to_numpy=True)
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True)
        normalized = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            normalized.append(v.tolist())
        return normalized

    @property
    def dimension(self) -> int:
        return 384  # all-MiniLM-L6-v2 outputs 384 dimensions


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI text-embedding-3-small API model wrapper.
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return 1536  # text-embedding-3-small standard dimension


def get_embedding_model(is_mock: bool = True, api_key: str = "") -> BaseEmbeddingModel:
    """
    Factory to return configured embedding model.
    """
    if not is_mock and api_key:
        return OpenAIEmbeddingModel(api_key=api_key)
    elif not is_mock:
        # Fallback to local HuggingFace model if no API key is provided but user wants local
        try:
            return SentenceTransformerEmbeddingModel()
        except ImportError:
            # Fallback to mock if sentence_transformers isn't installed
            return MockEmbeddingModel()
    else:
        return MockEmbeddingModel()
