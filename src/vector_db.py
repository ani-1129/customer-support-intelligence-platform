import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class QdrantVectorDB:
    def __init__(self, host: str = ":memory:", port: int = 6333):
        """
        Initializes Qdrant client. If host is ':memory:', Qdrant runs in-memory.
        """
        self.host = host
        self.port = port
        if host == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(host=host, port=port)

    def init_collection(self, collection_name: str, vector_size: int):
        """
        Creates collection if it doesn't exist.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def upsert_document(self, collection_name: str, doc_id: str, vector: List[float], payload: Dict[str, Any]):
        """
        Inserts or updates a document vector + metadata.
        """
        # Ensure we have a valid UUID or int for Qdrant IDs
        try:
            val_id = str(uuid.UUID(doc_id))
        except ValueError:
            # Deterministic uuid generation from string id
            val_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=val_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def search_similar(
        self, 
        collection_name: str, 
        vector: List[float], 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar vectors. Optionally applies metadata filtering.
        """
        qdrant_filter = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v)
                    )
                )
            qdrant_filter = models.Filter(must=conditions)

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter
        )

        results = []
        for hit in search_result.points:
            results.append({
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload
            })
        return results
