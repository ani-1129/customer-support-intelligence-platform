from typing import List, Dict, Any, Optional
from src.embeddings import BaseEmbeddingModel
from src.vector_db import QdrantVectorDB

class TicketRetriever:
    def __init__(self, embedding_model: BaseEmbeddingModel, vector_db: QdrantVectorDB, collection_name: str = "tickets"):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.collection_name = collection_name
        
        # Ensure collection is initialized
        self.vector_db.init_collection(
            collection_name=self.collection_name,
            vector_size=self.embedding_model.dimension
        )

    def index_ticket(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Embeds and indexes a ticket in Qdrant.
        """
        vector = self.embedding_model.embed_query(text)
        payload = {
            "text": text,
            **(metadata or {})
        }
        self.vector_db.upsert_document(
            collection_name=self.collection_name,
            doc_id=doc_id,
            vector=vector,
            payload=payload
        )

    def retrieve_similar(self, query_text: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Given a query text, retrieves similar tickets.
        """
        vector = self.embedding_model.embed_query(query_text)
        hits = self.vector_db.search_similar(
            collection_name=self.collection_name,
            vector=vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        retrieved_tickets = []
        for hit in hits:
            retrieved_tickets.append({
                "id": hit["id"],
                "score": hit["score"],
                "text": hit["payload"].get("text", ""),
                "metadata": {k: v for k, v in hit["payload"].items() if k != "text"}
            })
        return retrieved_tickets
