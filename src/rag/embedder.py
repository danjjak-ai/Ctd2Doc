import os
from typing import List, Any
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, settings: Any):
        self.settings = settings
        self.model_name = self.settings.model.embedding_model
        # CPU/GPU Device mapping
        self.device = "cuda" if self.settings.model.device == "cuda" else "cpu"
        
        print(f"[Embedder] Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single search query (prefix with 'query: ')"""
        formatted_query = f"query: {query}"
        embedding = self.model.encode([formatted_query], normalize_embeddings=True)[0]
        return embedding.tolist()

    def embed_passages(self, passages: List[str]) -> List[List[float]]:
        """Generates embeddings for list of documents/passages (prefix with 'passage: ')"""
        formatted_passages = [f"passage: {p}" for p in passages]
        embeddings = self.model.encode(formatted_passages, batch_size=32, normalize_embeddings=True)
        return embeddings.tolist()
