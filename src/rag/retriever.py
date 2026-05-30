from typing import List, Dict, Any

class Retriever:
    def __init__(self, vectorstore: Any, embedder: Any, settings: Any):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.settings = settings
        self.top_k = self.settings.rag.top_k

    def retrieve(self, query: str, japic_code: str) -> List[Dict[str, Any]]:
        """Retrieves top-K document chunks matching the query from ChromaDB collection."""
        collection_name = f"ctd_{japic_code}"
        
        try:
            collection = self.vectorstore.client.get_collection(name=collection_name)
        except Exception:
            print(f"[Retriever] Collection '{collection_name}' not found. Returning empty list.")
            return []

        # Generate query embedding with 'query: ' prefix via embedder
        query_embedding = self.embedder.embed_query(query)

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return formatted_results

    def retrieve_as_context(self, query: str, japic_code: str) -> str:
        """Retrieves and concatenates chunks into a single string to use as LLM prompt context."""
        results = self.retrieve(query, japic_code)
        if not results:
            return "No matching reference data found in CTD."
        
        context_parts = []
        for i, res in enumerate(results):
            context_parts.append(f"--- Reference Chunk {i+1} ---\n{res['text']}")
        return "\n\n".join(context_parts)
