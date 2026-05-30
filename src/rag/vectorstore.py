import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStore:
    def __init__(self, settings: Any):
        self.settings = settings
        self.db_path = self.settings.paths.vectordb
        os.makedirs(self.db_path, exist_ok=True)
        
        # Setup persistent client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.chunk_size = self.settings.rag.chunk_size
        self.chunk_overlap = self.settings.rag.chunk_overlap

        # Splitter fallback
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )

    def _split_markdown(self, markdown_text: str) -> List[str]:
        """Splits markdown semantically based on headings, falling back to character split if needed."""
        # Simple heading-based semantic divider, combined with Recursive split for maximum context safety
        return self.splitter.split_text(markdown_text)

    def get_or_create_collection(self, collection_name: str):
        """Gets or creates a collection. We will manage embeddings manually via the Embedder agent."""
        return self.client.get_or_create_collection(name=collection_name)

    def index_document(self, markdown_text: str, japic_code: str, embedder: Any) -> int:
        """Splits, embeds, and indexes a CTD markdown document into ChromaDB."""
        collection_name = f"ctd_{japic_code}"
        collection = self.get_or_create_collection(collection_name)

        chunks = self._split_markdown(markdown_text)
        if not chunks:
            return 0

        # Generate embeddings
        embeddings = embedder.embed_passages(chunks)
        
        ids = [f"{japic_code}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"japic_code": japic_code, "chunk_index": i} for i in range(len(chunks))]

        # Add to Chroma
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"[VectorStore] Indexed {len(chunks)} chunks into collection '{collection_name}'")
        return len(chunks)
