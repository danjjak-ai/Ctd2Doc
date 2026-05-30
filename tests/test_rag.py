import os
import pytest
import shutil
from src.config_helper import load_settings
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever

@pytest.fixture(scope="module")
def shared_resources():
    settings = load_settings()
    # Force cpu for test simplicity and clean temp paths
    settings.model.device = "cpu"
    settings.paths.vectordb = "data/test_vectordb"
    
    embedder = Embedder(settings)
    vectorstore = VectorStore(settings)
    retriever = Retriever(vectorstore, embedder, settings)
    
    yield embedder, vectorstore, retriever
    
    # Close chroma client to release file lock before deletion
    try:
        del vectorstore
        del retriever
    except Exception:
        pass
        
    # Cleanup temp test db
    if os.path.exists("data/test_vectordb"):
        try:
            shutil.rmtree("data/test_vectordb")
        except PermissionError:
            print("[Warning] PermissionError on cleaning up test db. Files locked by SQLite.")


def test_embedder_shapes(shared_resources):
    embedder, _, _ = shared_resources
    
    # Query embedding
    q_emb = embedder.embed_query("test query")
    assert isinstance(q_emb, list)
    assert len(q_emb) == 1024 # multilingual-e5-large is 1024-dim
    
    # Passage embeddings
    p_embs = embedder.embed_passages(["passage 1", "passage 2"])
    assert len(p_embs) == 2
    assert len(p_embs[0]) == 1024

def test_indexing_and_retrieval(shared_resources):
    embedder, vectorstore, retriever = shared_resources
    
    sample_ctd = """# CTD SECTION 2.1
This is a clinical trials paragraph detailing that Headache was reported in 12.0% of patients.
The primary dosage administered was 5.0mg once daily.
For pediatric cases, 0.5mg was evaluated.
"""
    japic_code = "TEST_99"
    
    # Indexing
    num_chunks = vectorstore.index_document(sample_ctd, japic_code, embedder)
    assert num_chunks > 0
    
    # Retrieve
    context = retriever.retrieve_as_context("What are the dosages?", japic_code)
    assert "Headache" in context
    assert "5.0mg" in context
    assert "0.5mg" in context
