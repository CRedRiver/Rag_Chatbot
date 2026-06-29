import os
from typing import List
import chromadb
from chromadb.config import Settings
from extractor.pdf_extractor import PdfExtractor

# Import your custom SentenceTransformer wrapper
from embeddings.sentenceTransformer import SentenceTransformerEmbedding
from embeddings.base import EmbeddingConfig

class BuildChromaDB:
    def __init__(self, collection_name:str, model_cache_dir: str, api_key: str, tenant: str, database: str,
                 model_name="BAAI/bge-m3", chunk_size=1000, chunk_overlap=300):
        self.client = chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=None)
        self.cache_dir = model_cache_dir
        self.extractor = PdfExtractor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = SentenceTransformerEmbedding(EmbeddingConfig(name=model_name))

    def build_(self, file_paths: List[str]):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        _, chil_chunks = self.extractor.extract_batch(file_paths)
        chil_docs = [str(chunk) for chunk in chil_chunks]
        chil_embeddings_raw = self.embedding_model.encode(chil_docs)
        chil_embeddings = [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in chil_embeddings_raw]

        chil_ids = [str(chunk.chunk_id) for chunk in chil_chunks]
        chil_metadata = [{"source": chunk.source, "section": chunk.metadata["section_header"]} for chunk in chil_chunks]

        if chil_docs:
            self.collection.add(ids=chil_ids, embeddings=chil_embeddings, documents=chil_docs, metadatas=chil_metadata)
            print(f"Successfully upserted {len(chil_docs)} child chunks into ChromaDB.")

        return chil_ids
    

if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    db = os.getenv("CHROMA_DATABASE")
    client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=db
        )
    collection = client.get_or_create_collection(
            name="test",
            embedding_function=None  
        )
    collection.add(
        ids = "1",
        embeddings=[3,2],
        documents="hello guys"
    )
