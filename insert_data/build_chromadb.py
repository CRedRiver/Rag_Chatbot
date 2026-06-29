import chromadb
from chromadb.config import Settings
from extractor.pdf_extractor import PdfExtractor
from embeddings.fastEmbed import FastEmbedding
import os
from typing import List

class BuildChromaDB:
    def __init__(self, model_cache_dir: str, api_key: str, tenant: str, database: str):
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
        self.collection = self.client.get_or_create_collection(
            name="collection_project1",
            embedding_function=None  
        )
        self.cache_dir = model_cache_dir
    
    def build_(self, file_paths: List[str], model_name="jinaai/jina-embeddings-v2-base-en",
               chunk_size=1000,chunk_overlap=300):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        extractor = PdfExtractor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        embedding_model = FastEmbedding(
            name=model_name,
            cache_dir=self.cache_dir
        )
        
        _, chil_chunks = extractor.extract_batch(file_paths)
        chil_docs = [str(chunk) for chunk in chil_chunks]  
        chil_embeddings = embedding_model.encode(chil_docs)
        chil_ids = [f"child_{idx}" for idx in range(len(chil_docs))]
        chil_metadata = [{"type": "child", "source_file": os.path.basename(file_paths[0])} for _ in chil_docs]
        
        if chil_docs:
            self.collection.add(
                ids=chil_ids,
                embeddings=chil_embeddings,
                documents=chil_docs,
                metadatas=chil_metadata
            )
            print(f"Successfully upserted {len(chil_docs)} child chunks into ChromaDB.")
            
        return chil_ids