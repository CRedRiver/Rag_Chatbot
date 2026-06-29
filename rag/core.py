import chromadb
from chromadb.config import Settings

from embeddings.sentenceTransformer import SentenceTransformerEmbedding
from embeddings.base import EmbeddingConfig
from rerank.core import Reranker

class RAG:
    def __init__(self,
        chromadb_api_key:str,
        tenant_key:str,
        db_key:str,
        collectionName:str="project1",
        embeddingName:str="BAAI/bge-m3",
        rerankerName:str="BAAI/bge-reranker-v2-m3"
        ):
        self.db_client = chromadb.CloudClient(
            api_key = chromadb_api_key,
            tenant=tenant_key,
            database=db_key
        )
        self.collection=self.db_client.get_or_create_collection(
            name=collectionName,
            embedding_function=None  
        )

        config = EmbeddingConfig(name=embeddingName)
        self.embedding = SentenceTransformerEmbedding(config)
        self.reranker = Reranker(model_name=rerankerName)

    def get_query_emb(self,text):
        if not text.strip():
            return []
        
        embedding = self.embedding.encode(text)
        return embedding.tolist()
    
    def vector_search(
            self,
            user_query,
            query_res=100,
            limit=35
    ):
        if limit > query_res:
            raise ValueError("Cant retrieve a limit greater than number of docs")

        user_query_emb = self.get_query_emb(user_query)
        query_dict = self.collection.query(
            query_embeddings=[user_query_emb],
            n_results=query_res
        )
        
        _, ranked_chunks, ranked_indices = self.reranker(user_query, query_dict["documents"][0])

        results = []
        for i in range(limit):
            orig_idx = ranked_indices[i]
            result = {
                "id": query_dict["ids"][0][orig_idx],
                "chunk": ranked_chunks[i],
                "metadata": query_dict["metadatas"][0][orig_idx]
            }
            results.append(result)

        return results