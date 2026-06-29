from pydantic.v1 import BaseModel, Field, validator
from embeddings.base import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = SentenceTransformer(self.config.name, trust_remote_code=True)

    def encode(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts)