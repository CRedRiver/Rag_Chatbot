from fastembed import TextEmbedding
from typing import List, Optional
from embeddings.base import BaseEmbedding
import pandas as pd

class FastEmbedding(BaseEmbedding):
    def __init__(self, name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__(name=name)
        self.model = TextEmbedding(model_name=self.name,
                                   cache_dir = r"D:\fastembed_cache")

    def encode(self, contents: List[str]) -> List[List[float]]:
        if not contents:
            return []
        if isinstance(contents, str):
            contents = [contents]
        try:
            embeddings_generator = self.model.embed(contents)
            return [embedding.tolist() for embedding in embeddings_generator]
        except Exception as e:
            print(f"Error embedding contents: {e}")
