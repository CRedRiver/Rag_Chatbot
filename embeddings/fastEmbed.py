import torch
import pandas as pd
from typing import List, Optional
from fastembed import TextEmbedding
from embeddings.base import BaseEmbedding

class FastEmbedding(BaseEmbedding):
    def __init__(self, name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__(name=name)
        self.use_gpu = torch.cuda.is_available()
        
        try:
            self.model = TextEmbedding(
                model_name=self.name,
                cache_dir=r"D:\fastembed_cache",
                cuda=self.use_gpu
            )
        except Exception as e:
            print(f"Failed to initialize FastEmbed model: {e}")
            raise e

    def encode(self, contents: List[str]) -> List[List[float]]:
        if not contents:
            return []
            
        if isinstance(contents, str):
            contents = [contents]
            
        try:
            embeddings = self.model.embed(contents)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            print(f"Error embedding contents: {e}")
            return []