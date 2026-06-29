import torch
from typing import List, Optional
from fastembed import TextEmbedding
from embeddings.base import BaseEmbedding

class FastEmbedding(BaseEmbedding):
    """
    * input cache_dir for downloaded models in cache
    * input specific_model_path for specific models with .onnx format
    """
    def __init__(self, name: str, cache_dir: Optional[str] = None, specific_model_path: Optional[str] = None):
        super().__init__(name=name)
        self.use_gpu = torch.cuda.is_available()
        
        self.init_args = {
            "model_name": self.name,
            "cuda": self.use_gpu
        }
        if cache_dir is not None:
            self.init_args["cache_dir"] = cache_dir
        if specific_model_path is not None:
            self.init_args["specific_model_path"] = specific_model_path
            
        try:
            self.model = TextEmbedding(**self.init_args)
        except Exception as e:
            print(f"Failed to initialize FastEmbed model {name}: {e}")
            raise e

    def encode(self, contents: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if not contents:
            return []
        if isinstance(contents, str):
            contents = [contents]
        if batch_size is None:
            batch_size = 256 if self.use_gpu else 64

        try:
            generator = self.model.embed(contents, batch_size=batch_size)
            return [embedding.tolist() for embedding in generator]
        except Exception as e:
            print(f"Error embedding contents: {e}")
            if self.use_gpu:
                try:
                    print("GPU embedding failed; retrying on CPU...")
                    self.init_args["cuda"] = False
                    self.model = TextEmbedding(**self.init_args)
                    self.use_gpu = False
                    small_batch = max(8, batch_size // 4)
                    fallback_generator = self.model.embed(contents, batch_size=small_batch)
                    return [embedding.tolist() for embedding in fallback_generator]
                except Exception as e2:
                    print(f"CPU fallback also failed: {e2}")
                    raise
            raise