from sentence_transformers import CrossEncoder
import numpy as np

class Reranker():
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = CrossEncoder(model_name, trust_remote_code=True,
                                    max_length=512)

    def __call__(self, query: str, passages: list[str], batch_size = 32):
        # Combine query and passages into pairs
        if len(passages)==1:
            passages=passages[0]
        query_passage_pairs = [[query, passage] for passage in passages]

        # Get scores from the reranker model
        scores = self.reranker.predict(
            query_passage_pairs, 
            batch_size=batch_size, 
            show_progress_bar=False
        )

        # Sort passages based on scores
        ranked = sorted(zip(scores, passages, range(len(passages))), key=lambda x: x[0], reverse=True)
        ranked_scores, ranked_passages, ranked_indices = map(list, zip(*ranked))
        return ranked_scores, ranked_passages, ranked_indices