import numpy as np
from semantic_router.encoders import FastEmbedEncoder

class SemanticRouter():
    def __init__(self, routes, 
                 embedding=FastEmbedEncoder(name="sentence-transformers/all-MiniLM-L6-v2")):
        self.routes = routes
        self.embedding = embedding
        self.routesEmbedding = {}

        for route in self.routes:
            self.routesEmbedding[
                route.name
            ] = self.embedding(route.samples)

    def get_routes(self):
        return self.routes

    def guide(self, query):
        queryEmbedding = self.embedding([query]) 
        queryEmbedding = queryEmbedding / np.linalg.norm(queryEmbedding)
        scores = []

        for route in self.routes:
            routesEmbedding = self.routesEmbedding[route.name] / np.linalg.norm(self.routesEmbedding[route.name])
            score = np.mean(np.dot(routesEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0][1]