from evaluation.Metric import Metric
from models.ClusteringModel import ClusteringModel
from kneed import KneeLocator
from typing import List

class Eval:
    def __init__(self, metric: Metric, model: ClusteringModel):
        self.metric = metric
        self.model = model

    def elbow_method(self, max_clusters: int) -> int:
        scores: List[float] = []

        for k in range(1, max_clusters + 1):
            self.model.do_clustering(k)
            score = self.model.eval(self.metric)
            scores.append(score)
            # print(f"Clusters: {k}, Score: {score}")

        # KneeLocator cerca il "ginocchio" (elbow) nel grafico clusters vs. score
        kl = KneeLocator(
            x=range(1, max_clusters + 1),
            y=scores,
            curve="convex",  # in genere la curva è convessa per inertia
            direction="decreasing"
        )

        optimal_k = kl.elbow
        if optimal_k is None:
            # fallback se KneeLocator non trova un punto di ginocchio
            optimal_k = 1
        return optimal_k
