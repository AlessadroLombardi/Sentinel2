import numpy as np
from evaluation.Metric import Metric
from input.DataSet import DataSet

class Inertia(Metric):
    def compute(self, ds: DataSet, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calcola l'inertia (somma delle distanze quadrate intra-cluster).
        """
        data = ds.get_data()  # Ottieni direttamente l'array numpy
        inertia = 0.0
        for i in range(len(data)):
            cluster_idx = labels[i][0]  # labels è di forma (n, 1)
            centroid = centroids[cluster_idx]
            inertia += np.sum((data[i] - centroid) ** 2)
        return inertia
