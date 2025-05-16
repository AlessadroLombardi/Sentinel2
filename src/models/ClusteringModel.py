from abc import ABC, abstractmethod
from input.DataSet import DataSet
from evaluation.Metric import Metric
import numpy as np
import pandas as pd


class ClusteringModel(ABC):
    def __init__(self, dataset: 'DataSet') -> None:
        self.dataset = dataset
        self.clusteringOutput = None

    @abstractmethod
    def inertia(self, n_clusters: int) -> float:
        pass

    @abstractmethod
    def do_clustering(self, n_clusters: int) -> None:
        pass

    def eval(self, metric: Metric) -> float:
        """
        Valuta il clustering utilizzando la metrica fornita.
        """
        if self.clusteringOutput is None:
            raise ValueError("Clustering non ancora eseguito.")
        return metric.compute(self.dataset, self.clusteringOutput, self.centroids())

    def centroids(self) -> np.ndarray:
        if self.clusteringOutput is None:
            raise ValueError("Devi prima eseguire il fit per avere le etichette del clustering.")
        
        df = pd.DataFrame(self.dataset.get_data())
        df['cluster'] = self.clusteringOutput.flatten()
        
        centroids_df = df.groupby('cluster').mean()
        
        return centroids_df.values