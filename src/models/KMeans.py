import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from models.ClusteringModel import ClusteringModel
from input.DataSet import DataSet

class KMeans(ClusteringModel):
    def __init__(self, dataset: 'DataSet'):
        super().__init__(dataset)

    def inertia(self, n_clusters: int):
        # Ottieni i dati dal dataset
        data = self.dataset.get_data()

        # Inizializza e addestra il modello KMeans
        kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data)

        # Ottieni le etichette e i centroidi trovati dal KMeans
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Calcola la somma delle distanze quadratiche dei punti dal proprio centroide
        return np.sum([
            np.sum((data[labels == i] - centroids[i]) ** 2)
            for i in range(n_clusters)
        ])

    def do_clustering(self, n_clusters: int):
        # Verifica che il dataset non sia vuoto
        if self.dataset.is_empty():
            raise ValueError("Il dataset è vuoto. Clustering non eseguibile.")

        # Esegui il clustering con KMeans e salva le etichette nel risultato
        data = self.dataset.get_data()
        kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data)
        self.clusteringOutput = kmeans.labels_.reshape(-1, 1)
