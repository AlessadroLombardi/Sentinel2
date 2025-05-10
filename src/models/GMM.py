from sklearn.mixture import GaussianMixture
import numpy as np
from models.ClusteringModel import ClusteringModel
from input.DataSet import DataSet


class GMM(ClusteringModel):
    def __init__(self, dataset: 'DataSet'):
        self.dataset = dataset
        self.clusteringOutput = None

    def inertia(self, n_clusters: int):
        # Ottieni i dati dal dataset
        data = self.dataset.get_data()

        # Inizializza e addestra il modello GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        gmm.fit(data)

        # Prevedi le etichette dei cluster e ottieni i centroidi (media dei componenti)
        labels = gmm.predict(data)
        centroids = gmm.means_

        # Calcola la somma delle distanze quadratiche dei punti dal proprio centroide
        return np.sum([
            np.sum((data[labels == i] - centroids[i]) ** 2)
            for i in range(n_clusters)
        ])

    def do_clustering(self, n_clusters: int):
        # Verifica che il dataset non sia vuoto
        if self.dataset.is_empty():
            raise ValueError("Il dataset è vuoto. Clustering non eseguibile.")

        # Esegui il clustering con GMM e salva le etichette nel risultato
        data = self.dataset.get_data()
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        gmm.fit(data)
        labels = gmm.predict(data)
        self.clusteringOutput = labels.reshape(-1, 1)
