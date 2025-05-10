from input.DataSet import DataSet
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from kneed import KneeLocator
from typing import Optional

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

    def elbow_method(self, max_clusters: int = 10, image_name: str = "", algorithm_name: str = "") -> int:
        # Ottieni i dati dal dataset
        data = self.dataset.get_data()
        n_samples = data.shape[0]

        # Controlla se il dataset è troppo piccolo per il clustering
        if n_samples < 2:
            print(f"[!] Dataset troppo piccolo ({n_samples} campioni) per {image_name}. Skipping.")
            return 1

        # Definisci il range valido di cluster in base alla dimensione del dataset
        max_valid_clusters = min(max_clusters, n_samples)
        n_clusters_range = range(1, max_valid_clusters + 1)
        inertias = []

        # Calcola l'inertia per ogni numero di cluster nel range
        for n in n_clusters_range:
            try:
                inertias.append(self.inertia(n))
            except ValueError as e:
                print(f"[!] Errore calcolando inertia per {image_name} con {algorithm_name}, n={n}: {e}")
                break

        # Usa il KneeLocator per determinare il numero ottimale di cluster
        kl = KneeLocator(n_clusters_range, inertias, curve='convex', direction='decreasing')
        k_ottimale = kl.knee or 2

        # STAMPA VALORI [K OTTIMALE] E [CORRISPETTIVA INERTIA]
        # ##############################################################################
        # if kl.knee is not None:
        #     inertia_ottimale = inertias[k_ottimale - 1]
        #     print(f"\n• Numero ottimale di cluster per {image_name} con {algorithm_name}: {k_ottimale}")
        #     print(f"• Pseudo-inertia corrispondente: {inertia_ottimale:.2f}")
        # else:
        #     print(f"\n• [!] Nessun 'gomito' rilevato per {image_name} con {algorithm_name}. Valore di default utilizzato: {k_ottimale}")
        #     print(f"• Pseudo-inertia corrispondente: {inertias[k_ottimale - 1]:.2f}")
        # ##############################################################################

        # STAMPA GRAFICO ELBOW METHOD
        # ##############################################################################
        # plt.figure(figsize=(8, 5))
        # plt.plot(n_clusters_range, inertias, marker='o', label='Pseudo-Inertia')
        # if kl.knee:
        #     plt.vlines(kl.knee, plt.ylim()[0], plt.ylim()[1],
        #                linestyles='dashed', colors='r', label=f'Elbow a k={kl.knee}')
        #     plt.legend()
        
        # # Modifica del titolo per includere il nome dell'immagine e l'algoritmo
        # plt.title(f'{algorithm_name} - Elbow Method\n{image_name}')
        # plt.xlabel('Numero di Cluster')
        # plt.ylabel('Pseudo-Inertia')
        # plt.grid(True)
        # plt.show()
        # ##############################################################################

        return k_ottimale
