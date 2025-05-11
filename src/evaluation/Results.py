import pandas as pd
from input.DataSet import DataSet
from input.Image import Image
from input.ForestType import ForestType
from models.KMeans import KMeans
from models.GMM import GMM

class Results:
    def __init__(self):
        self.risultati = pd.DataFrame(columns=[
            'NOME IMMAGINE',
            'KMEANS CLUSTERS',
            'KMEANS INERTIA',
            'GMM CLUSTERS',
            'GMM INERTIA'
        ])
        self.non_processate = pd.DataFrame(columns=['NOME IMMAGINE'])

    def processa_immagine(self, nome_immagine: str, max_clusters: int = 10):
        # Inizializza Image e ForestType usando solo il nome del file
        image = Image(nome_immagine)
        forest_type = ForestType(nome_immagine)  # Supponiamo che il nome sia lo stesso

        # Estrai dataset da Image e ForestType
        df = image.to_dataset(forest_type)
        dataset = DataSet(df)

        # Se dataset è vuoto passa all'immagine successiva
        if dataset.is_empty():
            print(f"[!] Dataset vuoto per immagine {nome_immagine}, saltata.")
            # Aggiungi riga a non_processate
            self.non_processate = pd.concat([self.non_processate, pd.DataFrame([[nome_immagine]], columns=['NOME IMMAGINE'])], ignore_index=True)
            return

        # Dividi i valori del dataset per 10_000
        dataset.normalize()

        # KMeans
        kmeans_model = KMeans(dataset)
        k_opt_kmeans = kmeans_model.elbow_method(
            max_clusters=max_clusters,
            image_name=nome_immagine,
            algorithm_name="K MEANS")
        inertia_kmeans = kmeans_model.inertia(k_opt_kmeans)

        # GMM
        gmm_model = GMM(dataset)
        k_opt_gmm = gmm_model.elbow_method(
            max_clusters=max_clusters,
            image_name=nome_immagine,
            algorithm_name="GMM")
        inertia_gmm = gmm_model.inertia(k_opt_gmm)

        # Salta se inertia nulla (Errore nel Clustering)
        if round(inertia_kmeans, 5) == 0.0 or round(inertia_gmm, 5) == 0.0:
            print(f"[!] Risultato non valido per immagine {nome_immagine}: inertia nulla. Immagine scartata.")
            # Aggiungi riga a non_processate
            self.non_processate = pd.concat([self.non_processate, pd.DataFrame([[nome_immagine]], columns=['NOME IMMAGINE'])], ignore_index=True)
            return

        # Aggiungi riga a risultati
        self.risultati.loc[len(self.risultati)] = [
            nome_immagine,
            k_opt_kmeans,
            round(inertia_kmeans,5),
            k_opt_gmm,
            round(inertia_gmm,5)
        ]

    def stampa_risultati(self):
        print("\n======== RISULTATI DEL CLUSTERING ========\n")

        df = self.risultati.copy()
        totale = 200
        df['NOME IMMAGINE'] = [
            f"{nome} ({i+1}/{totale})"
            for i, nome in enumerate(df['NOME IMMAGINE'])
        ]

        print(df.to_string(index=False))

    def stampa_non_processate(self):
        print("\n======== IMMAGINI NON PROCESSATE ========")
        if not self.non_processate.empty:
            print(self.non_processate.to_string(index=False))
        else:
            print("\nTutte le immagini sono state processate correttamente.\n")