import pandas as pd


class DataSet:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def get_data(self):
        return self.data.values  # AGGIUNTO: Necessario per GMM e KMeans

    def normalize(self):
        self.data = self.data / 10000.0

    def is_empty(self):
        return self.data.empty   # AGGIUNTO

    def stampa_dataset(self, n=5):
        print(f"\nPrime {n} righe del dataset ({self.data.shape[0]} PIXEL)")
        print("-" * 38)
        print(self.data.head(n))
