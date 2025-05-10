import rasterio
import numpy as np
import pandas as pd
import os


class ForestType:
    def __init__(self, file_name: str):
        # Percorso assoluto alla cartella "data/masks"
        masks_folder = '/Users/imac/Documents/Python/data/masks'

        # Costruisci il percorso assoluto del file
        file_path = os.path.join(masks_folder, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        # Usa rasterio per estrarre ft dal file .tif
        with rasterio.open(file_path) as src:
            self.ft = src.read(1)
            self.height = src.height
            self.width = src.width

        # Controllo sulla validità dei valori in ft
        if not np.all(np.isin(self.ft, [0, 1, 2, 3])):
            raise ValueError("La maschera ForestType deve contenere solo valori 0, 1, 2, 3.")



    # def stampa_matrice(self):                 # stampa ft
    #     print(f"\nMatrice ForestType ({self.height}x{self.width}):")
    #     print(self.ft)



    # def conta_occorrenze(self):               # stampa tabella riassuntiva dei valori in ft
    #     descrizioni = {
    #         0: "Pixel Mascherati",
    #         1: "Bosco di Latifoglie",
    #         2: "Bosco di Conifere",
    #         3: "Bosco Misto"
    #     }

    #     ft_series = pd.Series(self.ft.flatten())
    #     counts = ft_series.value_counts().sort_index()

    #     totale_pixel = self.ft.size
    #     print("\n")
    #     print(f"{'DESCRIZIONE':<25} {'VALORE':<10} {'OCCORRENZE':<15} {'PERCENTUALE':<10}")
    #     print("-" * 65)
    #     for valore in [0, 1, 2, 3]:
    #         count = counts.get(valore, 0)
    #         percentuale = (count / totale_pixel) * 100
    #         descrizione = descrizioni.get(valore, "Tipo sconosciuto")
    #         print(f"{descrizione:<25} {valore:<10} {count:<15} {percentuale:.2f}%")
