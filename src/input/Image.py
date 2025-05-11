import rasterio
import numpy as np
import pandas as pd
import os


class Image:
    def __init__(self, file_name: str):
        # Risali due livelli dalla cartella corrente per arrivare a Python/
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Tre livelli su
        image_folder = os.path.join(base_dir, 'data', 'images')  # Percorso relativo alla cartella "images"

        # Costruisci il percorso assoluto del file
        file_path = os.path.join(image_folder, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        # Usa rasterio per estrarre pixels dal file .tif
        with rasterio.open(file_path) as src:
            img = src.read()
            self.height = src.height
            self.width = src.width
            self.pixels = np.moveaxis(img, 0, -1).astype(np.float64)

        # Controllo sulla validità dei canali in pixels
        if self.pixels.shape[2] != 12:
            print(f"Attenzione: l'immagine caricata ha {self.pixels.shape[2]} canali, non 12.")


    # def stampa_canale(self, canale: int):                 # stampa i valori di tutti i pixel per un canale specificato
    #     if not (1 <= canale <= 12):
    #         raise ValueError("Il numero del canale deve essere compreso tra 1 e 12.")
    #     print(f"\n================= CANALE {canale} =================")
    #     print(self.pixels[:, :, canale - 1])


    def to_dataset(self, forest_type):
        # Controllo sulla compatibilità tra Image e ForestType
        if self.height != forest_type.height or self.width != forest_type.width:
            raise ValueError("Dimensioni di Image e ForestType non coincidono.")

        # Crea una maschera per selezionare solo i pixel appartenenti alla foresta
        mask = forest_type.ft > 0       
        # Estrai i pixel corrispondenti alla maschera
        pixel_list = self.pixels[mask]  

        # Restituisci un DataFrame pandas con i dati dei pixel selezionati
        column_names = [f'Banda {i+1}' for i in range(12)]
        return pd.DataFrame(pixel_list, columns=column_names)
