import os
from evaluation.Results import Results

# Ottieni la cartella principale del progetto (Python)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Risali di due livelli

# Costruisci il percorso completo verso la cartella "data/images"
image_folder = os.path.join(base_dir, 'data', 'images')

def main():
    # Ottieni tutti i nomi dei file nella cartella images
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])

    # Crea l'oggetto Results
    report = Results()

    # Numero Totale di immagini
    totale_immagini = 200

    # Cicla su ogni immagine e processa il clustering
    for idx, image_name in enumerate(image_files, start=1):
        print(f"\nProcessando immagine ({idx}/{totale_immagini}): {image_name}")
        report.processa_immagine(image_name)

    # Stampa i risultati
    report.stampa_risultati()

    # Esporta i risultati in un file CSV
    # #####################################################################
    report.risultati.to_csv("clustering_results.csv", index=False)
    print("\nI risultati sono stati salvati in 'clustering_results.csv'.")

    # Stampa immagini non processate
    report.stampa_non_processate()

    # Esporta le immagini non processate in un file CSV
    # #####################################################################
    if not report.non_processate.empty:
        report.non_processate.to_csv("non_processate.csv", index=False)
        print("\nLe immagini non processate sono state salvate in 'non_processate.csv'.")
    else:
        print("\nTutte le immagini sono state processate correttamente.")

if __name__ == "__main__":
    main()
