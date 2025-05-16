import os
import json
from input.Image import Image
from input.ForestType import ForestType
from input.DataSet import DataSet
from models.KMeans import KMeans
from models.GMM import GMM
from evaluation.Inertia import Inertia
from evaluation.Eval import Eval

# Percorso corretto al file di configurazione
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Risale da src/ a root Python/
config_path = os.path.join(base_dir, "config_file.json")

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def get_all_image_files(images_dir):
    return [f for f in os.listdir(images_dir) if f.endswith(".tif")]

def instantiate_model(model_name, dataset):
    models = {
        "KMeans": KMeans,
        "GMM": GMM
    }
    if model_name not in models:
        raise ValueError(f"Modello non supportato: {model_name}")
    return models[model_name](dataset)

def instantiate_metric(metric_name):
    metrics = {
        "Inertia": Inertia
    }
    if metric_name not in metrics:
        raise ValueError(f"Metrica non supportata: {metric_name}")
    return metrics[metric_name]()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    config = load_config(config_path)  # <-- usa il path completo definito sopra

    # Ottieni lista immagini da processare
    if config.get("use_all_data", False):
        image_files = get_all_image_files(images_dir)
    else:
        image_files = config.get("files", [])

    if not image_files:
        print("Nessuna immagine trovata per l'elaborazione.")
        return

    max_clusters = config.get("max_clusters", 10)
    models = config.get("models", [])
    metrics = config.get("metrics", [])

    for image_file in image_files:
        print(f"\n=== Elaborazione immagine: {image_file} ===")

        mask_file = image_file  # Supponiamo che la maschera abbia lo stesso nome

        try:
            img = Image(image_file)
            mask = ForestType(mask_file)
        except Exception as e:
            print(f"Errore caricando dati per {image_file}: {e}")
            continue

        try:
            df = img.to_dataset(mask)
            dataset = DataSet(df)
            dataset.normalize()

            if dataset.is_empty():
                print(f"Dataset vuoto per immagine {image_file}. Skipping.")
                continue

        except Exception as e:
            print(f"Errore preparando dataset per {image_file}: {e}")
            continue

        for model_name in models:
            try:
                model = instantiate_model(model_name, dataset)
            except Exception as e:
                print(f"Errore istanziando modello {model_name}: {e}")
                continue

            for metric_name in metrics:
                try:
                    metric = instantiate_metric(metric_name)
                except Exception as e:
                    print(f"Errore istanziando metrica {metric_name}: {e}")
                    continue

                evaluator = Eval(metric, model)
                print(f"\nModello: {model_name}, Metrica: {metric_name}")

                try:
                    optimal_k = evaluator.elbow_method(max_clusters)
                    score = model.eval(metric)
                    print(f"  -> Numero ottimale cluster: {optimal_k}")
                    print(f"  -> Valore metrica: {score:.4f}")
                except Exception as e:
                    print(f"Errore durante valutazione: {e}")

if __name__ == "__main__":
    main()
