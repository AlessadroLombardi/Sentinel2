from abc import ABC, abstractmethod
from input.DataSet import DataSet
import numpy as np


class Metric(ABC):
    @abstractmethod
    def compute(self, ds: DataSet, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calcola una metrica tra i dati, le etichette e i centroidi.
        """
        pass
