import numpy as np
from typing import Optional
import pickle


class HalfLifePredictor:
    
    def __init__(self, model=None, reference_cdf=None):
        self.model = model
        self.reference_cdf = reference_cdf or self._default_cdf()
    
    def predict(self, peptide: str) -> float:
        if self.model is None:
            return np.random.uniform(0.0, 1.0)
        return self.model.predict(peptide)
    
    def normalize(self, value: float) -> float:
        if self.reference_cdf is None:
            return max(0.0, min(1.0, value))
        percentile = np.searchsorted(self.reference_cdf, value) / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        return np.linspace(0.0, 1.0, 1000)
    
    @classmethod
    def load(cls, path: str):
        from pathlib import Path
        model_path = Path(path)
        if not model_path.exists():
            print(f"Warning: Half-life predictor file not found: {path}")
            print("Half-life predictor will return random scores.")
            return cls(model=None, reference_cdf=None)
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return cls(model=data.get('model'), reference_cdf=data.get('cdf'))
        except Exception as e:
            print(f"Warning: Could not load half-life predictor from {path}: {e}")
            print("Half-life predictor will return random scores.")
            return cls(model=None, reference_cdf=None)

