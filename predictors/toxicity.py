"""
Toxicity Predictor

Predicts peptide toxicity (lower is better).
"""

import numpy as np
from typing import Optional
import pickle


class ToxicityPredictor:
    """Predictor for peptide toxicity."""
    
    def __init__(self, model=None, reference_cdf=None):
        """
        Initialize toxicity predictor.
        
        Args:
            model: Trained toxicity prediction model
            reference_cdf: Empirical CDF for normalization
        """
        self.model = model
        self.reference_cdf = reference_cdf or self._default_cdf()
    
    def predict(self, peptide: str) -> float:
        """
        Predict toxicity for a peptide.
        
        Args:
            peptide: Peptide SMILES string
            
        Returns:
            Raw toxicity score (lower is better)
        """
        if self.model is None:
            # Placeholder: return random value
            return np.random.uniform(0.0, 1.0)
        
        return self.model.predict(peptide)
    
    def normalize(self, value: float) -> float:
        """
        Normalize prediction to [0, 1] using empirical CDF.
        
        Args:
            value: Raw prediction value
            
        Returns:
            Normalized value in [0, 1] (lower is better)
        """
        if self.reference_cdf is None:
            return max(0.0, min(1.0, value))
        
        percentile = np.searchsorted(self.reference_cdf, value) / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        """Default reference distribution (placeholder)."""
        return np.linspace(0.0, 1.0, 1000)
    
    @classmethod
    def load(cls, path: str):
        """Load predictor from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(model=data.get('model'), reference_cdf=data.get('cdf'))

