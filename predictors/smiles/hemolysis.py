"""
Hemolysis Predictor

Predicts hemolytic activity of peptides (red blood cell lysis).
Lower values indicate lower hemolytic activity (better for therapeutic use).
"""

import numpy as np
import torch
from typing import Optional
import pickle
from pathlib import Path


class HemolysisPredictor:
    """
    Predictor for hemolytic activity.
    
    Returns values in [0, 1] where:
    - 0 = no hemolysis (best)
    - 1 = high hemolysis (worst)
    """
    
    def __init__(self, model=None, reference_cdf=None, device=None):
        self.model = model
        self.reference_cdf = reference_cdf or self._default_cdf()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, peptide: str) -> float:
        """
        Predict hemolytic activity for a peptide.
        
        NOTE: This is a placeholder implementation.
        For real predictions, load a trained model.
        """
        if self.model is None:
            # Placeholder: return random value in [0, 1]
            # Use hash-based deterministic value for reproducibility
            import hashlib
            peptide_hash = int(hashlib.md5(peptide.encode()).hexdigest()[:8], 16)
            value = (peptide_hash % 10000) / 10000.0  # Maps to [0, 1]
            return float(value)
        # In practice, this would call the actual model
        return np.random.uniform(0.0, 1.0)
    
    def normalize(self, value: float) -> float:
        """
        Normalize hemolytic value to [0, 1] using empirical CDF.
        Lower is better for hemolysis.
        """
        if self.reference_cdf is None:
            return float(np.clip(value, 0.0, 1.0))
        
        # Compute percentile rank
        idx = np.searchsorted(self.reference_cdf, value, side='right')
        percentile = idx / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        """Default CDF for normalization (uniform distribution)."""
        return np.linspace(0.0, 1.0, 1000)
    
    @classmethod
    def load(cls, path: Optional[str] = None, device: Optional[str] = None):
        """
        Load hemolysis predictor from file.
        
        Args:
            path: Path to model file (.pkl or .pt), or None for placeholder
            device: Device to load model on
            
        Returns:
            HemolysisPredictor instance
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle None/null path - use placeholder
        if path is None or path == 'null':
            print("Hemolysis predictor: Using placeholder (hash-based deterministic values)")
            return cls(model=None, device=device)
        
        model_path = Path(path)
        if not model_path.exists():
            print(f"Warning: Hemolysis predictor file not found: {path}")
            print("Hemolysis predictor will return hash-based placeholder values.")
            return cls(model=None, device=device)
        
        print(f"Found hemolysis predictor file: {path} ({model_path.stat().st_size / 1e6:.2f} MB)")
        
        try:
            if path.endswith('.pt'):
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict):
                    model_state = checkpoint.get('model_state_dict', checkpoint)
                    reference_cdf = checkpoint.get('reference_cdf', None)
                else:
                    model_state = checkpoint
                    reference_cdf = None
                return cls(model=model_state, reference_cdf=reference_cdf, device=device)
            else:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                return cls(model=data.get('model'), reference_cdf=data.get('cdf'), device=device)
        except Exception as e:
            print(f"Warning: Could not load hemolysis predictor from {path}: {e}")
            print("Hemolysis predictor will return hash-based placeholder values.")
            return cls(model=None, device=device)

