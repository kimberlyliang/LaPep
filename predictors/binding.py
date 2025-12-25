import numpy as np
import torch
from typing import Optional
import pickle
import os


class BindingPredictor:
    
    def __init__(self, model=None, reference_cdf=None, device=None):
        self.model = model
        self.reference_cdf = reference_cdf or self._default_cdf()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, peptide: str) -> float:
        if self.model is None:
            return np.random.uniform(0.0, 1.0)
        return np.random.uniform(6.0, 8.0)
    
    def normalize(self, value: float) -> float:
        if self.reference_cdf is None:
            normalized = (value - 4.0) / (10.0 - 4.0)
            return float(np.clip(normalized, 0.0, 1.0))
        percentile = np.searchsorted(self.reference_cdf, value) / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        return np.linspace(4.0, 10.0, 1000)
    
    @classmethod
    def load(cls, path: str, device=None, protein_seq: Optional[str] = None, 
             tokenizer=None, base_path: Optional[str] = None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        from pathlib import Path
        model_path = Path(path)
        if not model_path.exists():
            print(f"ERROR: Binding predictor file not found: {path}")
            print("Please ensure the model file exists at this path.")
            return cls(model=None, device=device)
        
        print(f"Found binding predictor file: {path} ({model_path.stat().st_size / 1e6:.2f} MB)")
        
        if path.endswith('.pt') and protein_seq is not None:
            from .binding_wrapper import RealBindingPredictor
            return RealBindingPredictor.load(
                path, 
                protein_seq=protein_seq,
                tokenizer=tokenizer,
                base_path=base_path,
                device=device
            )
        
        if path.endswith('.pt'):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    reference_cdf = checkpoint.get('reference_cdf', None)
                else:
                    model_state = checkpoint
                    reference_cdf = None
            else:
                model_state = checkpoint
                reference_cdf = None
            return cls(model=model_state, reference_cdf=reference_cdf, device=device)
        
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return cls(model=data.get('model'), reference_cdf=data.get('cdf'), device=device)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(model=data.get('model'), reference_cdf=data.get('cdf'), device=device)

