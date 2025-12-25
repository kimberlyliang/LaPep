"""
Binding Affinity Predictor

Predicts peptide binding affinity to target receptors.
"""

import numpy as np
import torch
from typing import Optional
import pickle
import os


class BindingPredictor:
    """Predictor for peptide binding affinity."""
    
    def __init__(self, model=None, reference_cdf=None, device=None):
        """
        Initialize binding predictor.
        
        Args:
            model: Trained binding prediction model
            reference_cdf: Empirical CDF for normalization
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.reference_cdf = reference_cdf or self._default_cdf()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, peptide: str) -> float:
        """
        Predict binding affinity for a peptide.
        
        Args:
            peptide: Peptide SMILES string
            
        Returns:
            Raw binding affinity score
        """
        if self.model is None:
            # Placeholder: return random value for now
            return np.random.uniform(0.0, 1.0)
        
        # In practice, would use self.model to predict
        # For now, return a placeholder value
        # TODO: Implement actual prediction using the loaded model
        return np.random.uniform(6.0, 8.0)  # Typical binding affinity range
    
    def normalize(self, value: float) -> float:
        """
        Normalize prediction to [0, 1] using empirical CDF.
        
        Args:
            value: Raw prediction value
            
        Returns:
            Normalized value in [0, 1]
        """
        # Use reference CDF to compute percentile
        if self.reference_cdf is None:
            # Simple normalization: assume binding affinity is in range [4.0, 10.0]
            # Higher is better, so normalize to [0, 1]
            normalized = (value - 4.0) / (10.0 - 4.0)
            return float(np.clip(normalized, 0.0, 1.0))
        
        # Compute percentile
        percentile = np.searchsorted(self.reference_cdf, value) / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        """Default reference distribution (placeholder)."""
        # In practice, would load from training data
        # For binding affinity, typical range is [4.0, 10.0] (pKd/pKi)
        return np.linspace(4.0, 10.0, 1000)
    
    @classmethod
    def load(cls, path: str, device=None, protein_seq: Optional[str] = None, 
             tokenizer=None, base_path: Optional[str] = None):
        """
        Load predictor from file.
        
        Args:
            path: Path to model file (.pt for PyTorch, .pkl for pickle)
            device: Device to load on
            protein_seq: Protein target sequence (optional, for real predictions)
            tokenizer: Peptide tokenizer (optional)
            base_path: Base path for TR2-D2 (optional)
            
        Returns:
            BindingPredictor instance
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if file exists
        from pathlib import Path
        model_path = Path(path)
        if not model_path.exists():
            print(f"ERROR: Binding predictor file not found: {path}")
            print("Please ensure the model file exists at this path.")
            return cls(model=None, device=device)
        
        print(f"Found binding predictor file: {path} ({model_path.stat().st_size / 1e6:.2f} MB)")
        
        # Try to load real binding predictor first
        if path.endswith('.pt') and protein_seq is not None:
            try:
                from .binding_wrapper import RealBindingPredictor
                return RealBindingPredictor.load(
                    path, 
                    protein_seq=protein_seq,
                    tokenizer=tokenizer,
                    base_path=base_path,
                    device=device
                )
            except Exception as e:
                print(f"Warning: Could not load real binding predictor: {e}")
                print("This is likely because TR2-D2 dependencies are missing (pandas, esm, etc.)")
                print("Falling back to placeholder. Install TR2-D2 dependencies to use real model.")
        
        if path.endswith('.pt'):
            # PyTorch model file
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                
                # The checkpoint might contain 'model_state_dict' or just be the state dict
                if isinstance(checkpoint, dict):
                    # Check if it's a full checkpoint with model_state_dict
                    if 'model_state_dict' in checkpoint:
                        model_state = checkpoint['model_state_dict']
                        reference_cdf = checkpoint.get('reference_cdf', None)
                    else:
                        # Might just be the state dict
                        model_state = checkpoint
                        reference_cdf = None
                else:
                    model_state = checkpoint
                    reference_cdf = None
                
                # For now, we'll store the checkpoint but not instantiate the full model
                # The actual model architecture would need to be imported from TR2-D2
                # This allows the code to run while we work on proper integration
                return cls(model=model_state, reference_cdf=reference_cdf, device=device)
                
            except Exception as e:
                print(f"Warning: Could not load PyTorch model from {path}: {e}")
                print("Using placeholder predictor. Actual predictions will be random.")
                return cls(model=None, device=device)
        
        elif path.endswith('.pkl'):
            # Pickle file
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                return cls(model=data.get('model'), reference_cdf=data.get('cdf'), device=device)
            except Exception as e:
                print(f"Warning: Could not load pickle file from {path}: {e}")
                return cls(model=None, device=device)
        else:
            # Unknown format, try pickle first, then PyTorch
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                return cls(model=data.get('model'), reference_cdf=data.get('cdf'), device=device)
            except:
                try:
                    checkpoint = torch.load(path, map_location=device, weights_only=False)
                    return cls(model=checkpoint, device=device)
                except Exception as e:
                    print(f"Warning: Could not load model from {path}: {e}")
                    return cls(model=None, device=device)

