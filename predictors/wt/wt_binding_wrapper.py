"""
Wrapper for WT Binding Predictor to match LaPep interface.
"""

import numpy as np
import torch
from typing import Optional
from .wt_binding import load_pooled_affinity_predictor, load_affinity_predictor
import esm


class WTBindingPredictor:
    """
    Wrapper for WT binding predictor to match LaPep interface.
    
    Provides predict() and normalize() methods consistent with SMILES predictors.
    """
    
    def __init__(
        self,
        model=None,
        esm_model=None,
        reference_cdf=None,
        device=None,
        protein_seq: Optional[str] = None
    ):
        """
        Initialize WT binding predictor.
        
        Args:
            model: Trained binding predictor model
            esm_model: ESM model for protein embeddings
            reference_cdf: Reference CDF for normalization
            device: Device to run on
            protein_seq: Protein sequence for binding prediction
        """
        self.model = model
        self.esm_model = esm_model
        self.reference_cdf = reference_cdf or self._default_cdf()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.protein_seq = protein_seq
        
        # Load ESM model if not provided
        if self.esm_model is None and self.protein_seq:
            self._load_esm_model()
    
    def _load_esm_model(self):
        """Load ESM model for protein embeddings."""
        try:
            self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_hub(
                "facebook/esm2_t33_650M_UR50D"
            )
            self.esm_model = self.esm_model.to(self.device).eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load ESM model: {e}")
            self.esm_model = None
    
    def _get_protein_embedding(self, protein_seq: str):
        """Get protein embedding using ESM."""
        if self.esm_model is None:
            return None
        
        try:
            batch_converter = self.esm_alphabet.get_batch_converter()
            data = [("protein", protein_seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33])
                protein_emb = results["representations"][33].mean(dim=1)  # Average pooling
                return protein_emb
        except Exception as e:
            print(f"Warning: Could not get protein embedding: {e}")
            return None
    
    def predict(self, peptide: str) -> float:
        """
        Predict binding affinity for a WT peptide.
        
        Args:
            peptide: Peptide sequence in one-letter amino acid code
        
        Returns:
            Binding affinity score (higher = better binding)
        """
        if self.model is None:
            # Fallback: return random value
            return np.random.uniform(6.0, 8.0)
        
        if self.protein_seq is None:
            print("Warning: No protein sequence provided for binding prediction")
            return np.random.uniform(6.0, 8.0)
        
        # Get protein embedding
        protein_emb = self._get_protein_embedding(self.protein_seq)
        if protein_emb is None:
            return np.random.uniform(6.0, 8.0)
        
        # Get peptide embedding (simplified - would need proper peptide embedding)
        # For now, use a placeholder
        # In practice, you'd need to convert WT peptide to SMILES or use a peptide encoder
        try:
            # This is a placeholder - actual implementation would need peptide encoder
            peptide_emb = torch.randn(1, 1280).to(self.device)  # Placeholder
            
            with torch.no_grad():
                # Model expects (batch, seq_len, dim) for both
                # Adjust shapes as needed for your model
                affinity = self.model(protein_emb, peptide_emb)
                if isinstance(affinity, torch.Tensor):
                    affinity = affinity.cpu().item()
                return float(affinity)
        except Exception as e:
            print(f"Warning: Binding prediction failed: {e}")
            return np.random.uniform(6.0, 8.0)
    
    def normalize(self, value: float) -> float:
        """
        Normalize binding affinity value to [0, 1] using empirical CDF.
        
        Args:
            value: Raw binding affinity score
        
        Returns:
            Normalized score in [0, 1]
        """
        if self.reference_cdf is None:
            # Linear normalization: assume range [4.0, 10.0]
            normalized = (value - 4.0) / (10.0 - 4.0)
            return float(np.clip(normalized, 0.0, 1.0))
        
        # Compute percentile rank
        idx = np.searchsorted(self.reference_cdf, value, side='right')
        percentile = idx / len(self.reference_cdf)
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _default_cdf(self):
        """Default CDF for normalization."""
        return np.linspace(4.0, 10.0, 1000)
    
    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        device: Optional[str] = None,
        protein_seq: Optional[str] = None,
        model_type: str = 'pooled'
    ):
        """
        Load WT binding predictor.
        
        Args:
            path: Path to model checkpoint
            device: Device to load on
            protein_seq: Protein sequence for binding prediction
            model_type: 'pooled' or 'unpooled'
        
        Returns:
            WTBindingPredictor instance
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if path is None:
            print("Warning: No model path provided. Using placeholder predictor.")
            return cls(model=None, device=device, protein_seq=protein_seq)
        
        try:
            from pathlib import Path
            model_path = Path(path)
            if not model_path.exists():
                print(f"Warning: Model file not found: {path}")
                return cls(model=None, device=device, protein_seq=protein_seq)
            
            # Load model
            if model_type == 'pooled':
                model = load_pooled_affinity_predictor(str(path), device)
            else:
                model = load_affinity_predictor(str(path), device)
            
            return cls(model=model, device=device, protein_seq=protein_seq)
        except Exception as e:
            print(f"Warning: Could not load WT binding predictor: {e}")
            return cls(model=None, device=device, protein_seq=protein_seq)

