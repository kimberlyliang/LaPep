"""base generator interface for discrete generative models."""

import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """abstract base class for discrete peptide generators."""
    
    @abstractmethod
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        """compute base proposal probability b_θ(x'|x,τ)."""
        pass
    
    @abstractmethod
    def get_neighbors(self, x: str) -> List[str]:
        """get local edit neighborhood N(x)."""
        pass
    
    @abstractmethod
    def sample_initial_state(self) -> str:
        """sample initial state from prior μ_0."""
        pass
    
    def sample_unconditioned(self) -> str:
        """sample unconditionally from base generator."""
        return self.sample_initial_state()


def load_base_generator(path: str, device: str = 'cpu') -> BaseGenerator:
    """load base generator from file."""
    from pathlib import Path
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    print(f"Found model file: {path} ({model_path.stat().st_size / 1e9:.2f} GB)")
    
    try:
        from .peptune_wrapper import load_peptune_generator
        generator = load_peptune_generator(path, device=device)
        
        if generator.model is None:
            from .peptune_wrapper import TR2D2_AVAILABLE
            if not TR2D2_AVAILABLE:
                raise RuntimeError(
                    "Failed to load PepMDLM model: TR2-D2 modules could not be imported.\n"
                    "Install with: pip install pandas pytorch-lightning fair-esm hydra-core omegaconf torchmetrics timm SmilesPE fsspec"
                )
            else:
                raise RuntimeError(
                    "Failed to load PepMDLM model. Check error messages above for details."
                )
        
        return generator
    except Exception as e:
        raise RuntimeError(
            f"Could not load PepMDLM model from {path}: {e}\n"
            "Install TR2-D2 dependencies: pip install pandas pytorch-lightning fair-esm hydra-core omegaconf torchmetrics timm SmilesPE"
        ) from e
