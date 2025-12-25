import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    #interface
    @abstractmethod
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        pass
    
    @abstractmethod
    def get_neighbors(self, x: str) -> List[str]:
        pass
    
    @abstractmethod
    def sample_initial_state(self) -> str:
        pass
    
    def sample_unconditioned(self) -> str:
        return self.sample_initial_state()


def load_base_generator(path: str, device: str = 'cpu') -> BaseGenerator:
    from pathlib import Path
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    print(f"Found model file: {path} ({model_path.stat().st_size / 1e9:.2f} GB)")
    
    from .peptune_wrapper import load_peptune_generator, TR2D2_AVAILABLE
    generator = load_peptune_generator(path, device=device)
    
    if generator.model is None:
        if not TR2D2_AVAILABLE:
            raise RuntimeError(
                "Failed to load PepMDLM model: TR2-D2 modules could not be imported.\n"
                "Install with: pip install pandas pytorch-lightning fair-esm hydra-core omegaconf torchmetrics timm SmilesPE fsspec"
            )
    return generator
