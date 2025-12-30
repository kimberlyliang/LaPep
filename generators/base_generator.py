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


# Note: load_base_generator() has been removed.
# Use load_peptune_generator() or load_dfm_model() directly based on generator_type.
# Example:
#   if generator_type == 'pepmdlm':
#       generator = load_peptune_generator(config['base_generator_path'], device=device)
#   elif generator_type == 'pepdfm':
#       generator = load_dfm_model(config['dfm_model_path'], device=device)
