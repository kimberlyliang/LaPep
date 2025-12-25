"""
Discrete Flow Matching (DFM) Wrapper

Wrapper for discrete flow matching models.
"""

from .base_generator import BaseGenerator


class DFMWrapper(BaseGenerator):
    """Wrapper for discrete flow matching models."""
    
    def __init__(self, model):
        """
        Initialize DFM wrapper.
        
        Args:
            model: Trained discrete flow matching model
        """
        self.model = model
    
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        """Compute proposal probability from flow model."""
        # In practice, would use model to compute flow probabilities
        return self.model.proposal_probability(x_prime, x, tau)
    
    def get_neighbors(self, x: str) -> list:
        """Get neighbors (token edits)."""
        return self.model.get_edit_neighbors(x)
    
    def sample_initial_state(self) -> str:
        """Sample from flow prior."""
        return self.model.sample_prior()


def load_dfm_model(path: str) -> DFMWrapper:
    """Load DFM model from checkpoint."""
    raise NotImplementedError(
        "DFM model loading not yet implemented. "
        "Currently only PepMDLM (peptune_wrapper) is supported."
    )

