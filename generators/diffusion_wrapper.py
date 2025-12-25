"""
Diffusion Wrapper

Wrapper for masked discrete diffusion models.
"""

from .base_generator import BaseGenerator


class DiffusionWrapper(BaseGenerator):
    """Wrapper for masked discrete diffusion models."""
    
    def __init__(self, model):
        """
        Initialize diffusion wrapper.
        
        Args:
            model: Trained masked discrete diffusion model
        """
        self.model = model
    
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        """Compute proposal probability from diffusion model."""
        # In practice, would use model to compute unmasking probabilities
        return self.model.proposal_probability(x_prime, x, tau)
    
    def get_neighbors(self, x: str) -> list:
        """Get neighbors (unmasking operations)."""
        return self.model.get_unmasking_candidates(x)
    
    def sample_initial_state(self) -> str:
        """Sample fully masked state."""
        return self.model.get_masked_state()


def load_diffusion_model(path: str) -> DiffusionWrapper:
    """Load diffusion model from checkpoint."""
    raise NotImplementedError(
        "Generic diffusion model loading not yet implemented. "
        "Use PepMDLM (peptune_wrapper) instead, which is a masked discrete diffusion model."
    )

