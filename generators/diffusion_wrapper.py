from .base_generator import BaseGenerator


class DiffusionWrapper(BaseGenerator):
    
    def __init__(self, model):
        self.model = model
    
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        return self.model.proposal_probability(x_prime, x, tau)
    
    def get_neighbors(self, x: str) -> list:
        return self.model.get_unmasking_candidates(x)
    
    def sample_initial_state(self) -> str:
        return self.model.get_masked_state()


def load_diffusion_model(path: str) -> DiffusionWrapper:
    raise NotImplementedError(
        "Generic diffusion model loading not yet implemented. "
        "Use PepMDLM (peptune_wrapper) instead, which is a masked discrete diffusion model."
    )

