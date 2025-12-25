from .base_generator import BaseGenerator


class DFMWrapper(BaseGenerator):
    
    def __init__(self, model):
        self.model = model
    
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        return self.model.proposal_probability(x_prime, x, tau)
    
    def get_neighbors(self, x: str) -> list:
        return self.model.get_edit_neighbors(x)
    
    def sample_initial_state(self) -> str:
        return self.model.sample_prior()


def load_dfm_model(path: str) -> DFMWrapper:
    raise NotImplementedError(
        "DFM model loading not yet implemented. "
        "Currently only PepMDLM (peptune_wrapper) is supported."
    )

