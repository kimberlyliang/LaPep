import numpy as np
import torch
from typing import Optional, Dict, List

from .potential import compute_potential


def compute_transition_kernel(
    x: str,
    candidates: List[str],
    base_generator: any,
    text_encoder: Optional[any],
    preference_net: Optional[any],
    predictors: Dict,
    prompt: Optional[str],
    tau: int,
    constraints: Optional[Dict] = None,
    use_linear_preferences: bool = False,
    eta: Optional[any] = None
) -> np.ndarray:
    """
    Compute the LaPep transition kernel q_θ(x'|x,t,τ) from Eq (10).
    q_θ(x'|x,t,τ) = b_θ(x'|x,τ) * exp(0.5[U(x;t) - U(x';t)]) / Z(x)
    Args:
        x: Current state
        candidates: List of candidate next states (including x itself)
        base_generator: Base generator with proposal kernel b_θ
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        predictors: Dict of predictor objects
        prompt: Natural language prompt
        tau: Time step index
        constraints: Optional constraint configuration
        use_linear_preferences: Whether to use linear preference functional
        eta: Pre-computed preference parameters (should be computed once by caller for performance)
        
    Returns:
        Array of probabilities for each candidate (normalized)
    """
    # Note: eta should be pre-computed by the caller to avoid redundant encoding
    
    # potential at current state
    U_x = compute_potential(
        x, prompt, predictors, text_encoder, preference_net,
        constraints, use_linear_preferences, eta=eta
    )
    
    base_probs = []
    for candidate in candidates:
        prob = base_generator.proposal_probability(candidate, x, tau)
        base_probs.append(prob)
    
    base_probs = np.array(base_probs)
    
    # unnormalized weights with potential tilt
    log_weights = []
    for candidate in candidates:
        U_candidate = compute_potential(
            candidate, prompt, predictors, text_encoder, preference_net,
            constraints, use_linear_preferences, eta=eta
        )
        
        # base log probability
        idx = candidates.index(candidate)
        log_base = np.log(base_probs[idx] + 1e-10)
        
        # 0.5 * [U(x) - U(x')]
        potential_tilt = 0.5 * (U_x - U_candidate)
        
        log_weight = log_base + potential_tilt
        log_weights.append(log_weight)
    
    # normalize via log-sum-exp for numerical stability
    log_weights = np.array(log_weights)
    log_Z = np.logaddexp.reduce(log_weights)
    log_probs = log_weights - log_Z
    probs = np.exp(log_probs)
    
    return probs


def compute_edge_flow(
    x: str,
    x_prime: str,
    base_generator: any,
    text_encoder: Optional[any],
    preference_net: Optional[any],
    predictors: Dict,
    prompt: Optional[str],
    tau: int,
    constraints: Optional[Dict] = None,
    use_linear_preferences: bool = False,
    eta: Optional[any] = None
) -> float:
    """
    Compute antisymmetric edge flow F(x,x') = log(q(x'|x) / q(x|x')).
    
    For conservative flows, this should be a gradient (exact difference).
    
    Args:
        x, x_prime: Adjacent states
        base_generator: Base generator
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        predictors: Dict of predictors
        prompt: Natural language prompt
        tau: Time step
        constraints: Optional constraints
        use_linear_preferences: Whether to use linear preference functional
        eta: Pre-computed preference parameters (computed once if not provided)
        
    Returns:
        Edge flow value
    """
    # Compute eta once for both transition kernel calls
    if eta is None and prompt is not None and text_encoder is not None and preference_net is not None:
        e = text_encoder.encode(prompt)
        if isinstance(e, torch.Tensor):
            if len(e.shape) == 1:
                e = e.unsqueeze(0)
        eta = preference_net(e)
    
    # compute transition probabilities in both directions
    candidates_forward = [x_prime, x] 
    probs_forward = compute_transition_kernel(
        x, candidates_forward, base_generator,
        text_encoder, preference_net, predictors,
        prompt, tau, constraints, use_linear_preferences, eta=eta
    )
    q_forward = probs_forward[0]  # probability of x -> x'
    
    candidates_backward = [x, x_prime]
    probs_backward = compute_transition_kernel(
        x_prime, candidates_backward, base_generator,
        text_encoder, preference_net, predictors,
        prompt, tau, constraints, use_linear_preferences, eta=eta
    )
    q_backward = probs_backward[0]  # probability of x' -> x
    
    # edge flow
    if q_forward > 0 and q_backward > 0:
        flow = np.log(q_forward / q_backward)
    else:
        flow = 0.0
    
    return float(flow)
