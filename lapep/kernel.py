"""
LaPep Transition Kernel: Eq (10)

This module implements the conservative transition kernel that modifies base generator
dynamics through the scalar potential.
"""

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
    use_linear_preferences: bool = False
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
        
    Returns:
        Array of probabilities for each candidate (normalized)
    """
    # Compute potential at current state
    U_x = compute_potential(
        x, prompt, predictors, text_encoder, preference_net,
        constraints, use_linear_preferences
    )
    
    # Get base proposal probabilities
    base_probs = []
    for candidate in candidates:
        prob = base_generator.proposal_probability(candidate, x, tau)
        base_probs.append(prob)
    
    base_probs = np.array(base_probs)
    
    # Compute unnormalized weights with potential tilt
    log_weights = []
    for candidate in candidates:
        # Compute potential at candidate state
        U_candidate = compute_potential(
            candidate, prompt, predictors, text_encoder, preference_net,
            constraints, use_linear_preferences
        )
        
        # Base log probability
        idx = candidates.index(candidate)
        log_base = np.log(base_probs[idx] + 1e-10)
        
        # Potential tilt: 0.5 * [U(x) - U(x')]
        potential_tilt = 0.5 * (U_x - U_candidate)
        
        log_weight = log_base + potential_tilt
        log_weights.append(log_weight)
    
    # Normalize via log-sum-exp for numerical stability
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
    constraints: Optional[Dict] = None
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
        
    Returns:
        Edge flow value
    """
    # Compute transition probabilities in both directions
    candidates_forward = [x_prime, x]  # Include self-transition
    probs_forward = compute_transition_kernel(
        x, candidates_forward, base_generator,
        text_encoder, preference_net, predictors,
        prompt, tau, constraints
    )
    q_forward = probs_forward[0]  # Probability of x -> x'
    
    candidates_backward = [x, x_prime]
    probs_backward = compute_transition_kernel(
        x_prime, candidates_backward, base_generator,
        text_encoder, preference_net, predictors,
        prompt, tau, constraints
    )
    q_backward = probs_backward[0]  # Probability of x' -> x
    
    # Edge flow
    if q_forward > 0 and q_backward > 0:
        flow = np.log(q_forward / q_backward)
    else:
        flow = 0.0
    
    return float(flow)
