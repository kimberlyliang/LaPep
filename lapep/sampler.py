"""
LaPep Sampler: Prompt-Conditioned Peptide Generation

This module implements the sampling procedure using the LaPep transition kernel.
"""

import numpy as np
import torch
from typing import Optional, Dict, List

from .kernel import compute_transition_kernel
from .potential import compute_potential


def sample_peptide(
    base_generator: any,
    prompt: Optional[str],
    predictors: Dict,
    constraints: Optional[Dict],
    text_encoder: Optional[any] = None,
    preference_net: Optional[any] = None,
    num_steps: int = 50,
    use_linear_preferences: bool = False,
    seed: Optional[int] = None
) -> str:
    """
    Sample a peptide using LaPep conditioning.
    
    Args:
        base_generator: Base generator with proposal kernel b_θ
        prompt: Natural language prompt (None for predictor-only)
        predictors: Dict of predictor objects
        constraints: Optional constraint configuration
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        num_steps: Number of sampling steps
        use_linear_preferences: Whether to use linear preference functional
        seed: Random seed
        
    Returns:
        Generated peptide SMILES string
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Initialize from base generator prior
    x = base_generator.sample_initial_state()
    
    # Sample trajectory
    for step in range(num_steps):
        tau = step
        
        # Get candidate states (neighbors + self)
        candidates = base_generator.get_neighbors(x) + [x]
        
        # Compute transition probabilities
        probs = compute_transition_kernel(
            x, candidates, base_generator,
            text_encoder, preference_net, predictors,
            prompt, tau, constraints, use_linear_preferences
        )
        
        # Sample next state
        x = np.random.choice(candidates, p=probs)
    
    return x


def sample_step(
    current_state: str,
    base_generator: any,
    text_encoder: Optional[any],
    preference_net: Optional[any],
    predictors: Dict,
    prompt: Optional[str],
    tau: int,
    constraints: Optional[Dict] = None,
    use_linear_preferences: bool = False
) -> str:
    """
    Sample a single step of the LaPep process.
    
    Args:
        current_state: Current peptide state
        base_generator: Base generator
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        predictors: Dict of predictors
        prompt: Natural language prompt
        tau: Current time step
        constraints: Optional constraints
        use_linear_preferences: Whether to use linear preferences
        
    Returns:
        Next state
    """
    # Get candidates
    candidates = base_generator.get_neighbors(current_state) + [current_state]
    
    # Compute transition probabilities
    probs = compute_transition_kernel(
        current_state, candidates, base_generator,
        text_encoder, preference_net, predictors,
        prompt, tau, constraints, use_linear_preferences
    )
    
    # Sample
    next_state = np.random.choice(candidates, p=probs)
    
    return next_state

