"""
LaPep Sampler: Prompt-Conditioned Peptide Generation

This module implements Algorithm 2: LaPep Sampling - Conservative Conditioning of a Base Generator.
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
    seed: Optional[int] = None,
    schedule: Optional[List[int]] = None
) -> str:
    """
    Algorithm 2: LaPep Sampling - Conservative Conditioning of a Base Generator.
    
    Args:
        base_generator: Base proposal kernel b_θ(x' | x, τ)
        prompt: Natural language prompt t
        predictors: K predictors {f_k}^K_{k=1}
        constraints: Dict with penalty functions {ψ_k}^K_{k=1} and weights {λ_k}^K_{k=1}
        text_encoder: Frozen text encoder E_text (Qwen model)
        preference_net: Trained preference module g_ψ*
        num_steps: Number of steps T
        use_linear_preferences: Whether to use linear preference functional
        seed: Random seed
        schedule: Step schedule {τ_s}^{T-1}_{s=0} (default: [0, 1, ..., T-1])
        
    Returns:
        Generated peptide SMILES string X_T (prompt-aligned peptide SMILES satisfying hard predictor constraints)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Algorithm Lines 5-10: Compile prompt into preference functional
    # Line 6: e ← E_text(t) - Text embeddings using Qwen model
    if prompt is not None and text_encoder is not None:
        e = text_encoder.encode(prompt)
        if isinstance(e, torch.Tensor):
            if len(e.shape) == 1:
                e = e.unsqueeze(0)
    else:
        e = None
    
    # Line 7: η ← g_ψ*(e) - Preference parameters from trained module
    if e is not None and preference_net is not None:
        eta = preference_net(e)
    else:
        eta = None
    
    # Line 8: Define R(x; t) ← G_η(u(x)) where u_k(x) = F_k(f_k(x))
    # Line 9: Define Ψ(x) ← Σ^K_{k=1} λ_k ψ_k(u_k(x))
    # Line 10: Define U(x; t) ← -R(x; t) + Ψ(x)
    # These are computed via compute_potential function
    
    # Algorithm Line 12: Sample X_0 ~ μ_0 (e.g., fully masked SMILES or factorized token prior)
    X_s = base_generator.sample_initial_state()
    
    # Algorithm Line 13: for s = 0 to T - 1 do
    if schedule is None:
        schedule = list(range(num_steps))
    
    for s in range(num_steps):
        # Algorithm Line 14: Set current step index τ ← τ_s
        tau = schedule[s] if s < len(schedule) else s
        
        # Algorithm Line 15-16: Obtain candidate set C_s ⊆ N(X_s) ∪ {X_s} (e.g., top-p under b_θ)
        # Form conservative reweighted transition kernel on local candidates
        neighbors = base_generator.get_neighbors(X_s)
        C_s = neighbors + [X_s]  # Include current state
        
        # Algorithm Lines 17-19: Compute log-unnormalized weights
        # log w(x') ← log b_θ(x' | X_s, τ) + ½ [U(X_s; t) – U(x'; t)]
        log_weights = []
        U_X_s = compute_potential(
            X_s, prompt, predictors, text_encoder, preference_net,
            constraints, use_linear_preferences, eta=eta
        )
        
        for x_prime in C_s:
            # Base proposal probability
            log_b_theta = np.log(base_generator.proposal_probability(x_prime, X_s, tau) + 1e-10)
            
            # Potential difference
            U_x_prime = compute_potential(
                x_prime, prompt, predictors, text_encoder, preference_net,
                constraints, use_linear_preferences, eta=eta
            )
            potential_tilt = 0.5 * (U_X_s - U_x_prime)
            
            # Log-unnormalized weight
            log_w = log_b_theta + potential_tilt
            log_weights.append(log_w)
        
        # Algorithm Line 20: Normalize log w(·) via softmax over C_s to obtain q_θ(· | X_s, t, τ)
        log_weights = np.array(log_weights)
        log_Z = np.logaddexp.reduce(log_weights)  # log-sum-exp for numerical stability
        log_probs = log_weights - log_Z
        probs = np.exp(log_probs)
        
        # Algorithm Line 21: Sample X_{s+1} ~ q_θ(· | X_s, t, τ)
        X_s = np.random.choice(C_s, p=probs)
    
    # Algorithm Line 23: return X_T (prompt-aligned peptide SMILES satisfying hard predictor constraints)
    return X_s


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
    Sample a single step of the LaPep process (one iteration of Algorithm 2 Lines 14-21).
    
    Args:
        current_state: Current state X_s
        base_generator: Base generator with proposal kernel b_θ
        text_encoder: Frozen text encoder E_text
        preference_net: Trained preference network g_ψ*
        predictors: Dict of predictors
        prompt: Natural language prompt t
        tau: Current step index τ_s
        constraints: Optional constraints with penalties and weights
        use_linear_preferences: Whether to use linear preference functional
        
    Returns:
        Next state X_{s+1}
    """
    # Algorithm Line 16: Obtain candidate set C_s ⊆ N(X_s) ∪ {X_s}
    C_s = base_generator.get_neighbors(current_state) + [current_state]
    
    # Algorithm Lines 17-19: Compute log-unnormalized weights
    # log w(x') ← log b_θ(x' | X_s, τ) + ½ [U(X_s; t) – U(x'; t)]
    # Pre-compute eta once for performance (avoids re-encoding prompt)
    eta = None
    if prompt is not None and text_encoder is not None and preference_net is not None:
        e = text_encoder.encode(prompt)
        if isinstance(e, torch.Tensor):
            if len(e.shape) == 1:
                e = e.unsqueeze(0)
        eta = preference_net(e)
    
    log_weights = []
    U_X_s = compute_potential(
        current_state, prompt, predictors, text_encoder, preference_net,
        constraints, use_linear_preferences, eta=eta
    )
    
    for x_prime in C_s:
        # Base proposal probability
        log_b_theta = np.log(base_generator.proposal_probability(x_prime, current_state, tau) + 1e-10)
        
        # Potential difference
        U_x_prime = compute_potential(
            x_prime, prompt, predictors, text_encoder, preference_net,
            constraints, use_linear_preferences, eta=eta
        )
        potential_tilt = 0.5 * (U_X_s - U_x_prime)
        
        # Log-unnormalized weight
        log_w = log_b_theta + potential_tilt
        log_weights.append(log_w)
    
    # Algorithm Line 20: Normalize via softmax over C_s
    log_weights = np.array(log_weights)
    log_Z = np.logaddexp.reduce(log_weights)
    log_probs = log_weights - log_Z
    probs = np.exp(log_probs)
    
    # Algorithm Line 21: Sample X_{s+1} ~ q_θ(· | X_s, t, τ)
    next_state = np.random.choice(C_s, p=probs)
    
    return next_state

