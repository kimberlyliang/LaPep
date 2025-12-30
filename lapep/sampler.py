"""
LaPep Sampler: Prompt-Conditioned Peptide Generation

This module implements:
- Algorithm 2: LaPep Sampling - Conservative Conditioning of a Base Generator
- Algorithm 3: LaPep Sampling from Fixed Partially Masked Seeds (Controlled Evaluation)
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Union, Callable

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
    schedule: Optional[List[int]] = None,
    language_weight: float = 1.0
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
            constraints, use_linear_preferences, eta=eta, language_weight=language_weight
        )
        
        for x_prime in C_s:
            # Base proposal probability
            log_b_theta = np.log(base_generator.proposal_probability(x_prime, X_s, tau) + 1e-10)
            
            # Potential difference
            U_x_prime = compute_potential(
                x_prime, prompt, predictors, text_encoder, preference_net,
                constraints, use_linear_preferences, eta=eta, language_weight=language_weight
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
    use_linear_preferences: bool = False,
    language_weight: float = 1.0
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
        constraints, use_linear_preferences, eta=eta, language_weight=language_weight
    )
    
    for x_prime in C_s:
        # Base proposal probability
        log_b_theta = np.log(base_generator.proposal_probability(x_prime, current_state, tau) + 1e-10)
        
        # Potential difference
        U_x_prime = compute_potential(
            x_prime, prompt, predictors, text_encoder, preference_net,
            constraints, use_linear_preferences, eta=eta, language_weight=language_weight
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


def mask_sequence(
    sequence: str,
    mask_rate: Optional[float] = None,
    mask_positions: Optional[List[int]] = None,
    mask_token: Optional[str] = None,
    generator: Optional[any] = None
) -> str:
    """
    Apply mask to a sequence (Algorithm 3: Mask operator).
    
    Args:
        sequence: Input sequence (SMILES or WT)
        mask_rate: Fraction of tokens to mask (if mask_positions not provided)
        mask_positions: Explicit list of positions to mask (0-indexed)
        mask_token: Token to use for masking (if None, uses generator's mask token)
        generator: Base generator (needed to get mask token if mask_token is None)
        
    Returns:
        Partially masked sequence
    """
    if mask_positions is not None:
        # Explicit mask positions
        positions_to_mask = set(mask_positions)
    elif mask_rate is not None:
        # Random masking based on mask rate
        seq_len = len(sequence)
        num_to_mask = int(seq_len * mask_rate)
        positions_to_mask = set(np.random.choice(seq_len, size=num_to_mask, replace=False))
    else:
        # Default: mask 50% randomly
        seq_len = len(sequence)
        num_to_mask = seq_len // 2
        positions_to_mask = set(np.random.choice(seq_len, size=num_to_mask, replace=False))
    
    # Get mask token
    if mask_token is None:
        if generator is not None and hasattr(generator, 'tokenizer'):
            # Try to get mask token from generator
            if hasattr(generator.tokenizer, 'mask_token_id'):
                mask_id = generator.tokenizer.mask_token_id
                # Decode mask token
                mask_token = generator.tokenizer.decode([mask_id])
            elif hasattr(generator.tokenizer, 'mask_token'):
                mask_token = generator.tokenizer.mask_token
            else:
                # Fallback: use [MASK] for SMILES or 'X' for WT
                mask_token = '[MASK]' if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence.upper()) else 'X'
        else:
            # Fallback
            mask_token = '[MASK]' if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence.upper()) else 'X'
    
    # Apply mask
    masked_seq = list(sequence)
    for pos in positions_to_mask:
        if 0 <= pos < len(masked_seq):
            masked_seq[pos] = mask_token
    
    return ''.join(masked_seq)


def sample_from_fixed_seeds(
    base_generator: any,
    seed_set: List[str],
    prompt: Optional[str],
    predictors: Dict,
    constraints: Optional[Dict],
    text_encoder: Optional[any] = None,
    preference_net: Optional[any] = None,
    num_steps: int = 50,
    use_linear_preferences: bool = False,
    schedule: Optional[List[int]] = None,
    mask_rate: Optional[float] = None,
    mask_positions: Optional[List[int]] = None,
    completions_per_seed: int = 1,
    seed: Optional[int] = None,
    language_weight: float = 1.0
) -> List[Dict[str, Union[str, int]]]:
    """
    Algorithm 3: LaPep Sampling from Fixed Partially Masked Seeds (Controlled Evaluation).
    
    Generates controlled completions from fixed partially masked seeds, holding the seed
    and mask pattern fixed across all conditioning variants for fair comparison.
    
    Args:
        base_generator: Base proposal kernel b_θ(x' | x, τ)
        seed_set: Set of M seed sequences S = {x_seed^(i)}_{i=1}^M
        prompt: Natural language prompt t
        predictors: K predictors {f_k}^K_{k=1}
        constraints: Dict with penalty functions {ψ_k}^K_{k=1} and weights {λ_k}^K_{k=1}
        text_encoder: Frozen text encoder E_text
        preference_net: Trained preference module g_ψ*
        num_steps: Number of steps T
        use_linear_preferences: Whether to use linear preference functional
        schedule: Step schedule {τ_s}^{T-1}_{s=0} (default: [0, 1, ..., T-1])
        mask_rate: Fraction of tokens to mask (if mask_positions not provided)
        mask_positions: Explicit list of positions to mask (fixed across all seeds)
        completions_per_seed: Number of completions J per seed
        seed: Random seed for reproducibility
        
    Returns:
        List of dicts with keys: 'completion', 'seed_index', 'completion_index', 'prompt'
        Returns {X_T^(i,j)}_{i=1,j=1}^{M,J}
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Algorithm Lines 7-12: Compile prompt into preference functional (same as Algorithm 2)
    if prompt is not None and text_encoder is not None:
        e = text_encoder.encode(prompt)
        if isinstance(e, torch.Tensor):
            if len(e.shape) == 1:
                e = e.unsqueeze(0)
    else:
        e = None
    
    # Line 9: η ← g_ψ*(e)
    if e is not None and preference_net is not None:
        eta = preference_net(e)
    else:
        eta = None
    
    # Lines 10-12: Define R(x;t), Ψ(x), U(x;t) (computed via compute_potential)
    
    # Algorithm Line 13: Generate controlled completions from fixed partially masked seeds
    all_completions = []
    
    # Line 14: for i = 1 to M do
    for seed_idx, x_seed in enumerate(seed_set):
        # Line 15: Set partially masked initial state X_0^(i) ← Mask(x_seed^(i); m)
        # Hold the seed and mask pattern fixed across all conditioning variants
        X_0_i = mask_sequence(
            x_seed,
            mask_rate=mask_rate,
            mask_positions=mask_positions,
            generator=base_generator
        )
        
        # Line 17: for j = 1 to J do
        for completion_idx in range(completions_per_seed):
            # Line 18: Set X_0 ← X_0^(i)
            X_s = X_0_i
            
            # Line 19: for s = 0 to T - 1 do
            if schedule is None:
                schedule_list = list(range(num_steps))
            else:
                schedule_list = schedule
            
            for s in range(num_steps):
                # Line 20: Set current step index τ ← τ_s
                tau = schedule_list[s] if s < len(schedule_list) else s
                
                # Line 21: Obtain candidate set C_s ⊆ N(X_s) ∪ {X_s}
                neighbors = base_generator.get_neighbors(X_s)
                C_s = neighbors + [X_s]
                
                # Line 22-24: Compute log-unnormalized weights for each candidate
                log_weights = []
                U_X_s = compute_potential(
                    X_s, prompt, predictors, text_encoder, preference_net,
                    constraints, use_linear_preferences, eta=eta, language_weight=language_weight
                )
                
                for x_prime in C_s:
                    # Line 23: log w(x') ← log b_θ(x' | X_s, τ) + ½ [U(X_s; t) – U(x'; t)]
                    log_b_theta = np.log(base_generator.proposal_probability(x_prime, X_s, tau) + 1e-10)
                    
                    U_x_prime = compute_potential(
                        x_prime, prompt, predictors, text_encoder, preference_net,
                        constraints, use_linear_preferences, eta=eta, language_weight=language_weight
                    )
                    potential_tilt = 0.5 * (U_X_s - U_x_prime)
                    
                    log_w = log_b_theta + potential_tilt
                    log_weights.append(log_w)
                
                # Line 25: Normalize log w(·) via softmax over C_s
                log_weights = np.array(log_weights)
                log_Z = np.logaddexp.reduce(log_weights)
                log_probs = log_weights - log_Z
                probs = np.exp(log_probs)
                
                # Line 26: Sample X_{s+1} ~ q_θ(· | X_s, t, τ)
                X_s = np.random.choice(C_s, p=probs)
            
            # Line 28: Store completion X_T^(i,j) together with seed index i and prompt t
            completion_data = {
                'completion': X_s,
                'seed_index': seed_idx,
                'completion_index': completion_idx,
                'seed': x_seed,
                'masked_seed': X_0_i,
                'prompt': prompt if prompt else None
            }
            all_completions.append(completion_data)
    
    # Line 31: return {X_T^(i,j)}_{i=1,j=1}^{M,J}
    return all_completions

