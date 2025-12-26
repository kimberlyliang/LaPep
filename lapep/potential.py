"""
LaPep Potential: U(x;t) = -R(x;t) + Ψ(x)

This module implements the scalar potential that unifies soft language preferences
with hard predictor constraints.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Callable


def compute_potential(
    x: str,
    prompt: Optional[str],
    predictors: Dict,
    text_encoder: Optional[any],
    preference_net: Optional[any],
    constraints: Optional[Dict] = None,
    use_linear_preferences: bool = False,
    eta: Optional[any] = None
) -> float:
    """
    Compute the LaPep potential U(x;t) = -R(x;t) + Ψ(x).
    
    Args:
        x: Peptide SMILES string
        prompt: Natural language prompt (None for predictor-only)
        predictors: Dict mapping predictor names to predictor objects
        text_encoder: Frozen text encoder E_text
        preference_net: Trained preference network g_ψ
        constraints: Optional dict with constraint weights and penalty functions
        use_linear_preferences: If True, use linear preference functional
        eta: Optional pre-computed preference parameters (avoids re-encoding prompt)
        
    Returns:
        Scalar potential value (lower is better)
    """
    psi = compute_constraint_penalty(x, predictors, constraints)

    if eta is not None:
        # Use pre-computed eta (performance optimization)
        r = compute_preference_score(
            x, predictors, use_linear_preferences, eta=eta
        )
    elif prompt is not None and text_encoder is not None and preference_net is not None:
        r = compute_preference_score(
            x, predictors, use_linear_preferences,
            prompt=prompt, text_encoder=text_encoder, preference_net=preference_net
        )
    else:
        r = 0.0
    
    U = -r + psi
    
    return U


def compute_constraint_penalty(
    x: str,
    predictors: Dict,
    constraints: Optional[Dict] = None
) -> float:
    """
    Algorithm 1, Line 24: Compute hard constraint penalty Ψ(x) = Σ_k λ_k ψ_k(u_k(x)).
    
    This implements the constraint penalty term that enforces hard constraints on predictor values.
    Each predictor k has:
    - A penalty function ψ_k that penalizes undesirable values
    - A weight λ_k that controls the strength of the penalty
    
    Args:
        x: Peptide SMILES string
        predictors: Dict of predictor objects
        constraints: Optional dict with:
            - 'weights': Dict mapping predictor names to weights λ_k
            - 'penalties': Dict mapping predictor names to penalty functions ψ_k
            - 'strength': Global constraint strength multiplier
            
    Returns:
        Total constraint penalty (nonnegative)
    """
    if not predictors:
        return 0.0
    
    if constraints is None:
        constraints = {}
    
    weights = constraints.get('weights', {})
    penalties = constraints.get('penalties', {})
    strength = constraints.get('strength', 1.0)
    
    total_penalty = 0.0
    
    for pred_name, predictor in predictors.items():
        raw_value = predictor.predict(x)
        normalized = predictor.normalize(raw_value)
        weight = weights.get(pred_name, 1.0) * strength
        penalty_fn = penalties.get(pred_name, default_penalty_function)
        penalty = penalty_fn(normalized, pred_name)
        total_penalty += weight * penalty
    
    return total_penalty


def default_penalty_function(u: float, pred_name: str) -> float:
    """
    Default penalty function ψ_k(u_k) for Algorithm 1, Line 24.
    
    These are reasonable defaults based on typical therapeutic peptide requirements:
    - Toxicity: Should be low (penalize u > 0.3, normalized scale)
    - Binding: Should be high (penalize u < 0.7, normalized scale)  
    - Half-life: Should be in optimal range (penalize deviations from [0.5, 0.8])
    
    The penalty uses squared distance from threshold/target, which provides:
    - Smooth gradients for optimization
    - Stronger penalty for larger violations
    
    You can override these by providing custom penalty functions in the constraints dict.
    
    Args:
        u: Normalized predictor value u_k(x) = F_k(f_k(x)) in [0, 1]
        pred_name: Name of the predictor (used to determine penalty type)
        
    Returns:
        Penalty value (nonnegative, 0 if constraint satisfied)
    """
    if 'toxicity' in pred_name.lower():
        # Penalize toxicity > 0.3 (normalized scale)
        # Threshold chosen to allow some toxicity but penalize high values
        threshold = 0.3
        if u > threshold:
            return (u - threshold) ** 2
        return 0.0
    
    elif 'binding' in pred_name.lower():
        # Penalize binding < 0.7 (normalized scale)
        # Threshold chosen to require reasonably strong binding
        threshold = 0.7
        if u < threshold:
            return (threshold - u) ** 2
        return 0.0
    
    elif 'halflife' in pred_name.lower() or 'half_life' in pred_name.lower():
        # Target range: [0.5, 0.8] (normalized scale)
        # Penalize deviations from this optimal range
        target_min, target_max = 0.5, 0.8
        if u < target_min:
            return (target_min - u) ** 2
        elif u > target_max:
            return (u - target_max) ** 2
        return 0.0
    
    else:
        # Unknown predictor type: no penalty
        return 0.0


def compute_preference_score(
    x: str,
    predictors: Dict,
    use_linear_preferences: bool = False,
    prompt: Optional[str] = None,
    text_encoder: Optional[any] = None,
    preference_net: Optional[any] = None,
    eta: Optional[any] = None
) -> float:
    """
    Algorithm 1, Line 23: Compute prompt-dependent preference score R(x;t) = G_η(t)(u(x)).
    
    This implements the soft language preference term that depends on the prompt t.
    The preference functional G_η is parameterized by η = g_ψ*(e) where:
    - e = E_text(t) is the text embedding
    - g_ψ* is the trained preference network
    - u(x) = [u_1(x), ..., u_K(x)] are normalized predictor coordinates
    
    Two modes:
    - Linear: R(x;t) = η^T u (simple dot product)
    - Nonlinear: R(x;t) = G_η(u) (quadratic or learned function)
    
    Args:
        x: Peptide SMILES string
        predictors: Dict of predictor objects
        use_linear_preferences: If True, use linear functional η^T u instead of nonlinear
        prompt: Natural language prompt t (required if eta not provided)
        text_encoder: Frozen text encoder E_text (required if eta not provided)
        preference_net: Trained preference network g_ψ* (required if eta not provided)
        eta: Optional pre-computed preference parameters (performance optimization)
        
    Returns:
        Preference score R(x;t) (higher is better for the prompt)
    """
    # Compute eta if not provided (for backward compatibility)
    if eta is None:
        if prompt is None or text_encoder is None or preference_net is None:
            return 0.0
        prompt_embedding = text_encoder.encode(prompt)
        if isinstance(prompt_embedding, torch.Tensor):
            prompt_embedding = prompt_embedding.unsqueeze(0)
        eta = preference_net(prompt_embedding)
    
    # Compute normalized predictor coordinates u(x)
    u = []
    for pred_name, predictor in predictors.items():
        raw_value = predictor.predict(x)
        normalized = predictor.normalize(raw_value)
        u.append(normalized)
    
    u = np.array(u)
    
    # Compute preference functional
    if use_linear_preferences:
        # Linear: R(x;t) = η^T u
        if isinstance(eta, torch.Tensor):
            eta_np = eta.detach().cpu().numpy().flatten()
        else:
            eta_np = np.array(eta).flatten()
        
        if len(eta_np) == len(u):
            score = np.dot(eta_np, u)
        else:
            min_len = min(len(eta_np), len(u))
            score = np.dot(eta_np[:min_len], u[:min_len])
    else:
        # nonlinear preference functional
        score = apply_nonlinear_preference(u, eta)
    
    return float(score)


def apply_nonlinear_preference(
    u: np.ndarray,
    eta: any
) -> float:
    """
    Apply nonlinear preference functional G_η(u).
    
    This can be a neural network, polynomial, or other nonlinear function.
    For now, implements a simple MLP-like quadratic transformation.
    """
    if isinstance(eta, torch.Tensor):
        eta = eta.detach().cpu().numpy()
    
    
    if len(eta.shape) == 1:
        # Assume eta defines a quadratic form: u^T diag(eta) u + linear term
        quadratic = np.sum(eta[:len(u)] * u ** 2)
        linear = np.sum(eta[len(u):2*len(u)] if len(eta) >= 2*len(u) else eta[len(u):] * u)
        return quadratic + linear
    else:
        # Matrix form: u^T eta u
        if len(eta.shape) == 2 and eta.shape[0] == len(u) and eta.shape[1] == len(u):
            return float(u @ eta @ u)
        else:
            eta_flat = eta.flatten() if len(eta.shape) > 1 else eta
            return float(np.dot(eta_flat[:len(u)], u))
