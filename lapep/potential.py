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
    use_linear_preferences: bool = False
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
        
    Returns:
        Scalar potential value (lower is better)
    """
    # Compute constraint penalty Ψ(x)
    psi = compute_constraint_penalty(x, predictors, constraints)
    
    # Compute language preference R(x;t)
    if prompt is not None and text_encoder is not None and preference_net is not None:
        r = compute_preference_score(
            x, prompt, predictors, text_encoder, preference_net, use_linear_preferences
        )
    else:
        r = 0.0
    
    # LaPep potential
    U = -r + psi
    
    return U


def compute_constraint_penalty(
    x: str,
    predictors: Dict,
    constraints: Optional[Dict] = None
) -> float:
    """
    Compute hard constraint penalty Ψ(x) = Σ_k λ_k ψ_k(u_k(x)).
    
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
        # Get raw prediction
        raw_value = predictor.predict(x)
        
        # Normalize to [0, 1] using empirical CDF
        normalized = predictor.normalize(raw_value)
        
        # Get weight (default to 1.0)
        weight = weights.get(pred_name, 1.0) * strength
        
        # Get penalty function (default to squared distance from target)
        penalty_fn = penalties.get(pred_name, default_penalty_function)
        
        # Compute penalty
        penalty = penalty_fn(normalized, pred_name)
        total_penalty += weight * penalty
    
    return total_penalty


def default_penalty_function(u: float, pred_name: str) -> float:
    """
    Default penalty function based on predictor type.
    
    For toxicity: penalize high values (u > threshold)
    For binding: penalize low values (u < threshold)
    For half-life: penalize deviations from target range
    """
    if 'toxicity' in pred_name.lower():
        # Penalize toxicity > 0.3
        threshold = 0.3
        if u > threshold:
            return (u - threshold) ** 2
        return 0.0
    
    elif 'binding' in pred_name.lower():
        # Penalize binding < 0.7
        threshold = 0.7
        if u < threshold:
            return (threshold - u) ** 2
        return 0.0
    
    elif 'halflife' in pred_name.lower() or 'half_life' in pred_name.lower():
        # Target range: [0.5, 0.8]
        target_min, target_max = 0.5, 0.8
        if u < target_min:
            return (target_min - u) ** 2
        elif u > target_max:
            return (u - target_max) ** 2
        return 0.0
    
    else:
        # Generic: no penalty
        return 0.0


def compute_preference_score(
    x: str,
    prompt: str,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    use_linear_preferences: bool = False
) -> float:
    """
    Compute language preference score R(x;t) = G_η(t)(u(x)).
    
    Args:
        x: Peptide SMILES string
        prompt: Natural language prompt
        predictors: Dict of predictor objects
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        use_linear_preferences: If True, use linear functional instead of nonlinear
        
    Returns:
        Preference score (higher is better)
    """
    # Encode prompt
    prompt_embedding = text_encoder.encode(prompt)
    
    # Get preference parameters
    if isinstance(prompt_embedding, torch.Tensor):
        prompt_embedding = prompt_embedding.unsqueeze(0)  # Add batch dimension
    
    eta = preference_net(prompt_embedding)
    
    # Compute normalized predictor coordinates u(x)
    u = []
    for pred_name, predictor in predictors.items():
        raw_value = predictor.predict(x)
        normalized = predictor.normalize(raw_value)
        u.append(normalized)
    
    u = np.array(u)
    
    # Apply preference functional
    if use_linear_preferences:
        # Linear: R(x;t) = η^T u
        if isinstance(eta, torch.Tensor):
            eta_np = eta.detach().cpu().numpy().flatten()
        else:
            eta_np = np.array(eta).flatten()
        
        # Ensure dimensions match
        if len(eta_np) == len(u):
            score = np.dot(eta_np, u)
        else:
            # Use first len(u) components or pad u
            min_len = min(len(eta_np), len(u))
            score = np.dot(eta_np[:min_len], u[:min_len])
    else:
        # Nonlinear preference functional
        score = apply_nonlinear_preference(u, eta)
    
    return float(score)


def apply_nonlinear_preference(
    u: np.ndarray,
    eta: any
) -> float:
    """
    Apply nonlinear preference functional G_η(u).
    
    This can be a neural network, polynomial, or other nonlinear function.
    For now, implements a simple MLP-like transformation.
    """
    if isinstance(eta, torch.Tensor):
        eta = eta.detach().cpu().numpy()
    
    # Simple nonlinear transformation
    # In practice, this would be a learned neural network
    # For now, use a quadratic form with learned parameters
    
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
            # Fallback to linear
            eta_flat = eta.flatten() if len(eta.shape) > 1 else eta
            return float(np.dot(eta_flat[:len(u)], u))
