"""
Evaluation: Ablations on Preference and Constraint Formulation

This module implements experiments from Section 4.4 to isolate contributions of
different LaPep components.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
import torch

from lapep.potential import compute_potential
from lapep.sampler import sample_peptide
from lapep.kernel import compute_transition_kernel


def run_ablation_studies(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    num_samples: int = 500,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Run comprehensive ablation studies on LaPep components.
    
    Tests:
    1. Linear vs nonlinear preferences
    2. Conservative reweighting vs direct guidance
    3. Constraint strength variations
    4. Individual component contributions
    
    Returns:
        Dict mapping ablation names to evaluation metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    results = {}
    
    # Full LaPep (baseline)
    print("Evaluating full LaPep...")
    results['full_lapep'] = _evaluate_configuration(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_samples,
        use_linear_preferences=False,
        use_conservative_reweighting=True,
        constraint_strength=1.0
    )
    
    # Ablation 1: Linear preferences
    print("Evaluating linear preferences ablation...")
    results['linear_preferences'] = _evaluate_configuration(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_samples,
        use_linear_preferences=True,
        use_conservative_reweighting=True,
        constraint_strength=1.0
    )
    
    # Ablation 2: No conservative reweighting
    print("Evaluating no conservative reweighting ablation...")
    results['no_conservative_reweighting'] = _evaluate_configuration(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_samples,
        use_linear_preferences=False,
        use_conservative_reweighting=False,
        constraint_strength=1.0
    )
    
    # Ablation 3: Reduced constraint strength
    print("Evaluating reduced constraint strength...")
    results['reduced_constraint_strength'] = _evaluate_configuration(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_samples,
        use_linear_preferences=False,
        use_conservative_reweighting=True,
        constraint_strength=0.5
    )
    
    # Ablation 4: No language (predictor-only)
    print("Evaluating predictor-only (no language)...")
    results['no_language'] = _evaluate_configuration(
        base_generator,
        None,
        None,
        predictors,
        None,
        num_samples,
        use_linear_preferences=False,
        use_conservative_reweighting=True,
        constraint_strength=1.0
    )
    
    return results


def _evaluate_configuration(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: Optional[str],
    num_samples: int,
    use_linear_preferences: bool,
    use_conservative_reweighting: bool,
    constraint_strength: float
) -> Dict[str, float]:
    """
    Evaluate a specific configuration of LaPep components.
    
    Returns:
        Dict with constraint satisfaction, prompt alignment, and stability metrics
    """
    samples = []
    
    for _ in range(num_samples):
        if use_conservative_reweighting:
            # Use LaPep sampling
            peptide = sample_peptide(
                base_generator,
                prompt=prompt,
                predictors=predictors,
                constraints={'strength': constraint_strength},
                text_encoder=text_encoder,
                preference_net=preference_net,
                use_linear_preferences=use_linear_preferences
            )
        else:
            # Use non-conservative direct guidance
            peptide = _sample_with_direct_guidance(
                base_generator,
                text_encoder,
                preference_net,
                predictors,
                prompt,
                constraint_strength
            )
        samples.append(peptide)
    
    # Compute metrics
    metrics = {}
    
    # Constraint satisfaction
    constraint_satisfaction = _compute_constraint_satisfaction(
        samples, predictors, constraint_strength
    )
    metrics['constraint_satisfaction'] = constraint_satisfaction
    
    # Prompt alignment (if language is used)
    if prompt is not None and text_encoder is not None and preference_net is not None:
        prompt_alignment = _compute_prompt_alignment(
            samples, text_encoder, preference_net, predictors, prompt
        )
        metrics['prompt_alignment'] = prompt_alignment
    else:
        metrics['prompt_alignment'] = 0.0
    
    # Stability (path variance)
    stability = _compute_stability_metric(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        use_conservative_reweighting,
        constraint_strength
    )
    metrics['stability'] = stability
    
    return metrics


def _sample_with_direct_guidance(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    constraint_strength: float,
    num_steps: int = 50
) -> str:
    """
    Sample using non-conservative direct guidance (baseline for ablation).
    
    This applies language and predictor signals directly to proposal probabilities
    without the conservative reweighting scheme. Instead of using the LaPep kernel
    that ensures path independence, we directly reweight proposals by exp(-U(x;t)).
    
    This is the "naive guidance" baseline that applies signals directly without
    conservative reweighting.
    """
    import numpy as np
    from lapep.potential import compute_potential
    
    # Pre-compute eta for efficiency
    if prompt is not None and text_encoder is not None and preference_net is not None:
        prompt_embedding = text_encoder.encode(prompt)
        if isinstance(prompt_embedding, torch.Tensor):
            if len(prompt_embedding.shape) == 1:
                prompt_embedding = prompt_embedding.unsqueeze(0)
        eta = preference_net(prompt_embedding)
    else:
        eta = None
    
    constraints = {'strength': constraint_strength}
    
    # Start from initial state
    X_s = base_generator.sample_initial_state()
    
    # Sample for num_steps
    for s in range(num_steps):
        tau = s
        
        # Get neighbors (candidate next states)
        neighbors = base_generator.get_neighbors(X_s)
        C_s = neighbors + [X_s]  # Include current state
        
        # Compute potential for current state
        U_X_s = compute_potential(
            X_s, prompt, predictors, text_encoder, preference_net,
            constraints, use_linear_preferences=False, eta=eta, language_weight=1.0
        )
        
        # Compute unnormalized weights using direct guidance (non-conservative)
        # Weight = proposal_prob * exp(-U(x';t)) / exp(-U(x;t))
        # This directly applies the potential without conservative reweighting
        log_weights = []
        for x_prime in C_s:
            # Base proposal probability
            log_b_theta = np.log(base_generator.proposal_probability(x_prime, X_s, tau) + 1e-10)
            
            # Direct potential-based reweighting (non-conservative)
            U_x_prime = compute_potential(
                x_prime, prompt, predictors, text_encoder, preference_net,
                constraints, use_linear_preferences=False, eta=eta, language_weight=1.0
            )
            
            # Direct guidance: weight ∝ proposal * exp(-(U(x') - U(x)))
            # This is NOT conservative - it doesn't ensure path independence
            log_weight = log_b_theta - (U_x_prime - U_X_s)
            log_weights.append(log_weight)
        
        # Normalize and sample
        log_weights = np.array(log_weights)
        log_weights = log_weights - np.max(log_weights)  # Numerical stability
        weights = np.exp(log_weights)
        weights = weights / (weights.sum() + 1e-10)
        
        # Sample next state
        idx = np.random.choice(len(C_s), p=weights)
        X_s = C_s[idx]
    
    return X_s


def _compute_constraint_satisfaction(
    samples: List[str],
    predictors: Dict,
    constraint_strength: float
) -> float:
    """
    Compute the fraction of samples that satisfy hard predictor constraints.
    """
    if not predictors:
        return 1.0
    
    satisfied_count = 0
    
    for sample in samples:
        satisfies_all = True
        
        for pred_name, predictor in predictors.items():
            value = predictor.predict(sample)
            normalized = predictor.normalize(value)
            
            # Check if constraint is satisfied
            # (Assuming constraints are thresholds, e.g., toxicity < 0.3)
            if 'toxicity' in pred_name.lower():
                if normalized > 0.3 * constraint_strength:
                    satisfies_all = False
                    break
            elif 'binding' in pred_name.lower():
                if normalized < 0.7 * constraint_strength:
                    satisfies_all = False
                    break
        
        if satisfies_all:
            satisfied_count += 1
    
    return satisfied_count / len(samples) if samples else 0.0


def _compute_prompt_alignment(
    samples: List[str],
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str
) -> float:
    """
    Compute alignment between generated samples and the prompt.
    
    Uses the preference network to score samples under the prompt.
    """
    from lapep.potential import compute_preference_score
    
    # Pre-compute eta once for efficiency
    prompt_embedding = text_encoder.encode(prompt)
    if isinstance(prompt_embedding, torch.Tensor):
        if len(prompt_embedding.shape) == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
    eta = preference_net(prompt_embedding)
    
    # Score samples using preference functional R(x;t) = G_η(u(x))
    scores = []
    for sample in samples:
        # Compute preference score R(x;t) using normalized predictor coordinates
        score = compute_preference_score(
            sample,
            predictors=predictors,
            use_linear_preferences=False,
            eta=eta
        )
        scores.append(score)
    
    # Return mean alignment score (normalized to [0, 1] if needed)
    mean_score = np.mean(scores) if scores else 0.0
    # Normalize to reasonable range (assuming scores are typically in [-10, 10])
    # Use sigmoid-like normalization for better interpretability
    normalized = 1.0 / (1.0 + np.exp(-mean_score / 2.0))  # Soft sigmoid
    return float(normalized)


def _compute_stability_metric(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: Optional[str],
    use_conservative: bool,
    constraint_strength: float
) -> float:
    """
    Compute stability metric (inverse of path variance).
    
    Lower variance = higher stability.
    """
    from .circulation import evaluate_path_variance
    
    if prompt is None:
        prompt = ""
    
    path_var_results = evaluate_path_variance(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_trajectories=50,
        trajectory_length=20
    )
    
    path_variance = path_var_results.get('mean_final_variance', 0.0)
    
    # Convert to stability (inverse, normalized)
    stability = 1.0 / (1.0 + path_variance)
    
    return stability


def format_ablation_table(
    results: Dict[str, Dict[str, float]]
) -> str:
    """
    Format ablation results as LaTeX table matching Table 4 in the paper.
    """
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Ablation study on LaPep components.}",
        "\\label{tab:ablations}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Ablation} & \\textbf{Constraint Satisfaction} & ",
        "\\textbf{Prompt Alignment} & \\textbf{Stability} \\\\",
        "\\midrule"
    ]
    
    ablation_names = {
        'full_lapep': 'Full LaPep',
        'linear_preferences': 'Linear Preferences',
        'no_conservative_reweighting': 'No Conservative Reweighting',
        'reduced_constraint_strength': 'Reduced Constraint Strength',
        'no_language': 'No Language (Predictor-only)'
    }
    
    for ablation_key, ablation_display in ablation_names.items():
        if ablation_key in results:
            metrics = results[ablation_key]
            constraint = metrics.get('constraint_satisfaction', 0.0)
            alignment = metrics.get('prompt_alignment', 0.0)
            stability = metrics.get('stability', 0.0)
            
            lines.append(
                f"{ablation_display} & {constraint:.3f} & {alignment:.3f} & {stability:.3f} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def analyze_ablation_contributions(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze the contribution of each component by comparing to full LaPep.
    
    Returns:
        Dict with relative performance drops for each ablation
    """
    if 'full_lapep' not in results:
        return {}
    
    baseline = results['full_lapep']
    contributions = {}
    
    for ablation_name, ablation_metrics in results.items():
        if ablation_name == 'full_lapep':
            continue
        
        contribution = {}
        for metric_name in baseline:
            if metric_name in ablation_metrics:
                baseline_val = baseline[metric_name]
                ablation_val = ablation_metrics[metric_name]
                
                if baseline_val > 0:
                    relative_drop = (baseline_val - ablation_val) / baseline_val
                else:
                    relative_drop = 0.0
                
                contribution[metric_name] = {
                    'baseline': baseline_val,
                    'ablation': ablation_val,
                    'relative_drop': relative_drop
                }
        
        contributions[ablation_name] = contribution
    
    return contributions

