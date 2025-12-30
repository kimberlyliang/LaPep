"""
Evaluation: Effect of Language Conditioning on Generated Peptide Distributions

This module implements experiments from Section 4.1 to assess whether natural-language
conditioning produces systematic changes in generated peptides beyond hard predictor constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch

from lapep.potential import compute_potential
from lapep.sampler import sample_peptide


def evaluate_language_conditioning_effect(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict[str, any],
    prompts: List[str],
    num_samples: int = 1000,
    predictor_constraints: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Evaluate the effect of language conditioning on generated peptide distributions.
    
    Args:
        base_generator: Base generator with proposal kernel b_θ
        text_encoder: Frozen text encoder E_text
        preference_net: Trained preference network g_ψ
        predictors: Dict mapping predictor names to predictor objects
        prompts: List of natural language prompts to test
        num_samples: Number of peptides to generate per prompt
        predictor_constraints: Optional dict of constraint weights and penalties
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping prompt names to arrays of normalized predictor coordinates
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    results = {}
    
    # First, generate with predictor-only conditioning (baseline)
    print("Generating predictor-only baseline samples...")
    baseline_samples = _generate_predictor_only(
        base_generator,
        predictors,
        predictor_constraints,
        num_samples
    )
    baseline_coords = _compute_predictor_coordinates(baseline_samples, predictors)
    results['predictor_only'] = baseline_coords
    
    # Generate samples for each language prompt
    for prompt in prompts:
        print(f"Generating samples for prompt: {prompt}")
        samples = _generate_language_conditioned(
            base_generator,
            text_encoder,
            preference_net,
            predictors,
            prompt,
            predictor_constraints,
            num_samples
        )
        coords = _compute_predictor_coordinates(samples, predictors)
        results[prompt] = coords
    
    return results


def _generate_predictor_only(
    base_generator,
    predictors: Dict,
    constraints: Optional[Dict],
    num_samples: int
) -> List[str]:
    """Generate samples using only predictor constraints (no language)."""
    samples = []
    for _ in range(num_samples):
        # Sample using only predictor-based potential
        peptide = sample_peptide(
            base_generator,
            prompt=None,  # No language prompt
            predictors=predictors,
            constraints=constraints,
            text_encoder=None,
            preference_net=None
        )
        samples.append(peptide)
    return samples


def _generate_language_conditioned(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    constraints: Optional[Dict],
    num_samples: int
) -> List[str]:
    """Generate samples using both language and predictor conditioning."""
    samples = []
    for _ in range(num_samples):
        peptide = sample_peptide(
            base_generator,
            prompt=prompt,
            predictors=predictors,
            constraints=constraints,
            text_encoder=text_encoder,
            preference_net=preference_net
        )
        samples.append(peptide)
    return samples


def _compute_predictor_coordinates(
    samples: List[str],
    predictors: Dict[str, any]
) -> np.ndarray:
    """
    Compute normalized predictor coordinates u(x) for each sample.
    
    Returns:
        Array of shape (num_samples, num_predictors) with values in [0, 1]
    """
    coords = []
    for sample in samples:
        sample_coords = []
        for pred_name, predictor in predictors.items():
            # Get raw prediction
            raw_value = predictor.predict(sample)
            # Normalize using empirical CDF (percentile transform)
            normalized = predictor.normalize(raw_value)
            sample_coords.append(normalized)
        coords.append(sample_coords)
    
    return np.array(coords)


def compute_distribution_statistics(
    coords: np.ndarray,
    predictor_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for predictor coordinate distributions.
    
    Args:
        coords: Array of shape (num_samples, num_predictors)
        predictor_names: List of predictor names
        
    Returns:
        Dict mapping predictor names to statistics (mean, std, percentiles)
    """
    stats = {}
    for i, pred_name in enumerate(predictor_names):
        values = coords[:, i]
        stats[pred_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    return stats


def compare_distributions(
    baseline_coords: np.ndarray,
    conditioned_coords: np.ndarray,
    predictor_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline and language-conditioned distributions.
    
    Returns:
        Dict with statistical tests and effect sizes for each predictor
    """
    comparisons = {}
    
    for i, pred_name in enumerate(predictor_names):
        baseline_vals = baseline_coords[:, i]
        conditioned_vals = conditioned_coords[:, i]
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(baseline_vals) + np.var(conditioned_vals)) / 2
        )
        if pooled_std > 0:
            cohens_d = (np.mean(conditioned_vals) - np.mean(baseline_vals)) / pooled_std
        else:
            cohens_d = 0.0
        
        # Kolmogorov-Smirnov test statistic
        from scipy import stats
        ks_stat, ks_pvalue = stats.ks_2samp(baseline_vals, conditioned_vals)
        
        comparisons[pred_name] = {
            'mean_shift': float(np.mean(conditioned_vals) - np.mean(baseline_vals)),
            'cohens_d': float(cohens_d),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'baseline_mean': float(np.mean(baseline_vals)),
            'conditioned_mean': float(np.mean(conditioned_vals))
        }
    
    return comparisons


def format_results_table(
    results: Dict[str, np.ndarray],
    predictor_names: List[str]
) -> str:
    """
    Format evaluation results as a LaTeX table matching Table 1 in the paper.
    
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Effect of language conditioning under fixed predictor constraints.}",
        "\\label{tab:language_effect}",
        "\\begin{tabular}{l" + "c" * len(predictor_names) + "}",
        "\\toprule",
        "\\textbf{Conditioning Scheme} & " + " & ".join([
            f"\\textbf{{{name.replace('_', ' ').title()} Percentile}}"
            for name in predictor_names
        ]) + " \\\\",
        "\\midrule"
    ]
    
    for scheme_name, coords in results.items():
        stats = compute_distribution_statistics(coords, predictor_names)
        mean_strs = [
            f"{stats[pred]['mean']:.3f}"
            for pred in predictor_names
        ]
        scheme_display = scheme_name.replace('_', ' ').title()
        if 'predictor' in scheme_name.lower() and 'only' in scheme_name.lower():
            scheme_display = "Predictor-only Conditioning"
        elif 'lapep' in scheme_name.lower():
            scheme_display = f"LaPep ({scheme_name.replace('lapep_', '').replace('_', ' ').title()})"
        
        lines.append(f"{scheme_display} & " + " & ".join(mean_strs) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

